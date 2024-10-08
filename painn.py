import warnings

import numpy as np
import torch
from torch_scatter import scatter


class Painn(torch.nn.Module):
    def __init__(
        self,
        n_features=128,
        n_features_in=None,
        n_features_out=None,
        n_layers=5,
        length_scale=10.0,
    ):
        n_features_in = n_features_in or n_features
        n_features_out = n_features_out or n_features
        self.n_features = n_features

        super().__init__()
        self.embedding = torch.nn.Embedding(15, n_features - 1)
        layers = []

        for l in range(n_layers):
            layers.append(
                Message(
                    n_features=n_features_in if l == 0 else n_features,
                    length_scale=length_scale,
                )
            )
            layers.append(Update(n_features))

        layers.append(
            Readout(
                n_features,
                n_features_out,
            )
        )
        self.layers = torch.nn.Sequential(*layers)

    def forward(self, t, batch):
        batch = batch.clone()
        t_batch = t[batch.batch, None]
        atom_embeddings = self.embedding(batch.atom_idx)
        batch.invariant_node_features = torch.concatenate(
            [atom_embeddings, t_batch], dim=-1
        )

        r = batch.pos[batch.edge_index[0]] - batch.pos[batch.edge_index[1]]
        edge_dist = r.norm(dim=-1)
        edge_dir = r / (1 + edge_dist.unsqueeze(-1))

        batch.edge_dist = edge_dist
        batch.edge_dir = edge_dir

        batch.equivariant_node_features = torch.zeros(
            batch.pos.shape[0], self.n_features, 3
        )

        return self.layers(batch)


class Message(torch.nn.Module):
    def __init__(
        self,
        n_features=64,
        length_scale=10.0,
    ):
        super().__init__()
        self.n_features = n_features

        self.positional_encoder = PositionalEncoder(n_features, max_length=length_scale)

        phi_in_features = n_features
        self.phi = MLP(phi_in_features, n_features, 3 * n_features)
        self.w = MLP(n_features, n_features, 3 * n_features)

    def forward(self, batch):
        src_node = batch.edge_index[0]
        dst_node = batch.edge_index[1]

        in_features = torch.cat(
            [
                batch.invariant_node_features[src_node],
            ],
            dim=-1,
        )

        positional_encoding = self.positional_encoder(batch.edge_dist).to(
            in_features.device
        )

        gates, scale_edge_dir, ds = torch.split(
            self.phi(in_features) * self.w(positional_encoding),
            self.n_features,
            dim=-1,
        )
        gated_features = multiply_first_dim(
            gates, batch.equivariant_node_features[src_node]
        )
        scaled_edge_dir = multiply_first_dim(
            scale_edge_dir, batch.edge_dir.unsqueeze(1).repeat(1, self.n_features, 1)
        )

        dv = scaled_edge_dir + gated_features
        dv = scatter(dv, dst_node, dim=0)
        ds = scatter(ds, dst_node, dim=0)

        batch.equivariant_node_features = batch.equivariant_node_features + dv
        batch.invariant_node_features = batch.invariant_node_features + ds

        return batch


class PositionalEncoder(torch.nn.Module):
    def __init__(self, dim, max_length):
        super().__init__()
        assert dim % 2 == 0, "dim must be even for positional encoding for sin/cos"

        self.dim = dim
        self.max_length = max_length
        self.max_rank = dim // 2

    def forward(self, x):
        encodings = [
            self.positional_encoding(x, rank) for rank in range(1, self.max_rank + 1)
        ]
        encodings = torch.cat(
            encodings,
            axis=1,
        )
        return encodings

    def positional_encoding(self, x, rank):
        sin = torch.sin(x / self.max_length * rank * np.pi)
        cos = torch.cos(x / self.max_length * rank * np.pi)
        return torch.stack((cos, sin), axis=1)


class Update(torch.nn.Module):
    def __init__(self, n_features=128):
        super().__init__()
        self.u = EquivariantLinear(n_features, n_features)
        self.v = EquivariantLinear(n_features, n_features)
        self.n_features = n_features
        self.mlp = MLP(2 * n_features, n_features, 3 * n_features)

    def forward(self, batch):
        v = batch.equivariant_node_features
        s = batch.invariant_node_features

        vv = self.v(v)
        uv = self.u(v)

        vv_norm = vv.norm(dim=-1)
        vv_squared_norm = vv_norm**2

        mlp_in = torch.cat([vv_norm, s], dim=-1)

        gates, scale_squared_norm, add_invariant_features = torch.split(
            self.mlp(mlp_in), self.n_features, dim=-1
        )

        dv = multiply_first_dim(uv, gates)
        ds = vv_squared_norm * scale_squared_norm + add_invariant_features

        batch.invariant_node_features = batch.invariant_node_features + ds
        batch.equivariant_node_features = batch.equivariant_node_features + dv

        return batch


class Readout(torch.nn.Module):
    def __init__(self, n_features=128, n_features_out=13):
        super().__init__()
        self.mlp = MLP(n_features, n_features, 2 * n_features_out)
        self.V = EquivariantLinear(n_features, n_features_out)
        self.n_features_out = n_features_out

    def forward(self, batch):
        invariant_node_features_out, gates = torch.split(
            self.mlp(batch.invariant_node_features), self.n_features_out, dim=-1
        )

        equivariant_node_features = self.V(batch.equivariant_node_features)
        equivariant_node_features_out = multiply_first_dim(
            equivariant_node_features, gates
        )

        batch.invariant_node_features = invariant_node_features_out
        batch.equivariant_node_features = equivariant_node_features_out
        enf = batch.equivariant_node_features
        return enf[:, 0, :]


def multiply_first_dim(w, x):
    with warnings.catch_warnings(record=True):
        return (w.T * x.T).T


class EquivariantLinear(torch.nn.Module):
    def __init__(self, n_features_in, n_features_out):
        super().__init__()
        self.linear = torch.nn.Linear(n_features_in, n_features_out, bias=False)

    def forward(self, x):
        return self.linear(x.swapaxes(-1, -2)).swapaxes(-1, -2)


class MLP(torch.nn.Module):
    def __init__(self, f_in, f_hidden, f_out):
        super().__init__()

        self.f_out = f_out

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(f_in, f_hidden),
            torch.nn.LayerNorm(f_hidden),
            torch.nn.SiLU(),
            torch.nn.Linear(f_hidden, f_hidden),
            torch.nn.LayerNorm(f_hidden),
            torch.nn.SiLU(),
            torch.nn.Linear(f_hidden, f_out),
        )

    def forward(self, x):
        return self.mlp(x)
