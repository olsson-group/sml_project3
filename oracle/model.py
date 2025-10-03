from typing import List, NamedTuple
import torch


class BatchSequential(torch.nn.Module):
    def __init__(self, layers: List[torch.nn.Module]):
        super().__init__()
        self.layers = torch.nn.ModuleList(layers)

    def forward(self, batch: "Batch") -> "Batch":
        for layer in self.layers:
            batch = layer(batch)
        return batch


class Batch(NamedTuple):
    equivariant_node_features: torch.Tensor
    invariant_node_features: torch.Tensor
    edge_index: torch.Tensor
    pos: torch.Tensor
    atom_idx: torch.Tensor
    batch: torch.Tensor
    edge_dist: torch.Tensor
    edge_dir: torch.Tensor


class Oracle(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        
        self.device = next(model.parameters()).device
        self.dtype = next(model.parameters()).dtype

    def forward(self, batch):
        t = torch.ones_like(batch.atom_idx)
        t = t.to(device=self.device, dtype=self.dtype)

        pos = batch.pos.to(self.device, dtype=self.dtype)

        equivariant_node_features = torch.zeros(batch.pos.shape[0], self.model.n_features, 3).to(self.device, dtype=self.dtype)
        invariant_node_features = torch.zeros(batch.pos.shape[0], self.model.n_features).to(self.device, dtype=self.dtype)

        edge_index = batch.edge_index.to(self.device)
        atom_index = batch.atom_idx.to(self.device)

        batch_attr = batch.batch.to(self.device)

        edge_dist = torch.zeros(batch.edge_index.shape[1]).to(self.device, dtype=self.dtype)
        edge_dir = torch.zeros(batch.edge_index.shape[1], 3).to(self.device, dtype=self.dtype)
        
        batch = Batch(
            equivariant_node_features=equivariant_node_features,
            invariant_node_features=invariant_node_features,
            edge_index=edge_index,
            pos=pos,
            atom_idx=atom_index,
            batch=batch_attr,
            edge_dist=edge_dist,
            edge_dir=edge_dir,
        )
        return self.model(t, batch)

    def get_energy(self, batch):
        return self.forward(batch).detach().cpu().numpy()


