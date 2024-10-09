import torch
import torch_geometric as geom
from torchdyn.core import NeuralODE

from sml_project3 import utils


class CFM(torch.nn.Module):
    def __init__(self, score, basedistribution):
        super().__init__()
        self.score = score
        self.sigma = 0.01
        self.basedistribution = basedistribution

        self.setup_optimizer()

    def training_step(self, loss):
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

    def setup_optimizer(self, lr=0.001):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def forward(self, t, batch):
        batch = batch.clone()
        return self.score(t, batch)

    def get_loss(self, t, batch):
        corr_batch = batch.clone()
        x0 = self.basedistribution.sample(len(batch))
        x1 = batch.pos
        t_batch = t[batch.batch]

        xt = self.sample_conditional_pt(t_batch, x0, x1, batch_size=len(batch))
        corr_batch.pos = xt
        ut = self.compute_conditional_vector_field(x0, x1)
        vt = self.forward(t, corr_batch)

        norms = torch.norm(vt - ut, dim=1)
        loss = torch.mean(norms**2)
        return loss

    def sample_conditional_pt(self, t, x0, x1, batch_size):
        epsilon = self.basedistribution.sample(batch_size) * 0.01
        mu_t = (t * x1.T + (1 - t) * x0.T).T
        return mu_t + self.sigma * epsilon

    def compute_conditional_vector_field(self, x0, x1):
        return x1 - x0

    def sample_forward(self, t, x, *args, **kwargs):
        while True:
            t = t.repeat(len(x))
            return self.forward(t, x)

    def sample(self, example_batch):
        print("Sampling ... ")
        x0 = self.basedistribution.sample(len(example_batch)).reshape(
            example_batch.pos.shape
        )

        node = NeuralODE(
            SampleHandler(example_batch, self.sample_forward),
            solver="dopri5",
            sensitivity="adjoint",
            atol=1e-4,
            rtol=1e-4,
        )

        with torch.no_grad():
            traj = node.trajectory(x0, t_span=torch.linspace(0, 1, 100))
            return traj[-1].reshape(len(example_batch), -1, 3).numpy()


class SampleHandler:
    def __init__(self, batch, forward):
        self.forward = forward
        self.batch = batch

    def __call__(self, t, pos, *args, **kwargs):
        self.batch.pos = pos
        return self.forward(t, self.batch)
