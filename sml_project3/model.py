import torch


class CFM(torch.nn.Module):
    def __init__(self, score, basedistribution):
        super().__init__()
        self.score = score
        self.sigma = 0.01
        self.basedistribution = basedistribution

        self.setup_optimizer()
        self.setup_lr_scheduler()

    def training_step(self, loss):
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

    def on_epoch_end(self, epoch_loss):
        self.lr_scheduler.step(epoch_loss)

    def setup_optimizer(self, lr=0.001):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def setup_lr_scheduler(self):
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-7
        )

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
        epsilon = self.basedistribution.sample(batch_size)# * 0.01
        mu_t = (t * x1.T + (1 - t) * x0.T).T
        return mu_t + self.sigma * epsilon

    def compute_conditional_vector_field(self, x0, x1):
        return x1 - x0

    def sample_batch(self, times, batch):
        n_steps = len(times)

        # simple Euler solver
        with torch.no_grad():
            traj = torch.zeros((n_steps, *batch.pos.shape), device=batch.pos.device)
            for i, t in enumerate(times):
                ts = torch.ones_like(batch.atom_idx)*t
                velocity = self(ts, batch)
                batch.pos = batch.pos + velocity*(times[i+1]-times[i]) if i < n_steps-1 else batch.pos
                traj[i] = batch.pos

        traj = traj.view(n_steps, -1, 15, 3)
        return traj
    

    def sample_batch_dlogp(self, times, batch):
        # TODO: implement dlogp sampling
        pass

 
    def sample(self, loader, n_steps=100, dlogp=False):
        # TODO: add dlogp to sampling.
        trajs = []
        for i, batch in enumerate(loader):
            times = torch.linspace(0, 1, n_steps)
            traj = self.sample_batch(times, batch)
            trajs.append(traj)
            print(f"Batch {i+1}/{len(loader)} sampled", end="\r")

        trajs = torch.cat(trajs, dim=1)
        return trajs
    