from datetime import datetime

from torch_scatter import scatter


def center_coordinates(x):
    com = x.mean(dim=1, keepdim=True).repeat(1, 15, 1)
    return x - com


def center_batch(batch):
    com = scatter(batch.x, batch.batch, dim=0, reduce="mean")
    batch.x -= com[batch.batch]
    return batch


class Timer:
    def __init__(self):
        self.start = datetime.now()

    def __str__(self):
        now = datetime.now()
        time_passed = now - self.start
        return str(time_passed).split(".")[0]
