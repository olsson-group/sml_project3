import data
import mlops
import utils
from train_painn import *
from vis import nglview

cfm = mlops.load("results/model/painn_4999.pkl")

dataset = data.Pentene1Dataset("data")
example_batch = utils.get_example_batch(dataset, 10)

traj = cfm.sample(example_batch)
__import__("pdb").set_trace()  # TODO delme

nglview(traj, torch.tensor(data.PENTENE_ATOMS))
__import__("pdb").set_trace()  # TODO delme
