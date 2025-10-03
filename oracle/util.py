import torch
from oracle.model import Oracle


def load_jit_compiled(path='oracle/.weights/model.pt', device='cpu'):
    compiled_model = torch.jit.load(path)
    compiled_model = compiled_model.to(device=device)
    return compiled_model
    

def load_oracle(path='oracle/.weights/model.pt', device='cpu'):
    compiled_model = load_jit_compiled(path, device)
    oracle = Oracle(compiled_model)
    return oracle

