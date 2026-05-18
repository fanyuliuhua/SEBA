import pickle
import torch
from typing import Optional

def save(filename:str,data):
    with open(filename,"wb") as f:
        pickle.dump(data,f)
def load(filename):
    with open(filename,"rb") as f:
        return pickle.load(f)
def get_plot_coords(points):
    if not points:
        return [], []
    x_coords = [c + 0.5 for r, c in points]
    y_coords = [r + 0.5 for r, c in points]
    return x_coords, y_coords
def compute(TP,FP,FN):
    if TP==0 and FP==0:
        return 0,0,0
    pr=TP/(TP+FP)
    rc=TP/(TP+FN)
    if pr+rc==0:
        return 0,0,0
    f1=2*(pr*rc)/(pr+rc)
    return pr,rc,f1

def get_device(device_name: Optional[str] = None) -> torch.device:
    """Return the requested device, or automatically select CUDA when available."""
    if device_name is not None:
        return torch.device(device_name)
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")