import pickle
import torch
import numpy as np
from typing import Optional
import json
import score_matrices as sm
from smoothing import smooth_diag_numpy



def save(filename:str,data):
    with open(filename,"wb") as f:
        pickle.dump(data,f)
def load(filename):
    with open(filename,"rb") as f:
        return pickle.load(f)
def json_save(name,data):
    with open(name, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
def json_load(data):
    with open(data, "r", encoding="utf-8") as f:
        loaded_data = json.load(f)
        return loaded_data
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
def build_similarity_matrix(emb1: np.ndarray, emb2: np.ndarray, use_smoothing: bool = True) -> np.ndarray:
    """Build and optionally smooth the residue-residue similarity matrix."""

    similarity_matrix = sm.compute_similarity_matrix2(emb1, emb2)
    if use_smoothing:
        similarity_matrix = smooth_diag_numpy(similarity_matrix)
    return similarity_matrix
def get_device(device_name: Optional[str] = None) -> torch.device:
    """Return the requested device, or automatically select CUDA when available."""
    if device_name is not None:
        return torch.device(device_name)
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def normalize(matrix):
    sm = torch.tensor(matrix, dtype=torch.float32)
    sm= torch.exp(sm)
    columns_avg = torch.sum(sm, 0) / sm.shape[0]
    rows_avg = torch.sum(sm, 1) / sm.shape[1]

    columns_std = torch.std(sm, 0)
    rows_std = torch.std(sm, 1)

    z_rows = (sm - rows_avg.unsqueeze(1)) / rows_std.unsqueeze(1)
    z_columns = (sm - columns_avg) / columns_std
    res=(z_rows+z_columns)/2

    return res.cpu().numpy()