import torch
import numpy as np
from scipy import spatial
from numba import njit

@njit
def consine_similiarity(x, y):
    num = np.dot(x, y)
    denom = np.linalg.norm(x) * np.linalg.norm(y)
    return num / denom


@njit(parallel=True)
def getEemCosSim(emb1, emb2):
    m1, n1 = emb1.shape
    m2, n2 = emb2.shape
    matrix = np.zeros((m1, m2))

    for i in range(m1):
        for j in range(m2):
            matrix[i, j] = consine_similiarity(emb1[i, :], emb2[j, :])

    return matrix

def compute_similarity_matrix(embedding1, embedding2, l=1, p=2):
    """ Take as input 2 sequence embeddings (at a residue level) and returns the similarity matrix
        with the signal enhancement based on Z-scores.

        :param embedding1: residues embedding representation for sequence 1
        :param embedding2: residues embedding representation for sequence 2
        :param l: scalar that can be use to regularize the similarity matrix (no effect with l=1)
        :param p: Minkowski distance order (ex. p=1:Manhattan, p=2:Euclidean)

        :type embedding1: pytorch tensor
        :type embedding2: pytorch tensor
        :type l: float
        :type p: integer
    """
    embedding1 = embedding1.astype(np.float32)
    embedding2 = embedding2.astype(np.float32)
    sm=getEemCosSim(embedding1,embedding2)

    #update
    sm = torch.tensor(sm, dtype=torch.float32)
    # sm = compute_similarity_matrix_plain(embedding1, embedding2, l=l, p=p)
    columns_avg = torch.sum(sm,0)/sm.shape[0]
    rows_avg = torch.sum(sm,1)/sm.shape[1]
    #
    columns_std = torch.std(sm,0)
    rows_std = torch.std(sm,1)
    #
    z_rows = (sm-rows_avg.unsqueeze(1))/rows_std.unsqueeze(1)
    z_columns = (sm-columns_avg)/columns_std
    ans=(z_rows+z_columns)/2
    ans=ans.cpu().numpy()
    return ans
    # return sm


def compute_similarity_matrix2(embedding1, embedding2, l=1, p=2):
    """ Take as input 2 sequence embeddings (at a residue level) and returns the similarity matrix
        with the signal enhancement based on Z-scores.

        :param embedding1: residues embedding representation for sequence 1
        :param embedding2: residues embedding representation for sequence 2
        :param l: scalar that can be use to regularize the similarity matrix (no effect with l=1)
        :param p: Minkowski distance order (ex. p=1:Manhattan, p=2:Euclidean)

        :type embedding1: pytorch tensor
        :type embedding2: pytorch tensor
        :type l: float
        :type p: integer
    """
    embedding1=torch.tensor(embedding1,dtype=torch.float32)
    embedding2=torch.tensor(embedding2,dtype=torch.float32)
    sm = compute_similarity_matrix_plain(embedding1, embedding2, l=l, p=p)
    columns_avg = torch.sum(sm,0)/sm.shape[0]
    rows_avg = torch.sum(sm,1)/sm.shape[1]

    columns_std = torch.std(sm,0)
    rows_std = torch.std(sm,1)

    z_rows = (sm-rows_avg.unsqueeze(1))/rows_std.unsqueeze(1)
    z_columns = (sm-columns_avg)/columns_std
    ans=(z_rows+z_columns)/2
    ans=ans.cpu().numpy()
    return ans

def compute_similarity_matrix3(embedding1, embedding2):
    """ Take as input 2 sequence embeddings (at a residue level) and returns the similarity matrix
        with the signal enhancement based on Z-scores.

        :param embedding1: residues embedding representation for sequence 1
        :param embedding2: residues embedding representation for sequence 2
        :param l: scalar that can be use to regularize the similarity matrix (no effect with l=1)
        :param p: Minkowski distance order (ex. p=1:Manhattan, p=2:Euclidean)

        :type embedding1: pytorch tensor
        :type embedding2: pytorch tensor
        :type l: float
        :type p: integer
    """
    eps=1e-8
    embedding1=torch.tensor(embedding1,dtype=torch.float32)
    embedding2=torch.tensor(embedding2,dtype=torch.float32)

    # A_centered = (embedding1 + embedding1.mean(dim=1, keepdim=True))
    # B_centered = (embedding2 + embedding2.mean(dim=1, keepdim=True))
    A_centered = embedding1
    B_centered = embedding2

    A_norm = torch.norm(A_centered, dim=1, keepdim=True)  # (len1,1)
    B_norm = torch.norm(B_centered, dim=1, keepdim=True)

    # A_norm = A_norm.clamp(min=eps)
    # B_norm = B_norm.clamp(min=eps)

    # 归一化
    A_normalized = A_centered/A_norm
    B_normalized = B_centered/B_norm



    # sm = torch.exp(torch.matmul(A_normalized, B_normalized.T))
    sm = 1000**(torch.matmul(A_normalized, B_normalized.T))
    columns_avg = torch.sum(sm,0)/sm.shape[0]
    rows_avg = torch.sum(sm,1)/sm.shape[1]

    columns_std = torch.std(sm,0)
    rows_std = torch.std(sm,1)

    z_rows = (sm-rows_avg.unsqueeze(1))/rows_std.unsqueeze(1)
    z_columns = (sm-columns_avg)/columns_std
    ans=(z_rows+z_columns)/2
    ans=ans.cpu().numpy()
    return ans



def compute_similarity_matrix_plain(embedding1, embedding2, l=1, p=2):
    """ Take as input 2 sequence embeddings (at a residue level) and returns the plain
        similarity matrix.

        :param embedding1: residues embedding representation for sequence 1
        :param embedding2: residues embedding representation for sequence 2
        :param l: scalar that can be use to regularize the similarity matrix (no effect wit l=1)
        :param p: Minkowski distance order (ex. p=1:Manhattan, p=2:Euclidean)

        :type embedding1: pytorch tensor
        :type embedding2: pytorch tensor
        :type l: float
        :type p: integer
    """
    # return -l*torch.cdist(embedding1, embedding2, p=p)
    return torch.exp(-l*torch.cdist(embedding1, embedding2, p=p))



def compute_cosine_similarity_matrix(embedding1, embedding2):
    """ Take as input 2 sequence embeddings (at a residue level) and returns the cosine similarity matrix
        with the signal enhancement based on Z-scores. The signal enhancement seems to be redundant 
        when used with the cosine similarity score, therefore we don't recommend this version.

        :param embedding1: residues embedding representation for sequence 1
        :param embedding2: residues embedding representation for sequence 2

        :type embedding1: pytorch tensor
        :type embedding2: pytorch tensor
    """
    
    sm = compute_cosine_similarity_matrix_plain(embedding1, embedding2)
    columns_avg = torch.sum(sm,0)/sm.shape[0]
    rows_avg = torch.sum(sm,1)/sm.shape[1]
    
    columns_std = torch.std(sm,0)
    rows_std = torch.std(sm,1)

    z_rows = (sm-rows_avg.unsqueeze(1))/rows_std.unsqueeze(1)
    z_columns = (sm-columns_avg)/columns_std
    return (z_rows+z_columns)/2


def compute_cosine_similarity_matrix_plain(embedding1, embedding2):
    """ Take as input 2 sequence embeddings (at a residue level) and returns the cosine
        similarity matrix.

        :param embedding1: residues embedding representation for sequence 1
        :param embedding2: residues embedding representation for sequence 2

        :type embedding1: pytorch tensor
        :type embedding2: pytorch tensor
    """
    
    return torch.tensor(1-spatial.distance.cdist(embedding1.cpu().numpy(), embedding2.cpu().numpy(), 'cosine'))

