import methods
import numpy as np
from utils import json_load,build_similarity_matrix

gap_open_penalty=0
gap_extend_penalty=0

seqA=json_load("../data/SeqA.json")
seqB=json_load("../data/SeqB.json")
seqA_emb=json_load("../data/SeqA_Emb.json")
seqB_emb=json_load("../data/SeqB_Emb.json")

seqA_emb=np.array(seqA_emb)
seqB_emb=np.array(seqB_emb)

similarity=build_similarity_matrix(seqA_emb,seqB_emb)
results=methods.compute_seba(similarity,gap_open_penalty=gap_open_penalty,gap_extend_penalty=gap_extend_penalty,extensive_output=True)

print(seqA)
print(results['aln_1'])
print(seqB)
print(results['aln_2'])

