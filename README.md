# Improving Protein Remote Homology Alignment via Data Smoothing of Embedding Similarity Matrices
## Introduction 
Protein remote homology alignment remains a challenging problem in bioinformatics, particularly in low-identity settings where weak evolutionary signals often limit the reliability of 
conventional sequence-based methods. Recent protein language model–based approaches have improved sequence-only alignment by introducing embedding-derived residue similarity matrices, 
but these matrices can still be affected by score bias, local noise, and fragmented matching patterns, thereby constraining alignment quality.In this study, we present SEBA (Smoothed 
Embedding-Based Alignment), an unsupervised sequence-only framework designed to refine residue-level similarity estimation for remote homology alignment. SEBA combines bidirectional 
Z-score normalization with alignment-aware diagonal smoothing to improve the comparability, continuity, and interpretability of embedding-derived similarity matrices before 
dynamic programming. This method was designed to suppress spurious local responses while reinforcing spatially coherent homologous signals that are more consistent with plausible 
alignment paths. Evaluations across multiple benchmark settings show that SEBA improves alignment performance over existing sequence-only baselines while preserving sensitivity to 
structural similarity. Thus, SEBA provides an effective and computationally simple refinement strategy for embedding-based alignment in sequence-only remote homology analysis.
## Method
