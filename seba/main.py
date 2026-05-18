
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import time

import numpy as np
import torch

from seba import methods
from seba import score_matrices as sm
from smoothing import smooth_diag_numpy
from utils import compute, load


AlignmentPair = Tuple[int, int]


@dataclass(frozen=True)
class DatasetPaths:
    """Input files used by the MALIDUP evaluation pipeline."""

    seq1_embeddings: Path
    seq2_embeddings: Path
    references: Path
    seq1: Path
    seq2: Path


@dataclass
class SampleResult:
    """Evaluation result for one sequence pair."""

    sample_id: int
    precision: float
    recall: float
    f1: float
    num_tp: int
    num_fp: int
    num_fn: int
    seba_alignment: Tuple[str, str]
    reference_alignment: Tuple[str, str]


# -----------------------------------------------------------------------------
# Matrix utilities
# -----------------------------------------------------------------------------


def bidirectional_zscore(matrix: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Apply row-wise and column-wise z-score normalization.

    This is the original ``process`` logic with clearer naming and a small
    numerical guard for rows or columns with zero standard deviation.
    """

    tensor = torch.as_tensor(matrix, dtype=torch.float32)

    row_mean = tensor.mean(dim=1, keepdim=True)
    col_mean = tensor.mean(dim=0, keepdim=True)
    row_std = tensor.std(dim=1, keepdim=True).clamp_min(eps)
    col_std = tensor.std(dim=0, keepdim=True).clamp_min(eps)

    z_by_row = (tensor - row_mean) / row_std
    z_by_col = (tensor - col_mean) / col_std
    normalized = (z_by_row + z_by_col) / 2.0

    return normalized.cpu().numpy()


def cosine_similarity(x: np.ndarray, y: np.ndarray, eps: float = 1e-8) -> float:
    """Compute cosine similarity between two vectors."""

    denominator = np.linalg.norm(x) * np.linalg.norm(y)
    if denominator < eps:
        return 0.0
    return float(np.dot(x, y) / denominator)


def cosine_similarity_matrix(emb1: np.ndarray, emb2: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Vectorized cosine similarity matrix for two embedding arrays.

    This replaces the nested-loop implementation in ``getEemCorr`` and removes
    per-cell debug printing.
    """

    emb1 = np.asarray(emb1, dtype=np.float32)
    emb2 = np.asarray(emb2, dtype=np.float32)

    emb1_norm = emb1 / np.maximum(np.linalg.norm(emb1, axis=1, keepdims=True), eps)
    emb2_norm = emb2 / np.maximum(np.linalg.norm(emb2, axis=1, keepdims=True), eps)
    return emb1_norm @ emb2_norm.T


def build_similarity_matrix(emb1: np.ndarray, emb2: np.ndarray, use_smoothing: bool = True) -> np.ndarray:
    """Build and optionally smooth the residue-residue similarity matrix."""

    similarity_matrix = sm.compute_similarity_matrix2(emb1, emb2)
    if use_smoothing:
        similarity_matrix = smooth_diag_numpy(similarity_matrix)
    return similarity_matrix


# -----------------------------------------------------------------------------
# Alignment reconstruction and pair extraction
# -----------------------------------------------------------------------------


def rebuild_predicted_alignment(
    eba_result: Dict[str, Sequence[int]], seq1: Sequence[str], seq2: Sequence[str]
) -> Tuple[str, str]:
    """Convert SEBA alignment index arrays into two aligned strings."""

    aligned_seq1: List[str] = []
    aligned_seq2: List[str] = []

    for idx1, idx2 in zip(eba_result["aln_1"], eba_result["aln_2"]):
        aligned_seq1.append("-" if idx1 == -1 else str(seq1[idx1]))
        aligned_seq2.append("-" if idx2 == -1 else str(seq2[idx2]))

    return "".join(aligned_seq1), "".join(aligned_seq2)


def rebuild_reference_alignment(ref: Sequence[str], seq1: Sequence[str], seq2: Sequence[str]) -> Tuple[str, str]:
    """Convert compact reference symbols into two aligned strings.

    Reference encoding:
    - ``"1"``: gap in sequence 1;
    - ``"2"``: gap in sequence 2;
    - ``":"`` or ``"."``: aligned residue pair.
    """

    aligned_seq1: List[str] = []
    aligned_seq2: List[str] = []
    pos1 = 0
    pos2 = 0

    for symbol in ref:
        if symbol == "1":
            aligned_seq1.append("-")
            aligned_seq2.append(str(seq2[pos2]))
            pos2 += 1
        elif symbol == "2":
            aligned_seq1.append(str(seq1[pos1]))
            aligned_seq2.append("-")
            pos1 += 1
        elif symbol in {":", "."}:
            aligned_seq1.append(str(seq1[pos1]))
            aligned_seq2.append(str(seq2[pos2]))
            pos1 += 1
            pos2 += 1
        else:
            raise ValueError(f"Unexpected reference symbol: {symbol!r}")

    return "".join(aligned_seq1), "".join(aligned_seq2)


def extract_predicted_pairs(eba_result: Dict[str, Sequence[int]]) -> List[AlignmentPair]:
    """Extract non-gap residue pairs from an EBA result."""

    return [
        (int(idx1), int(idx2))
        for idx1, idx2 in zip(eba_result["aln_1"], eba_result["aln_2"])
        if idx1 != -1 and idx2 != -1
    ]


def extract_reference_pairs(ref: Sequence[str]) -> List[AlignmentPair]:
    """Extract reference residue pairs from compact alignment symbols."""

    pairs: List[AlignmentPair] = []
    pos1 = 0
    pos2 = 0

    for symbol in ref:
        if symbol == "1":
            pos2 += 1
        elif symbol == "2":
            pos1 += 1
        elif symbol in {":", "."}:
            pairs.append((pos1, pos2))
            pos1 += 1
            pos2 += 1
        else:
            raise ValueError(f"Unexpected reference symbol: {symbol!r}")

    return pairs


# -----------------------------------------------------------------------------
# Evaluation pipeline
# -----------------------------------------------------------------------------


def evaluate_pairs(
    predicted_pairs: Sequence[AlignmentPair], reference_pairs: Sequence[AlignmentPair]
) -> Tuple[float, float, float, int, int, int]:
    """Compute precision, recall, and F1 from predicted/reference pairs."""

    predicted_set = set(predicted_pairs)
    reference_set = set(reference_pairs)

    num_tp = len(predicted_set & reference_set)
    num_fp = len(predicted_set - reference_set)
    num_fn = len(reference_set - predicted_set)

    precision, recall, f1 = compute(num_tp, num_fp, num_fn)
    return precision, recall, f1, num_tp, num_fp, num_fn


def run_seba_alignment(
    similarity_matrix: np.ndarray,
    gap_open_penalty: float = 0,
    gap_extend_penalty: float = 0,
) -> Dict[str, Sequence[int]]:
    """Run EBA alignment with a consistent parameter interface."""

    return methods.compute_seba(
        similarity_matrix,
        gap_open_penalty=gap_open_penalty,
        gap_extend_penalty=gap_extend_penalty,
        extensive_output=True,
    )


def evaluate_sample(
    sample_id: int,
    emb1: np.ndarray,
    emb2: np.ndarray,
    seq1: Sequence[str],
    seq2: Sequence[str],
    ref: Sequence[str],
    use_smoothing: bool = True,
) -> SampleResult:
    """Run the full SEBA/EBA pipeline for one sample."""

    similarity_matrix = build_similarity_matrix(emb1, emb2, use_smoothing=use_smoothing)
    eba_result = run_seba_alignment(similarity_matrix)

    predicted_pairs = extract_predicted_pairs(eba_result)
    reference_pairs = extract_reference_pairs(ref)
    precision, recall, f1, num_tp, num_fp, num_fn = evaluate_pairs(predicted_pairs, reference_pairs)

    return SampleResult(
        sample_id=sample_id,
        precision=precision,
        recall=recall,
        f1=f1,
        num_tp=num_tp,
        num_fp=num_fp,
        num_fn=num_fn,
        seba_alignment=rebuild_predicted_alignment(eba_result, seq1, seq2),
        reference_alignment=rebuild_reference_alignment(ref, seq1, seq2),
    )


def load_dataset(paths: DatasetPaths):
    """Load all required MALIDUP data files."""

    emb1s = load(paths.seq1_embeddings)
    emb2s = load(paths.seq2_embeddings)
    refs = load(paths.references)
    seq1s = load(paths.seq1)
    seq2s = load(paths.seq2)
    return emb1s, emb2s, refs, seq1s, seq2s


def select_sample_indices(num_samples: int, sample_index: Optional[int] = None) -> range:
    """Return indices for all samples or for one selected sample."""

    if sample_index is None:
        return range(num_samples)
    if not 0 <= sample_index < num_samples:
        raise IndexError(f"sample_index={sample_index} is outside [0, {num_samples})")
    return range(sample_index, sample_index + 1)


def summarize_results(results: Sequence[SampleResult]) -> None:
    """Print a compact summary of evaluation scores."""

    if not results:
        print("No samples were evaluated.")
        return

    f1_scores = np.array([item.f1 for item in results], dtype=np.float32)
    precision_scores = np.array([item.precision for item in results], dtype=np.float32)
    recall_scores = np.array([item.recall for item in results], dtype=np.float32)

    print(f"Evaluated samples: {len(results)}")
    print(f"Mean precision: {precision_scores.mean():.4f}")
    print(f"Mean recall:    {recall_scores.mean():.4f}")
    print(f"Mean F1:        {f1_scores.mean():.4f}")


def main(
    paths: DatasetPaths,
    sample_index: Optional[int] = None,
    use_smoothing: bool = True,
) -> List[SampleResult]:
    """

    Args:
        paths: locations of embedding, sequence, and reference files.
        sample_index: evaluate one sample when provided; evaluate all samples when
            set to ``None``.
        use_smoothing: whether to apply ``smooth_diag_numpy`` before EBA.
    """

    emb1s, emb2s, refs, seq1s, seq2s = load_dataset(paths)
    num_samples = len(emb1s)

    print(f"Loaded {num_samples} samples.")
    start_time = time.time()

    results: List[SampleResult] = []
    for sample_id in select_sample_indices(num_samples, sample_index=sample_index):
        result = evaluate_sample(
            sample_id=sample_id,
            emb1=emb1s[sample_id],
            emb2=emb2s[sample_id],
            seq1=seq1s[sample_id],
            seq2=seq2s[sample_id],
            ref=refs[sample_id],
            use_smoothing=use_smoothing,
        )
        results.append(result)
        print(
            f"sample={sample_id} | "
            f"P={result.precision:.4f}, R={result.recall:.4f}, F1={result.f1:.4f} | "
            f"TP={result.num_tp}, FP={result.num_fp}, FN={result.num_fn}"
        )

    summarize_results(results)
    print(f"Elapsed time: {time.time() - start_time:.2f}s")
    return results


if __name__ == "__main__":
    # Use sample_index=1 to reproduce the original debugging behavior for one
    # selected sample. Use sample_index=None to evaluate the whole dataset.
    dataset_paths = DatasetPaths(
        seq1_embeddings=Path("./data/Malidup2Seq1Emb.pkl"),
        seq2_embeddings=Path("./data/Malidup2Seq2Emb.pkl"),
        references=Path("./data/Malidup2Alignment.pkl"),
        seq1=Path("./data/Malidup2Seq1.pkl"),
        seq2=Path("./data/Malidup2Seq2.pkl"),
    )
    main(paths=dataset_paths, sample_index=None, use_smoothing=True)
