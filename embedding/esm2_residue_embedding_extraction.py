"""Extract ESM residue embeddings for paired protein sequences.

This refactor keeps the original execution logic unchanged:
1. Load seq1s and seq2s from pickle files.
2. Load an ESM model and tokenizer.
3. Extract residue-level embeddings for seq1 and seq2 one by one.
4. Save the two embedding lists as pickle files.

All paths and model settings are exposed as function parameters instead of being
hard-coded in the script body.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Any, Iterable, List, Optional, Sequence, Tuple, Union

import torch
from transformers import EsmModel, EsmTokenizer

PathLike = Union[str, Path]


def save_pickle(filename: PathLike, data: Any) -> None:
    """Save Python data to a pickle file."""
    with open(filename, "wb") as file:
        pickle.dump(data, file)
    print(f"finish: {filename}")


def load_pickle(filename: PathLike) -> Any:
    """Load Python data from a pickle file."""
    with open(filename, "rb") as file:
        return pickle.load(file)


def get_device(device_name: Optional[str] = None) -> torch.device:
    """Return the requested device, or automatically select CUDA when available."""
    if device_name is not None:
        return torch.device(device_name)
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_esm_model_and_tokenizer(
    model_name: PathLike,
    device: torch.device,
) -> Tuple[EsmModel, EsmTokenizer]:
    """Load the ESM model and tokenizer."""
    model_name = str(model_name)
    model = EsmModel.from_pretrained(model_name).to(device)
    tokenizer = EsmTokenizer.from_pretrained(model_name)
    model.eval()
    return model, tokenizer


def extract_embedding(
    seq: str,
    model: EsmModel,
    tokenizer: EsmTokenizer,
    device: torch.device,
):
    """Extract residue-level embedding for one protein sequence.

    The slicing rule is the same as in the original script: remove the first
    special token and keep exactly ``len(seq)`` residue embeddings.
    """
    inputs = tokenizer(seq, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    embedding = outputs.last_hidden_state
    return embedding[0][1 : 1 + len(seq)].cpu().numpy()


def extract_pair_embeddings(
    seq1s: Sequence[str],
    seq2s: Sequence[str],
    model: EsmModel,
    tokenizer: EsmTokenizer,
    device: torch.device,
) -> Tuple[List[Any], List[Any]]:
    """Extract embeddings for two sequence lists using the original paired loop."""
    if len(seq1s) != len(seq2s):
        raise ValueError(
            f"seq1s and seq2s must have the same length, "
            f"got {len(seq1s)} and {len(seq2s)}."
        )

    save_seq1s = []
    save_seq2s = []

    for idx in range(len(seq1s)):
        emb_seq1 = seq1s[idx]
        emb1 = extract_embedding(emb_seq1, model, tokenizer, device)
        save_seq1s.append(emb1)

        emb_seq2 = seq2s[idx]
        emb2 = extract_embedding(emb_seq2, model, tokenizer, device)
        save_seq2s.append(emb2)

    return save_seq1s, save_seq2s


def run_embedding_extraction(
    model_name: PathLike,
    seq1_input_path: PathLike,
    seq2_input_path: PathLike,
    seq1_output_path: PathLike,
    seq2_output_path: PathLike,
    device_name: Optional[str] = None,
) -> Tuple[List[Any], List[Any]]:
    """Run the full embedding extraction pipeline.

    Parameters
    ----------
    model_name:
        Hugging Face model name or local model directory, such as
        ``"./PretrainedModel/ESM2-3B"``.
    seq1_input_path, seq2_input_path:
        Pickle files containing the two protein sequence lists.
    seq1_output_path, seq2_output_path:
        Output pickle files for the extracted embedding lists.
    device_name:
        Optional device name, such as ``"cuda:0"`` or ``"cpu"``. When omitted,
        CUDA is used automatically if available.
    """
    device = get_device(device_name)
    model, tokenizer = load_esm_model_and_tokenizer(model_name, device)

    seq1s = load_pickle(seq1_input_path)
    seq2s = load_pickle(seq2_input_path)

    save_seq1s, save_seq2s = extract_pair_embeddings(
        seq1s=seq1s,
        seq2s=seq2s,
        model=model,
        tokenizer=tokenizer,
        device=device,
    )

    save_pickle(seq1_output_path, save_seq1s)
    save_pickle(seq2_output_path, save_seq2s)

    return save_seq1s, save_seq2s


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Extract ESM residue embeddings for paired protein sequences."
    )
    parser.add_argument("--model-name", required=True, help="ESM model name or path.")
    parser.add_argument("--seq1-input", required=True, help="Input pickle for seq1s.")
    parser.add_argument("--seq2-input", required=True, help="Input pickle for seq2s.")
    parser.add_argument("--seq1-output", required=True, help="Output pickle for seq1 embeddings.")
    parser.add_argument("--seq2-output", required=True, help="Output pickle for seq2 embeddings.")
    parser.add_argument(
        "--device",
        default=None,
        help="Optional device, e.g. 'cuda:0' or 'cpu'. Defaults to auto selection.",
    )
    return parser.parse_args()


def main() -> None:
    """Command-line entry point."""
    args = parse_args()
    run_embedding_extraction(
        model_name=args.model_name,
        seq1_input_path=args.seq1_input,
        seq2_input_path=args.seq2_input,
        seq1_output_path=args.seq1_output,
        seq2_output_path=args.seq2_output,
        device_name=args.device,
    )


if __name__ == "__main__":
    main()
