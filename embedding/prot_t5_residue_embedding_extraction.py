"""Extract residue-level ProtT5 embeddings from protein sequences.

This script keeps the original extraction logic:
1. load protein sequences from a pickle file;
2. encode each sequence with ProtT5;
3. remove the final special token from the hidden states;
4. save the per-residue embeddings to a pickle file.

The hard-coded paths in the original script are exposed as function
parameters and command-line arguments for easier reuse in experiments.
"""

from __future__ import annotations

import argparse
import pickle
import re
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from transformers import T5EncoderModel, T5Tokenizer


DEFAULT_MODEL_PATH = "/mnt/remote_home/yzhu/PretrainedModel/ProtT5"
DEFAULT_INPUT_PATH = "PiscesSeq2s.pkl"
DEFAULT_OUTPUT_PATH = "/mnt/remote_home/yzhu/Benchmark/pisces/PiscesSeq2EmbsProtT5.pkl"


def save_pickle(file_path: str | Path, data: object) -> None:
    """Save data to a pickle file."""

    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with file_path.open("wb") as file:
        pickle.dump(data, file)

    print(f"Saved: {file_path}")


def load_pickle(file_path: str | Path) -> object:
    """Load data from a pickle file."""

    with Path(file_path).open("rb") as file:
        return pickle.load(file)


def resolve_device(device_name: str | None = None) -> torch.device:
    """Resolve the target device.

    If ``device_name`` is not provided, use the original behavior:
    CUDA device 0 when CUDA is available, otherwise CPU.
    """

    if device_name is None:
        device_name = "cuda:0" if torch.cuda.is_available() else "cpu"

    return torch.device(device_name)


def load_prot_t5_model(
    model_path: str | Path,
    device: torch.device,
) -> tuple[T5Tokenizer, T5EncoderModel]:
    """Load the ProtT5 tokenizer and encoder."""

    tokenizer = T5Tokenizer.from_pretrained(
        str(model_path),
        do_lower_case=False,
    )
    model = T5EncoderModel.from_pretrained(str(model_path))
    model.to(device)
    model.eval()

    return tokenizer, model


def format_protein_sequence(sequence: str) -> list[str]:
    """Prepare a protein sequence for ProtT5 tokenization.

    The logic follows the original script:
    - replace rare or ambiguous residues U, Z, O, and B with X;
    - insert spaces between amino-acid tokens.
    """

    sequence = re.sub(r"[UZOB]", "X", sequence)
    return [" ".join(sequence)]


def extract_residue_embedding(
    sequence: str,
    tokenizer: T5Tokenizer,
    model: T5EncoderModel,
    device: torch.device,
) -> np.ndarray:
    """Extract residue-level ProtT5 embeddings for one protein sequence."""

    formatted_sequence = format_protein_sequence(sequence)

    tokenized = tokenizer.batch_encode_plus(
        formatted_sequence,
        add_special_tokens=True,
        padding=True,
    )
    input_ids = torch.tensor(tokenized["input_ids"]).to(device)
    attention_mask = torch.tensor(tokenized["attention_mask"]).to(device)

    with torch.no_grad():
        output = model(input_ids=input_ids, attention_mask=attention_mask)

    hidden_states = output.last_hidden_state.cpu().numpy()

    # Original logic: remove padding and the final special token.
    features: list[np.ndarray] = []
    for sequence_index in range(len(hidden_states)):
        sequence_length = int((attention_mask[sequence_index] == 1).sum())
        residue_embedding = hidden_states[sequence_index][: sequence_length - 1]
        features.append(residue_embedding)

    return features[0]


def extract_embeddings(
    sequences: Sequence[str],
    tokenizer: T5Tokenizer,
    model: T5EncoderModel,
    device: torch.device,
    progress_interval: int = 1000,
) -> list[np.ndarray]:
    """Extract ProtT5 embeddings for a sequence list."""

    embeddings: list[np.ndarray] = []

    for index, sequence in enumerate(sequences):
        if progress_interval > 0 and index % progress_interval == 0:
            print(index)

        embedding = extract_residue_embedding(
            sequence=sequence,
            tokenizer=tokenizer,
            model=model,
            device=device,
        )
        embeddings.append(embedding)

    return embeddings


def run_embedding_extraction(
    input_path: str | Path = DEFAULT_INPUT_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    device_name: str | None = None,
    progress_interval: int = 1000,
) -> list[np.ndarray]:
    """Run the complete ProtT5 residue-embedding extraction workflow."""

    sequences = load_pickle(input_path)
    if not isinstance(sequences, Sequence):
        raise TypeError("The input pickle file should contain a sequence list.")

    print(f"Number of sequences: {len(sequences)}")

    device = resolve_device(device_name)
    tokenizer, model = load_prot_t5_model(model_path=model_path, device=device)

    embeddings = extract_embeddings(
        sequences=sequences,
        tokenizer=tokenizer,
        model=model,
        device=device,
        progress_interval=progress_interval,
    )

    print(f"Number of embeddings: {len(embeddings)}")
    save_pickle(output_path, embeddings)

    return embeddings


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""

    parser = argparse.ArgumentParser(
        description="Extract residue-level embeddings using ProtT5.",
    )
    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT_PATH,
        help="Path to the input pickle file containing protein sequences.",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT_PATH,
        help="Path to save the output pickle file.",
    )
    parser.add_argument(
        "--model-path",
        default=DEFAULT_MODEL_PATH,
        help="Path or Hugging Face name of the ProtT5 model.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device name, such as cuda:0 or cpu. Defaults to cuda:0 if available.",
    )
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=1000,
        help="Print progress every N sequences. Set to 0 to disable.",
    )

    return parser


def main() -> None:
    """Command-line entry point."""

    args = build_arg_parser().parse_args()

    run_embedding_extraction(
        input_path=args.input,
        output_path=args.output,
        model_path=args.model_path,
        device_name=args.device,
        progress_interval=args.progress_interval,
    )


if __name__ == "__main__":
    main()
