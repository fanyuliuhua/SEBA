"""Extract residue-level embeddings from ProstT5.

This script keeps the original extraction workflow:
1. load a pickled list of protein sequences;
2. convert each sequence to the ProstT5 input format;
3. extract residue-level hidden states from T5EncoderModel;
4. save the resulting list of embedding matrices as a pickle file.

The original hard-coded paths and device settings are exposed as function
arguments and command-line options for easier reuse in experiments.
"""

from __future__ import annotations

import argparse
import pickle
import re
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np
import torch
from transformers import T5EncoderModel, T5Tokenizer


ArrayLike = np.ndarray
AttentionMaps = Tuple[ArrayLike, ...]


DEFAULT_MODEL_PATH = ""#"./PretrainedModel/ProstT5"
DEFAULT_INPUT_PATH = ""
DEFAULT_OUTPUT_PATH = ""
INVALID_AMINO_ACID_PATTERN = re.compile(r"[UZOB]")


def save_pickle(data, output_path: str | Path) -> None:
    """Save data to a pickle file."""
    output_path = Path(output_path)
    with output_path.open("wb") as file:
        pickle.dump(data, file)
    print(f"Saved: {output_path}")


def load_pickle(input_path: str | Path):
    """Load data from a pickle file."""
    input_path = Path(input_path)
    with input_path.open("rb") as file:
        return pickle.load(file)


def resolve_device(device_name: Optional[str] = None) -> torch.device:
    """Return the requested device, or use cuda:4 when CUDA is available."""
    if device_name is not None:
        return torch.device(device_name)
    return torch.device("cuda:4" if torch.cuda.is_available() else "cpu")


def load_prostt5_model(
    model_path: str | Path = DEFAULT_MODEL_PATH,
    device_name: Optional[str] = None,
    use_half_on_cuda: bool = True,
) -> tuple[T5Tokenizer, T5EncoderModel, torch.device]:
    """Load the ProstT5 tokenizer and encoder model."""
    device = resolve_device(device_name)
    model_path = str(model_path)

    tokenizer = T5Tokenizer.from_pretrained(model_path, do_lower_case=False)
    model = T5EncoderModel.from_pretrained(model_path).to(device)

    if device.type == "cpu":
        model.full()
    elif use_half_on_cuda:
        model.half()

    model.eval()
    return tokenizer, model, device


def format_prostt5_sequence(sequence: str) -> str:
    """Convert one protein sequence into the input format required by ProstT5."""
    sequence = INVALID_AMINO_ACID_PATTERN.sub("X", sequence)
    spaced_sequence = " ".join(list(sequence))

    if spaced_sequence.isupper():
        return f"<AA2fold> {spaced_sequence}"
    return f"<fold2AA> {spaced_sequence}"


def tokenize_sequence(
    sequence: str,
    tokenizer: T5Tokenizer,
    device: torch.device,
):
    """Tokenize one sequence and move the encoded tensors to the target device."""
    formatted_sequence = format_prostt5_sequence(sequence)
    return tokenizer.batch_encode_plus(
        [formatted_sequence],
        add_special_tokens=True,
        padding="longest",
        return_tensors="pt",
    ).to(device)


def extract_residue_embedding(
    sequence: str,
    tokenizer: T5Tokenizer,
    model: T5EncoderModel,
    device: torch.device,
) -> ArrayLike:
    """Extract residue-level ProstT5 embeddings for one sequence."""
    encoded_inputs = tokenize_sequence(sequence, tokenizer, device)

    with torch.no_grad():
        outputs = model(
            encoded_inputs.input_ids,
            attention_mask=encoded_inputs.attention_mask,
        )

    embedding = outputs.last_hidden_state[0, 1 : 1 + len(sequence)]
    return embedding.cpu().numpy()


def extract_attention_maps(
    sequence: str,
    tokenizer: T5Tokenizer,
    model: T5EncoderModel,
    device: torch.device,
) -> AttentionMaps:
    """Extract residue-level attention maps for one sequence."""
    encoded_inputs = tokenize_sequence(sequence, tokenizer, device)

    with torch.no_grad():
        outputs = model(
            encoded_inputs.input_ids,
            attention_mask=encoded_inputs.attention_mask,
            output_attentions=True,
        )

    return tuple(
        attention[:, :, 1 : 1 + len(sequence), 1 : 1 + len(sequence)]
        .cpu()
        .numpy()
        for attention in outputs.attentions
    )


def get_residue_features_from_attention(
    attention_maps: AttentionMaps,
    sequence_length: Optional[int] = None,
) -> ArrayLike:
    """Summarize ProstT5 attention maps using the original mean operation.

    Parameters
    ----------
    attention_maps:
        Tuple of layer-wise attention maps. Each element is expected to have
        shape ``(1, num_heads, L, L)`` after residue-level slicing.
    sequence_length:
        Kept for compatibility with the original function signature. The value
        is not required by the original computation.
    """
    del sequence_length
    return np.mean(attention_maps, axis=0)


def extract_embeddings_from_sequences(
    sequences: Sequence[str],
    tokenizer: T5Tokenizer,
    model: T5EncoderModel,
    device: torch.device,
    to_upper: bool = True,
    progress_interval: int = 100,
) -> list[ArrayLike]:
    """Extract embeddings from a sequence list in the original one-by-one order."""
    embeddings: list[ArrayLike] = []

    for index, sequence in enumerate(sequences, start=1):
        sequence = sequence.upper() if to_upper else sequence
        embeddings.append(extract_residue_embedding(sequence, tokenizer, model, device))

        if progress_interval > 0 and index % progress_interval == 0:
            print(f"Processed {index} sequences")

    print(f"Total processed sequences: {len(embeddings)}")
    return embeddings


def run_embedding_extraction(
    input_path: str | Path = DEFAULT_INPUT_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    device_name: Optional[str] = None,
    to_upper: bool = True,
    use_half_on_cuda: bool = True,
    progress_interval: int = 100,
) -> list[ArrayLike]:
    """Load sequences, extract ProstT5 residue embeddings, and save results."""
    sequences = load_pickle(input_path)
    tokenizer, model, device = load_prostt5_model(
        model_path=model_path,
        device_name=device_name,
        use_half_on_cuda=use_half_on_cuda,
    )

    embeddings = extract_embeddings_from_sequences(
        sequences=sequences,
        tokenizer=tokenizer,
        model=model,
        device=device,
        to_upper=to_upper,
        progress_interval=progress_interval,
    )
    save_pickle(embeddings, output_path)
    return embeddings


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Extract residue-level embeddings from ProstT5."
    )
    parser.add_argument("--input", default=DEFAULT_INPUT_PATH, help="Input pickle file.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH, help="Output pickle file.")
    parser.add_argument(
        "--model-path",
        default=DEFAULT_MODEL_PATH,
        help="Path or Hugging Face name of the ProstT5 model.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device name, such as cuda:4 or cpu. Default: cuda:4 if available.",
    )
    parser.add_argument(
        "--keep-case",
        action="store_true",
        help="Do not convert sequences to uppercase before extraction.",
    )
    parser.add_argument(
        "--no-half",
        action="store_true",
        help="Do not convert the model to half precision on CUDA.",
    )
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=100,
        help="Print progress every N sequences. Use 0 to disable progress logs.",
    )
    return parser.parse_args()


def main() -> None:
    """Command-line entry point."""
    args = parse_args()
    run_embedding_extraction(
        input_path=args.input,
        output_path=args.output,
        model_path=args.model_path,
        device_name=args.device,
        to_upper=not args.keep_case,
        use_half_on_cuda=not args.no_half,
        progress_interval=args.progress_interval,
    )


if __name__ == "__main__":
    main()
