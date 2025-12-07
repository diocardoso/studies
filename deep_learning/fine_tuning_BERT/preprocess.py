"""
preprocess.py
Handles data loading and tokenization using Hugging Face datasets library.
"""

from typing import Dict, Any

from datasets import load_dataset, DatasetDict
from transformers import PreTrainedTokenizerFast, BatchEncoding

COLUMN_NAMES = ["tweet_id", "entity", "label", "text"]


def load_and_tokenize_data(
    train_path: str,
    val_path: str,
    tokenizer: PreTrainedTokenizerFast,
    max_length: int = 128,
) -> tuple[DatasetDict, dict, dict]:
    """
    Loads CSV data, maps labels to integers, and tokenizes text.
    Returns the tokenized dataset and label mappings.
    """
    data_files = {"train": str(train_path), "validation": str(val_path)}
    dataset = load_dataset("csv", data_files=data_files, column_names=COLUMN_NAMES)
    dataset = dataset.filter(lambda x: x["text"] is not None and x["label"] is not None)

    dataset["train"] = dataset["train"]
    dataset["validation"] = dataset["validation"]

    labels = sorted(set(dataset["train"]["label"]))
    label2id = {label: i for i, label in enumerate(labels)}
    id2label = {i: label for label, i in label2id.items()}

    def preprocess_function(examples: Dict[str, Any]) -> BatchEncoding:
        tokenized = tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )
        tokenized["labels"] = [label2id[label] for label in examples["label"]]
        return tokenized

    tokenized_datasets = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=COLUMN_NAMES,
    )

    tokenized_datasets.set_format("torch")
    return tokenized_datasets, label2id, id2label
