"""
train.py
Main entry point for fine-tuning DistilBERT on Sentiment Analysis.


export TWITTER_DATA_PATH="./data"

python train_model.py \
    --model_name distilbert-base-uncased \
    --train_file "$TWITTER_DATA_PATH/twitter_training.csv" \
    --validation_file "$TWITTER_DATA_PATH/twitter_validation.csv" \
    --output_dir ./sentiment-model \
    --num_train_epochs 5 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 32 \
    --eval_strategy epoch \
    --save_strategy epoch \
    --load_best_model_at_end True \
    --metric_for_best_model f1 \
    --save_total_limit 2 \
    --do_eval True \
    --report_to mlflow
"""

import logging
import sys
from dataclasses import dataclass, field

import numpy as np
import mlflow
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    HfArgumentParser,
)

from preprocess import load_and_tokenize_data

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.INFO,
)


@dataclass
class ModelArguments:
    model_name: str = field(
        default="distilbert-base-uncased",
        metadata={"help": "Model identifier from huggingface.co/models"},
    )
    max_seq_length: int = field(
        default=128,
        metadata={"help": "The maximum total input sequence length after tokenization"},
    )


@dataclass
class DataTrainingArguments:
    train_file: str = field(metadata={"help": "Path to training csv file"})
    validation_file: str = field(metadata={"help": "Path to validation csv file"})


def compute_metrics(eval_pred):
    """
    Computes accuracy, precision, recall, and F1-score for evaluation.
    """
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="weighted", zero_division=0
    )
    acc = accuracy_score(labels, predictions)

    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}


def main():
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if "mlflow" in training_args.report_to:
        mlflow.set_experiment("distilbert-sentiment-analysis")

    logger.info(f"Training/evaluation parameters {training_args}")

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name)

    logger.info("Loading and tokenizing datasets...")
    tokenized_datasets, label2id, id2label = load_and_tokenize_data(
        data_args.train_file,
        data_args.validation_file,
        tokenizer,
        model_args.max_seq_length,
    )
    config = AutoConfig.from_pretrained(
        model_args.model_name,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id,
        finetuning_task="sentiment-analysis",
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name,
        config=config,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        compute_metrics=compute_metrics,
    )

    logger.info("*** Starting Training ***")
    train_result = trainer.train()
    trainer.save_model()
    tokenizer.save_pretrained(training_args.output_dir)

    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)

    logger.info("*** Starting Evaluation ***")
    metrics = trainer.evaluate()
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)

    logger.info(f"✔ Process complete. Model saved to {training_args.output_dir}")


if __name__ == "__main__":
    main()
