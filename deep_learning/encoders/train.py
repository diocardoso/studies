import hydra
import pandas as pd
import torch
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from dataloader import MedicalDataset
from model import MultiModalClassifier


def train_epoch(model, dataloader, criterion, optimizer):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch in dataloader:

        # Forward pass
        logits = model(batch['input_ids'], batch['labels'])
        loss = criterion(logits, batch['diagnosis'])

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Track metrics
        total_loss += loss.item()
        preds = logits.argmax(dim=-1)
        correct += (preds == batch['diagnosis']).sum().item()
        total += batch['diagnosis'].size(0)

    return total_loss / len(dataloader), correct / total


def evaluate(model, dataloader, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in dataloader:
            logits = model(batch['input_ids'], batch['labels'])
            loss = criterion(logits, batch['diagnosis'])

            total_loss += loss.item()
            preds = logits.argmax(dim=-1)
            correct += (preds == batch['diagnosis']).sum().item()
            total += batch['diagnosis'].size(0)

    return total_loss / len(dataloader), correct / total


@hydra.main(version_base=None, config_path="conf", config_name="config")
def train(cfg: DictConfig) -> None:

    model = MultiModalClassifier(
        vocab_size=cfg.architecture.vocab_size,
        num_labels=cfg.architecture.num_labels,
        num_classes=cfg.architecture.num_classes,
        dim=cfg.architecture.dim,
        heads=cfg.architecture.heads,
        layers=cfg.architecture.layers,
        dropout=cfg.architecture.dropout,
    )
    tokenizer = AutoTokenizer.from_pretrained(cfg.train.tokenizer)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.learning_rate, weight_decay=cfg.train.weight_decay)

    df = pd.read_csv(cfg.train.data, sep=";", nrows=100)
    train_df, test_df = train_test_split(df, test_size=cfg.train.test_size)

    train_dataset = MedicalDataset(train_df.reset_index(drop=True), tokenizer)
    test_dataset = MedicalDataset(test_df.reset_index(drop=True), tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=cfg.train.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=cfg.train.batch_size, shuffle=False)

    best_test_acc = 0
    for epoch in range(cfg.train.num_epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
        test_loss, test_acc = evaluate(model, test_loader, criterion)

        print(f"Epoch {epoch + 1}/{cfg.train.num_epochs}")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"  Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")

        # Save best model
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'test_acc': test_acc,
                'tokenizer_name': cfg.train.tokenizer,
                'vocab_size': tokenizer.vocab_size,
            }, 'best_model.pt')
            print("  → Saved new best model")


if __name__ == "__main__":
    train()