import torch
from torch import nn

from layers import Encoder


class MultiModalClassifier(nn.Module):
    def __init__(self, vocab_size, num_classes, num_labels, dim, heads, layers, dropout, pool_type='cls'):
        super().__init__()
        self.encoder = Encoder(vocab_size, dim, heads, layers, dropout)
        self.pool_type = pool_type

        self.label = nn.Sequential(
            nn.Linear(num_labels, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim)
        )

        self.fusion = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(dim, num_classes)

    def forward(self, input_ids, labels):
        x = self.encoder(input_ids)

        text_pooled = x[:, 0, :]

        label_features = self.label(labels.float())

        combined = torch.cat([text_pooled, label_features], dim=-1)
        fused = self.fusion(combined)

        fused = self.dropout(fused)
        logits = self.classifier(fused)
        return logits
