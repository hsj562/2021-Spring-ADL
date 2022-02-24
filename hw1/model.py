from typing import Dict

import torch
from torch.nn import Embedding
import torch.nn as nn

INTENT = 0
SLOT = 1

class SeqClassifier(torch.nn.Module):
    def __init__(
        self,
        embeddings: torch.tensor,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        bidirectional: bool,
        num_class: int,
    ) -> None:
        super(SeqClassifier, self).__init__()
        self.embed = Embedding.from_pretrained(embeddings, freeze=False)
        self.gru = nn.GRU(input_size=embeddings.shape[1], hidden_size=hidden_size, num_layers=num_layers, bidirectional=bidirectional, dropout=dropout, batch_first=True)
        
        self.dim = hidden_size * ( 2 if (bidirectional == True) else 1 )

        self.fc = nn.Linear(128 * self.dim, num_class)
        self.net = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.dim, 2048),
            nn.BatchNorm1d(128),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(2048, 9)
        )
        self.soft = nn.Softmax()
        self.relu = nn.ReLU()
        self.type = SLOT if num_class == 9 else INTENT

    @property
    def encoder_output_size(self) -> int:
        if self.type == INTENT:
            return 128 * self.dim
        else:
            return self.dim

    def forward(self, batch) -> Dict[str, torch.Tensor]:
        # TODO: implement model forward
        # raise NotImplementedError
        embedded = self.embed(batch)
        output, hidden = self.gru(embedded)
        if self.type == SLOT:
            output = self.net(output)
        elif self.type == INTENT:
            output = torch.reshape(output, (output.shape[0], -1))
            output = self.fc(self.relu(output))
        return output
