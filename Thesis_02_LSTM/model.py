import torch
from torch import nn


class FlowLSTM(nn.Module):
    def __init__(
        self,
        input_dim: int = 5,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.output_layer = nn.Linear(hidden_dim, 1)

    def forward(self, sequences: torch.Tensor) -> torch.Tensor:
        outputs, _ = self.lstm(sequences)
        return self.output_layer(outputs[:, -1, :])
