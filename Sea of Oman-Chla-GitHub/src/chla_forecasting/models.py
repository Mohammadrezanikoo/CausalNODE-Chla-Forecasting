"""Three manuscript-related neural forecasters, with shared interfaces."""
from __future__ import annotations
import math
import torch
from torch import nn


class ODERightHandSide(nn.Module):
    def __init__(self, n_drivers: int, hidden: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1 + n_drivers, hidden), nn.Tanh(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1),
        )

    def forward(self, state: torch.Tensor, drivers: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat((state, drivers), dim=-1))


class CausalNeuralODE(nn.Module):
    """RK4 integration from t to t+h, holding observed exogenous inputs at t fixed."""
    def __init__(self, n_drivers: int, hidden: int = 64,
                 integration_steps: int = 4, dropout: float = 0.15):
        super().__init__()
        if integration_steps < 1:
            raise ValueError("RK4 substeps must be >= 1")
        self.rhs = ODERightHandSide(n_drivers, hidden, dropout)
        self.integration_steps = integration_steps

    def forward(self, current: torch.Tensor, drivers: torch.Tensor, horizon: int = 1) -> torch.Tensor:
        state = current
        dt = 1.0 / self.integration_steps
        for _day in range(horizon):
            for _step in range(self.integration_steps):
                k1 = self.rhs(state, drivers)
                k2 = self.rhs(state + dt * k1 / 2, drivers)
                k3 = self.rhs(state + dt * k2 / 2, drivers)
                k4 = self.rhs(state + dt * k3, drivers)
                state = state + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        return state


class LSTMForecaster(nn.Module):
    def __init__(self, n_features: int, hidden: int = 96,
                 layers: int = 1, dropout: float = 0.25):
        super().__init__()
        self.rnn = nn.LSTM(n_features, hidden, layers, batch_first=True,
                           dropout=dropout if layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, 1)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(sequence)
        return self.head(self.dropout(out[:, -1, :]))


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, length: int = 512):
        super().__init__()
        pos = torch.arange(length, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32)
                        * (-math.log(10000.0) / d_model))
        enc = torch.zeros(length, d_model)
        enc[:, 0::2] = torch.sin(pos * div)
        enc[:, 1::2] = torch.cos(pos * div[:enc[:, 1::2].shape[1]])
        self.register_buffer("encoding", enc.unsqueeze(0), persistent=False)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        return sequence + self.encoding[:, :sequence.shape[1], :]


class TransformerForecaster(nn.Module):
    def __init__(self, n_features: int, hidden: int = 64,
                 layers: int = 2, heads: int = 4, dropout: float = 0.2):
        super().__init__()
        if hidden % heads:
            raise ValueError("Transformer hidden width must be divisible by number of heads")
        self.projection = nn.Linear(n_features, hidden)
        self.position = PositionalEncoding(hidden)
        layer = nn.TransformerEncoderLayer(d_model=hidden, nhead=heads,
                                           dim_feedforward=hidden * 2,
                                           dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers,
                                              enable_nested_tensor=False)
        self.head = nn.Linear(hidden, 1)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        latent = self.encoder(self.position(self.projection(sequence)))
        return self.head(latent[:, -1, :])


def create_model(name: str, n_features: int, n_drivers: int,
                 hidden: int = 64, dropout: float = 0.2,
                 layers: int = 1, heads: int = 4, integration_steps: int = 4) -> nn.Module:
    if name == "causal_ode":
        return CausalNeuralODE(n_drivers, hidden, integration_steps, dropout)
    if name == "lstm":
        return LSTMForecaster(n_features, hidden, layers, dropout)
    if name == "transformer":
        return TransformerForecaster(n_features, hidden, layers, heads, dropout)
    raise ValueError(f"Unsupported model: {name}")
