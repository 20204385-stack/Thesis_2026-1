"""A two-layer GraphSAGE model with bounded sampled neighborhoods."""

from __future__ import annotations

import torch
from torch import nn


class GraphSAGE(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 64) -> None:
        super().__init__()
        self.first_self = nn.Linear(input_size, hidden_size)
        self.first_neighbors = nn.Linear(input_size, hidden_size, bias=False)
        self.second_self = nn.Linear(hidden_size, hidden_size)
        self.second_neighbors = nn.Linear(hidden_size, hidden_size, bias=False)
        self.classifier = nn.Linear(hidden_size, 1)

    @staticmethod
    def _neighbor_mean(
        features: torch.Tensor, neighbor_ids: torch.Tensor
    ) -> torch.Tensor:
        valid = neighbor_ids >= 0
        safe_ids = neighbor_ids.clamp_min(0)
        gathered = features[safe_ids]
        gathered = gathered * valid.unsqueeze(-1)
        counts = valid.sum(dim=1, keepdim=True).clamp_min(1)
        return gathered.sum(dim=1) / counts

    def forward(
        self,
        node_ids: torch.Tensor,
        features: torch.Tensor,
        adjacency: torch.Tensor,
    ) -> torch.Tensor:
        seed_neighbors = adjacency[node_ids]
        included_ids = torch.cat(
            (node_ids, seed_neighbors[seed_neighbors >= 0]), dim=0
        )
        local_node_ids = torch.unique(included_ids, sorted=True)

        first_neighbor_mean = self._neighbor_mean(
            features, adjacency[local_node_ids]
        )
        first_layer = torch.relu(
            self.first_self(features[local_node_ids])
            + self.first_neighbors(first_neighbor_mean)
        )

        seed_positions = torch.searchsorted(local_node_ids, node_ids)
        neighbor_positions = torch.searchsorted(
            local_node_ids, seed_neighbors.clamp_min(0)
        )
        neighbor_positions = neighbor_positions.masked_fill(seed_neighbors < 0, -1)
        second_neighbor_mean = self._neighbor_mean(first_layer, neighbor_positions)
        second_layer = torch.relu(
            self.second_self(first_layer[seed_positions])
            + self.second_neighbors(second_neighbor_mean)
        )
        return self.classifier(second_layer).squeeze(-1)
