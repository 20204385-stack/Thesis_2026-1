"""CSV loading, binary labels, and lazy direction-based neighbor sampling."""

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data


BASE_FEATURES = ("Dur", "TotPkts", "TotBytes", "SrcBytes")
NUM_NEIGHBORS = (5, 5)


@dataclass
class FlowGraph:
    x: torch.Tensor
    y: torch.Tensor
    group_members: list[np.ndarray]
    group_ids: np.ndarray
    group_positions: np.ndarray

    @property
    def num_nodes(self) -> int:
        return self.x.size(0)

    @property
    def num_features(self) -> int:
        return self.x.size(1)


def encode_labels(frame: pd.DataFrame) -> torch.Tensor:
    """Use BinaryLabel when present; otherwise derive labels from NormLabel."""
    label_column = "BinaryLabel" if "BinaryLabel" in frame.columns else "NormLabel"
    if label_column not in frame.columns:
        raise ValueError("CSV must contain either 'BinaryLabel' or 'NormLabel'.")

    labels = frame[label_column]
    if labels.isna().any():
        raise ValueError(f"{label_column} contains missing labels.")

    if pd.api.types.is_numeric_dtype(labels):
        numeric_labels = pd.to_numeric(labels, errors="raise")
        if not numeric_labels.isin((0, 1)).all():
            raise ValueError(f"Numeric {label_column} values must be 0 or 1.")
        return torch.tensor(numeric_labels.to_numpy(dtype=np.int64), dtype=torch.long)

    mapping = {
        "malicious": 1,
        "benign": 0,
        "background": 0,
        "1": 1,
        "1.0": 1,
        "0": 0,
        "0.0": 0,
    }
    normalized = labels.astype("string").str.strip().str.lower()
    encoded = normalized.map(mapping)
    if encoded.isna().any():
        unknown = sorted(normalized[encoded.isna()].unique().tolist())
        raise ValueError(f"Unsupported values in {label_column}: {unknown[:10]}")
    return torch.tensor(encoded.to_numpy(dtype=np.int64), dtype=torch.long)


def build_flow_graph(frame: pd.DataFrame, split_name: str) -> FlowGraph:
    """Store flow features and direction groups without materializing clique edges."""
    required = {*BASE_FEATURES, "SrcAddr", "DstAddr"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{split_name} CSV is missing required columns: {missing}")

    feature_columns = list(BASE_FEATURES) + [
        column for column in frame.columns if column.startswith("State_")
    ]
    if frame.loc[:, feature_columns].isna().any().any():
        raise ValueError(f"{split_name} feature columns contain missing values.")
    if frame[["SrcAddr", "DstAddr"]].isna().any().any():
        raise ValueError(f"{split_name} SrcAddr/DstAddr values cannot be missing.")

    x_values = frame.loc[:, feature_columns].apply(
        pd.to_numeric, errors="raise"
    ).to_numpy(dtype=np.float32, copy=True)
    if not np.isfinite(x_values).all():
        raise ValueError(f"{split_name} feature columns contain non-finite values.")

    groups = frame.groupby(["SrcAddr", "DstAddr"], sort=False).indices.values()
    group_members = [np.asarray(indices, dtype=np.int64) for indices in groups]
    group_ids = np.empty(len(frame), dtype=np.int32)
    group_positions = np.empty(len(frame), dtype=np.int32)
    undirected_pair_count = 0
    for group_id, members in enumerate(group_members):
        group_ids[members] = group_id
        group_positions[members] = np.arange(len(members), dtype=np.int32)
        undirected_pair_count += len(members) * (len(members) - 1) // 2

    print(
        f"{split_name}: {len(frame):,} nodes in {len(group_members):,} "
        f"direction groups; {undirected_pair_count:,} possible undirected "
        "pairs represented implicitly.",
        flush=True,
    )
    return FlowGraph(
        x=torch.from_numpy(x_values),
        y=encode_labels(frame),
        group_members=group_members,
        group_ids=group_ids,
        group_positions=group_positions,
    )


class DirectionNeighborLoader:
    """Sample a bounded two-hop subgraph from implicit direction-group cliques."""

    def __init__(
        self,
        graph: FlowGraph,
        batch_size: int,
        shuffle: bool,
        num_neighbors: tuple[int, ...] = NUM_NEIGHBORS,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be positive.")
        if not num_neighbors or any(size < 0 for size in num_neighbors):
            raise ValueError("num_neighbors must contain non-negative fanouts.")
        self.graph = graph
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_neighbors = num_neighbors
        self.rng = np.random.default_rng()

    def __iter__(self) -> Iterator[Data]:
        if self.shuffle:
            node_order = torch.randperm(self.graph.num_nodes).tolist()
        else:
            node_order = range(self.graph.num_nodes)

        for start in range(0, self.graph.num_nodes, self.batch_size):
            seeds = list(node_order[start : start + self.batch_size])
            nodes = list(seeds)
            node_to_local = {node: i for i, node in enumerate(seeds)}
            edges: set[tuple[int, int]] = set()
            frontier = seeds

            for fanout in self.num_neighbors:
                next_frontier = []
                next_frontier_seen = set()
                for target in frontier:
                    group_id = int(self.graph.group_ids[target])
                    members = self.graph.group_members[group_id]
                    population = len(members) - 1
                    sample_count = min(fanout, population)
                    if sample_count == 0:
                        continue

                    target_position = int(self.graph.group_positions[target])
                    if sample_count * 2 > population:
                        sampled_positions = self.rng.choice(
                            population, size=sample_count, replace=False
                        )
                    else:
                        selected_positions: set[int] = set()
                        while len(selected_positions) < sample_count:
                            draws = self.rng.integers(
                                population,
                                size=sample_count - len(selected_positions),
                            )
                            selected_positions.update(draws.tolist())
                        sampled_positions = np.fromiter(
                            selected_positions, dtype=np.int64
                        )

                    sampled_positions = np.where(
                        sampled_positions >= target_position,
                        sampled_positions + 1,
                        sampled_positions,
                    )
                    for neighbor in members[sampled_positions].tolist():
                        neighbor = int(neighbor)
                        neighbor_local = node_to_local.get(neighbor)
                        if neighbor_local is None:
                            neighbor_local = len(nodes)
                            node_to_local[neighbor] = neighbor_local
                            nodes.append(neighbor)
                        target_local = node_to_local[target]
                        edge = (
                            min(target_local, neighbor_local),
                            max(target_local, neighbor_local),
                        )
                        edges.add(edge)
                        if neighbor not in next_frontier_seen:
                            next_frontier_seen.add(neighbor)
                            next_frontier.append(neighbor)
                frontier = next_frontier

            if edges:
                undirected_edges = np.asarray(sorted(edges), dtype=np.int64)
                edge_index = torch.from_numpy(
                    np.concatenate(
                        (undirected_edges.T, undirected_edges[:, ::-1].T), axis=1
                    ).copy()
                )
            else:
                edge_index = torch.empty((2, 0), dtype=torch.long)

            local_nodes = torch.tensor(nodes, dtype=torch.long)
            yield Data(
                x=self.graph.x[local_nodes],
                y=self.graph.y[local_nodes],
                edge_index=edge_index,
                batch_size=len(seeds),
            )


def make_loader(
    data: FlowGraph, batch_size: int, shuffle: bool
) -> DirectionNeighborLoader:
    return DirectionNeighborLoader(data, batch_size=batch_size, shuffle=shuffle)
