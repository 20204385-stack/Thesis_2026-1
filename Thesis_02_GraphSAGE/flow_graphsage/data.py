"""CSV loading, feature preparation, and communication-channel graph construction."""

from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


NUMERIC_FEATURES = ("Dur", "TotPkts", "TotBytes", "SrcBytes")
LABEL_COLUMN = "NormLabel"
ADDRESS_COLUMNS = ("SrcAddr", "DstAddr")


@dataclass
class FlowDataset:
    features: np.ndarray
    labels: np.ndarray
    source_addresses: list[str]
    destination_addresses: list[str]
    feature_columns: list[str]


def load_split(
    csv_path: Path, feature_columns: list[str] | None = None
) -> FlowDataset:
    """Load one split and map Benign/Background to 0 and Malicious to 1."""
    with csv_path.open("r", newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file)
        columns = reader.fieldnames
        if columns is None:
            raise ValueError(f"{csv_path} is missing a CSV header.")

        required_columns = {*NUMERIC_FEATURES, LABEL_COLUMN, *ADDRESS_COLUMNS}
        missing = sorted(required_columns.difference(columns))
        if missing:
            raise ValueError(f"{csv_path} is missing required columns: {missing}")

        if feature_columns is None:
            state_columns = [column for column in columns if column.startswith("State_")]
            if not state_columns:
                raise ValueError(f"{csv_path} has no State_* feature columns.")
            feature_columns = [*NUMERIC_FEATURES, *state_columns]
        else:
            missing_features = sorted(set(feature_columns).difference(columns))
            if missing_features:
                raise ValueError(
                    f"{csv_path} is missing feature columns: {missing_features}"
                )

        features: list[list[float]] = []
        labels: list[int] = []
        source_addresses: list[str] = []
        destination_addresses: list[str] = []

        for row_number, row in enumerate(reader, start=2):
            values: list[float] = []
            for column in feature_columns:
                raw_value = row[column]
                try:
                    value = float(raw_value)
                except (TypeError, ValueError) as error:
                    raise ValueError(
                        f"{csv_path}:{row_number} has an invalid value for "
                        f"{column}: {raw_value!r}"
                    ) from error
                if not np.isfinite(value):
                    raise ValueError(
                        f"{csv_path}:{row_number} has a non-finite value for "
                        f"{column}: {raw_value!r}"
                    )
                values.append(value)

            label = row[LABEL_COLUMN].strip().lower()
            if label not in {"benign", "background", "malicious"}:
                raise ValueError(
                    f"{csv_path}:{row_number} has an unsupported "
                    f"{LABEL_COLUMN}: {row[LABEL_COLUMN]!r}"
                )
            features.append(values)
            labels.append(int(label == "malicious"))
            source_addresses.append(row["SrcAddr"].strip())
            destination_addresses.append(row["DstAddr"].strip())

    if not features:
        raise ValueError(f"{csv_path} contains no flow rows.")

    return FlowDataset(
        features=np.asarray(features, dtype=np.float32),
        labels=np.asarray(labels, dtype=np.float32),
        source_addresses=source_addresses,
        destination_addresses=destination_addresses,
        feature_columns=feature_columns,
    )


def standardize_features(
    train_features: np.ndarray, other_features: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Standardize both splits using only training-split statistics."""
    mean = train_features.mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = train_features.std(axis=0, dtype=np.float64).astype(np.float32)
    scale[scale == 0] = 1.0
    return (
        (train_features - mean) / scale,
        (other_features - mean) / scale,
        mean,
        scale,
    )


def build_edge_index(
    source_addresses: list[str],
    destination_addresses: list[str],
    max_neighbors: int = 10,
) -> torch.Tensor:
    """Build a bounded edge index by joining flows on both address columns.

    Each node connects to up to ``max_neighbors`` other rows with the same
    source and destination addresses. Candidate rows closest in CSV order are
    selected deterministically.
    """
    if len(source_addresses) != len(destination_addresses):
        raise ValueError("Source and destination address lists must have equal length.")
    if max_neighbors < 1:
        raise ValueError("max_neighbors must be at least 1.")

    channel_rows: dict[tuple[str, str], list[int]] = defaultdict(list)
    for node_id, (source, destination) in enumerate(
        zip(source_addresses, destination_addresses)
    ):
        channel = (source.strip(), destination.strip())
        if any(
            not address or address.lower() in {"nan", "none"}
            for address in channel
        ):
            continue
        channel_rows[channel].append(node_id)

    edges: list[tuple[int, int]] = []
    for rows in channel_rows.values():
        for position, node_id in enumerate(rows):
            candidates = [
                *rows[max(0, position - max_neighbors) : position],
                *rows[position + 1 : position + max_neighbors + 1],
            ]
            selected = sorted(
                candidates, key=lambda candidate: (abs(candidate - node_id), candidate)
            )[:max_neighbors]
            edges.extend((node_id, neighbor_id) for neighbor_id in selected)

    if not edges:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.tensor(edges, dtype=torch.long).t().contiguous()


def edge_index_to_neighbor_table(
    edge_index: torch.Tensor, num_nodes: int, max_neighbors: int
) -> np.ndarray:
    """Convert a bounded edge index to the padded table used by GraphSAGE."""
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, num_edges].")
    if num_nodes < 0 or max_neighbors < 1:
        raise ValueError("num_nodes must be nonnegative and max_neighbors positive.")

    edges = edge_index.detach().cpu().numpy()
    source_ids, target_ids = edges
    if (
        np.any(source_ids < 0)
        or np.any(source_ids >= num_nodes)
        or np.any(target_ids < 0)
        or np.any(target_ids >= num_nodes)
    ):
        raise ValueError("edge_index contains a node ID outside the graph.")
    if np.any(source_ids == target_ids):
        raise ValueError("edge_index must not contain self-loops.")
    counts = np.bincount(source_ids, minlength=num_nodes)
    if np.any(counts > max_neighbors):
        raise ValueError("edge_index exceeds max_neighbors for a node.")

    neighbors = np.full((num_nodes, max_neighbors), -1, dtype=np.int64)
    positions = np.zeros(num_nodes, dtype=np.int64)
    for source_id, target_id in zip(source_ids, target_ids):
        neighbors[source_id, positions[source_id]] = target_id
        positions[source_id] += 1
    return neighbors
