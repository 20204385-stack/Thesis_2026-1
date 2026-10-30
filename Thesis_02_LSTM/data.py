from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


FEATURE_COLUMNS = ["Dur", "TotPkts", "TotBytes", "SrcBytes", "TimeDiff"]
LABEL_COLUMN = "NormLabel"
LABEL_MAP = {"Benign": 0.0, "Background": 0.0, "Malicious": 1.0}


class FlowSequenceDataset(Dataset):
    """Create overlapping windows, using each window's final flow label."""

    def __init__(self, csv_path: Path, window_size: int = 10) -> None:
        if window_size < 1:
            raise ValueError("window_size must be at least 1")

        required_columns = FEATURE_COLUMNS + [LABEL_COLUMN]
        frame = pd.read_csv(csv_path, usecols=required_columns)
        frame = frame[required_columns]

        labels = frame[LABEL_COLUMN].astype("string").str.strip()
        mapped_labels = labels.map(LABEL_MAP)
        if mapped_labels.isna().any():
            invalid_labels = sorted(labels[mapped_labels.isna()].dropna().unique().tolist())
            raise ValueError(
                f"{csv_path} contains missing or unsupported labels: {invalid_labels}"
            )

        try:
            features = frame[FEATURE_COLUMNS].apply(pd.to_numeric, errors="raise")
        except (TypeError, ValueError) as error:
            raise ValueError(f"{csv_path} contains non-numeric feature values") from error

        feature_values = np.array(features, dtype=np.float32, copy=True)
        label_values = np.array(mapped_labels, dtype=np.float32, copy=True)
        if not np.isfinite(feature_values).all():
            raise ValueError(f"{csv_path} contains non-finite feature values")
        if len(feature_values) < window_size:
            raise ValueError(
                f"{csv_path} has {len(feature_values)} rows, fewer than "
                f"window_size={window_size}"
            )

        self.features = torch.from_numpy(feature_values)
        self.labels = torch.from_numpy(label_values)
        self.window_size = window_size

    def __len__(self) -> int:
        return len(self.features) - self.window_size + 1

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        window_end = index + self.window_size
        sequence = self.features[index:window_end]
        label = self.labels[window_end - 1]
        return sequence, label
