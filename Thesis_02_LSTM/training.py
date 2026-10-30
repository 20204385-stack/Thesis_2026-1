import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from data import FEATURE_COLUMNS, FlowSequenceDataset
from model import FlowLSTM


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_function: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    total_samples = 0

    for sequences, labels in loader:
        sequences = sequences.to(device)
        labels = labels.to(device).unsqueeze(1)

        optimizer.zero_grad()
        logits = model(sequences)
        loss = loss_function(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / total_samples


def evaluate(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> np.ndarray:
    model.eval()
    confusion_matrix = np.zeros((2, 2), dtype=np.int64)

    with torch.no_grad():
        for sequences, labels in loader:
            sequences = sequences.to(device)
            labels = labels.to(device).bool()
            predictions = model(sequences).squeeze(1) >= 0

            true_positives = (predictions & labels).sum().item()
            false_positives = (predictions & ~labels).sum().item()
            false_negatives = (~predictions & labels).sum().item()
            true_negatives = (~predictions & ~labels).sum().item()
            confusion_matrix += np.array(
                [
                    [true_negatives, false_positives],
                    [false_negatives, true_positives],
                ],
                dtype=np.int64,
            )

    return confusion_matrix


def metrics_from_confusion_matrix(
    confusion_matrix: np.ndarray,
) -> tuple[float, float]:
    true_negatives, false_positives = confusion_matrix[0]
    false_negatives, true_positives = confusion_matrix[1]
    total = confusion_matrix.sum()
    accuracy = (true_positives + true_negatives) / total
    f1_denominator = 2 * true_positives + false_positives + false_negatives
    f1_score = 2 * true_positives / f1_denominator if f1_denominator else 0.0
    return accuracy, f1_score


def print_final_metrics(confusion_matrix: np.ndarray) -> None:
    total = confusion_matrix.sum()
    accuracy, _ = metrics_from_confusion_matrix(confusion_matrix)
    class_names = ("Benign/Background (0)", "Malicious (1)")

    print("\nFinal test metrics")
    print(f"Overall accuracy: {accuracy:.4f}")
    print(
        f"{'Class':<24} {'Accuracy':>10} {'Precision':>10} "
        f"{'Recall':>10} {'F1-score':>10} {'Support':>10}"
    )
    for class_index, class_name in enumerate(class_names):
        true_positive = confusion_matrix[class_index, class_index]
        false_positive = confusion_matrix[:, class_index].sum() - true_positive
        false_negative = confusion_matrix[class_index, :].sum() - true_positive
        true_negative = total - true_positive - false_positive - false_negative

        class_accuracy = (
            (true_positive + true_negative) / total if total else 0.0
        )
        precision = (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive
            else 0.0
        )
        recall = (
            true_positive / (true_positive + false_negative)
            if true_positive + false_negative
            else 0.0
        )
        f1_denominator = precision + recall
        f1_score = (
            2 * precision * recall / f1_denominator if f1_denominator else 0.0
        )
        support = confusion_matrix[class_index, :].sum()
        print(
            f"{class_name:<24} {class_accuracy:>10.4f} {precision:>10.4f} "
            f"{recall:>10.4f} {f1_score:>10.4f} {support:>10}"
        )

    print("\nConfusion matrix (rows = actual, columns = predicted)")
    print("                    Predicted 0  Predicted 1")
    print(
        f"Actual 0 (Benign)   {confusion_matrix[0, 0]:>11} "
        f"{confusion_matrix[0, 1]:>12}"
    )
    print(
        f"Actual 1 (Malicious){confusion_matrix[1, 0]:>12} "
        f"{confusion_matrix[1, 1]:>12}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train an LSTM binary classifier on chronological network flows."
    )
    parser.add_argument(
        "--train-csv", type=Path, default=Path("unified_captures_train.csv")
    )
    parser.add_argument(
        "--test-csv", type=Path, default=Path("unified_captures_test.csv")
    )
    parser.add_argument("--window-size", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.epochs < 1:
        raise ValueError("epochs must be at least 1")
    if args.batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if args.learning_rate <= 0:
        raise ValueError("learning_rate must be greater than 0")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_dataset = FlowSequenceDataset(args.train_csv, args.window_size)
    test_dataset = FlowSequenceDataset(args.test_csv, args.window_size)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True
    )
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FlowLSTM(input_dim=len(FEATURE_COLUMNS)).to(device)
    loss_function = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    print(f"Training on {device}")
    final_confusion_matrix = None
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, loss_function, optimizer, device
        )
        final_confusion_matrix = evaluate(model, test_loader, device)
        accuracy, f1_score = metrics_from_confusion_matrix(final_confusion_matrix)
        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Test Accuracy: {accuracy:.4f} | "
            f"Test F1: {f1_score:.4f}"
        )
    if final_confusion_matrix is not None:
        print_final_metrics(final_confusion_matrix)


if __name__ == "__main__":
    main()
