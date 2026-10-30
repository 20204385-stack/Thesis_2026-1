"""Train the flow GCN and report metrics on the test split."""

import argparse
import gc
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch import nn

from data import DirectionNeighborLoader, build_flow_graph, make_loader
from model import GCNClassifier


BASE_DIR = Path(__file__).resolve().parent
TRAIN_CSV = BASE_DIR / "unified_captures_train.csv"
TEST_CSV = BASE_DIR / "unified_captures_test.csv"


def train_epoch(
    model: GCNClassifier,
    loader: DirectionNeighborLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.CrossEntropyLoss,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    total_examples = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        logits = model(batch.x, batch.edge_index)[: batch.batch_size]
        targets = batch.y[: batch.batch_size]
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.batch_size
        total_examples += batch.batch_size
    return total_loss / total_examples


@torch.no_grad()
def evaluate(
    model: GCNClassifier, loader: DirectionNeighborLoader, device: torch.device
) -> tuple[list[int], list[int]]:
    model.eval()
    predictions = []
    targets = []
    for batch in loader:
        batch = batch.to(device)
        logits = model(batch.x, batch.edge_index)[: batch.batch_size]
        predictions.extend(logits.argmax(dim=1).cpu().tolist())
        targets.extend(batch.y[: batch.batch_size].cpu().tolist())

    return targets, predictions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, default=TRAIN_CSV)
    parser.add_argument("--test-csv", type=Path, default=TEST_CSV)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--hidden-dim", type=int, default=64)
    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 1 or args.hidden_dim < 1:
        parser.error("--epochs, --batch-size, and --hidden-dim must be positive.")
    if not args.train_csv.is_file():
        raise FileNotFoundError(f"Training CSV not found: {args.train_csv}")
    if not args.test_csv.is_file():
        raise FileNotFoundError(f"Test CSV not found: {args.test_csv}")

    train_frame = pd.read_csv(args.train_csv)
    train_data = build_flow_graph(train_frame, "Training split")
    del train_frame
    train_loader = make_loader(train_data, args.batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GCNClassifier(train_data.num_features, args.hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(model, train_loader, optimizer, criterion, device)
        print(f"Epoch {epoch:02d}/{args.epochs} - loss: {loss:.4f}")

    del train_loader, train_data
    gc.collect()

    test_frame = pd.read_csv(args.test_csv)
    test_data = build_flow_graph(test_frame, "Test split")
    del test_frame
    test_loader = make_loader(test_data, args.batch_size, shuffle=False)

    targets, predictions = evaluate(model, test_loader, device)
    class_labels = [0, 1]
    class_names = ["Benign/Background", "Malicious"]
    matrix = confusion_matrix(targets, predictions, labels=class_labels)

    print("\nTest metrics:")
    print(f"Accuracy: {accuracy_score(targets, predictions):.4f}")
    print(
        "F1 score (malicious): "
        f"{f1_score(targets, predictions, pos_label=1, zero_division=0):.4f}"
    )
    print("\nPer-class metrics:")
    print(
        classification_report(
            targets,
            predictions,
            labels=class_labels,
            target_names=class_names,
            zero_division=0,
            digits=4,
        )
    )
    print("Confusion matrix (rows = actual, columns = predicted):")
    print(f"{'':20}{class_names[0]:>22}{class_names[1]:>14}")
    for class_name, counts in zip(class_names, matrix):
        print(f"{class_name:20}{counts[0]:>22,}{counts[1]:>14,}")
    del test_loader, test_data
    gc.collect()


if __name__ == "__main__":
    main()
