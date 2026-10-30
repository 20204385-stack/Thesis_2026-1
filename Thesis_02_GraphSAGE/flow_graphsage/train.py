"""Train and evaluate the binary flow GraphSAGE classifier."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from torch import nn

from flow_graphsage.data import (
    build_edge_index,
    edge_index_to_neighbor_table,
    load_split,
    standardize_features,
)
from flow_graphsage.model import GraphSAGE


MAX_NEIGHBORS = 10
SEED = 42


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description="Train a communication-channel GraphSAGE flow classifier."
    )
    parser.add_argument(
        "--train-csv",
        type=Path,
        default=project_root / "unified_captures_train.csv",
    )
    parser.add_argument(
        "--test-csv",
        type=Path,
        default=project_root / "unified_captures_test.csv",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "graphsage_binary.pt",
    )
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.hidden_size < 1:
        parser.error("--epochs, --batch-size, and --hidden-size must be positive.")
    if args.learning_rate <= 0:
        parser.error("--learning-rate must be positive.")
    return args


def evaluate(
    model: GraphSAGE,
    node_ids: torch.Tensor,
    features: torch.Tensor,
    labels: torch.Tensor,
    adjacency: torch.Tensor,
    batch_size: int,
) -> tuple[float, np.ndarray]:
    model.eval()
    total_loss = 0.0
    predictions: list[np.ndarray] = []
    loss_function = nn.BCEWithLogitsLoss()

    with torch.no_grad():
        for start in range(0, len(node_ids), batch_size):
            batch_ids = node_ids[start : start + batch_size]
            logits = model(batch_ids, features, adjacency)
            batch_labels = labels[batch_ids]
            total_loss += loss_function(logits, batch_labels).item() * len(batch_ids)
            predictions.append((logits >= 0).cpu().numpy().astype(np.int64))

    return total_loss / len(node_ids), np.concatenate(predictions)


def main() -> None:
    args = parse_args()
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    print(f"Loading training flows from {args.train_csv}")
    train_data = load_split(args.train_csv)
    print(f"Loading test flows from {args.test_csv}")
    test_data = load_split(args.test_csv, train_data.feature_columns)

    train_features, test_features, feature_mean, feature_scale = standardize_features(
        train_data.features, test_data.features
    )
    print("Building separate train and test communication-channel graphs")
    train_edge_index = build_edge_index(
        train_data.source_addresses,
        train_data.destination_addresses,
        MAX_NEIGHBORS,
    )
    test_edge_index = build_edge_index(
        test_data.source_addresses,
        test_data.destination_addresses,
        MAX_NEIGHBORS,
    )
    train_adjacency = edge_index_to_neighbor_table(
        train_edge_index, len(train_data.labels), MAX_NEIGHBORS
    )
    test_adjacency = edge_index_to_neighbor_table(
        test_edge_index, len(test_data.labels), MAX_NEIGHBORS
    )

    all_train_ids = np.arange(len(train_data.labels))
    fit_ids, validation_ids = train_test_split(
        all_train_ids,
        test_size=0.1,
        random_state=SEED,
        stratify=train_data.labels,
    )
    fit_labels = train_data.labels[fit_ids]
    positives = int(fit_labels.sum())
    negatives = len(fit_labels) - positives
    if positives == 0 or negatives == 0:
        raise ValueError("The training partition must contain both target classes.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}; sampled neighbors per node: {MAX_NEIGHBORS}")

    train_features_tensor = torch.as_tensor(
        train_features, dtype=torch.float32, device=device
    )
    train_labels_tensor = torch.as_tensor(
        train_data.labels, dtype=torch.float32, device=device
    )
    train_adjacency_tensor = torch.as_tensor(
        train_adjacency, dtype=torch.long, device=device
    )
    test_features_tensor = torch.as_tensor(
        test_features, dtype=torch.float32, device=device
    )
    test_labels_tensor = torch.as_tensor(
        test_data.labels, dtype=torch.float32, device=device
    )
    test_adjacency_tensor = torch.as_tensor(
        test_adjacency, dtype=torch.long, device=device
    )
    validation_ids_tensor = torch.as_tensor(
        validation_ids, dtype=torch.long, device=device
    )

    model = GraphSAGE(len(train_data.feature_columns), args.hidden_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    positive_weight = torch.tensor([negatives / positives], device=device)
    loss_function = nn.BCEWithLogitsLoss(pos_weight=positive_weight)
    rng = np.random.default_rng(SEED)

    for epoch in range(1, args.epochs + 1):
        model.train()
        shuffled_ids = rng.permutation(fit_ids)
        total_loss = 0.0
        for start in range(0, len(shuffled_ids), args.batch_size):
            batch_ids = torch.as_tensor(
                shuffled_ids[start : start + args.batch_size],
                dtype=torch.long,
                device=device,
            )
            logits = model(batch_ids, train_features_tensor, train_adjacency_tensor)
            loss = loss_function(logits, train_labels_tensor[batch_ids])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch_ids)

        validation_loss, validation_predictions = evaluate(
            model,
            validation_ids_tensor,
            train_features_tensor,
            train_labels_tensor,
            train_adjacency_tensor,
            args.batch_size,
        )
        validation_labels = train_data.labels[validation_ids]
        validation_f1 = f1_score(
            validation_labels, validation_predictions, zero_division=0
        )
        print(
            f"Epoch {epoch:02d}/{args.epochs} "
            f"train_loss={total_loss / len(fit_ids):.4f} "
            f"validation_loss={validation_loss:.4f} "
            f"validation_f1={validation_f1:.4f}"
        )

    test_loss, test_predictions = evaluate(
        model,
        torch.arange(len(test_data.labels), device=device),
        test_features_tensor,
        test_labels_tensor,
        test_adjacency_tensor,
        args.batch_size,
    )
    test_labels = test_data.labels.astype(np.int64)
    print(f"\nTest loss: {test_loss:.4f}")
    print(f"Accuracy:  {accuracy_score(test_labels, test_predictions):.4f}")
    print(
        "Precision: "
        f"{precision_score(test_labels, test_predictions, zero_division=0):.4f}"
    )
    print(
        "Recall:    "
        f"{recall_score(test_labels, test_predictions, zero_division=0):.4f}"
    )
    print(f"F1:        {f1_score(test_labels, test_predictions, zero_division=0):.4f}")
    print(
        "Confusion matrix (rows=true, columns=predicted; "
        "0=benign/background, 1=malicious):"
    )
    print(confusion_matrix(test_labels, test_predictions, labels=[0, 1]))
    print(
        classification_report(
            test_labels,
            test_predictions,
            labels=[0, 1],
            target_names=["benign/background", "malicious"],
            zero_division=0,
        )
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "feature_columns": train_data.feature_columns,
            "feature_mean": feature_mean,
            "feature_scale": feature_scale,
            "hidden_size": args.hidden_size,
            "max_neighbors": MAX_NEIGHBORS,
            "target_mapping": {"benign_or_background": 0, "malicious": 1},
        },
        args.output,
    )
    print(f"Saved model checkpoint to {args.output}")


if __name__ == "__main__":
    main()
