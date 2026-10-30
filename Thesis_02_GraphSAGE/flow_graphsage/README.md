# Flow GraphSAGE classifier

This project classifies each flow as **benign/background** (class 0) or
**malicious** (class 1). Each flow is a graph node. Directed edges join flows whose
`SrcAddr` and `DstAddr` both match exactly, excluding self-loops. Each node
uses at most 10 deterministically sampled neighbors from that exact address
pair.

Node features are `Dur`, `TotPkts`, `TotBytes`, `SrcBytes`, and every
`State_*` column. `NormLabel` supplies the target; `Label` is not used as a
feature. `Benign` and `Background` map to class 0, while `Malicious` maps to
class 1. The test graph is built separately from the training graph.

## Run

From the dataset folder, install the dependencies and train:

```powershell
python -m pip install -r .\flow_graphsage\requirements.txt
python -m flow_graphsage.train --epochs 10
```

The two CSVs are expected in the dataset folder by default. Override their
locations with `--train-csv` and `--test-csv`. The trained checkpoint is saved
to `flow_graphsage\graphsage_binary.pt` by default; change that with `--output`.

Training uses a stratified 90/10 split of the training CSV for fitting and
validation. The saved checkpoint includes the learned model weights and
training feature-normalization statistics.
