# Contributing

Use Python 3.12 and install requirements.txt in an isolated environment. Run `python -m unittest discover -s tests -v` and `python verify_artifacts.py artifacts` before proposing changes.

Keep units and receiving-body sign conventions explicit. When changing the solver, verify all three bodies' force and moment balances and an independently calculated case. Changes to feature order, graph structure, output order or model architecture require compatible new weights and a documented retraining run.

Use a new directory under runs/ for experiments. Do not commit virtual environments, TFRecords, local logs or secrets. Report validation and test metrics separately; do not use the test set to select a checkpoint. State when a result is synthetic rather than experimental.
