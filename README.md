# Learned search direction for incremental cycle detection

## Motivation
"A new approach to incremental cycle detection and related problems" (2015) studies detecting a cycle in a directed graph that grows by arc insertions, using bounded two-way searches. This experiment asks whether a cheap learned model can choose which direction to search and so reduce the work per insertion.

## Question
Can a LightGBM model that picks the cheaper search direction from graph-degree features cut the arcs traversed by exact incremental cycle detection on a real, structured arc-insertion stream, compared with a balanced bidirectional search?

## Dataset
SNAP cit-HepTh citation graph (https://snap.stanford.edu/data/cit-HepTh.txt.gz), used as a stand-in for a software dependency graph. It is **not** the paper's dataset. Arcs are inserted in node-id order, and every 20th insertion (u,v) is a query asking whether v already reaches u. The first 60% of queries train the model and the last 40% are the test queries. The graph keeps growing through the stream.

## Baseline vs improved
- **Baseline:** balanced bidirectional search. It alternates one arc forward from v and one arc backward from u. It stops when a cycle is found or either side is exhausted. This is a simplified stand-in, not the paper's exact algorithm.
- **Improved:** a LightGBM regressor, tuned with time-ordered cross-validation on the training queries only. It uses degree, 2-hop degree and graph-size features to predict which side has the smaller search cost. Only that one-sided exact search is run. Cycle answers are checked to be identical for both methods.

Main metric: total arcs traversed over test queries (lower is better). Secondary metrics: mean arcs per query and the share of queries where the model picks the cheaper side.

## How to run
```
pip install -r requirements.txt
python analysis.py
```

## Results
Results are written to `results.json` (and a chart to `figures/`).

## Reference

This project is motivated by the following paper:

> MA Bender, JT Fineman, S Gilbert (2015). *A new approach to incremental cycle detection and related problems*. ACM Transactions on. https://dl.acm.org/doi/abs/10.1145/2756553

Paper author: Prof. Jeremy Fineman, Georgetown University (jfineman@cs.georgetown.edu)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
