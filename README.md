# Learned search-direction choice for incremental cycle detection

Question: can a LightGBM model that picks the cheaper search direction (from degree features) reduce the arcs traversed by exact incremental cycle detection, compared with a balanced bidirectional search?

Data: SNAP cit-HepTh (arcs citing->cited, inserted in node-id order). The script downloads the official file from snap.stanford.edu; if that host is unreachable it falls back to http and to Internet Archive snapshots of the same file. If none are reachable it exits with an error (no synthetic data is ever used).

Method: every 20th insertion (u, v) is a query "does v reach u?". Chronological 60/40 train/test query split. The LightGBM regressor is tuned by cross-validation on the training queries only (seed 42), and the baseline is balanced bidirectional search. Results are written to `results.json` and `figures/arcs.png`.

Run: `pip install -r requirements.txt && python analysis.py`

## Reference

This project is motivated by the following paper:

> MA Bender, JT Fineman, S Gilbert (2015). *A new approach to incremental cycle detection and related problems*. ACM Transactions on. https://dl.acm.org/doi/abs/10.1145/2756553

Paper author: Prof. Jeremy Fineman, Georgetown University (jfineman@cs.georgetown.edu)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
