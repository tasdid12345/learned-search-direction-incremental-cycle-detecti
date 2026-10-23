# Learned search-direction choice for incremental cycle detection

Question: can a LightGBM model that picks the cheaper search direction (from degree features) reduce the arcs traversed by exact incremental cycle detection, compared with a balanced bidirectional search?

Data: SNAP cit-HepTh (arcs citing->cited, inserted in node-id order). The script downloads the official file from snap.stanford.edu; if that host is unreachable it falls back to http and to Internet Archive snapshots of the same file. If none are reachable it exits with an error (no synthetic data is ever used).

Method: every 20th insertion (u, v) is a query "does v reach u?". Chronological 60/40 train/test query split. The LightGBM regressor is tuned by cross-validation on the training queries only (seed 42), and the baseline is balanced bidirectional search. Results are written to `results.json` and `figures/arcs.png`.

Run: `pip install -r requirements.txt && python analysis.py`

## Results (latest run)

On the SNAP cit-HepTh arc-insertion stream, total arcs traversed over 7056 test queries fell from 173617401 for a standard balanced bidirectional search to 161775713 for the LightGBM-chosen one-sided search, a small gain.

| Metric | Value |
|---|---|
| baseline_score | 173617401 |
| improved_score | 161775713 |
| improvement | 11841688 |
| baseline_score_std | 0 |
| improved_score_std | 318200 |
| baseline_mean_arcs_per_query | 24610 |
| improved_mean_arcs_per_query | 22930 |
| improved_cheaper_side_share | 0.8151 |
| oracle_total_arcs | 126901615 |
| always_forward_total_arcs | 197933138 |
| always_backward_total_arcs | 286971683 |
| test_cycle_share | 0.112 |

Run on 2026-10-23. Full output: [results.json](results.json) · [run.log](run.log)

Notes: Arcs are citing->cited, so the stream is almost a DAG and one search side is often empty; this limits the room for any method. Train labels cap each one-sided search at 20000 arcs; test costs are exact. Both one-sided costs are computed per query on the evolving graph, so the improved cost is the cost of the chosen side. Baseline is deterministic (std 0); improved varies with LightGBM subsampling seeds 42-44. Answers verified identical. cit-HepTh is a citation graph used as a stand-in for dependency graphs. Data file was fetched from: https://web.archive.org/web/2023id_/https://snap.stanford.edu/data/cit-HepTh.txt.gz (same SNAP file; fallback mirrors are used only if the official host is unreachable).

## Reference

This project is motivated by the following paper:

> MA Bender, JT Fineman, S Gilbert (2015). *A new approach to incremental cycle detection and related problems*. ACM Transactions on. https://dl.acm.org/doi/abs/10.1145/2756553

Paper author: Prof. Jeremy Fineman, Georgetown University (jfineman@cs.georgetown.edu)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
