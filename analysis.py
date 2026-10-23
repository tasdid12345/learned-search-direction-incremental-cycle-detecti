"""Learned search-direction choice vs balanced bidirectional search for incremental cycle detection
on the SNAP cit-HepTh citation graph (arcs citing->cited, inserted in node-id order)."""
import gzip, io, json, os, sys, time
import numpy as np, pandas as pd, requests
import lightgbm as lgb
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

SEED, K, CAP = 42, 20, 20000   # every K-th insertion is a query; CAP only limits TRAIN labels
URL = "https://snap.stanford.edu/data/cit-HepTh.txt.gz"
np.random.seed(SEED)

def download(url):
    for i in range(5):  # retry with backoff on 429/5xx
        try:
            r = requests.get(url, timeout=60)
            if r.status_code == 429 or r.status_code >= 500:
                raise requests.HTTPError(str(r.status_code))
            r.raise_for_status()
            return r.content
        except requests.RequestException as e:
            print("download retry", i, e); time.sleep(2 ** i)
    sys.exit("download failed")

raw = gzip.decompress(download(URL))
df = pd.read_csv(io.BytesIO(raw), sep=r"\s+", comment="#", names=["u", "v"], dtype="int64")
df = df[df.u != df.v].drop_duplicates().sort_values(["u", "v"])  # node-id order stream
arcs = list(zip(df.u.tolist(), df.v.tolist()))
print("arcs in stream:", len(arcs))

def side(adj, s, seen):
    """Lazy DFS: yields one scanned arc head per step (shared by all methods)."""
    stack = [s]
    while stack:
        x = stack.pop()
        for y in adj.get(x, ()):
            yield y
            if y not in seen:
                seen.add(y); stack.append(y)

def one_sided(adj, s, t, cap=None):
    """Exact one-sided search from s for t. Returns (found, arcs traversed)."""
    cost = 0
    for y in side(adj, s, {s}):
        cost += 1
        if y == t: return True, cost
        if cap and cost >= cap: break
    return False, cost

def balanced(out, inn, u, v):
    """Alternate one arc forward from v and one backward from u; stop on meeting or exhaustion."""
    fs, bs = {v}, {u}
    gens = [(side(out, v, fs), bs), (side(inn, u, bs), fs)]
    cost, t = 0, 0
    while True:
        g, other = gens[t]
        try: y = next(g)
        except StopIteration: return False, cost
        cost += 1
        if y in other: return True, cost
        t ^= 1

# ---- single pass over the growing stream; searches are deterministic given the graph state
nq = len(range(0, len(arcs), K)); n_train = int(0.6 * nq)
out, inn, rows = {}, {}, []
for i, (u, v) in enumerate(arcs):
    if i % K == 0:
        qi = i // K; test = qi >= n_train
        fo, fi = out.get(v, []), inn.get(u, [])
        feat = [len(fo), sum(len(out.get(w, ())) for w in fo),      # forward side: v out-deg, 2-hop
                len(fi), sum(len(inn.get(w, ())) for w in fi),      # backward side: u in-deg, 2-hop
                len(inn.get(v, ())), len(out.get(u, ())), i, len(out)]  # other degrees, graph size
        cap = None if test else CAP
        ff, cf = one_sided(out, v, u, cap); fb, cb = one_sided(inn, u, v, cap)
        bf, bc = balanced(out, inn, u, v) if test else (None, 0)
        rows.append(feat + [cf, cb, ff, fb, bf, bc, test])
    out.setdefault(u, []).append(v); inn.setdefault(v, []).append(u)
cols = ["f_od", "f_od2", "b_id", "b_id2", "v_id", "u_od", "n_arcs", "n_nodes",
        "cf", "cb", "ff", "fb", "bf", "bc", "test"]
Q = pd.DataFrame(rows, columns=cols); FEATS = cols[:8]
tr, te = Q[~Q.test.astype(bool)], Q[Q.test.astype(bool)]
print("queries train/test:", len(tr), len(te))

# ---- correctness: all exact methods must give the same cycle answer on test queries
assert (te.ff == te.fb).all() and (te.ff == te.bf).all(), "answers differ between methods"

# ---- LightGBM regressor: log1p(forward cost) - log1p(backward cost); <0 -> search forward
ytr = np.log1p(tr.cf) - np.log1p(tr.cb)
gs = GridSearchCV(lgb.LGBMRegressor(random_state=SEED, subsample=0.8, subsample_freq=1,
                                    colsample_bytree=0.8, verbose=-1, n_jobs=2),
                  {"num_leaves": [7, 15], "n_estimators": [100, 300]},
                  cv=TimeSeriesSplit(3), scoring="neg_mean_squared_error")  # CV on train only
gs.fit(tr[FEATS], ytr)
print("best params:", gs.best_params_)

base_total = int(te.bc.sum())
oracle_total = int(np.minimum(te.cf, te.cb).sum())
tot, share = [], []
for s in (42, 43, 44):
    m = lgb.LGBMRegressor(**{**gs.best_estimator_.get_params(), "random_state": s}).fit(tr[FEATS], ytr)
    fwd = m.predict(te[FEATS]) < 0   # test set used once, here
    chosen = np.where(fwd, te.cf, te.cb)
    tot.append(int(chosen.sum()))
    share.append(float((chosen <= np.minimum(te.cf, te.cb)).mean()))
    print("seed", s, "improved total arcs", tot[-1])
if base_total == 0 or len(te) < 100:
    sys.exit("no meaningful metrics")

b, imp = float(base_total), float(np.mean(tot))
always_f, always_b = int(te.cf.sum()), int(te.cb.sum())
res = {
    "question": "Can a LightGBM model that picks the cheaper search direction from graph-degree features cut the arcs traversed by exact incremental cycle detection on a real arc-insertion stream, compared with a balanced bidirectional search?",
    "method": "cit-HepTh arcs inserted in node-id order; every K-th insertion (u,v) queries whether v reaches u. Chronological 60/40 query split. LightGBM regressor (CV-tuned on train) predicts which side has the cheaper exact one-sided search; compared with balanced bidirectional search.",
    "dataset": {"name": "SNAP cit-HepTh", "source_url": URL, "n_rows": len(arcs),
                "n_train": len(tr), "n_test": len(te), "same_as_paper": False,
                "baseline_is_papers_method": False},
    "metric_name": "total arcs traversed over test queries", "higher_is_better": False,
    "baseline": "Balanced bidirectional search (alternating one arc forward from v and backward from u; simplified stand-in, not the paper's algorithm).",
    "improved": "LightGBM regressor on degree/2-hop/graph-size features picks one side; only that exact one-sided search is run.",
    "metrics": {
        "baseline_score": b, "improved_score": imp, "improvement": b - imp,
        "baseline_score_std": 0.0, "improved_score_std": float(np.std(tot)),
        "baseline_mean_arcs_per_query": b / len(te), "improved_mean_arcs_per_query": imp / len(te),
        "improved_cheaper_side_share": float(np.mean(share)),
        "oracle_total_arcs": float(oracle_total),
        "always_forward_total_arcs": float(always_f), "always_backward_total_arcs": float(always_b),
        "test_cycle_share": float(te.ff.astype(bool).mean())},
    "notes": ("Arcs are citing->cited, so the stream is almost a DAG and one search side is often empty; "
              "this limits the room for any method. Train labels cap each one-sided search at "
              f"{CAP} arcs; test costs are exact. Both one-sided costs are computed per query on the "
              "evolving graph, so the improved cost is the cost of the chosen side. Baseline is deterministic "
              "(std 0); improved varies with LightGBM subsampling seeds 42-44. Answers verified identical. "
              "cit-HepTh is a citation graph used as a stand-in for dependency graphs."),
}
with open("results.json", "w", encoding="utf-8") as f: json.dump(res, f, indent=2)
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
os.makedirs("figures", exist_ok=True)
plt.bar(["balanced", "LightGBM side", "oracle"], [b, imp, oracle_total])
plt.ylabel("arcs traversed (test)"); plt.tight_layout(); plt.savefig("figures/arcs.png")
print(json.dumps(res["metrics"], indent=2))
