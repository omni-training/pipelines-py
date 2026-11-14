# Original author: Ilaria Billato (https://github.com/billila)
# Source: billila/pca_scwf_paper, wfsc/sc_mix/scanpy_sc_mix.py @ commit 99bdef3
# Modified by: Tom Kitak (Omnibenchmark compatibility)
# Modified by: Mark Robinson (bringing to Omnibenchmark 0.8.0 API)
# License: MIT (see repository LICENSE)

import os
import sys
import argparse
import json
import time
from pathlib import Path

import pandas as pd
import polars as pl
from brisc import SingleCell

# Parse command line arguments
parser = argparse.ArgumentParser(description="Scanpy single-cell analysis pipeline")

# dummy args (old api)
parser.add_argument("--output_dir", "-o", type=str, required=True, help="Output directory")
parser.add_argument("--name", "-n", type=str, required=True, help="Dataset name")

# inputs
parser.add_argument("--data_h5ad", dest="data_h5ad", type=str, required=True, help="Input h5ad file")
parser.add_argument("--data_clusters_truth_num", dest="data_clusters_truth_num", type=str,
                    required=True, help="input true number of clusters path")

# parameters
parser.add_argument("--method_name", type=str, choices=["scanpy", "rapids"], help="Method to run")
parser.add_argument("--d_cluster", type=int, help="delta to true number of clusters")
parser.add_argument("--n_comp", type=int, default=50, help="number of PCA components")
parser.add_argument("--n_neig", type=int, default=15, help="number of neighbors")
parser.add_argument("--n_hvg", type=int, default=1000, help="number of highly variable genes")
parser.add_argument("--filter", type=str, choices=["manual"], default="manual", 
                    help="filtering strategy (manual uses suggested cutoffs; auto uses package QC)")

def key_path(s):
    key, sep, path = s.partition("=")  # splits on the first "=" only
    if not sep or not key or not path:
        raise argparse.ArgumentTypeError(f"expected key=path, got {s!r}")
    return key, path

parser.add_argument("--output", type=key_path, action="append", default=[],
                    help="key=pair list of outputs")
args, _ = parser.parse_known_args()
outputs = dict(args.output)


# Logging
print(f"Full command: {' '.join(sys.argv)}")
for k in ("method_name", "d_cluster", "n_comp", "n_hvg", "filter"):
    print(f"  {k}: {getattr(args, k)}")

for key, path in outputs.items():
    print(f"[outputs] {key}: {path}")

# time object to store time involved (in seconds) in each step
timings: dict[str, float | None] = {
    "gpu_load": None,
    "find_mit_gene": None,
    "filter": None,
    "normalization": None,
    "hvg": None,
    "scaling": None,
    "pca": None,
    "t_sne": None,
    "umap": None,
    "louvain": None,
    "leiden": None,
}

clustering_info: dict[str, dict[str, float | int | None]] = {
    "resolutions": {
        "louvain": None,
        "leiden": None,
    },
    "num_runs": {
        "louvain": None,
        "leiden": None,
    },
}

def timed(key, fn):
    """Run fn(), store elapsed seconds in timings[key], return the result."""
    start = time.perf_counter()
    result = fn()
    timings[key] = time.perf_counter() - start
    return result


def leiden_to_target(sc, n_target, max_runs=50, max_resolution=100.0):
    """
    Search the Leiden resolution that gives n_target clusters.
    First doubles the resolution until there are enough clusters, then bisects.
    Returns (dataset with obs['leiden'], resolution used, number of runs).
    """
    runs = 0
    best = None  # (abs diff, dataset, resolution)

    def run(res):
        nonlocal runs, best
        runs += 1
        out = sc.cluster(resolution=res, cluster_column="leiden", overwrite=True)
        k = out.obs["leiden"].drop_nulls().n_unique()
        diff = abs(k - n_target)
        if best is None or diff < best[0]:
            best = (diff, out, res)
        return k

    # expand upper bound
    lo, hi = 0.0, 1.0
    k = run(hi)
    while k < n_target and hi < max_resolution and runs < max_runs:
        lo, hi = hi, hi * 2
        k = run(hi)

    # bisect
    while k != n_target and runs < max_runs:
        mid = (lo + hi) / 2
        k = run(mid)
        if k < n_target:
            lo = mid
        elif k > n_target:
            hi = mid

    return best[1], best[2], runs


# data ####
sc = SingleCell(args.data_h5ad)
# only dedupe when needed: brisc refuses if any name already contains the
# separator, and gene names like "MT-CO1" contain "-"
if sc.var_names.is_duplicated().any():
    sc = sc.make_var_names_unique(separator="__")
with open(args.data_clusters_truth_num, "r") as f:
    clusters_truth_num = int(f.read())
n_cluster = clusters_truth_num + args.d_cluster

# QC metrics (mitochondrial fraction, genes and counts per cell)
# allow_float=True: many h5ad files store raw counts as float32
sc = timed("find_mit_gene", lambda: sc.qc_metrics(allow_float=True))

qc = sc.uns['qc_thresholds']
min_genes = int(sc.uns['qc_thresholds']["min"][0])
#max_genes = int(sc.uns['qc_thresholds']["max"][0])
#max_counts = int(sc.uns['qc_thresholds']["max"][0])
max_mt = sc.uns['qc_thresholds']["max"][2]/100.

# filtering: brisc defaults (<=5% mito, >=100 genes, non-zero MALAT1);
# subset=True drops failing cells, like scanpy does
if args.filter == "manual":
    sc = timed("filter", lambda: sc.qc(subset=True, allow_float=True, 
                                       max_mito_fraction = max_mt,
                                       min_genes = min_genes, nonzero_MALAT1=False))


# HVGs (brisc runs this on raw counts, so before normalization)
sc = timed("hvg", lambda: sc.hvg(num_genes=args.n_hvg))

# normalization (log1pPF, brisc default)
sc = timed("normalization", lambda: sc.normalize())

# PCA on the HVGs
sc = timed("pca", lambda: sc.pca(num_PCs=args.n_comp))

# KNN graph + shared nearest neighbor graph
sc = timed(
    "neighbors",
    lambda: sc.neighbors(num_neighbors=args.n_neig).shared_neighbors(),
)

# UMAP
sc = timed("umap", lambda: sc.umap())

# Leiden clustering, tuned to n_cluster
start = time.perf_counter()
sc, leiden_res, leiden_runs = leiden_to_target(sc, n_cluster)
timings["leiden"] = time.perf_counter() - start
clustering_info["resolutions"]["leiden"] = leiden_res
clustering_info["num_runs"]["leiden"] = leiden_runs

# Save timings as JSON
with open(outputs["methods_timings"], "w") as f:
    json.dump(timings, f, indent=2)

print(f"after timings dump")

# Save clustering metadata as JSON
with open(outputs["methods_clustering_info"], "w") as f:
    json.dump(clustering_info, f, indent=2)

# Save cluster assignments as TSV
clusters = sc.obs.select(pl.col("_index").alias("cell_id"),
                         pl.col("leiden").cast(pl.String),
                         pl.col("leiden").cast(pl.String).alias("louvain")) # hack: no louvain - brisc 
clusters.write_csv(outputs["methods_clusters"], separator="\t")


hvgs = sc.var.filter(pl.col("highly_variable")).sort("highly_variable_rank")
hvgs.select("_index").write_csv(outputs["methods_hvgs"], separator="\t", include_header=False)

print(f"after HVGs TSV")

# Save PCA coordinates as TSV
pca_df = pd.DataFrame(
    sc.obsm["pca"],
    index=sc.obs_names,
    columns=[f"PC{i + 1}" for i in range(sc.obsm["pca"].shape[1])],
)
pca_df.to_csv(outputs["methods_pca"], sep="\t", index=True, index_label="cell_id")
print(f"after PCA TSV")
