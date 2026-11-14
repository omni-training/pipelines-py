# Original author: Ilaria Billato (https://github.com/billila)
# Source: billila/pca_scwf_paper, wfsc/sc_mix/scanpy_sc_mix.py @ commit 99bdef3
# Modified by: Tom Kitak (Omnibenchmark compatibility)
# Modified by: Mark Robinson (bringing to Omnibenchmark 0.8.0 API)
# License: MIT (see repository LICENSE)

import os
import sys
import argparse
import json
from pathlib import Path

import pandas as pd
import scanpy as sc

#repo_dir = Path(__file__).parent
#sys.path.insert(0, str(repo_dir))

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
for k in ("method_name", "d_cluster", "n_comp", "n_hvg", "filter", "output"):
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

sc.settings.verbosity = 3

# data ####
adata = sc.read_h5ad(args.data_h5ad)
adata.var_names_make_unique()
with open(args.data_clusters_truth_num, "r") as f:
    clusters_truth_num = int(f.read())
n_cluster = clusters_truth_num + args.d_cluster


if args.method_name == "scanpy":
    from methods.run_scanpy import run_scanpy
    adata = run_scanpy(
        adata,
        n_cluster,
        args.n_comp,
        args.n_neig,
        args.n_hvg,
        args.filter,
        timings,
        clustering_info,
    )
elif args.method_name == "rapids":
    from methods.run_rapids import run_rapids
    adata = run_rapids(
        adata,
        n_cluster,
        args.n_comp,
        args.n_neig,
        args.n_hvg,
        args.filter,
        timings,
        clustering_info,
    )

# Save timings as JSON
with open(outputs["methods_timings"], "w") as f:
    json.dump(timings, f, indent=2)

# Save clustering metadata as JSON
with open(outputs["methods_clustering_info"], "w") as f:
    json.dump(clustering_info, f, indent=2)

# Save cluster assignments as TSV
adata.obs[["louvain", "leiden"]].reset_index(names="cell_id").to_csv(
    outputs["methods_clusters"], sep="\t", index=False)

hvg_list = adata.var.highly_variable.index.to_series()
hvg_list.to_csv(outputs["methods_hvgs"], sep="\t", index=False, header=False)

# Save PCA coordinates as TSV
pca_df = pd.DataFrame(
    adata.obsm["X_pca"],
    index=adata.obs_names,
    columns=[f"PC{i + 1}" for i in range(adata.obsm["X_pca"].shape[1])],
)
pca_df.to_csv(outputs["methods_pca"], sep="\t", index=True, index_label="cell_id")
