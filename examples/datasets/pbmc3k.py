import scanpy as sc
import numpy as np

adata = sc.datasets.pbmc3k()
# remove cells with too few genes
sc.pp.filter_cells(adata, min_genes=200)

# remove genes expressed in too few cells
sc.pp.filter_genes(adata, min_cells=3)

# normalize each cell to the same total count
sc.pp.normalize_total(adata, target_sum=1e4)

# log(1 + x)
sc.pp.log1p(adata)

sc.pp.highly_variable_genes(adata, n_top_genes=2000, flavor="seurat")

# keep only HVGs
adata = adata[:, adata.var.highly_variable]
sc.pp.scale(adata, max_value=10)
sc.tl.pca(adata, n_comps=50, svd_solver="arpack")

X = adata.obsm["X_pca"]  # shape: (cells, 50)
np.savetxt("pbmc3k_pca50.txt", X)
