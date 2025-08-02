import scanpy as sc
import numpy as np
import scipy.sparse
from UNI_histology_extractor import UNIExtractor
import os
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2, 40).__str__()

def process(ref_dir, tgt_dir, ref_name, tgt_name, n_genes=3000, overlap=None, preprocess=False):
    #ref = []
    ref, ref_img, ref_pos = [], [], []
    UNI = UNIExtractor()

    for r in ref_name:
        adata = sc.read(ref_dir + r + '.h5ad')
        position = adata.obsm['spatial']
        spot_diameter = int(adata.uns['spot_diameter'])
        image = UNI.extract(ref_dir + r + '.jpg', position, crop_size=spot_diameter)
        ref.append(adata)
        ref_img.append(image)
        ref_pos.append(position)
        print(f'Section {r} is processed...')

    tgt_adata = sc.read_visium(tgt_dir)
    tgt_adata.var_names_make_unique()

    mask = tgt_adata.obs['in_tissue'] == 1
    tgt_adata._inplace_subset_obs(mask)
    tgt_pos = tgt_adata.obsm['spatial']

    sample_id = [k for k in tgt_adata.uns["spatial"].keys() if k != "is_single"][0]
    spot_diameter = int(tgt_adata.uns["spatial"][sample_id]["scalefactors"]["spot_diameter_fullres"])
    tgt_img = UNI.extract(tgt_dir + '\\image.tif', tgt_pos, crop_size=spot_diameter)
    print(f'Target Section is processed...')

    ref[0].var_names_make_unique()
    overlap_genes = set(ref[0].var_names)
    for r in ref[1:]:
        r.var_names_make_unique()
        overlap_genes &= set(r.var_names)

    tgt_adata.var_names_make_unique()
    overlap_genes &= set(tgt_adata.var_names)
    overlap_genes = list(overlap_genes)

    ref = [r[:, overlap_genes] for r in ref]
    tgt_adata = tgt_adata[:, overlap_genes]

    for d in ref:
        sc.pp.normalize_total(d, target_sum=1e4)
        sc.pp.log1p(d)
    sc.pp.normalize_total(tgt_adata, target_sum=1e4)
    sc.pp.log1p(tgt_adata)

    sc.pp.highly_variable_genes(ref[0], flavor="seurat_v3", n_top_genes=n_genes, subset=True)

    ref = [d[:, ref[0].var_names] for d in ref]
    tgt_adata = tgt_adata[:, ref[0].var_names]

    ref_expr = [adata.X.toarray() if scipy.sparse.issparse(adata.X) else adata.X for adata in ref]
    del ref
    tgt_expr = tgt_adata.X.toarray() if scipy.sparse.issparse(tgt_adata.X) else tgt_adata.X
    del tgt_adata

    np.savez_compressed(
        f"data\\train_dataset_{tgt_name}1111.npz",
        img_feats=np.array(ref_img, dtype=object),
        gene_exprs=np.array(ref_expr, dtype=object),
        coords=np.array(ref_pos, dtype=object)
    )

    np.savez_compressed(
        f"data\\test_dataset_{tgt_name}1111.npz",
        img_feats=np.array(tgt_img),
        gene_exprs=np.array(tgt_expr),
        coords=np.array(tgt_pos)
    )

ref_dir = "data\\HER2ST\\"
ref_name = ['A1', 'B1', 'C1', 'D1', 'E1', 'F1', 'G2', 'H1']
tgt_dir = "data\\Visium_HBC"
tgt_name = 'Visium_HBC'
process(ref_dir, tgt_dir, ref_name, tgt_name)