import scanpy as sc
import numpy as np
import scipy.sparse
from torch.utils.data import Dataset
from PIL import Image
from UNI_histology_extractor import UNIExtractor
import os
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2, 40).__str__()


def row_min_max_normalize(features: np.ndarray) -> np.ndarray:

    row_min = features.min(axis=1, keepdims=True)
    row_max = features.max(axis=1, keepdims=True)
    normalized = (features - row_min) / (row_max - row_min + 1e-8)
    return normalized

class ROIDataset(Dataset):
    def __init__(self, img_list, transform):
        super().__init__()
        self.images_lst = img_list
        self.transform = transform

    def __len__(self):
        return len(self.images_lst)

    def __getitem__(self, idx):
        pil_image = Image.fromarray(self.images_lst[idx].astype('uint8'))
        image = self.transform(pil_image)
        return image

def process(ref_dir, ref_name, n_genes=3000, overlap=None, preprocess=False):
    ref, ref_img, ref_pos = [], [], []
    UNI = UNIExtractor()
    for r in ref_name:
        adata = sc.read_visium(ref_dir + r)
        adata.var_names_make_unique()

        mask = adata.obs['in_tissue'] == 1
        adata._inplace_subset_obs(mask)
        position = adata.obsm['spatial']

        sample_id = [k for k in adata.uns["spatial"].keys() if k != "is_single"][0]
        spot_diameter = int(adata.uns["spatial"][sample_id]["scalefactors"]["spot_diameter_fullres"])
        image = UNI.extract(ref_dir + r + '\\spatial\\tissue_hires_image.png', position, crop_size=224)

        ref.append(adata)
        ref_img.append(image)
        ref_pos.append(position)
        print(f'Section {r} is processed...')


    overlap_genes = set(ref[0].var_names)
    for r in ref[1:]:
        overlap_genes &= set(r.var_names)
    overlap_genes = list(overlap_genes)

    ref = [r[:, overlap_genes] for r in ref]

    for d in ref:
        sc.pp.normalize_total(d, target_sum=1e4)
        sc.pp.log1p(d)

    sc.pp.highly_variable_genes(ref[0], flavor="seurat_v3", n_top_genes=n_genes, subset=True)

    ref = [d[:, ref[0].var_names] for d in ref]

    ref_expr = [adata.X.toarray() if scipy.sparse.issparse(adata.X) else adata.X for adata in ref]
    del ref

    np.savez_compressed(
        f"{ref_dir}train_dataset.npz",
        img_feats=np.array(ref_img, dtype=object),
        gene_exprs=np.array(ref_expr, dtype=object),
        coords=np.array(ref_pos, dtype=object)
    )


ref_dir = "data/CRC/"
ref_name = ['A1', 'A2', 'B1', 'B2', 'C1', 'C2', 'D1', 'D2', 'E1', 'E2', 'F1', 'F2', 'G1', 'G2']

process(ref_dir, ref_name)
