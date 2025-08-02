from torch.utils.data import Dataset
import numpy as np
from scipy.spatial import cKDTree
from scipy.interpolate import interp1d

def normalize(cnts):

    cnts = cnts.copy()

    cnts_min = cnts.min(0)
    cnts_max = cnts.max(0)
    cnts -= cnts_min
    cnts /= (cnts_max - cnts_min) + 1e-12

    return cnts, (cnts_min, cnts_max)

class SpotDataset_CL(Dataset):
    def __init__(self, img_feats, gene_exprs, coords):
        self.img_feats = img_feats
        self.gene_exprs, (_, _) = normalize(gene_exprs)
        self.coords = coords

    def __len__(self):
        return self.img_feats.shape[0]

    def __getitem__(self, idx):
        return (
            self.img_feats[idx],
            self.gene_exprs[idx],
            self.coords[idx]
        )

class SpotDataset_Rec(Dataset):
    def __init__(self, img_feats, gene_exprs, coords, y):
        gene_exprs, (_, _) = normalize(gene_exprs)
        self.img_feats, self.gene_exprs, self.coords, self.y = get_samples_data(img_feats, gene_exprs, coords, y)

    def __len__(self):
        return self.img_feats.shape[0]

    def __getitem__(self, idx):
        return (
            self.img_feats[idx],
            self.gene_exprs[idx],
            self.coords[idx],
            self.y[idx]
        )

def get_samples_data(x, exp, locs, y, k=7):
    """
    Selects the k nearest neighboring cells (including itself) based on coordinates and extracts gene expression data for each cell.

    Parameters:

    locs: np.ndarray, shape (n, 2), coordinates of cells.

    y: np.ndarray, shape (n, 1000), gene expression data of cells.

    k: int, number of nearest neighbors to select for each cell.

    Returns:

    patches: list, each element is a numpy array of shape (k, 3000).
    """
    tree = cKDTree(locs)

    _, indices = tree.query(locs, k=k)

    genes_list = [exp[idx] for idx in indices]
    genes_list = np.stack(genes_list)

    pos_list = [locs[idx] for idx in indices]
    pos_list = np.stack(pos_list)

    img_list = [x[idx] for idx in indices]
    img_list = np.stack(img_list)

    y_list = [y[idx] for idx in indices]
    y_list = np.stack(y_list)
    y_list = y_list[:, 0]

    return img_list, genes_list, pos_list, y_list