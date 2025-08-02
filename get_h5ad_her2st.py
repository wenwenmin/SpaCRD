import numpy as np
from PIL import Image
from skimage.transform import rescale
import pandas as pd
import os
from anndata import AnnData

samples = ['H1']
root_dir = r"data\\HER2ST\HE2Exp_cluster"

for sample in samples:
    print(f"Processing {sample}...")
    dir = os.path.join(root_dir, sample)

    counts = pd.read_csv(f"{dir}\\ut_{sample}_stdata_filtered.tsv", sep='\t', index_col=0, header=0)
    coords = pd.read_csv(f"{dir}\\{sample}_selection.tsv", sep='\t', index_col=None, header=0)

    xy = np.zeros((len(counts), 2))
    for i in range(len(counts)):
        index_1 = counts.index[i]
        for j in range(len(coords)):
            index_2 = f"{coords.iloc[j, 0]}x{coords.iloc[j, 1]}"
            if index_1 == index_2:
                xy[i, 0] = coords.iloc[j, 4]
                xy[i, 1] = coords.iloc[j, 5]
                break

    loc = pd.DataFrame(xy, columns=['x', 'y'])
    loc = loc.round().astype(int)

    adata = AnnData(X=counts.values)
    adata.obs_names = counts.index
    adata.var_names = counts.columns

    adata.obsm["spatial"] = loc[["x", "y"]].values

    raw_pix_size = np.loadtxt(f'{dir}\\pixel-size-raw.txt')
    spot_diameter = 100 // raw_pix_size
    adata.uns["spot_diameter"] = spot_diameter

    adata.write_h5ad(f"{dir}\\{sample}.h5ad")