import torch
import numpy as np
from train import get_model_CL as train_load_model_CL
from train import get_model_Rec as train_load_model_Rec
from model import ContrastiveModel, ReconstructionModel
from Dataset import SpotDataset_CL, SpotDataset_Rec
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from utils import evaluate, load_label
import random
import os
import time

def set_all_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

def get_model_kwargs(kwargs):
    return get_model(**kwargs)

def get_model(
        x, exp, locs, y, prefix, batch_size, epochs, lr, device='cuda', CL_output=512):

    x = x.copy()

    datasets_CA = []
    for i in range(len(x)):
        datasets_CA.append(SpotDataset_CL(x[i], exp[i], locs[i]))

    model_CL, checkpoint_file = train_load_model_CL(
            model_class=ContrastiveModel,
            model_kwargs=dict(
                lr=lr,
                img_dim=x[0].shape[-1],
                gene_dim=exp[0].shape[-1],
                proj_dim=CL_output,
                temperature=0.07),
            datasets=datasets_CA, prefix=prefix,
            batch_size=batch_size, epochs=epochs, device=device)
    model_CL.eval()
    if device == 'cuda':
        torch.cuda.empty_cache()


    datasets_Rec = []
    for i in range(len(x)):
        datasets_Rec.append(SpotDataset_Rec(x[i], exp[i], locs[i], y[i]))

    model_Rec = train_load_model_Rec(
        model_class=ReconstructionModel,
        model_kwargs=dict(
            contrastive_checkpoint_path=checkpoint_file,
            proj_dim=CL_output,
            img_dim=x[0].shape[-1],
            gene_dim=exp[0].shape[-1],
            lr=lr),
        datasets=datasets_Rec, prefix=prefix,
        batch_size=batch_size, epochs=epochs, device=device)
    model_Rec.eval()
    if device == 'cuda':
        torch.cuda.empty_cache()
    return model_Rec

def normalize(data):

    data = data.copy()

    data_min = data.min(0)
    data_max = data.max(0)
    data -= data_min
    data /= (data_max - data_min) + 1e-12

    return data

def training(
        imgs, exps, locs, y, epochs, lr, batch_size, prefix, device='cuda', n_jobs=1, CL_output=512):

    imgs = [img.astype(np.float32) for img in imgs]
    imgs = [normalize(img) for img in imgs]

    kwargs = dict(x=imgs, exp=exps, locs=locs, y=y, batch_size=batch_size, epochs=epochs, lr=lr,
                       prefix=f'{prefix}states/0/', device=device, CL_output=CL_output)

    model = get_model_kwargs(kwargs)
    return model

def inference(model, imgs, exps, locs, y, batch_size):

    trainer = pl.Trainer(
        accelerator='gpu',
        devices=1,
        deterministic=True
    )

    imgs = imgs.astype(np.float32)
    imgs = normalize(imgs)

    model.eval()

    dataloader = DataLoader(SpotDataset_Rec(imgs, exps, locs, y), batch_size=batch_size, shuffle=False)
    trainer.test(model, dataloader)
    results = model.scores

    min_score = results.min()
    max_score = results.max()
    results = (results - min_score) / (max_score - min_score)

    return results

def main(dir, train_names, test_name, epoch, device='cuda', lr=1e-4, batch_size=64, CL_output=256, seed=42):
    set_all_seeds(seed)

    all_samples = train_names
    data = np.load(f"{dir}train_dataset_ST_HER2ST.npz", allow_pickle=True)

    all_img = data['img_feats']
    all_expr = data['gene_exprs']
    all_pos = data['coords']


    train_labels = []
    for s in all_samples:
        label = load_label(f'{dir}{s}_label.txt')
        train_labels.append(np.array(label))
    train_labels = np.array(train_labels, dtype=object)

    start = time.time()

    model = training(
        imgs=all_img, exps=all_expr, locs=all_pos, y=train_labels,
        epochs=epoch, lr=lr, batch_size=batch_size,
        prefix=dir, device=device, n_jobs=1, CL_output=CL_output
    )
    end = time.time()
    print(f'training time: {end-start}')

    test_data = np.load(f"{dir}test_dataset_Visium_HBC.npz", allow_pickle=True)
    test_img = test_data['img_feats']
    test_expr = np.float32(test_data['gene_exprs'])
    test_pos = test_data['coords']
    test_label = np.array(load_label(f"{dir}{test_name}_label.txt"))

    start = time.time()
    scores = inference(model=model, imgs=test_img, exps=test_expr, locs=test_pos, y=test_label, batch_size=batch_size)
    end = time.time()
    print(f'test time: {end-start}')
    output_str = ' '.join([f'{x:.4f}' for x in scores.tolist()])
    with open(f'{dir}results/scores_{test_name}.txt', 'w') as f:
        f.write(output_str)

    auc, ap, f1 = evaluate(test_label, scores)

    print(f"Seed {seed:4d}: AUC={auc:.4f}, AP={ap:.4f}, F1={f1:.4f}")
    return auc, ap, f1


# ###############################################################################################
# ###############################################################################################
seeds = [42, 123, 456, 789, 2026]
aucs, aps, f1s = [], [], []
pretrain_dir = 'data/'
sections = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
test_name = 'Visium_HBC'
for i in seeds:
    auc, ap, f1 = main(pretrain_dir, sections, test_name,
                       epoch=100, device='cuda', lr=1e-5, batch_size=512, CL_output=512, seed=i)
    aucs.append(auc)
    aps.append(ap)
    f1s.append(f1)

print("="*60)
print("Individual Results for Each Seed:")
print("="*60)
for i, seed in enumerate(seeds):
   print(f"Seed {seed:4d}: AUC={aucs[i]:.4f}, AP={aps[i]:.4f}, F1={f1s[i]:.4f}")

auc_mean, auc_std = np.mean(aucs), np.std(aucs)
ap_mean, ap_std = np.mean(aps), np.std(aps)
f1_mean, f1_std = np.mean(f1s), np.std(f1s)

print("\n" + "="*60)
print("Results for Paper (Mean ± Std):")
print("="*60)
print(f"AUC: {auc_mean:.3f}±{auc_std:.3f}")
print(f"AP:  {ap_mean:.3f}±{ap_std:.3f}")
print(f"F1:  {f1_mean:.3f}±{f1_std:.3f}")

results_dict = {
   'seeds': seeds,
   'aucs': aucs,
   'aps': aps,
   'f1s': f1s,
   'summary': {
       'auc_mean': auc_mean, 'auc_std': auc_std,
       'ap_mean': ap_mean, 'ap_std': ap_std,
       'f1_mean': f1_mean, 'f1_std': f1_std
   }
}

print(f"Peak memory usage: {torch.cuda.max_memory_allocated() / (1024 ** 2):.2f} MB")