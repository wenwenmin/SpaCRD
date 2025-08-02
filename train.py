import os
from copy import deepcopy
from time import time

import numpy as np
import torch
from torch.utils.data import DataLoader, ConcatDataset
import pytorch_lightning as pl
import matplotlib
import matplotlib.pyplot as plt

from utils import load_pickle, save_pickle

matplotlib.use('Agg')


class MetricTracker(pl.Callback):

    def __init__(self):
        self.collection = []

    def on_train_epoch_end(self, trainer, *args, **kwargs):
        metrics = deepcopy(trainer.logged_metrics)
        self.collection.append(metrics)

    def clean(self):
        keys = [set(e.keys()) for e in self.collection]
        keys = set().union(*keys)
        for elem in self.collection:
            for ke in keys:
                if ke in elem.keys():
                    if isinstance(elem[ke], torch.Tensor):
                        elem[ke] = elem[ke].item()
                else:
                    elem[ke] = float('nan')


def get_model_CL(
        model_class, model_kwargs, datasets, prefix,
        epochs=None, device='cuda', load_saved=False, **kwargs):
    checkpoint_file = prefix + 'model_CL.pt'
    history_file = prefix + 'history_CL.pickle'

    if load_saved and os.path.exists(checkpoint_file):
        model = model_class.load_from_checkpoint(checkpoint_file)
        print(f'Model loaded from {checkpoint_file}')
        history = load_pickle(history_file)
    else:
        model = None
        history = []

    if (epochs is not None) and (epochs > 0):
        model, hist, trainer = train_model_CL(
            model=model,
            model_class=model_class, model_kwargs=model_kwargs,
            datasets=datasets, epochs=epochs, device=device,
            **kwargs)
        trainer.save_checkpoint(checkpoint_file)
        print(f'Model_CL saved to {checkpoint_file}')
        history += hist
        save_pickle(history, history_file)
        print(f'History_CL saved to {history_file}')
        plot_history(history, prefix, 'CA')

    return model, checkpoint_file


def train_model_CL(
        datasets, batch_size, epochs,
        model=None, model_class=None, model_kwargs={},
        device='cuda'):

    if model is None:
        model = model_class(**model_kwargs)

    tracker = MetricTracker()
    device_accelerator_dict = {
            'cuda': 'gpu',
            'cpu': 'cpu'}
    accelerator = device_accelerator_dict[device]

    trainer = pl.Trainer(
        max_epochs=epochs,
        callbacks=[tracker],
        deterministic=True,
        accelerator=accelerator,
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=True)

    model.train()
    t0 = time()

    combined_dataset = ConcatDataset(datasets)
    dataloader = DataLoader(combined_dataset, batch_size=batch_size, shuffle=True)

    trainer.fit(model=model, train_dataloaders=dataloader)

    print(int(time() - t0), 'sec')
    tracker.clean()
    history = tracker.collection
    return model, history, trainer


def get_model_Rec(
        model_class, model_kwargs, datasets, prefix,
        epochs=None, device='cuda', load_saved=False, **kwargs):
    checkpoint_file = prefix + 'model_Rec.pt'
    history_file = prefix + 'history_Rec.pickle'

    if load_saved and os.path.exists(checkpoint_file):
        model = model_class.load_from_checkpoint(checkpoint_file)
        print(f'Model loaded from {checkpoint_file}')
        history = load_pickle(history_file)
    else:
        model = None
        history = []

    if (epochs is not None) and (epochs > 0):
        model, hist, trainer = train_model_Rec(
            model=model,
            model_class=model_class, model_kwargs=model_kwargs,
            datasets=datasets, epochs=epochs, device=device,
            **kwargs)
        trainer.save_checkpoint(checkpoint_file)
        print(f'Model_Rec saved to {checkpoint_file}')
        history += hist
        save_pickle(history, history_file)
        print(f'History_Rec saved to {history_file}')
        plot_history(history, prefix, 'Rec')

    return model


def train_model_Rec(
        datasets, batch_size, epochs,
        model=None, model_class=None, model_kwargs={},
        device='cuda'):
    if model is None:
        model = model_class(**model_kwargs)

    tracker = MetricTracker()
    device_accelerator_dict = {
        'cuda': 'gpu',
        'cpu': 'cpu'}
    accelerator = device_accelerator_dict[device]

    # # 设置随机种子
    # pl.seed_everything(42, workers=True)

    trainer = pl.Trainer(
        max_epochs=epochs,
        callbacks=[tracker],
        deterministic=True,
        accelerator=accelerator,
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=True)

    model.train()
    t0 = time()

    combined_dataset = ConcatDataset(datasets)
    dataloader = DataLoader(combined_dataset, batch_size=batch_size, shuffle=True)

    trainer.fit(model=model, train_dataloaders=dataloader)

    print(int(time() - t0), 'sec')
    tracker.clean()
    history = tracker.collection
    return model, history, trainer


def plot_history(history, prefix, stage):
    plt.figure(figsize=(16, 16))
    groups = set([e.split('_')[-1] for e in history[0].keys()])
    groups = np.sort(list(groups))
    for i, grp in enumerate(groups):
        plt.subplot(len(groups), 1, 1+i)
        for metric in history[0].keys():
            if metric.endswith(grp):
                hist = np.array([e[metric] for e in history])
                hmin, hmax = hist.min(), hist.max()
                label = f'{metric} ({hmin:+013.6f}, {hmax:+013.6f})'
                hist -= hmin
                hist /= hmax + 1e-12
                plt.plot(hist, label=label)
        plt.legend()
        plt.ylim(0, 1)
    outfile = f'{prefix}history_{stage}.png'
    plt.savefig(outfile, dpi=300)
    plt.close()
    print(outfile)
