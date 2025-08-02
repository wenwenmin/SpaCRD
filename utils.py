import pickle
import os
import pandas as pd
from sklearn import metrics
import numpy as np
def mkdir(path):
    dirname = os.path.dirname(path)
    if dirname != '':
        os.makedirs(dirname, exist_ok=True)

def save_pickle(x, filename):
    mkdir(filename)
    with open(filename, 'wb') as file:
        pickle.dump(x, file)


def load_pickle(filename, verbose=True):
    with open(filename, 'rb') as file:
        x = pickle.load(file)
    return x


def evaluate(y_true, y_score, pre=False):
    y_true = pd.Series(y_true)
    y_score = pd.Series(y_score)

    roc_auc = metrics.roc_auc_score(y_true, y_score)
    ap = metrics.average_precision_score(y_true, y_score)

    ratio = 100.0 * len(np.where(y_true == 0)[0]) / len(y_true)
    thres = np.percentile(y_score, ratio)
    y_pred = (y_score >= thres).astype(int)
    y_true = y_true.astype(int)
    _, _, f1, _ = metrics.precision_recall_fscore_support(y_true, y_pred, average='binary')
    if pre:
        return roc_auc, ap, f1, y_pred
    else:
        return roc_auc, ap, f1

def load_label(data_path):
    with open(data_path, "r") as tf:
        content = tf.read().strip()
        tokens = content.split()
        label = [int(i) for i in tokens if i != '']
    return label