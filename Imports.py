import numpy as np
from sklearn.datasets import make_blobs
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import precision_score,recall_score,f1_score,confusion_matrix,\
    precision_recall_curve,roc_curve,auc
import sklearn.metrics as sklm
import matplotlib.pyplot as plt
from pandas import DataFrame
