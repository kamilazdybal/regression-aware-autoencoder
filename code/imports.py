import numpy as np
import pandas as pd
import time
import random
import csv
import copy as cp
from tqdm import tqdm
from sklearn.datasets import make_spd_matrix

from PCAfold import preprocess
from PCAfold import reduction
from PCAfold import analysis
from PCAfold.styles import *

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
from matplotlib import cm
from matplotlib.colors import ListedColormap

import plotly as plty
from plotly import express as px
from plotly import graph_objects as go

from mpl_toolkits import mplot3d
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import tensorflow as tf
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, InputLayer
from keras import optimizers
from keras import metrics
from keras import losses
import keras.layers as kl
import keras
from keras import layers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import initializers

from bayes_opt import BayesianOptimization

# Common settings: ------------------------------------------------------------
random_seed = 100

def plot_nn_diagnostics(history, loss=True, val_loss=True, figsize=(12,4)):

    fig = plt.figure(figsize=figsize)
    if loss: plt.plot(history.history['loss'], 'k--', lw=3)
    if val_loss: plt.plot(history.history['val_loss'], 'r-', lw=3, alpha=0.6)
    plt.ylabel('MSE loss', fontsize=20)
    plt.xlabel('Number of epochs', fontsize=20)
    plt.grid(alpha=0.3)
    plt.title('Final validation MSE loss: ' + str(round(history.history['val_loss'][-1],5)), fontsize=20)
    plt.legend(['Train data', 'Validation data'], loc='upper right', fontsize=20, frameon=False)
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["bottom"].set_visible(True)
    plt.gca().spines["right"].set_visible(False)
    plt.gca().spines["left"].set_visible(True)
    
    return plt
        
def plot_3d(x, y, z, color, cmap='inferno', s=2):

    fig = go.Figure(data=[go.Scatter3d(
        x=x.ravel(),
        y=y.ravel(),
        z=z.ravel(),
        mode='markers',
        marker=dict(
            size=s,
            color=color.ravel(),
            colorscale=cmap,
            opacity=1,
            colorbar=dict(thickness=20)
        )
    )])
    
    fig.update_layout(autosize=False,
                width=1000, height=600,
                margin=dict(l=65, r=50, b=65, t=90),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                scene = dict(
                xaxis_title='x',
                yaxis_title='y',
                zaxis_title='z'))
    
    fig.show()

def plot_3d_regression(x, y, x_grid, y_grid, original, predicted):
    
    fig = go.Figure(data=[go.Scatter3d(
    x=x.ravel(),
    y=y.ravel(),
    z=original.ravel(),
    mode='markers',
    marker=dict(
        size=2,
        color='#aeaeae',
        opacity=1
    )
    ),
    go.Scatter3d(
        x=x_grid.ravel(),
        y=y_grid.ravel(),
        z=predicted.ravel(),
        mode='markers',
        marker=dict(
            size=4,
            color='#ff928b',
            opacity=0.5
        )
    )])
    
    fig.update_layout(autosize=False,
                width=1000, height=600,
                margin=dict(l=65, r=50, b=65, t=90),
                scene = dict(
                xaxis_title='x',
                yaxis_title='y',
                zaxis_title='Phi'))
    
    fig.show()