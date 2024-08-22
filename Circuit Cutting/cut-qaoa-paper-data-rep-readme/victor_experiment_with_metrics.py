import logging

import re
import networkx as nx
import numpy as np

from create_plots_for_dataset import plot_func, relabel_algorithms, truncate_colormap
#from qaoa.circuit_generation import create_qaoa_circ_parameterized
# from graphs import draw_graph
# from circuit_cutting import preprocess, util
# from circuit_cutting.execute import Executor
# from utils import mkdir
import logging
from ast import literal_eval

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.cm import ScalarMappable, Spectral
from matplotlib.colors import BoundaryNorm

from victor_thesis_metrics import calculate_metrics


def clean_value(dirty_string):
    clean_float = float(dirty_string.replace("[","").replace("]",""))
    return clean_float

def csv_to_landscapes(csv_path):
    df_param_maps = pd.read_csv(csv_path, index_col=0)
    df_param_maps['parameters'] = df_param_maps['parameters'].apply(lambda x: literal_eval(x))
    # df_param_maps['beta'] = df_param_maps['parameters'].apply(lambda p: p[0])
    # df_param_maps['gamma'] = df_param_maps['parameters'].apply(lambda p: p[1])
    num_columns = df_param_maps.shape[1]
    params = df_param_maps.iloc[:,0].to_numpy()
    df_loss_values = []
    dimensions = len(params[0])
    last_x_param = 0
    landscapes_non_square = []
    landscapes = []
    current_landscape_rows = []
    #fill empty landscape and current landscape rows
    for i in range(num_columns-1):
        current_landscape_rows.append([])
        landscapes_non_square.append([])
        landscapes.append([])
        df_loss_values.append(df_param_maps.iloc[:,i+1].to_numpy())
    #iterate over all entries, for every differend loss landscape fill a 2D array with the landscape data
    for idx, param in enumerate(params):
        if param[0] != last_x_param:
            last_x_param = param[0]
            for j in range(0,num_columns-1):
                landscapes_non_square[j].append(current_landscape_rows[j])
                current_landscape_rows[j] = []
        for k in range(num_columns-1):
            current_landscape_rows[k].append(clean_value(df_loss_values[k][idx]))
        if idx == len(params)-1:
            for j in range(0, num_columns-1):
                landscapes_non_square[j].append(current_landscape_rows[j])
    landscapes_non_square = np.array(landscapes_non_square)
    # make landscapes square by appending landscape to itself
    for i, landscape in enumerate(landscapes_non_square):
        landscapes[i] = (np.concatenate((landscape, landscape), axis=0))
    landscapes = np.array(landscapes)
    return landscapes


csv_path = "param_maps/0/aer_simulator_1660210830623176361/parameter_map.csv"

landscapes = csv_to_landscapes(csv_path)

print(calculate_metrics(landscapes[0]))
