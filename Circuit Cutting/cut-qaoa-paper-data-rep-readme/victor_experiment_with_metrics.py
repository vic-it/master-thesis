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

logger = logging.getLogger(__name__)

def calc_landscape(exp_id, plot_path, sim_path, qpu_path, cmap=Spectral, levels=100, cols=None, relabel=None):
    v_min, v_max = get_v_min_max(sim_path / 'parameter_map.csv',
                                 'qaoa')

    if cols is None:
        cols = ['qaoa-short_1000', 'cut-qaoa_1000', 'qaoa-short_10000', 'cut-qaoa_10000']

    df_param_maps = pd.read_csv(qpu_path / 'param_map/parameter_map.csv', index_col=0)
    df_param_maps['parameters'] = df_param_maps['parameters'].apply(lambda x: literal_eval(x))
    df_param_maps['beta'] = df_param_maps['parameters'].apply(lambda p: p[0])
    df_param_maps['gamma'] = df_param_maps['parameters'].apply(lambda p: p[1])
    average(df_param_maps, cols)
    cols_and_params = [f'{c}_avg' for c in cols]
    value_vars = cols_and_params
    cols_and_params.extend(['beta', 'gamma'])
    df_params_reduced = df_param_maps[cols_and_params].copy()
    df_params_reduced = df_params_reduced.melt(id_vars=['beta', 'gamma'], value_vars=value_vars)
    df_params_reduced['algorithm'] = df_params_reduced['variable'].apply(lambda s: s.split('_')[0])
    relabel_algorithms(df_params_reduced, relabel)
    df_params_reduced['shots'] = df_params_reduced['variable'].apply(lambda s: int(s.split('_')[1]))

    min_qpu, max_qpu = df_params_reduced['value'].min(), df_params_reduced['value'].max()




def average(df, columns):
    """
    Average of columns containing a list of numbers
    :param df: dataframe
    :param columns:
    """
    for col in columns:
        df[f'{col}_avg'] = df[col].apply(lambda values: np.average(literal_eval(values)))


def relabel_algorithms(df, relabel):
    if relabel is not None:
        for old_label, new_label in relabel.items():
            df['algorithm'] = df['algorithm'].apply(
                lambda name: re.sub('^' + old_label + '$', new_label, name, 1))

def get_v_min_max(csv_path, column):
    df = pd.read_csv(csv_path)
    average(df, [column])
    column_avg = f'{column}_avg'
    values = np.array(df[column_avg].to_list())
    return [np.min(values), np.max(values)]