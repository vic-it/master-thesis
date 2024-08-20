import json
import logging
import re
import shutil
from pathlib import Path
from typing import Any, Type

import networkx as nx
from qiskit.providers.aer import AerSimulator
from qiskit.providers.ibmq.runtime import UserMessenger, RuntimeEncoder

from graphs import get_graph_from_file
from json_utils import store_kwargs_as_json
from runtime_programs.runtime_with_imports import main

logger = logging.getLogger(__name__)


class StoreUserMessenger(UserMessenger):

    def __init__(self, path):
        self.path = path

    def publish(
            self,
            message: Any,
            encoder: Type[json.JSONEncoder] = RuntimeEncoder,
            final: bool = False
    ) -> None:
        if isinstance(message, dict):
            if '__final_result__' in message:
                return

            iteration = message['iteration']
            graph_path = self.path / str(iteration)
            graph_path.mkdir(parents=True, exist_ok=True)

            if '__qaoa_result__' in message:
                qaoa_result = message['result']
                store_kwargs_as_json(str(graph_path.resolve()), 'sim-qaoa', **qaoa_result)
            elif '__qaoa_short_result__' in message:
                qaoa_short_result = message['result']
                store_kwargs_as_json(str(graph_path.resolve()), 'sim-qaoa-short', **qaoa_short_result)
            elif '__cut_qaoa_result__' in message:
                cut_qaoa_result = message['result']
                store_kwargs_as_json(str(graph_path.resolve()), 'sim-cut-qaoa', **cut_qaoa_result)


def run(path, graph, n_rounds=10, shots=10000, shots_cut=10000, p=1):
    logger.info('Start')
    backend = AerSimulator()
    user_messenger = StoreUserMessenger(path)
    sub_graph_size = graph.number_of_nodes() // 2

    combined_inputs = {
        'edge_list': list(nx.to_edgelist(graph)),
        'n_rounds': n_rounds,
        'shots': shots,
        'shots_cut': shots_cut,
        'partitions': [sub_graph_size, sub_graph_size],
        'p': p,
        'retrieve_interval': 0.01,
        'retries': 0,
        'reduced': True,
        'log_level': 'INFO',
        'log_modules': ['program', 'circuit_cutting'],
        'algorithms': [
            "qaoa",
            "qaoa-short",
            "cut-qaoa"
        ]

    }

    result_dict = main(backend, user_messenger, **combined_inputs)
    store_kwargs_as_json(str(path.resolve()), 'sim-result', **result_dict)


def start_sim_exp(path):
    graph_path = path / 'graph.txt'
    graph = get_graph_from_file(graph_path)
    sim_path = path / 'qaoa_execution_sim'
    sim_path.mkdir(exist_ok=True)
    config_path = path / 'config.json'

    with open(config_path, 'r') as f:
        config = json.load(f)

    shutil.copy(graph_path, sim_path / 'graph.txt')
    shutil.copy(config_path, sim_path / 'config.json')

    run(sim_path, graph, config['n_rounds'], config['shots'], config['shots']*config['cut_shot_factor'], config['p'])


def get_paths(base_path, min_exp=None, max_exp=None, exclude_ids=None):
    if exclude_ids is None:
        exclude_ids = []
    path_list = []
    for exp_dir in base_path.iterdir():
        if not exp_dir.is_dir() or not re.match('^\d+$', exp_dir.name):
            continue
        if min_exp is not None and int(exp_dir.name) < min_exp:
            continue
        if max_exp is not None and int(exp_dir.name) > max_exp:
            continue
        if exp_dir.name in exclude_ids:
            continue
        dirs = list(exp_dir.iterdir())
        dirs = list(filter(lambda x: x.is_dir(), dirs))
        dirs = list(filter(lambda x: 'simulator' not in str(x.resolve()), dirs))
        path_list.extend(dirs)
    return path_list


if __name__ == '__main__':
    exclude_ids = ['19', '21', '30']
    paths = get_paths(Path('experiment_complete/'), min_exp=13, exclude_ids=exclude_ids)
    for p in paths:
        print(p)
        start_sim_exp(p)
