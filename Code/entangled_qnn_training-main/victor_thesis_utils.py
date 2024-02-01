import torch
import numpy as np
from data import *
import numpy as np
from utils import *


# get plot metadata for different modes
def get_meta_for_mode(mode, data, min_val, max_val, titles, o, gate_name, ansatz):
    low_threshold = 0.000000001
    if mode == "default":
        c_map = "plasma"
        sup_title = f"Loss Landscapes for {ansatz}($\\phi,\\lambda)$ Approximating {gate_name} for Different Datasets"
        title = titles[o]
        v_min = min(min_val, 0)
        v_max = max(max_val, 1)
    elif mode == "grad":
        c_map = "winter"
        sup_title = "Gradient Magnitudes"
        # average gradient magnitude adjusted for sample frequency - not sure how to call this.
        title = f"GM Score: {np.round(np.average(data)*len(data), 2)}"
        v_min = min(min_val, 0)
        v_max = math.ceil(max_val * 100.0) / 100.0
    elif mode == "log_scale":
        v_max = 1
        # v_min = min((min_val+low_threshold/18)*12, low_threshold)
        v_min = low_threshold
        c_map = "Greys"
        if min_val < low_threshold:
            min_text = "< 0.000000001"
        else:
            min_text = f"= {np.round(min_val, 10)}"
        sup_title = f"Logarithmic Loss (min. {min_text})"
        title = titles[o]
    return c_map, sup_title, title, v_min, v_max


# print expected output
def print_expected_output(U, x, name):
    print("====")
    expected_output = torch.matmul(U, x)
    np_arr = expected_output.detach().cpu().numpy()
    print("expected output for ", name, ":\n", np_arr)
    print("====")


# print datapoints
def print_datapoints(points, title):
    print("", title, " data points:")
    np_arr = points.detach().cpu().numpy()
    for i, row in enumerate(np_arr):
        print("---")
        for j, point in enumerate(row):
            # idx = i * len(row) + j + 1
            print("", i, " - ", j, ":", point)

def get_1_norm(arr):
    return np.sum(np.absolute(np.array(arr)))
