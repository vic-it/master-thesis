import torch
import numpy as np
from classic_training import cost_func
from data import *
from generate_experiments import get_qnn
import numpy as np
from utils import *
from victor_thesis_experiments import *
from victor_thesis_utils import *
from victor_thesis_landscapes import *
from victor_thesis_plots import *
from victor_thesis_metrics import *


# gen datapoints
def generate_random_datapoints(numb_points, s_rank, U):
    schmidt_rank = s_rank
    num_points = numb_points
    x_qbits = 1
    r_qbits = s_rank - x_qbits
    inputs = torch.from_numpy(
        np.array(uniform_random_data(schmidt_rank, num_points, x_qbits, r_qbits))
    )
    inputs = inputs.reshape(
        (inputs.shape[0], int(inputs.shape[1] / U.shape[0]), U.shape[0])
    ).permute(0, 2, 1)
    return inputs


# get zero/one datapoints
def get_zero_one_datapoints():
    zero_state = np.array([[1], [0]], dtype=complex)
    one_state = np.array([[0], [1]], dtype=complex)
    #super_pos_state = np.array([[1], [1]], dtype=complex) / np.sqrt(2)
    tensor = torch.tensor(np.array([zero_state, one_state]))
    return tensor
    # inputs = torch.from_numpy(np.array([zero_state, one_state], dtype=complex))
    # print(inputs.size())
    # inputs = inputs.reshape((inputs.shape[0], int(inputs.shape[1] / U.shape[0]), U.shape[0])).permute(0, 2, 1)
    # return inputs


# gen 2D loss landscape
def generate_loss_landscape(grid_size, inputs, U, qnn):
    landscape = []
    lanscape_limit = 2 * math.pi
    step_size = lanscape_limit / grid_size
    #step_size = lanscape_limit / (grid_size-1) # <- more evenly spread samples
    x = inputs
    expected_output = torch.matmul(U, x)
    y_true = expected_output.conj()
    for i in range(grid_size):
        # start at 2pi so y axis label still fits (upwards scaling instead of downards)
        arg_1 = lanscape_limit - i * step_size
        row = []
        for j in range(grid_size):
            # start at 0 because x axis label direction is correct
            arg_2 = j * step_size
            qnn.params = torch.tensor(
                [[[arg_1, arg_2]]], dtype=torch.float64, requires_grad=True
            )
            cost = cost_func(inputs, y_true, qnn, device="cpu")
            row.append(cost.item())
        landscape.append(row)
    return landscape


# gen 3D loss landscape for U3
def generate_3D_loss_landscape_with_labels(grid_size, inputs, U):
    qnn = get_qnn("CudaPennylane", list(range(1)), 1, device="cpu")
    landscape = []
    x_array = []
    y_array = []
    z_array = []
    points = []
    lanscape_limit = 2 * math.pi
    step_size = lanscape_limit / grid_size
    x = inputs
    expected_output = torch.matmul(U, x)
    y_true = expected_output.conj()
    for i in range(grid_size):
        # start at 2pi so y axis label still fits (upwards scaling instead of downards)
        arg_1 = i * step_size
        for j in range(grid_size):
            # start at 0 because x axis label direction is correct
            arg_2 = lanscape_limit - j * step_size
            for k in range(grid_size):
                # maybe change direction?
                arg_3 = k * step_size
                qnn.params = torch.tensor(
                    [[[arg_1, arg_2, arg_3]]], dtype=torch.float64, requires_grad=True
                )
                cost = cost_func(inputs, y_true, qnn, device="cpu")
                landscape.append(cost.item())
                x_array.append(arg_3)
                y_array.append(arg_2)
                z_array.append(arg_1)
    points.append(x_array)
    points.append(y_array)
    points.append(z_array)
    return landscape, points

# gen 3D loss landscape for U3
def generate_3D_loss_landscape(grid_size, inputs, U):
    qnn = get_qnn("CudaPennylane", list(range(1)), 1, device="cpu")
    landscape = []
    lanscape_limit = 2 * math.pi
    step_size = lanscape_limit / grid_size
    x = inputs
    expected_output = torch.matmul(U, x)
    y_true = expected_output.conj()
    for i in range(grid_size):
        row_x= []
        # start at 2pi so y axis label still fits (upwards scaling instead of downards)
        arg_1 = i * step_size
        for j in range(grid_size):
            row_y = []
            # start at 0 because x axis label direction is correct
            arg_2 = lanscape_limit - j * step_size
            for k in range(grid_size):
                # maybe change direction?
                arg_3 = k * step_size
                qnn.params = torch.tensor(
                    [[[arg_1, arg_2, arg_3]]], dtype=torch.float64, requires_grad=True
                )
                cost = cost_func(inputs, y_true, qnn, device="cpu")
                row_y.append(cost.item())
            row_x.append(row_y)
        landscape.append(row_x)
    return landscape
