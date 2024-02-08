import torch
import orqviz
import numpy as np
from classic_training import cost_func
from data import *
import numpy as np
from utils import *
from victor_thesis_experiments import *
from victor_thesis_utils import *
from victor_thesis_landscapes import *
from victor_thesis_plots import *
from victor_thesis_metrics import *


# n-dimensional - das war ein schmerz auf n-dimensionen zu generalisieren...
def calc_scalar_curvature(landscape):
    landscape = np.array(landscape)
    gradient_curvature = np.ndarray(landscape.shape)
    dims = len(landscape.shape)
    first_order_gradients = np.array(np.gradient(np.array(landscape)))
    second_order_gradients = []
    for grad in first_order_gradients:
        # should go like e.g. xx, xy, xz, yx, yy, yz, zx, zy, zz for 3d
        temp = np.array(np.gradient(np.array(grad)))
        for arr in temp:
            second_order_gradients.append(arr)
    second_order_gradients = np.array(second_order_gradients)
    # iterate over all landscape entries where idx is the exact position in the array (i.e: idx = (11, 2, 9, 10) -> arr[11][2][9][10] for a 4param qnn)
    for idx, _ in np.ndenumerate(landscape):
        # generate dimsXdims hessian and dims sized vector of gradients for a specific point of the loss landscape
        point_hessian = []
        gradient_vector = []
        for i in range(dims):
            gradient_vector.append(first_order_gradients[i][idx])
            row = []
            for j in range(dims):
                # append e.g. [[0],[1]],[[2],[3]] for 2d
                row.append(second_order_gradients[i * dims + j][idx])
            point_hessian.append(row)
        point_hessian = np.array(point_hessian)
        gradient_vector = np.array(gradient_vector)
        # calculate scalar curvature from here
        beta = 1 / (1 + np.linalg.norm(gradient_vector) ** 2)
        left_term = beta * (
            np.trace(point_hessian) ** 2
            - np.trace(np.matmul(point_hessian, point_hessian))
        )
        right_inner = np.matmul(point_hessian, point_hessian) - np.trace(
            point_hessian
        ) * np.array(point_hessian)
        # order of matmul with gradient does not matter
        right_term = (
            2
            * (beta**2)
            * (np.matmul(np.matmul(gradient_vector.T, right_inner), gradient_vector))
        )
        point_curv = left_term + right_term
        gradient_curvature[idx] = point_curv
    return gradient_curvature


# calculate Total Variation (n-dim)
def calc_total_variation(landscape):
    lanscape_limit = 2 * math.pi
    length = np.array(landscape).shape[0]
    step_size = lanscape_limit / length
    gradients = np.gradient(np.array(landscape))
    total_variation = np.sum(np.absolute(gradients))
    # normalize by adjusting for step size
    normalized_tv = total_variation * step_size
    return np.round(normalized_tv, 2)


# calculate IGSD (n-dim)
def calc_IGSD(landscape):
    gradients = np.gradient(np.array(landscape))
    # each array of the gradients encompasses the gradients for one dimension/direction/parameter
    gradient_standard_deviations = []
    for gradient in gradients:
        gradient_standard_deviations.append(np.std(gradient))
    inverse_gradient_standard_deviations = np.divide(1, gradient_standard_deviations)
    return np.round(inverse_gradient_standard_deviations, 2)


# calculate fourier densitiy (n-dim)
def calc_fourier_density(landscape):
    fourier_result = np.fft.fftn(landscape, norm="forward")
    # fourier_result = np.fft.fftshift(np.fft.fftn(landscape, norm="forward"))
    fourier_density = round(
        get_1_norm(fourier_result) ** 2 / np.linalg.norm(np.array(fourier_result)) ** 2,
        3,
    )
    return fourier_density


# get fourier landscape
def get_fourier_landscape(inputs, U, qnn):
    def loss_function(params):
        qnn.params = torch.tensor(
            [[[params[0], params[1]]]], dtype=torch.float64, requires_grad=True
        )
        x = inputs
        expected_output = torch.matmul(U, x)
        y_true = expected_output.conj()
        return cost_func(x, y_true, qnn, device="cpu")

    n_params = 2
    params = np.random.uniform(-np.pi, np.pi, size=n_params)
    dir1 = np.array([0.0, 1.0])
    dir2 = np.array([1.0, 0.0])
    end_points = (0, 2 * np.pi)
    fourier_result = orqviz.fourier.scan_2D_fourier(
        params,
        loss_function,
        direction_x=dir1,
        direction_y=dir2,
        n_steps_x=60,
        end_points_x=end_points,
    )
    # previous fourier density calculations
    print(
        "different versions of calculating fourier density - not sure which one is the correct one?"
    )
    fourier_density = round(
        np.linalg.norm(np.array(fourier_result.values), ord=1) ** 2
        / np.linalg.norm(np.array(fourier_result.values), ord=2) ** 2,
        3,
    )
    print("FD1:", fourier_density)
    fourier_density = round(
        get_1_norm(fourier_result.values) ** 2
        / np.linalg.norm(np.array(fourier_result.values), ord=2) ** 2,
        3,
    )
    print("FD2:", fourier_density)
    return fourier_result


# get grad curvature
def get_grad_curv(landscape):
    first_order_gradients = np.gradient(np.array(landscape))
    second_order_gradients = []
    for grad in first_order_gradients:
        grads_of_grad = np.gradient(np.array(grad))
        for sec_grad in grads_of_grad:
            second_order_gradients.append(sec_grad)
    magnitude_sum = 0
    for g in second_order_gradients:
        magnitude_sum += g**2
    curv_mag = np.sqrt(magnitude_sum)
    return curv_mag
