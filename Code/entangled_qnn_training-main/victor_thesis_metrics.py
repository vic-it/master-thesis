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


# get scalar curvature
def get_scalar_curvature(landscape):
    grad_xx_xy_yx_yy = []
    scalar_curvature = []
    gradients = np.array(np.gradient(np.array(landscape)))
    # maybe add gradients? not sure
    for gradient in gradients:
        second_grads = np.array(np.gradient(np.array(gradient)))
        for second_grad in second_grads:
            grad_xx_xy_yx_yy.append(second_grad)
    # calculate scalar curvature point by point
    for x_id in range(len(landscape)):
        row = []
        for y_id in range(len(landscape)):
            # hessian for point with entries: [d_xx, d_xy][d_yx, d_yy]
            point_hessian = [
                [grad_xx_xy_yx_yy[0][x_id][y_id], grad_xx_xy_yx_yy[1][x_id][y_id]],
                [grad_xx_xy_yx_yy[2][x_id][y_id], grad_xx_xy_yx_yy[3][x_id][y_id]],
            ]
            # gradients as 2 entry vector (x dir, y dir)
            gradient = np.array([gradients[1][x_id][y_id], gradients[0][x_id][y_id]])
            # take euclidean norm
            beta = 1 / (1 + np.linalg.norm(gradient) ** 2)
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
                * (np.matmul(np.matmul(gradient.T, right_inner), gradient))
            )
            point_curv = left_term + right_term
            # print(point_curv)
            # maybe sum, maybe not? point curv is 2 entry vector
            row.append(point_curv)
        scalar_curvature.append(row)
        # output absolute and root to compare visually to grad curvature
    return scalar_curvature


# calculate Total Variation
def calc_total_variation(landscape):
    lanscape_limit = 2 * math.pi
    step_size = lanscape_limit / len(landscape)
    gradients = np.gradient(np.array(landscape))
    total_variation = np.sum(np.absolute(gradients))
    # normalize by adjusting for step size
    normalized_tv = total_variation * step_size
    return np.round(normalized_tv, 2)


# calculate IGSD
def calc_IGSD(landscape):
    gradients = np.gradient(np.array(landscape))
    # each array of the gradients encompasses the gradients for one dimension/direction/parameter
    gradient_standard_deviations = []
    for gradient in gradients:
        gradient_standard_deviations.append(np.std(gradient))
    inverse_gradient_standard_deviations = np.divide(1, gradient_standard_deviations)
    return np.round(inverse_gradient_standard_deviations, 2)


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
    fourier_density = round(
        np.linalg.norm(np.array(fourier_result.values), ord=1) ** 2
        / np.linalg.norm(np.array(fourier_result.values), ord=2) ** 2,
        3,
    )
    print("Fourier Density:", fourier_density)
    return fourier_result
