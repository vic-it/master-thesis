import torch
import orqviz
import numpy as np
from classic_training import cost_func
from data import *
import numpy as np
from utils import *
from victor_thesis_utils import *
from victor_thesis_landscapes import *
from victor_thesis_plots import *
from victor_thesis_metrics import *


# n-dimensional scalar curvature
def calc_scalar_curvature(landscape):
    """calculates the scalar curvature of a loss landscape
    instead of calculating the whole n dimensional curvature array (same size as the input landscape)
    this function calculates the scalar curvature at each entry of the n dimensional landscape 
    and puts them back together into an output array

    Args:
        landscape (array): n dimensional loss landscape array

    Returns:
        array: n dimensional scalar curvature array
    """
    landscape = np.asarray(landscape)
    scalar_curvature = np.ndarray(landscape.shape)
    dims = len(landscape.shape)
    # iterate over all landscape entries where idx is the exact position in the array (i.e: idx = (11, 2, 9, 10) -> arr[11][2][9][10] for a 4param qnn)
    for idx, _ in np.ndenumerate(scalar_curvature):
        # generate dimsXdims hessian and dims sized vector of gradients for a specific point of the loss landscape
        point_hessian = []
        gradient_vector = []
        for i in range(dims):
            #get gradient vector
            gradient_vector.append(get_first_order_gradient_of_point(i, idx, landscape))
            row = []
            for j in range(dims):
                # append e.g. [[0],[1]],[[2],[3]] for 2d
                row.append(get_second_order_gradient_of_point(i,j,idx,landscape))
            point_hessian.append(row)
        point_hessian = np.asarray(point_hessian)
        gradient_vector = np.asarray(gradient_vector)
        # calculate scalar curvature from here
        beta = 1 / (1 + np.linalg.norm(gradient_vector) ** 2)
        left_term = beta * (
            np.trace(point_hessian) ** 2
            - np.trace(np.matmul(point_hessian, point_hessian))
        )
        right_inner = np.matmul(point_hessian, point_hessian) - np.trace(
            point_hessian
        ) * point_hessian
        # order of matmul with gradient does not matter
        right_term = (
            2
            * (beta**2)
            * (np.matmul(np.matmul(gradient_vector.T, right_inner), gradient_vector))
        )
        point_curv = left_term + right_term
        scalar_curvature[idx] = point_curv
    return scalar_curvature


def calc_total_variation(landscape):
    """calculates the total variation of a landscape

    Args:
        landscape (array): n dimensional loss landscape as an n dimensional array
    """
    dimensions = len(np.array(landscape).shape)
    #print(dimensions)
    lanscape_limit = 2 * math.pi
    length = np.array(landscape).shape[0]
    step_size = lanscape_limit / length
    gradients = np.gradient(np.array(landscape))
    total_variation = np.sum(np.absolute(gradients))
    # normalize it by step size
    #using dimensions -1 gives more stable results w.r.t. the number of samples per dimension
    total_variation = total_variation * step_size**(dimensions)
    return np.round(total_variation, 3)


def calc_IGSD(landscape):
    """calculates the inverse gradient standard deviation of a landscape

    Args:
        landscape (array): n dimensional loss landscape array

    Returns:
        array: returns a list of IGSDs, one for each dimension 
    """
    gradients = np.gradient(np.array(landscape))
    # each array of the gradients encompasses the gradients for one dimension/direction/parameter
    gradient_standard_deviations = []
    for dimension in gradients:
        gradient_standard_deviations.append(np.std(dimension))

    inverse_gradient_standard_deviations = np.divide(1, gradient_standard_deviations)

    #print(landscape)
    return np.round(inverse_gradient_standard_deviations, 3)


def calc_fourier_density(landscape) -> float:
    """same as calculate_fourier_density below 
    but with custom k-norm function and rounded to 6 digits

    Args:
        landscape (array): n dimensional landscape array

    """
    fourier_result = np.fft.fftshift(np.fft.fftn(landscape, norm="forward"))
    fourier_density = round(
        (get_k_norm(fourier_result, 1) ** 2) / (get_k_norm(fourier_result, 2) ** 2),
        6,
    )
    return fourier_density


# calculates the fourier density by reshaping the fourier result to get an vector of Fourier coefficients
def calculate_fourier_density(
    landscape,
) -> float:
    """calculates the fourier density of a given landscape

    Args:
        landscape (array): n-dim landscape

    """
    fourier_result = np.fft.fftshift(np.fft.fftn(landscape, norm="forward"))

    # reshape the fourier result into a vector according to the paper
    vector_fourier_result = fourier_result.reshape(-1)

    # sum the absolute values of each component of the vector
    one_norm = np.sum(np.abs(vector_fourier_result))

    # frobenius norm
    two_norm = np.linalg.norm(vector_fourier_result)
    return one_norm**2 / two_norm**2


def get_fourier_landscape(inputs, U, qnn, steps=60):
    """a much too complicated way to calculate the fourier landscape and density 
    with the orqviz scan 2d fourier function.

    Args:
        inputs (tensor): tensor representation of the data points used to train the qnn
        U (unitary): unitary
        steps (int, optional): how many frequencies do you want to look at. Defaults to 60.
    """
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
        n_steps_x=steps,
        end_points_x=end_points,
    )
    # previous fourier density calculations
    print(
        "different versions of calculating fourier density - not sure which one is the correct one?"
    )
    fourier_density = round(
        (np.linalg.norm(np.array(fourier_result.values), ord=1) ** 2)
        / (np.linalg.norm(np.array(fourier_result.values), ord=2) ** 2),
        3,
    )
    print("FD lib with np linalg norms:", fourier_density)
    fourier_density = round(
        (get_k_norm(fourier_result.values, 1) ** 2)
        / (np.linalg.norm(np.array(fourier_result.values), ord=2) ** 2),
        3,
    )
    print("FD lib with semi custom norms:", fourier_density)

    fourier_density = round(
        (get_k_norm(fourier_result.values, 1) ** 2)
        / (get_k_norm(fourier_result.values, 2) ** 2),
        3,
    )
    print("FD lib with full custom norms:", fourier_density)
    return fourier_result


def calc_grad_curv(landscape):
    """calculates the gradient curvature (custom metric consisting of the second order gradient magnitudes) for a given landscape

    Args:
        landscape (array): landscape of which you want to calculate the curvature

    Returns:
        array: array of curvature for every point in the landscape
    """
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
