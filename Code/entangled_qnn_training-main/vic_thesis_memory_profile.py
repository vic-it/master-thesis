import gc
import time
import torch
import orqviz
import numpy as np
from classic_training import cost_func
from data import *
import numpy as np
from qnns.cuda_qnn import CudaPennylane
from utils import *
from victor_thesis_experiments_main import process_sc_metrics
from victor_thesis_utils import *
from victor_thesis_landscapes import *
from victor_thesis_plots import *
from victor_thesis_metrics import *
from memory_profiler import memory_usage
from memory_profiler import profile

def get_first_order_gradient_of_point(i, target_point, landscape):
    grid_size = len(landscape[0])
    if target_point[i]==0:
        #forward diff
        leftid = list(target_point)
        leftid[i] = leftid[i]+1
        rightid = list(target_point)    
        leftidx = tuple(leftid)
        rightidx = tuple(rightid)
        return (landscape[leftidx]-landscape[rightidx])
    if target_point[i] >= grid_size-1:
        #backward diff        
        leftid = list(target_point)
        rightid = list(target_point)
        rightid[i] = rightid[i] -1
        
        leftidx = tuple(leftid)
        rightidx = tuple(rightid)
        return (landscape[leftidx]-landscape[rightidx])
    
    leftid = list(target_point)
    rightid = list(target_point)
    leftid[i] = leftid[i]+1
    rightid[i] = rightid[i] -1
    leftidx = tuple(leftid)
    rightidx = tuple(rightid)
    return (landscape[leftidx]-landscape[rightidx])/2

#ich bin ein genie
def get_second_order_gradient_of_point(i,j, target_point, landscape):
    grid_size = len(landscape[0])
    if target_point[j]==0:
        #forward diff
        leftid = list(target_point)
        leftid[j] = leftid[j]+1
        rightid = list(target_point)    
        leftidx = tuple(leftid)
        rightidx = tuple(rightid)
        return (get_first_order_gradient_of_point(i, leftidx, landscape)-get_first_order_gradient_of_point(i, rightidx, landscape))
    if target_point[j] >= grid_size-1:
        #backward diff        
        leftid = list(target_point)
        rightid = list(target_point)
        rightid[j] = rightid[j] -1        
        leftidx = tuple(leftid)
        rightidx = tuple(rightid)
        return (get_first_order_gradient_of_point(i, leftidx, landscape)-get_first_order_gradient_of_point(i, rightidx, landscape))    
    leftid = list(target_point)
    rightid = list(target_point)
    leftid[j] = leftid[j]+1
    rightid[j] = rightid[j] -1
    leftidx = tuple(leftid)
    rightidx = tuple(rightid)
    return (get_first_order_gradient_of_point(i, leftidx, landscape)-get_first_order_gradient_of_point(i, rightidx, landscape))/2

    
#@profile
def bad_calc_scalar_curvature(landscape):
    """calculates the scalar curvature of a loss landscape
    instead of calculating the whole n dimensional curvature array (same size as the input landscape)
    this function calculates the scalar curvature at each entry of the n dimensional landscape 
    and puts them back together into an output array

    Args:
        landscape (array): n dimensional loss landscape array

    Returns:
        array: n dimensional scalar curvature array
    """
    landscape = np.array(landscape)
    scalar_curvature = np.ndarray(landscape.shape)
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
        scalar_curvature[idx] = point_curv
    return scalar_curvature

#@profile
def good_calc_scalar_curvature(landscape):
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
    # del landscape
    # gc.collect()
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

if __name__ == "__main__":
    x_qubits = 2
    r_qubits = 2
    U = torch.tensor(np.array(random_unitary_matrix(2)), dtype=torch.complex128,device="cpu")
    raw_input = torch.from_numpy(np.array(uniform_random_data(3, 4, x_qubits, r_qubits)))
    data = raw_input.reshape((raw_input.shape[0], int(raw_input.shape[1] / U.shape[0]), U.shape[0])).permute(0, 2, 1)
    qnn = CudaPennylane(num_wires=2, num_layers=1, device="cpu")
    
    start = time.time()
    landscape = generate_loss_landscape(7, 6, data, U, qnn)
    end = time.time()
    print(f"ls  {end- start}")
    print("start SC")
    start = time.time()
    SC1 = good_calc_scalar_curvature(landscape)
    end = time.time()
    print(f"good  {end- start}")
    start = time.time()
    SC2 = bad_calc_scalar_curvature(landscape)
    end = time.time()
    print(f"bad  {end- start}")
    print(process_sc_metrics(SC1)[0])
    print(process_sc_metrics(SC2)[0])
    print(process_sc_metrics(SC1)[1])
    print(process_sc_metrics(SC2)[1])