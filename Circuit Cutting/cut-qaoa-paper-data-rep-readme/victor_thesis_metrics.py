import math
import numpy as np
import numpy as np
from utils import *
from victor_thesis_utils import *
from victor_thesis_plots import *
from victor_thesis_metrics import *

def calculate_metrics(landscape):
    metrics = []
    metrics.append(calc_total_variation(landscape))
    metrics.append(calc_fourier_density(landscape))
    metrics.append(calc_IGSD(landscape))
    metrics.append(process_sc_metrics(calc_scalar_curvature(landscape)))
    metrics.append(get_extra_fourier_metrics(landscape))
    return metrics


def process_sc_metrics(SC):
    """calculates the mean, standard deviation, absolute mean and absolute standard deviations, 
       as well as the positive and negative percentage of the scalar curvature values in the input array

    Args:
        SC (array): the scalar curvature values for each point of a sampled landscape, 
        each entry of the n-dimensional SC array corresponds to the value of the landscape at the same position in the array

    Returns:
        list: a list of all metrics that depend on the scalar curvature array
    """
    sc = np.array(SC).reshape(-1)
    sc_avg = np.mean(sc)
    sc_std = np.std(sc)
    sc_pos = (1.0 * np.sum(sc >= 0)) / (1.0 * len(sc))
    sc_neg = (1.0 * np.sum(sc < 0)) / (1.0 * len(sc))
    sc_abs = np.abs(sc)
    sc_avg_abs = np.mean(sc_abs)
    sc_std_abs = np.std(sc_abs)
    #add medians
    sc_med = np.median(sc)
    sc_med_abs = np.median(sc_abs)
    return [sc_avg,sc_std,sc_pos,sc_neg,sc_abs,sc_avg_abs,sc_std_abs, sc_med, sc_med_abs]

#needs to be adapted to circuit cutting library
def process_and_store_metrics(metrics,  experiment_id, name):
    """calculates, processes and stores the metrics of given landscapes into a txt file for later evaluation
       supposed to get five landscapes, corresponding to 5 different runs for the same configuration
       and unitaries but with different qubit data points
       beings by calculating the metrics for each run individually and then calculates the average and stdev for all runs together

    Args:
        landscapes (array): an array of n dimensional loss landscapes, one landscape for each run with this config
        conf_id (int): the id of the configuration used for these runs
        experiment_id (string): a string identifier for the file system to identify which experiment results and configs belong together
            contains mostly time and dimension/grid size info
    """
    os.makedirs(f"results/", exist_ok=True)
    file = open(
        f"results/{name}_{experiment_id}.txt", "w"
    )
    file.write(f"conf_id={experiment_id}\n---\n")
    file.close()
    TV = metrics[0]
    FD = metrics[1]
    IGSD = metrics[2]
    sc_metric = metrics[3]
    efms = metrics[4]
    # calculate and store individual sub-metric (avg, std,..)
    file = open(
        f"results/{name}_{experiment_id}.txt", "a"
    )
    file.write(f"run_{name}\n")
    file.write(f"TV={TV}\n")
    file.write(f"FD={FD}\n")
    igsd_string = (
        np.array2string(IGSD, separator=",")
        .replace("\n", "")
        .replace(" ", "")
    )
    file.write(f"IGSD={igsd_string}\n")
    # calculate SC sub-metrics
    # flatten SC
    sc_avg = sc_metric[0]
    sc_std = sc_metric[1]
    sc_pos = sc_metric[2]
    sc_neg = sc_metric[3]
    sc_abs = sc_metric[4]
    sc_avg_abs = sc_metric[5]
    sc_std_abs = sc_metric[6]
    sc_med = sc_metric[7]
    sc_med_abs = sc_metric[8]        
    file.write(f"SC_pos={sc_pos}\n")
    file.write(f"SC_neg={sc_neg}\n")
    file.write(f"SC_avg={sc_avg}\n")
    file.write(f"SC_std={sc_std}\n")
    file.write(f"SC_avg_abs={sc_avg_abs}\n")
    file.write(f"SC_std_abs={sc_std_abs}\n")
    file.write(f"SC_med={sc_med}\n")
    file.write(f"SC_med_abs={sc_med_abs}\n")
    # do extra fourier metrics stuff
    efm_lamps_avg = efms[0]
    efm_lamps_med = efms[1]
    efm_lamps_std = efms[2]
    efm_nzfreq_avg = efms[3]
    efm_nzfreq_med = efms[4]
    efm_nzfreq_num = efms[5]
    efm_total_coeffs = efms[6]
    file.write(f"efm_lamps_avg={efm_lamps_avg}\n")
    file.write(f"efm_lamps_med={efm_lamps_med}\n")
    file.write(f"efm_lamps_std={efm_lamps_std}\n")
    file.write(f"efm_nzfreq_avg={efm_nzfreq_avg}\n")
    file.write(f"efm_nzfreq_med={efm_nzfreq_med}\n")
    file.write(f"efm_nzfreq_num={efm_nzfreq_num}\n")
    file.write(f"efm_total_coeffs={efm_total_coeffs}\n---\n")
    file.write("combined\n")
    file.close()

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
    #using dimensions -1 gives more stable results w.r.t. the number of scoeffles per dimension
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

def calc_fourier_frequencies_coeffs(landscape):
    fourier_landscape = np.fft.fftshift(np.fft.fftn(landscape, norm="forward"))
    list_of_nonzero_adj_freq_coeffs = []
    list_of_non_zero_freq_lenghts = []
    length = len(fourier_landscape)
    fourier_landscape = fourier_landscape.T
    total_coeff = 0
    for idx, _ in np.ndenumerate(fourier_landscape):
        # adjust indizes to match frequencies
        freqs =np.asarray(idx)-np.repeat(int(length/2),len(idx))
        # get 2 norm length of frequencies
        freq_length = get_k_norm(freqs, 2)
        # get coefflitude (2norm of real and imag parts)
        coeff = np.round(np.absolute(fourier_landscape[idx].real),10)
        total_coeff += coeff
        #coeff = np.round(get_k_norm([fourier_landscape[idx].real,fourier_landscape[idx].imag], 2),7)
        if(coeff > 0):                
            list_of_nonzero_adj_freq_coeffs.append(freq_length*coeff)
            list_of_non_zero_freq_lenghts.append(freq_length)
    return list_of_non_zero_freq_lenghts, list_of_nonzero_adj_freq_coeffs, total_coeff

def get_extra_fourier_metrics(landscape):
    non_zero_freq_lengths, freq_length_coeffs, total_coeff = calc_fourier_frequencies_coeffs(landscape)
    avg_nonzero_freqs = np.mean(non_zero_freq_lengths)
    med_nonzero_freqs = np.median(non_zero_freq_lengths)
    num_nonzero_freqs = len(non_zero_freq_lengths)

    avg_length_coeffs = np.mean(freq_length_coeffs)
    median_length_coeffs = np.median(freq_length_coeffs)
    std_length_coeffs = np.std(freq_length_coeffs)
    return [avg_length_coeffs, median_length_coeffs, std_length_coeffs, avg_nonzero_freqs, med_nonzero_freqs, num_nonzero_freqs, total_coeff]