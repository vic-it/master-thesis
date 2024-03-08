import torch
import orqviz
import numpy as np
from data import *
from generate_experiments import get_qnn
import numpy as np
from utils import *
from victor_thesis_utils import *
from victor_thesis_landscapes import *
from victor_thesis_plots import *
from victor_thesis_metrics import *

def test_metrics_convergence():
    """a function which tests whether the different metrics converge with a low a mount of samples and displays the results in different line graphs
    """
    # hadamard U2
    qnn = get_qnn("CudaU2", list(range(1)), 1, device="cpu")
    num_qubits = 2
    # qnn = UnitaryParametrization(num_wires=num_qubits, num_layers=1, device='cpu')
    # print(qnn)
    ############
    data_points = 3
    num_random_unitaries = 5
    min_ticks = 2
    max_ticks = 20
    # generate data points
    unitary_for_shape = torch.tensor(np.array(random_unitary_matrix(1)), dtype=torch.complex128, device="cpu")
    non_entangled_inputs = generate_random_datapoints(data_points, 1, unitary_for_shape)
    entangled_inputs = generate_random_datapoints(data_points, 2, unitary_for_shape)

    # for each random unitary -> do 30 runs with 1 tick, 2 ticks, ... 30 ticks -> evaluate metrics in line charts
    # metric[#run][values by tick 1,...,30]
    non_entangled_igsds = []
    non_entangled_TVs = []
    non_entangled_FDs = []
    non_entangled_SC = []
    entangled_igsds = []
    entangled_TVs = []
    entangled_FDs = []
    entangled_SC = []
    for _ in range(num_random_unitaries):
        unitary = torch.tensor(np.array(random_unitary_matrix(1)) / np.sqrt(2), dtype=torch.complex128, device="cpu")
        non_entangled_igsds_row = []
        non_entangled_TVs_row = []
        non_entangled_FDs_row = []
        non_entangled_SC_row = []
        entangled_igsds_row = []
        entangled_TVs_row = []
        entangled_FDs_row = []
        entangled_SC_row = []
        # calculate metrics
        for ticks in range(min_ticks, max_ticks, 1):
            non_entangled_landscape, _= generate_loss_landscape(ticks, 2 , non_entangled_inputs, unitary, qnn)  
            entangled_landscape, _= generate_loss_landscape(ticks, 2 , entangled_inputs, unitary, qnn)   
            non_entangled_igsds_row.append(get_k_norm(calc_IGSD(non_entangled_landscape),1))
            non_entangled_TVs_row.append(calc_total_variation(non_entangled_landscape))
            non_entangled_FDs_row.append(calc_fourier_density(non_entangled_landscape))
            non_entangled_SC_row.append(get_k_norm(calc_scalar_curvature(non_entangled_landscape),1))
            entangled_igsds_row.append(get_k_norm(calc_IGSD(entangled_landscape),1))
            entangled_TVs_row.append(calc_total_variation(entangled_landscape))
            entangled_FDs_row.append(calc_fourier_density(entangled_landscape))
            entangled_SC_row.append(get_k_norm(calc_scalar_curvature(entangled_landscape),1))
        non_entangled_igsds.append(non_entangled_igsds_row)
        non_entangled_TVs.append(non_entangled_TVs_row)
        non_entangled_FDs.append(non_entangled_FDs_row)
        non_entangled_SC.append(non_entangled_SC_row)
        entangled_igsds.append(entangled_igsds_row)
        entangled_TVs.append(entangled_TVs_row)
        entangled_FDs.append(entangled_FDs_row)
        entangled_SC.append(entangled_SC_row)
    print(non_entangled_FDs)
    plot_metrics_convergence(non_entangled_TVs, entangled_TVs, "total variation", min_ticks)
    plot_metrics_convergence(non_entangled_FDs, non_entangled_FDs, "fourier density (non entangled)", min_ticks)
    plot_metrics_convergence(entangled_FDs, entangled_FDs, "fourier density (entangled)", min_ticks)
    plot_metrics_convergence(non_entangled_igsds, entangled_igsds, "igsd (sum of absolutes)", min_ticks)
    plot_metrics_convergence(non_entangled_SC, entangled_SC, "scalar curvature (sum of absolutes)", min_ticks)
    
# Experiments Framework
def run_experiment_on(
    gate_name,
    unitary,
    ansatz,
    print_info=True,
    num_data_points=1,
    num_ticks=20,
    fourier_plot=1,
):
    """runs various tests and representations for 2D and 3D loss landscapes 
    to test both metrics as well as the general implementation 
    and to give a first look at how different levels of entanglement may affect a loss landscape

    Args:
        gate_name (string): name of the gate used
        unitary (unitary): unitary that is emulated by the qnn
        ansatz (string): name of the ansatz
        print_info (bool, optional): whether or not you want to print the data points used as well as the expected outputs. Defaults to True.
        num_data_points (int, optional): how many datapoints should be generated and used. Defaults to 1.
        num_ticks (int, optional): the resolution of the loss landscape sampling. Defaults to 20.
        fourier_plot (int, optional): of which kind of datapoints you want the fourier landscape plotted 
            as the orqviz visualization has difficulties with more than one. Defaults to 1.
    """
    qnn = get_qnn("Cuda" + ansatz, list(range(1)), 1, device="cpu")
    # generate data points
    non_entangled_inputs = generate_random_datapoints(num_data_points, 1, unitary)
    entangled_inputs = generate_random_datapoints(num_data_points, 2, unitary)
    z_o_inputs = get_zero_one_datapoints()
    if print_info:
        # print data points
        print_datapoints(z_o_inputs, "zero-one")
        print_datapoints(non_entangled_inputs, "not entangled")
        print_datapoints(entangled_inputs, "entangled")
        # print expected output
        print_expected_output(unitary, z_o_inputs, "zero one")
    # calculate loss landscapes
    loss_z_o = generate_2d_loss_landscape(num_ticks, z_o_inputs, unitary, qnn)
    loss_non_ent = generate_2d_loss_landscape(
        num_ticks, non_entangled_inputs, unitary, qnn
    )
    loss_ent = generate_2d_loss_landscape(num_ticks, entangled_inputs, unitary, qnn)
    # multiplot lanscapes and gradients
    landscapes = [loss_z_o, loss_non_ent, loss_ent]
    names = [
        f"Zero-One, n = 2",
        f"Not Entangled, n = {num_data_points}",
        f"Entangled, n = {num_data_points}",
    ]
    multi_plot_landscape(landscapes, names, gate_name, ansatz)
    #print advanced metrics
    print(
        "TOTAL VARIATION: ",
        calc_total_variation(landscapes[0]),
        calc_total_variation(landscapes[1]),
        calc_total_variation(landscapes[2]),
    )
    for landscape in landscapes:
        igsd = calc_IGSD(landscape)
        print("IGSD (dir 1): ", igsd[0])
        print("IGSD (dir 2): ", igsd[1])
        print("---------")
    # plot fourier stuff (can only plot one at a time?)
    print("Frequency Domain for Plot", fourier_plot)
    fourier_result_z_o = get_fourier_landscape(z_o_inputs, unitary, qnn)
    fd0, _ = calc_fourier_density(landscapes[0])
    print("FD3", fd0)
    fourier_result_non_ent = get_fourier_landscape(non_entangled_inputs, unitary, qnn)
    fd1, _ = calc_fourier_density(landscapes[1])
    print("FD3", fd1)
    fourier_result_ent = get_fourier_landscape(entangled_inputs, unitary, qnn)
    fd2, _ = calc_fourier_density(landscapes[2])
    print("FD3", fd2)
    if fourier_plot == 1:
        orqviz.fourier.plot_2D_fourier_result(
            fourier_result_z_o, max_freq_x=10, max_freq_y=10
        )
    elif fourier_plot == 2:
        orqviz.fourier.plot_2D_fourier_result(
            fourier_result_non_ent, max_freq_x=10, max_freq_y=10
        )
    elif fourier_plot == 3:
        orqviz.fourier.plot_2D_fourier_result(
            fourier_result_ent, max_freq_x=10, max_freq_y=10
        )
    # plot 3d scatter plots for U3 gate minimization
    loss_z_o_3d, points_z_o = generate_3D_loss_landscape_with_labels(
        num_ticks, z_o_inputs, unitary
    )
    loss_non_ent_3d, points_non_ent = generate_3D_loss_landscape_with_labels(
        num_ticks, non_entangled_inputs, unitary
    )
    loss_ent_3d, points_ent = generate_3D_loss_landscape_with_labels(
        num_ticks, entangled_inputs, unitary
    )
    plot_scatter_of_U3(loss_z_o_3d, points_z_o, num_ticks)
    plot_scatter_of_U3(loss_non_ent_3d, points_non_ent, num_ticks)
    plot_scatter_of_U3(loss_ent_3d, points_ent, num_ticks)
    # plot basic 3d loss landscapes
    plot_3d_loss_landscape(loss_z_o, ansatz, f"{gate_name} (Zero-One, n = 2)")
    plot_3d_loss_landscape(
        loss_non_ent, ansatz, f"{gate_name} (Not Entangled, n = {num_data_points})"
    )
    plot_3d_loss_landscape(
        loss_ent, ansatz, f"{gate_name} (Entangled, n = {num_data_points})"
    )
    # plot 3d loss landscapes with curvature coloring
    plot_3d_loss_landscape_curv(loss_z_o, ansatz, "scalar")
    # plot_3d_loss_landscape_curv(loss_z_o, ansatz, "grad")
    plot_3d_loss_landscape_curv(loss_non_ent, ansatz, "scalar")
    # plot_3d_loss_landscape_curv(loss_non_ent, ansatz, "grad")
    plot_3d_loss_landscape_curv(loss_ent, ansatz, "scalar")
    # plot_3d_loss_landscape_curv(loss_ent, ansatz, "grad")


def run_hadamard():    
    """runs the experiments framework on the hadamard unitary
    """
    # EXP on Hadamard
    U = torch.tensor(
        np.array([[1, 1], [1, -1]]) / np.sqrt(2), dtype=torch.complex128, device="cpu"
    )
    run_experiment_on(
        "Hadamard",
        U,
        ansatz="U2",
        print_info=False,
        num_data_points=1,
        num_ticks=10,
        fourier_plot=3,
    )
    # U3 visualization has troubles with different tick numbers -> maybe sample errors?
    # red dots on logarithmic loss are true minima (below a certain threshold), sometimes they are just above the threshold but still true minima (due to sampling) and they will appear white (not grey, as grey means they are false minima)
    # if num_ticks is too large you cant see the red dots anymore (maybe due to aliasing?)


def run_pauli_x():
    """runs the experiments framework on the pauli X unitary
    """
    U = torch.tensor(np.array([[0, 1], [1, 0]]), dtype=torch.complex128, device="cpu")
    run_experiment_on(
        "Pauli-X",
        U,
        ansatz="R",
        print_info=False,
        num_data_points=1,
        num_ticks=20,
        fourier_plot=3,
    )


def run_pauli_y():
    """runs the experiments framework on the pauli Y unitary
    """
    U = torch.tensor(
        np.array([[0, -1j], [1j, 0]]), dtype=torch.complex128, device="cpu"
    )
    run_experiment_on(
        "Pauli-Y",
        U,
        ansatz="R",
        print_info=False,
        num_data_points=1,
        num_ticks=20,
        fourier_plot=3,
    )


def run_pauli_z():
    """runs the experiments framework on the pauli Z unitary
    """
    U = torch.tensor(np.array([[1, 0], [0, -1]]), dtype=torch.complex128, device="cpu")
    run_experiment_on(
        "Pauli-Z",
        U,
        ansatz="R",
        print_info=False,
        num_data_points=1,
        num_ticks=20,
        fourier_plot=3,
    )


def run_phase_s():
    """runs the experiments framework on the phase (S) unitary
    """
    U = torch.tensor(np.array([[1, 0], [0, 1j]]), dtype=torch.complex128, device="cpu")
    run_experiment_on(
        "Phase-S",
        U,
        ansatz="U2",
        print_info=False,
        num_data_points=1,
        num_ticks=20,
        fourier_plot=3,
    )


def run_random_unitary():   
    """runs the experiments framework on a randomly generated unitary
    """ 
    U = torch.tensor(np.array(random_unitary_matrix(1)), dtype=torch.complex128, device="cpu")
    run_experiment_on(
        "Random Unitary",
        U,
        ansatz="U2",
        print_info=False,
        num_data_points=1,
        num_ticks=20,
        fourier_plot=3,
    )


def main():
    run_hadamard()
    run_pauli_x()
    run_pauli_y()
    run_pauli_z()
    run_phase_s()

print("starting")


# run main() for all experiments
#run_hadamard()
print("done")
