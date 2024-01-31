import torch
import orqviz
import numpy as np
from data import *
from generate_experiments import get_qnn
import numpy as np
from utils import *
from victor_thesis_experiments import *
from victor_thesis_utils import *
from victor_thesis_landscapes import *
from victor_thesis_plots import *
from victor_thesis_metrics import *


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
    loss_z_o = generate_loss_landscape(num_ticks, z_o_inputs, unitary, qnn)
    loss_non_ent = generate_loss_landscape(
        num_ticks, non_entangled_inputs, unitary, qnn
    )
    loss_ent = generate_loss_landscape(num_ticks, entangled_inputs, unitary, qnn)
    # multiplot lanscapes and gradients
    landscapes = [loss_z_o, loss_non_ent, loss_ent]
    names = [
        f"Zero-One, n = 2",
        f"Not Entangled, n = {num_data_points}",
        f"Entangled, n = {num_data_points}",
    ]
    multi_plot_landscape(landscapes, names, gate_name, ansatz)
    # print advanced metrics
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
    if fourier_plot == 1:
        fourier_result_z_o = get_fourier_landscape(z_o_inputs, unitary, qnn)
        orqviz.fourier.plot_2D_fourier_result(
            fourier_result_z_o, max_freq_x=10, max_freq_y=10
        )
    elif fourier_plot == 2:
        fourier_result_non_ent = get_fourier_landscape(
            non_entangled_inputs, unitary, qnn
        )
        orqviz.fourier.plot_2D_fourier_result(
            fourier_result_non_ent, max_freq_x=10, max_freq_y=10
        )
    elif fourier_plot == 3:
        fourier_result_ent = get_fourier_landscape(entangled_inputs, unitary, qnn)
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
    #plot_3d_loss_landscape_curv(loss_z_o, ansatz, "grad")
    plot_3d_loss_landscape_curv(loss_non_ent, ansatz, "scalar")
    #plot_3d_loss_landscape_curv(loss_non_ent, ansatz, "grad")
    plot_3d_loss_landscape_curv(loss_ent, ansatz, "scalar")
    #plot_3d_loss_landscape_curv(loss_ent, ansatz, "grad")

def run_hadamard():
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
        num_ticks=20,
        fourier_plot=3,
    )
    # U3 visualization has troubles with different tick numbers -> maybe sample errors?
    # red dots on logarithmic loss are true minima (below a certain threshold), sometimes they are just above the threshold but still true minima (due to sampling) and they will appear white (not grey, as grey means they are false minima)
    # if num_ticks is too large you cant see the red dots anymore (maybe due to aliasing?)


def run_pauli_x():
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
    U = torch.tensor(
        np.array([[0, -1j], [1j, 0]]), dtype=torch.complex128, device="cpu"
    )
    run_experiment_on(
        "Pauli-X",
        U,
        ansatz="R",
        print_info=False,
        num_data_points=1,
        num_ticks=20,
        fourier_plot=3,
    )


def run_pauli_z():
    U = torch.tensor(np.array([[1, 0], [0, -1]]), dtype=torch.complex128, device="cpu")
    run_experiment_on(
        "Pauli-X",
        U,
        ansatz="R",
        print_info=False,
        num_data_points=1,
        num_ticks=20,
        fourier_plot=3,
    )


def run_phase_s():
    U = torch.tensor(np.array([[1, 0], [0, 1j]]), dtype=torch.complex128, device="cpu")
    run_experiment_on(
        "Pauli-X",
        U,
        ansatz="R",
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


# run main() for all experiments
# main()
