# a class containing reproducable methods used to generate the figures for the thesis
from datetime import datetime

from torch import tensor
from qnns.cuda_qnn import CudaPennylane
from victor_thesis_utils import *
from victor_thesis_landscapes import *
from victor_thesis_plots import *
from victor_thesis_metrics import *

def figure_basic_loss_landscape():
    qnn = get_qnn("CudaU2", list(range(1)), 1, device="cpu")
    U = torch.tensor(np.array([[1, 1], [1, -1]]) / np.sqrt(2), dtype=torch.complex128, device="cpu")
    #entangled_inputs = generate_random_datapoints(2, 1, U)
    entangled_inputs= tensor([[[ 0.1076-0.5764j], [-0.7628+0.2727j]], [[ 0.2477+0.3764j], [-0.8891+0.0808j]]], dtype=torch.complex128)
    print(entangled_inputs)
    loss_ent = generate_2d_loss_landscape(50, entangled_inputs, U, qnn)
    plot_3d_loss_landscape(
        loss_ent, "U2", f"Hadamard (on 2 data points)"
    )
figure_basic_loss_landscape()














