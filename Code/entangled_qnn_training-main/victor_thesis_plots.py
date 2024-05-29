from matplotlib.ticker import MaxNLocator
import numpy as np
from data import *
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import matplotlib as mpl
from utils import *
from matplotlib import cm

# from victor_thesis_experiments import *
from victor_thesis_utils import *
from victor_thesis_landscapes import *
from victor_thesis_plots import *
from victor_thesis_metrics import *

# from victor_thesis_utils import get_meta_for_mode


def plot_results_metric(mean_list, std_list, pos_list, neg_list, y_labels, x_label, sample_labels):
    """plots the metrics for multiple results

    Args:
        mean_list (list): list of mean values of all metrics
        std_list (list): list of stdv values of all metrics
        pos_list (list): list of positive percentage of scalar curvatur values
        neg_list (list): list of negative percentage of scalar curvatur values
        y_labels (list): labels for the y axis
        x_label (list): labels for the x axis
        title (list): title of plot
    """
    fig, axs = plt.subplots(3,2, figsize=(10,12))
    title_list = ["Total Variation", "Inverse Gradient Standard Deviation","Fourier Density", "Scalar Curvature", "Absolute Scalar Curvature", "% positive and negative Scalar Curvature"]
    #fig().gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    #fig.suptitle(title)
    for i in range(3):
        for j in range(2):
            index = 2*i + j
            axs[i,j].xaxis.set_major_locator(MaxNLocator(integer=True))
            axs[i,j].set_title(title_list[index])
            #axs[i,j].set_title(title)
            if index < 5:
                axs[i,j].errorbar(sample_labels, mean_list[index], std_list[index], linestyle='None', marker='o', capsize=5)
                axs[i,j].set(ylabel=title_list[index], xlabel=x_label)
            else:   
                axs[i,j].bar(sample_labels, neg_list, label="negative", color="cornflowerblue")
                axs[i,j].bar(sample_labels, pos_list, bottom=neg_list, label="positive", color="springgreen") 
                axs[i,j].set(ylabel="% pos/neg Scalar Curvature", xlabel=x_label)
                axs[i,j].legend()
    plt.tight_layout()
    plt.show()


# plot a row of datasets
def plot_row(in_data, titles, gate_name, ansatz, mode="default"):
    """gets a set of data (i.e. a few loss landscapes) and plots them in a row with a selected mode

    Args:
        in_data (array): set of 2D data (i.e. loss landscapes)
        titles (_type_): _description_
        ansatz (string): name of ansatz
        mode (str, optional): which mode to display in, default, logarithmic scale or gradient magnitudes. Defaults to "default".
    """
    width = len(in_data)
    min_val = np.min(in_data)
    max_val = np.max(in_data)
    fig, ax = plt.subplots(1, width, figsize=(9, 3))
    for data_idx in range(width):
        data = in_data[data_idx]
        # get mode dependent settings such as titles, color maps, thresholds, etc.
        c_map, sup_title, title, v_min, v_max = get_meta_for_mode(
            mode, data, min_val, max_val, titles, data_idx, gate_name, ansatz
        )
        length = len(data)
        x_labels = []
        # create labels
        for i in range(length):
            n = f"{np.round(i*2/length, 1)} $\\pi$"
            x_labels.append(n)
        y_labels = reversed(x_labels)
        # do plot stuff
        # for logarithmic scale
        if mode == "log_scale":
            data = data + v_min
            im = ax[data_idx].imshow(
                data, cmap=c_map, norm=matplotlib.colors.LogNorm(vmin=v_min, vmax=v_max)
            )
        else:
            # normal scale
            im = ax[data_idx].imshow(data, cmap=c_map, vmin=v_min, vmax=v_max)
        # what happens to values below the color bar (=legend) threshold
        cm_copy = im.cmap.copy()
        cm_copy.set_under("r", 1)
        im.cmap = cm_copy
        # set label ticks
        ax[data_idx].set_xticks(np.arange(len(x_labels)), labels=x_labels)
        ax[data_idx].set_yticks(np.arange(len(x_labels)), labels=y_labels)
        ax[data_idx].set_ylabel("$\\phi$", rotation=180, va="center")
        ax[data_idx].set_xlabel("$\\lambda$")
        tick_density = int(length / 4)
        # only display every x'th tick
        ax[data_idx].set_xticks(ax[data_idx].get_xticks()[::tick_density])
        ax[data_idx].set_yticks(ax[data_idx].get_yticks()[::tick_density])
        plt.setp(
            ax[data_idx].get_xticklabels(),
            rotation=45,
            ha="right",
            rotation_mode="anchor",
        )

        ax[data_idx].set_title(title)
    fig.colorbar(im, ax=ax.ravel().tolist(), shrink=0.58)
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.75, top=0.9, wspace=0.7)
    fig.suptitle(sup_title, x=0.43)
    plt.show()


def plot_metrics_convergence(
    non_entangled_metric_data, entangled_metric_data, metric_name, min_ticks
):
    """a helper function to visualize whether or not certain metrics converge with fewer ticks
    """
    fig, axes = plt.subplots(1, len(entangled_metric_data))
    fig().gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    fig.suptitle(f"{metric_name} for {len(entangled_metric_data)} runs")
    x = range(min_ticks, len(entangled_metric_data[0]) + min_ticks, 1)
    for run_id in range(len(entangled_metric_data)):
        axes[run_id].plot(x, non_entangled_metric_data[run_id], label="non entangled")
        axes[run_id].plot(x, entangled_metric_data[run_id], label="entangled")
    plt.ylabel("value of metric")  # add Y-axis label
    plt.xlabel("ticks")  # add X-axis label
    fig.tight_layout()
    plt.legend()
    plt.show()


# plots 2d fourier landscapes
def plot_fourier_row(landscapes, titles):
    """takes the frequency representation of a set of 2D loss landscapes and plots them in a row

    Args:
        landscapes (array): fourier representations of 2D loss landscapes
    """
    landscapes = np.array(landscapes)
    width = len(landscapes)
    min_val = 0
    max_val = 0.7
    fig, ax = plt.subplots(1, width, figsize=(9, 3))
    for data_idx in range(width):
        data = landscapes[data_idx]
        length = len(data)
        x_labels = []
        # create labels
        for i in range(length):
            n = f"{i}"
            x_labels.append(n)
        y_labels = x_labels
        # do plot stuff

        im = ax[data_idx].imshow(data, cmap="viridis", vmin=min_val, vmax=max_val)
        # what happens to values below the color bar (=legend) threshold
        # set label ticks
        ax[data_idx].set_xticks(np.arange(len(x_labels)), labels=x_labels)
        ax[data_idx].set_yticks(np.arange(len(x_labels)), labels=y_labels)
        ax[data_idx].set_ylabel("y freq", rotation=90, va="center")
        ax[data_idx].set_xlabel("x freq")
        tick_density = int(length / 4)
        # only display every x'th tick
        ax[data_idx].set_xticks(ax[data_idx].get_xticks()[::tick_density])
        ax[data_idx].set_yticks(ax[data_idx].get_yticks()[::tick_density])
        plt.setp(
            ax[data_idx].get_xticklabels(),
            rotation=45,
            ha="right",
            rotation_mode="anchor",
        )

        ax[data_idx].set_title(titles[data_idx])
    fig.colorbar(im, ax=ax.ravel().tolist(), shrink=0.58)
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.75, top=0.9, wspace=0.7)
    plt.show()


def plot_scatter_of_U3(landscape, points, ticks):
    """takes a 3D loss landscape (generated with U3 gate) and plots it as a 3D scatter plot,
      where the opacity of each sample is logarithmically proportional to its value, 
      highlighting global minima

    Args:
        landscape (array): 3D loss landscape
        points (array): contains the positions of the sampled points
    """
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(projection="3d")
    v_min = 0.000000001
    v_max = 1
    values = np.array(landscape) + v_min
    # the higher v_min, the more similar entangled and non entangled sample minima look

    x = points[0]
    y = points[1]
    z = points[2]
    c_white = matplotlib.colors.colorConverter.to_rgba("white", alpha=0)
    c_red = matplotlib.colors.colorConverter.to_rgba("red", alpha=1)
    cmap_rb = matplotlib.colors.LinearSegmentedColormap.from_list(
        "rb_cmap", [c_red, c_white], 512
    )
    im = ax.scatter(
        x,
        y,
        z,
        c=values,
        cmap=cmap_rb,
        norm=matplotlib.colors.LogNorm(vmin=v_min, vmax=v_max),
        depthshade=0,
    )
    # set labels
    length = 6
    x_labels = []
    # create labels
    for i in range(length + 1):
        n = f"{np.round(i*2/length,1)} $\\pi$"
        x_labels.append(n)
    x_labels = x_labels
    y_labels = x_labels
    z_labels = x_labels
    # set label ticks
    tick_density = int(length / 5)
    ax.set_xticks(np.arange(len(x_labels)), labels=x_labels)
    ax.set_yticks(np.arange(len(x_labels)), labels=y_labels)
    ax.set_zticks(np.arange(len(x_labels)), labels=z_labels)
    # ax.set_xticks(ax.get_xticks()[::tick_density])
    # ax.set_yticks(ax.get_yticks()[::tick_density])
    # ax.set_zticks(ax.get_zticks()[::tick_density])
    ax.set_ylabel("$\\phi$", rotation=180, va="center")
    ax.set_xlabel("$\\lambda$")
    ax.set_zlabel("der andere parameter")
    # only display every x'th tick

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    ax.set_title(f"U3 Minima for (finish title later)")
    # set colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Loss", rotation=-90, va="bottom")
    plt.show()


def plot_3d_loss_landscape(landscape, ansatz, title):
    """takes a 2D landscape and plots it in 3D where the third dimension 
    (as well as the coloring) corresponds to the value of each entry
    (https://matplotlib.org/stable/gallery/images_contours_and_fields/image_annotated_heatmap.html)

    Args:
        landscape (array): 2D array representing a 2D landscape
        ansatz (string): name of ansatz
    """
    ls = np.array(landscape)
    min_val = np.min(ls)
    max_val = np.max(ls)
    length = len(ls)
    x_labels = []
    # create labels
    for i in range(length):
        n = f"{np.round(i*2/length,1)} $\\pi$"
        x_labels.append(n)
    y_labels = reversed(x_labels)
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    X = np.arange(0, length, 1)
    Y = np.arange(0, length, 1)
    X, Y = np.meshgrid(X, Y)
    im = ax.plot_surface(
        X, Y, ls, cmap="plasma", vmin=min(min_val, 0), vmax=max(max_val, 1)
    )
    # set label ticks
    ax.set_xticks(np.arange(len(x_labels)), labels=x_labels)
    ax.set_yticks(np.arange(len(x_labels)), labels=y_labels)
    ax.set_ylabel("$\\phi$", rotation=-45, va="center")
    ax.set_xlabel("$\\lambda$", rotation=180, va="center")
    ax.set_zlabel("Loss")
    tick_density = int(length / 5)
    # only display every x'th tick
    ax.set_xticks(ax.get_xticks()[::tick_density])
    ax.set_yticks(ax.get_yticks()[::tick_density])
    plt.setp(ax.get_xticklabels(), rotation=0, ha="right", rotation_mode="anchor")
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Loss", rotation=-90, va="bottom")
    #ax.set_title(f"{ansatz}($\\phi,\\lambda)$ Approximating {title}")
    plt.show()


def plot_3d_loss_landscape_curv(landscape, ansatz, curv_mode="scalar"):
    """generates a 3D plot of a 2D landscape where the third dimension (height) 
    corresponds to the values of the landscape and the coloring corresponds to the curvature

    Args:
        landscape (array): 2D array of a loss landscape
        ansatz (string): name of the ansatz
        curv_mode (str, optional): What kind of curvature you want to display. Defaults to "scalar".
    """
    landscape = np.array(landscape)
    if curv_mode == "scalar":
        curv = calc_scalar_curvature(landscape)
    elif curv_mode == "grad":
        curv = get_grad_curv(landscape)
    min_val = np.min(curv)
    max_val = np.max(curv)
    length = len(curv)
    x_labels = []
    # create labels
    for i in range(length):
        n = f"{np.round(i*2/length,1)} $\\pi$"
        x_labels.append(n)
    y_labels = reversed(x_labels)
    _, ax = plt.subplots(subplot_kw={"projection": "3d"})
    X = np.arange(0, length, 1)
    Y = np.arange(0, length, 1)
    X, Y = np.meshgrid(X, Y)
    norm = mpl.colors.Normalize(vmin=min_val, vmax=max_val)
    c_map = mpl.cm.plasma
    im = ax.plot_surface(X, Y, landscape, cmap=c_map, facecolors=c_map(norm(curv)))
    # set label ticks
    ax.set_xticks(np.arange(len(x_labels)), labels=x_labels)
    ax.set_yticks(np.arange(len(x_labels)), labels=y_labels)
    ax.set_ylabel("$\\phi$", rotation=180, va="center")
    ax.set_xlabel("$\\lambda$")
    ax.set_zlabel("Loss")
    tick_density = int(length / 10)
    # only display every x'th tick
    ax.set_xticks(ax.get_xticks()[::tick_density])
    ax.set_yticks(ax.get_yticks()[::tick_density])
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    m = cm.ScalarMappable(cmap=c_map, norm=norm)
    m.set_array([])
    plt.colorbar(m)
    ax.set_title(f"{ansatz}($\\phi,\\lambda)$ Curvature - {curv_mode} curvature")
    plt.show()


def multi_plot_landscape(landscapes, titles, gate_name, ansatz):
    """takes a set of landscapes (usually three) and generates three rows, 
    each with all landscapes but in different representations 
    (default, logscale and gradient magnitude)

    Args:
        landscapes (array): array of landscapes (which also happen to be ndim arrays)
        titles (string array): the captions for the plots
    """
    data = np.array(landscapes)
    # calculate gradient magnitudes
    gradient_magnitudes = []
    for landscape_idx in range(len(data)):
        gradient = np.gradient(np.array(landscapes[landscape_idx]))
        grad_mag = np.sqrt(gradient[0] ** 2 + gradient[1] ** 2)
        gradient_magnitudes.append(grad_mag)
    # Plot rows for each mode you want to display
    plot_row(data, titles, gate_name, ansatz, mode="default")
    plot_row(data, titles, gate_name, ansatz, mode="log_scale")
    plot_row(gradient_magnitudes, ansatz, titles, gate_name, mode="grad")
