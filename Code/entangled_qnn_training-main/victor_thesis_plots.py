import numpy as np
from data import *
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import matplotlib as mpl
from utils import *
from matplotlib import cm
from victor_thesis_experiments import *
from victor_thesis_utils import *
from victor_thesis_landscapes import *
from victor_thesis_plots import *
from victor_thesis_metrics import *
from victor_thesis_utils import get_meta_for_mode


# plot a row of datasets
def plot_row(in_data, titles, gate_name, ansatz, mode="default"):
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

# plots 2d fourier landscapes
def plot_fourier_row(landscapes, titles):
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
    #fig.suptitle(sup_title, x=0.43)
    plt.show()






# 3D scatter plot of U3 Gate
def plot_scatter_of_U3(landscape, points, ticks):
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
        depthshade=0
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
    z_labels = x_labels  # just for fun
    # set label ticks
    # labels not really working
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


# plot 3D loss landscape
# https://matplotlib.org/stable/gallery/images_contours_and_fields/image_annotated_heatmap.html
def plot_3d_loss_landscape(landscape, ansatz, title):
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
    # im = ax.imshow(ls, cmap = "plasma",vmin=min(min_val, 0), vmax=max(max_val,1))
    X = np.arange(0, length, 1)
    Y = np.arange(0, length, 1)
    X, Y = np.meshgrid(X, Y)
    im = ax.plot_surface(
        X, Y, ls, cmap="plasma", vmin=min(min_val, 0), vmax=max(max_val, 1)
    )
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
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Loss", rotation=-90, va="bottom")
    ax.set_title(f"{ansatz}($\\phi,\\lambda)$ Approximating {title}")
    plt.show()


# plot 3D loss landscape with curvature coloring
def plot_3d_loss_landscape_curv(landscape, ansatz, curv_mode="scalar"):
    landscape = np.array(landscape)
    if curv_mode == "scalar":
        curv = calc_scalar_curvature(landscape)
    elif curv_mode == "grad":
        curv = get_grad_curv(landscape)
    # normalize from -1 to 1
    # max_entry = np.max(np.absolute(curv))
    # curv = curv/max_entry
    min_val = np.min(curv)
    max_val = np.max(curv)
    length = len(curv)
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
    # cbar = ax.figure.colorbar(im, ax=ax)
    # cbar.ax.set_ylabel("Loss", rotation=-90, va="bottom")
    m = cm.ScalarMappable(cmap=c_map, norm=norm)
    m.set_array([])
    plt.colorbar(m)
    ax.set_title(f"{ansatz}($\\phi,\\lambda)$ Curvature - {curv_mode} curvature")
    plt.show()


# multi plot with gradients
def multi_plot_landscape(landscapes, titles, gate_name, ansatz):
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
