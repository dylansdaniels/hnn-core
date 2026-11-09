# %% [code]
# ==================================================
# SETUP
# ==================================================

import pickle
import pprint
import time
import warnings
from copy import deepcopy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.utils.io import capture_output

from hnn_core import Dipole, MPIBackend, hnn_io, simulate_dipole
from hnn_core.network import pick_connection
from hnn_core.optimization import Optimizer
from hnn_core.viz import plot_dipole

dt = 0.025
n_procs = 8

beta10_datafile = "BETA10_4hnn.txt"
beta04_datafile = "BETA04_100_4hnn.txt"
beta44_datafile = "BETA44_4hnn.txt"
agg_datafile = "Final_HNN_dipole_S1_data_shifted.txt"

base_network_file = "agg_80_optim_Final.json"

analysis_dir = Path.cwd()

if analysis_dir.name != "tep_modeling_scripts":
    try:
        analysis_dir = next(
            path for path in analysis_dir.rglob("tep_modeling_scripts") if path.is_dir()
        )
    except StopIteration:
        raise FileNotFoundError("Could not resolve analysis directory")


def sub_dipole_data(subject_file):
    # load subject data
    df = pd.read_csv(
        analysis_dir / "data" / subject_file,
        sep=" ",
        header=None,
        names=["ms", "nAm"],
    )

    # instantiate dipole object from subject data
    x = df["ms"].to_numpy()
    y = df["nAm"].to_numpy()
    # y = y * 1e9  # scale the dipole
    dpl_data = Dipole(x, y)

    return dpl_data


beta10_dpl = sub_dipole_data(beta10_datafile)
beta04_dpl = sub_dipole_data(beta04_datafile)
beta44_dpl = sub_dipole_data(beta44_datafile)
agg_dpl = sub_dipole_data(agg_datafile)

# instantiate base network
base_net = hnn_io.read_network_configuration(
    analysis_dir / "networks" / base_network_file
)

# %%
# ==================================================
# FUNCTIONS
# ==================================================


def plot_hist_sim_data(
    net,
    data_dict,
    sim_dict,
    colors={},
):
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(8, 6),
        constrained_layout=True,
        height_ratios=[1, 3],
        sharex=True,
    )

    # plot histogram
    net.cell_response.plot_spikes_hist(
        ax=axes[0],
        spike_types=[
            "evprox",
            "evdist",
            # "ngfc",
        ],
        show=False,
    )

    for label, data in sim_dict.items():
        # plot simulation data
        plot_dipole(
            data,
            ax=axes[1],
            layer="agg",
            color=colors.get(label, "black"),
            show=False,
            average=True,
            label=label,
        )

    # plot subject data
    for label, data in data_dict.items():
        plot_dipole(
            data, ax=axes[1], color=colors.get(label, "blue"), show=False, label=label
        )

    # format plot
    axes[1].lines[-1].set_linewidth(1.5)

    # Add experimental data labels to the legend
    handles, labels = axes[1].get_legend_handles_labels()
    for label in data_dict:
        if label not in labels:
            handles.append(plt.Line2D([], [], color=colors.get(label, "blue")))
            labels.append(label)

    axes[1].legend(handles, labels)

    return fig


def plot_calcium(
    net,
    ax=None,
    trial_idx=0,
    cell_type="L5_pyramidal",
    compartment="apical_tuft",
    dpls=None,
    sort_by_peak=True,
    vmin=0,
    vmax=None,
    tick_interval=50,
    figsize=(8, 4),
    print_range=False,
    show=True,
):
    """
    Plot calcium concentration
    """
    cell_response = net.cell_response
    times = np.asarray(cell_response.times)
    gids = np.asarray(
        net.gid_ranges[cell_type],
        dtype=int,
    )

    ca = np.zeros((len(gids), len(times)))

    for i, gid in enumerate(gids):
        ca[i] = cell_response.ca[trial_idx][gid][compartment]

    if print_range:
        print(
            f"Calcium range ({cell_type}, {compartment}): "
            f"Ca min={ca.min():.8g} mM, Ca max={ca.max():.8g} mM"
        )

    if sort_by_peak:
        sort_idx = np.argsort(np.argmax(ca, axis=1))[::-1]
        ca = ca[sort_idx]

    if vmin is None:
        vmin = ca.min()
    if vmax is None:
        vmax = ca.max()

    if ax is None:
        fig, ax = plt.subplots(
            figsize=figsize,
            constrained_layout=True,
        )
    else:
        fig = ax.figure

    timevec = np.arange(len(times))

    im = ax.pcolormesh(
        timevec,
        np.arange(len(gids)),
        ca,
        shading="gouraud",
        vmin=vmin,
        vmax=vmax,
    )

    ax.invert_yaxis()
    ax.set_title(
        f"{cell_type} {compartment} Calcium\n"
        f"Ca min={ca.min():.8g} mM, Ca max={ca.max():.8g} mM"
    )
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Cells")

    cb = fig.colorbar(
        im,
        ax=ax,
        pad=0.03,
    )
    cb.set_ticks(np.linspace(vmin, vmax, 5))
    cb.set_label("Ca concentration $mM$")

    tick_times = np.arange(
        0,
        times[-1] + 1 - np.mod(times[-1], tick_interval),
        tick_interval,
    )
    tick_indices = np.searchsorted(times, tick_times)

    ax.set_xticks(tick_indices)
    ax.set_xticklabels(tick_times)
    ax.set_xlim(0, len(times) - 1)

    if dpls is not None:
        dpl = dpls[trial_idx] if isinstance(dpls, list) else dpls

        ax2 = ax.twinx()
        ax2.plot(
            timevec,
            dpl.data["agg"],
            color=(1, 1, 1),
            linewidth=2,
        )
        ax2.set_ylabel("Dipole Moment (nAm)")

    if show:
        plt.show()

    return fig, ax


def plot_calcium_fig(
    net,
    title=None,
    trial_idx=0,
    cell_type="L5_pyramidal",
    compartment="apical_tuft",
    spike_types=["evprox", "evdist"],
    dpls=None,
    sort_by_peak=True,
    vmin=0,
    vmax=None,
    tick_interval=50,
    inset_colorbar=False,
    figsize=(8, 6),
    font_scale=1.0,
    show=True,
):

    cell_response = net.cell_response
    times = np.asarray(cell_response.times)
    gids = np.asarray(
        net.gid_ranges[cell_type],
        dtype=int,
    )

    ca = np.zeros((len(gids), len(times)))

    for i, gid in enumerate(gids):
        ca[i] = cell_response.ca[trial_idx][gid][compartment]

    if sort_by_peak:
        sort_idx = np.argsort(np.argmax(ca, axis=1))[::-1]
        ca = ca[sort_idx]

    if vmin is None:
        vmin = ca.min()
    if vmax is None:
        vmax = ca.max()

    fig, axes = plt.subplots(
        2,
        1,
        sharex=False,
        figsize=figsize,
        constrained_layout=True,
        height_ratios=[1, 4],
    )

    if title is not None:
        fig.suptitle(title)

    cell_response.plot_spikes_hist(
        ax=axes[0],
        spike_types=spike_types,
        trial_idx=trial_idx,
        show=False,
    )

    timevec = np.arange(len(times))

    im = axes[1].pcolormesh(
        timevec,
        np.arange(len(gids)),
        ca,
        shading="gouraud",
        vmin=vmin,
        vmax=vmax,
    )

    axes[1].invert_yaxis()
    axes[1].set_title(f"{cell_type} {compartment} Calcium")
    axes[1].set_xlabel("Time (ms)")
    axes[1].set_ylabel("Cells")

    if not inset_colorbar:
        cb = fig.colorbar(
            im,
            ax=axes[1],
            pad=0.03,
        )
        cb.set_ticks(
            np.linspace(
                vmin,
                vmax,
                5,
            )
        )
        cb.set_label("Ca concentration $mM$")
    else:
        cbar_ax = axes[1].inset_axes(
            [0.80, 0.05, 0.025, 0.9],
        )
        cb = fig.colorbar(im, cax=cbar_ax)
        cb.set_ticks(
            np.linspace(
                vmin,
                vmax,
                5,
            )
        )
        cb.ax.tick_params(
            colors="white",
            labelsize=10,
        )
        for label in cb.ax.get_yticklabels():
            label.set_fontweight("bold")
        cb.set_label(
            "Ca concentration $mM$",
            color="white",
            fontweight="bold",
        )
        cb.ax.yaxis.get_offset_text().set_color("white")
        cb.ax.yaxis.get_offset_text().set_fontweight("bold")

    tick_times = np.arange(
        0,
        times[-1] + 1 - np.mod(times[-1], tick_interval),
        tick_interval,
    )
    tick_indices = np.searchsorted(
        times,
        tick_times,
    )

    axes[1].set_xticks(tick_indices)
    axes[1].set_xticklabels(tick_times)
    axes[1].set_xlim(0, len(times) - 1)

    if dpls is not None:
        dpl = dpls[trial_idx] if isinstance(dpls, list) else dpls

        ax2 = axes[1].twinx()
        ax2.plot(
            timevec,
            dpl.data["agg"],
            color=(1, 1, 1),
            linewidth=2,
        )
        ax2.set_ylabel("Dipole Moment (nAm)")

    # format figure
    for ax in fig.axes:
        ax.title.set_fontsize(16 * font_scale)
        ax.xaxis.label.set_size(14 * font_scale)
        ax.yaxis.label.set_size(14 * font_scale)

        ax.tick_params(
            axis="both",
            labelsize=12 * font_scale,
        )

        legend = ax.get_legend()
        if legend is not None:
            for text in legend.get_texts():
                text.set_fontsize(12 * font_scale)

            legend.get_title().set_fontsize(12 * font_scale)

        for text in ax.texts:
            text.set_fontsize(12 * font_scale)

    fig._suptitle.set_fontsize(18 * font_scale) if fig._suptitle else None

    if show:
        plt.show()

    return fig, axes


def plot_calcium_spiking_fig(
    net,
    title=None,
    trial_idx=0,
    cell_type="L5_pyramidal",
    compartment="apical_tuft",
    spike_types=["evprox", "evdist"],
    dpls=None,
    sort_by_peak=True,
    vmin=0,
    vmax=None,
    tick_interval=50,
    figsize=(10, 12),
    show=True,
):
    """
    Plot calcium spiking figure
    """
    cell_response = net.cell_response
    times = np.asarray(cell_response.times)
    gids = np.asarray(
        net.gid_ranges[cell_type],
        dtype=int,
    )

    ca = np.zeros((len(gids), len(times)))

    for i, gid in enumerate(gids):
        ca[i] = cell_response.ca[trial_idx][gid][compartment]

    if sort_by_peak:
        sort_idx = np.argsort(np.argmax(ca, axis=1))[::-1]
        ca = ca[sort_idx]

    if vmin is None:
        vmin = ca.min()
    if vmax is None:
        vmax = ca.max()

    fig, axes = plt.subplots(
        3,
        1,
        sharex=False,
        figsize=figsize,
        constrained_layout=True,
        height_ratios=[1, 4, 4],
    )

    if title is not None:
        fig.suptitle(title)

    cell_response.plot_spikes_hist(
        ax=axes[0],
        spike_types=spike_types,
        trial_idx=trial_idx,
        show=False,
    )

    timevec = np.arange(len(times))

    im = axes[1].pcolormesh(
        timevec,
        np.arange(len(gids)),
        ca,
        shading="gouraud",
        vmin=vmin,
        vmax=vmax,
    )

    axes[1].invert_yaxis()
    axes[1].set_title(f"{cell_type} {compartment} Calcium")
    axes[1].set_xlabel("Time (ms)")
    axes[1].set_ylabel("Cells")

    cb = fig.colorbar(
        im,
        ax=axes[1],
        pad=0.03,
    )
    cb.set_ticks(
        np.linspace(
            vmin,
            vmax,
            5,
        )
    )
    cb.set_label("Ca concentration $mM$")

    tick_times = np.arange(
        0,
        times[-1] + 1 - np.mod(times[-1], tick_interval),
        tick_interval,
    )
    tick_indices = np.searchsorted(
        times,
        tick_times,
    )

    axes[1].set_xticks(tick_indices)
    axes[1].set_xticklabels(tick_times)
    axes[1].set_xlim(0, len(times) - 1)

    if dpls is not None:
        dpl = dpls[trial_idx] if isinstance(dpls, list) else dpls

        ax2 = axes[1].twinx()
        ax2.plot(
            timevec,
            dpl.data["agg"],
            color=(1, 1, 1),
            linewidth=2,
        )
        ax2.set_ylabel("Dipole Moment (nAm)")

        raster_fig = cell_response.plot_spikes_raster(
            ax=axes[2],
            trial_idx=trial_idx,
            show=False,
            dpl=dpls,
            overlay_dipoles=True,
        )

    axes[2].get_legend().set_loc("lower right")

    if show:
        plt.show()

    return fig, axes


def format_val(v):
    if isinstance(v, float):
        return f"{v:.4g}"
    return v


def compare_drive_params(*drives):
    """Compare dynamics, synaptic delays, and AMPA/NMDA weights across drives."""

    params = [
        ("mu", lambda drive: drive["dynamics"]["mu"]),
        ("sigma", lambda drive: drive["dynamics"]["sigma"]),
        ("delay", lambda drive: drive["synaptic_delays"]),
        ("ampa", lambda drive: drive["weights_ampa"]),
        ("nmda", lambda drive: drive["weights_nmda"]),
    ]

    for param_name, get_value in params:
        print(f"{param_name}:")

        for drive_name, drive in drives:
            data = get_value(drive)

            if isinstance(data, dict):
                formatted_values = {
                    key: format_val(value) for key, value in data.items()
                }
            elif isinstance(data, float):
                formatted_values = format_val(data)
            else:
                formatted_values = data

            formatted_data = pprint.pformat(
                formatted_values,
                indent=4,
            )
            indented_data = "\n".join(
                f"      {line}" for line in formatted_data.splitlines()
            )
            print(f"   {drive_name}:")
            print(indented_data)
        print()


def evaluate_optimization(optim, initial_params):
    import math

    import numpy as np

    es = optim.es_
    result = es.result

    param_names = list(initial_params)
    initial_values = np.array(list(initial_params.values()), dtype=float)
    optimized_values = np.asarray(optim.opt_params_, dtype=float)

    popsize = es.opts["popsize"]
    best_eval = result.evals_best
    best_iteration = math.ceil(best_eval / popsize)

    lower_bounds = np.asarray(es.opts["bounds"][0], dtype=float)
    upper_bounds = np.asarray(es.opts["bounds"][1], dtype=float)

    print("=" * 75)
    print("OPTIMIZATION SUMMARY")
    print("=" * 75)
    print(f"Termination: {es.stop()}")
    print(f"Iterations: {es.countiter}")
    print(f"Evaluations: {es.countevals}")
    print(f"Population size: {popsize}")
    print(f"Best evaluation: {best_eval}")
    print(f"Best iteration: {best_iteration}")
    print(f"Best objective: {result.fbest:.8f}")

    print("\n" + "=" * 75)
    print("PARAMETER COMPARISON")
    print("=" * 75)
    print(
        f"{'Parameter':<27} {'Initial':>10} {'Optimized':>10} "
        f"{'Change':>10} {'Bounds':>22} {'Position':>10}"
    )

    for i, (name, initial, optimized) in enumerate(
        zip(param_names, initial_values, optimized_values)
    ):
        change = (optimized / initial - 1) * 100 if initial != 0 else np.nan
        lower = lower_bounds[i]
        upper = upper_bounds[i]
        width = upper - lower

        position = (optimized - lower) / width * 100 if width > 0 else np.nan

        status = ""
        if np.isclose(optimized, lower):
            status = " LOWER"
        elif np.isclose(optimized, upper):
            status = " UPPER"

        print(
            f"{name:<27} {initial:>10.4f} {optimized:>10.4f} "
            f"{change:>+9.2f}% {f'[{lower:.4g}, {upper:.4g}]':>22} "
            f"{position:>9.1f}%{status}"
        )

    return {
        "best_objective": result.fbest,
        "best_evaluation": best_eval,
        "best_iteration": best_iteration,
        "iterations": es.countiter,
        "evaluations": es.countevals,
        "termination": es.stop(),
        "optimized_parameters": dict(zip(param_names, optimized_values)),
    }


def format_bound(bound):
    if isinstance(bound, (list, tuple)) and len(bound) == 2 and bound[0] == "relative":
        return f"{float(bound[1]):g}×"

    if isinstance(bound, (list, tuple)):
        return "[" + ", ".join(f"{float(value):g}" for value in bound) + "]"

    return f"{float(bound):g}"


def get_drive_bounds(drive_config, initial_net):
    rows = []

    for drive, config in drive_config.items():
        initial_drive = initial_net.external_drives[drive]

        for param in ("mu", "sigma"):
            lower, upper = config[param]
            initial = initial_drive["dynamics"][param]

            rows.append(
                {
                    "Drive": drive,
                    "Category": "Timing",
                    "Parameter": param,
                    "Target": "-",
                    "Initial": f"{initial:.5e}",
                    "Lower": format_bound(lower),
                    "Upper": format_bound(upper),
                }
            )

        for receptor, targets in config.get("weights", {}).items():
            initial_weights = initial_drive[f"weights_{receptor}"]

            for target, (lower, upper) in targets.items():
                rows.append(
                    {
                        "Drive": drive,
                        "Category": "Weights",
                        "Parameter": receptor.upper(),
                        "Target": target,
                        "Initial": f"{initial_weights[target]:.3e}",
                        "Lower": format_bound(lower),
                        "Upper": format_bound(upper),
                    }
                )

    return pd.DataFrame(rows).set_index(["Drive", "Category", "Parameter", "Target"])


def plot_spikes_comparison(net_dict):

    _, axes = plt.subplots(
        4,
        1,
        figsize=(8, 12),
        constrained_layout=True,
        # sharex=True,
    )

    i = -1
    for label, net in net_dict.items():
        i += 1
        axes[i].set_title(label)
        net.cell_response.plot_spikes_hist(
            ax=axes[i],
            spike_types=["evprox", "evdist", "ngfc"],
            show=False,
        )
        i += 1
        net.cell_response.plot_spikes_raster(
            ax=axes[i],
            show=False,
            show_legend=False,
        )

    return


def plot_grant_figure(
    net,
    title=None,
    trial_idx=0,
    cell_type="L5_pyramidal",
    compartment="apical_tuft",
    spike_types=["evprox", "evdist"],
    sim_dict=None,
    data_dict=None,
    colors={},
    sim_dpls=None,
    sort_by_peak=True,
    vmin=0,
    vmax=None,
    example_L5_gid=None,
    tick_interval=50,
    figsize=(10, 20),
    show=True,
    font_scale=1.2,
    export_eps=False,
):

    cell_response = net.cell_response
    times = np.asarray(cell_response.times)
    gids = np.asarray(
        net.gid_ranges[cell_type],
        dtype=int,
    )

    ca = np.zeros((len(gids), len(times)))

    for i, gid in enumerate(gids):
        ca[i] = cell_response.ca[trial_idx][gid][compartment]

    if sort_by_peak:
        sort_idx = np.argsort(np.argmax(ca, axis=1))[::-1]
        ca = ca[sort_idx]

    if vmin is None:
        vmin = ca.min()
    if vmax is None:
        vmax = ca.max()

    fig, axes = plt.subplots(
        6,
        1,
        sharex=False,
        figsize=figsize,
        constrained_layout=True,
        height_ratios=[1, 3, 3, 3, 3, 3],
    )
    fig.set_constrained_layout_pads(
        h_pad=0.15,
        hspace=0.15,
    )

    if title is not None:
        fig.suptitle(title)

    # --------------------------------------------------
    # histogram
    # --------------------------------------------------
    cell_response.plot_spikes_hist(
        ax=axes[0],
        spike_types=spike_types,
        trial_idx=trial_idx,
        show=False,
    )

    # --------------------------------------------------
    # simulation dipoles
    # --------------------------------------------------

    for label, data in sim_dict.items():
        # plot simulation data
        plot_dipole(
            data,
            ax=axes[1],
            layer="agg",
            color=colors.get(label, "black"),
            show=False,
            average=True,
            label=label,
        )

    # plot subject data
    for label, data in data_dict.items():
        plot_dipole(
            data, ax=axes[1], color=colors.get(label, "blue"), show=False, label=label
        )

    # format plot
    axes[1].lines[-1].set_linewidth(1.5)

    # Add experimental data labels to the legend
    handles, labels = axes[1].get_legend_handles_labels()
    for label in data_dict:
        if label not in labels:
            handles.append(plt.Line2D([], [], color=colors.get(label, "blue")))
            labels.append(label)

    axes[1].legend(handles, labels)

    # --------------------------------------------------
    # calcium
    # --------------------------------------------------

    timevec = np.arange(len(times))

    im = axes[2].pcolormesh(
        timevec,
        np.arange(len(gids)),
        ca,
        shading="gouraud",
        vmin=vmin,
        vmax=vmax,
        rasterized=True,
    )

    # including a redundancy for good measure ...
    im.set_rasterized(True)

    axes[2].invert_yaxis()
    title_string = f"{cell_type} {compartment} Calcium"
    title_string = title_string.replace("_", " ")
    title_string = title_string.title()
    axes[2].set_title(title_string)
    axes[2].set_xlabel("Time (ms)")
    axes[2].set_ylabel("Cells")

    # cb = fig.colorbar(
    #     im,
    #     ax=axes[2],
    #     pad=0.03,
    # )
    # cb.set_ticks(
    #     np.linspace(
    #         vmin,
    #         vmax,
    #         5,
    #     )
    # )
    # cb.set_label("Ca concentration $mM$")

    cbar_ax = axes[2].inset_axes(
        [0.80, 0.15, 0.025, 0.7],
    )
    cb = fig.colorbar(im, cax=cbar_ax)
    cb.set_ticks(
        np.linspace(
            vmin,
            vmax,
            5,
        )
    )
    cb.ax.tick_params(colors="white")
    cb.set_label("Ca concentration $mM$", color="white")
    cb.ax.yaxis.get_offset_text().set_color("white")

    tick_times = np.arange(
        0,
        times[-1] + 1 - np.mod(times[-1], tick_interval),
        tick_interval,
    )
    tick_indices = np.searchsorted(
        times,
        tick_times,
    )

    axes[2].set_xticks(tick_indices)
    axes[2].set_xticklabels(tick_times)
    axes[2].set_xlim(0, len(times) - 1)

    dpl = sim_dpls[trial_idx] if isinstance(sim_dpls, list) else sim_dpls

    ax2 = axes[2].twinx()
    ax2.plot(
        timevec,
        dpl.data["agg"],
        color=(1, 1, 1),
        linewidth=2,
    )
    ax2.set_ylabel("Dipole Moment (nAm)")

    # --------------------------------------------------
    # spike raster
    # --------------------------------------------------
    raster_fig = cell_response.plot_spikes_raster(
        ax=axes[3],
        trial_idx=trial_idx,
        show=False,
        dpl=sim_dpls,
        overlay_dipoles=True,
    )

    axes[3].get_legend().set_loc("lower right")
    axes[3].set_title("Network Spiking Raster")

    # --------------------------------------------------
    # example L5 cell voltage/calcium
    # --------------------------------------------------
    soma_v_all = np.asarray(
        [cell_response.vsec[trial_idx][gid]["soma"] for gid in gids]
    )

    if example_L5_gid is None:
        gid_idx = np.argmax(np.ptp(soma_v_all, axis=1))
        gid = gids[gid_idx]
    else:
        gid = example_L5_gid

    soma_v = cell_response.vsec[trial_idx][gid]["soma"]
    tuft_v = cell_response.vsec[trial_idx][gid]["apical_tuft"]
    ca_cell = cell_response.ca[trial_idx][gid][compartment]

    axes[4].plot(
        times,
        soma_v,
        color="blue",
        label="Voltage (soma)",
    )
    axes[4].plot(
        times,
        tuft_v,
        color="green",
        label="Voltage (apical tuft)",
    )
    axes[4].set_ylabel("Membrane Voltage (mV)")

    ax_ca = axes[4].twinx()
    ax_ca.plot(
        times,
        ca_cell,
        color="red",
        label="Calcium concentration",
    )
    ax_ca.set_ylabel("Ca concentration (mM)")

    handles, labels = axes[4].get_legend_handles_labels()
    ca_handles, ca_labels = ax_ca.get_legend_handles_labels()
    axes[4].legend(handles + ca_handles, labels + ca_labels)

    axes[4].set_title(f"Example L5 Pyramidal Cell; Cell GID: {gid}")

    axes[4].set_xlim(axes[1].get_xlim())
    axes[4].set_xticks(axes[1].get_xticks())
    axes[4].set_xlabel("Time (ms)")

    # --------------------------------------------------
    # magnesium opening probability and apical tuft nmda current
    # --------------------------------------------------
    apical_tuft_v = np.asarray(
        [cell_response.vsec[trial_idx][gid]["apical_tuft"] for gid in gids]
    )
    apical_tuft_nmda = np.asarray(
        [
            cell_response.isec[trial_idx][gid]["apical_tuft"]["apical_tuft_nmda"]
            for gid in gids
        ]
    )

    apical_tuft_mg = 1 / (1 + np.exp(-0.072 * apical_tuft_v) / 3.57)

    mean_mg_opening = apical_tuft_mg.mean(axis=0)
    std_mg_opening = apical_tuft_mg.std(axis=0)

    mean_nmda = apical_tuft_nmda.mean(axis=0)
    std_nmda = apical_tuft_nmda.std(axis=0)

    if not export_eps:
        std_color = "blue"
        std_alpha = 0.1
    else:
        std_color = "lightblue"
        std_alpha = 1

    axes[5].plot(
        times,
        mean_mg_opening,
        color="blue",
        label="Mg block opening probability",
    )
    axes[5].fill_between(
        times,
        mean_mg_opening - std_mg_opening,
        mean_mg_opening + std_mg_opening,
        color=std_color,
        alpha=std_alpha,
    )
    axes[5].set_ylabel("Opening probability")

    if not export_eps:
        std_color = "red"
        std_alpha = 0.1
    else:
        std_color = "mistyrose"
        std_alpha = 1

    ax_nmda = axes[5].twinx()
    ax_nmda.plot(
        times,
        mean_nmda,
        color="red",
        label="NMDA current",
    )
    ax_nmda.fill_between(
        times,
        mean_nmda - std_nmda,
        mean_nmda + std_nmda,
        color=std_color,
        alpha=std_alpha,
    )
    ax_nmda.set_ylabel("NMDA current")

    handles, labels = axes[5].get_legend_handles_labels()
    nmda_handles, nmda_labels = ax_nmda.get_legend_handles_labels()

    if not export_eps:
        axes[5].legend(
            handles + nmda_handles,
            labels + nmda_labels,
        )
    else:
        # fix for transparency in .eps exports
        # ------------------------------
        legend = axes[5].legend(
            handles + nmda_handles,
            labels + nmda_labels,
            loc="best",
            framealpha=1,
        )
        axes[5].set_zorder(ax_nmda.get_zorder() + 1)
        axes[5].patch.set_visible(False)

    axes[5].set_title(
        "Magnesium Block Opening Probability\nand Apical Tuft NMDA Current"
    )
    axes[5].set_xlim(axes[1].get_xlim())
    axes[5].set_xticks(axes[1].get_xticks())
    axes[5].set_xlabel("Time (ms)")

    # --------------------------------------------------
    # formatting
    # --------------------------------------------------
    for ax in axes[:5]:
        ax.tick_params(axis="x", labelbottom=False)
        ax.set_xlabel("")

    axes[5].set_xlabel("Time (ms)")

    # font scaling
    for ax in fig.axes:
        ax.title.set_fontsize(16 * font_scale)
        ax.xaxis.label.set_size(14 * font_scale)
        ax.yaxis.label.set_size(14 * font_scale)

        ax.tick_params(
            axis="both",
            labelsize=12 * font_scale,
        )

        legend = ax.get_legend()
        if legend is not None:
            for text in legend.get_texts():
                text.set_fontsize(12 * font_scale)

            legend.get_title().set_fontsize(12 * font_scale)

        for text in ax.texts:
            text.set_fontsize(12 * font_scale)

    if title is not None:
        fig._suptitle.set_fontsize(18 * font_scale)

    for ax, label in zip(axes, "abcdef"):
        ax.text(
            -0.12,
            1.05,
            f"({label})",
            transform=ax.transAxes,
            fontsize=13 * font_scale,
            fontweight="bold",
            va="bottom",
            ha="right",
        )

    if show:
        plt.show()

    return fig, axes


def get_sims_dict():
    plot_simulations = {
        "BETA44 simulation": beta44_sim_dpl_processed,
    }
    for id in beta44_tracker.keys():
        plot_simulations[f"BETA44 simulation {id}"] = beta44_tracker[id][
            "dpl_processed"
        ]
    return plot_simulations


# %% [code]
# ==================================================
# SIMULATION 01 (BASE)
# ==================================================


def simulation_01_base():
    return


net_00 = deepcopy(base_net)

# check external_drives keys
net_00.external_drives.keys()

tstop = 200
with MPIBackend(n_procs=n_procs):
    base_dipole = simulate_dipole(
        net_00,
        tstop=tstop,
        dt=dt,
        n_trials=1,
        baseline_correction=True,
        record_ca="all",
        record_isec="all",
        record_vsec="all",
    )


figure_eeg_data = deepcopy(agg_dpl)

if figure_eeg_data.times[-1] > tstop:
    keep = figure_eeg_data.times <= tstop
    figure_eeg_data.times = figure_eeg_data.times[keep]

    for key in figure_eeg_data.data:
        figure_eeg_data.data[key] = figure_eeg_data.data[key][keep]

# %% [code]
# plot the simulation
# --------------------------------------------------

# process simulation dipole
dpl_00_raw = base_dipole[0].copy()

dpl_00_processed = dpl_00_raw.copy()
dpl_00_processed.smooth(20)
dpl_00_processed.scale(0.8)


plot_hist_sim_data(
    net=net_00,
    data_dict={
        "EEG data": figure_eeg_data,
    },
    sim_dict={
        "HNN Simulation": dpl_00_processed,
    },
    colors={
        "EEG data": "blue",
        "HNN Simulation": "black",
    },
)

# %% [code]

fig, axes = plot_calcium_fig(
    net=net_00,
    title="Simulation: `agg_80_optim_Final`",
    dpls=dpl_00_processed,
    trial_idx=0,
    cell_type="L5_pyramidal",
    compartment="apical_tuft",
)

# %% [code]

fig, axes = plot_grant_figure(
    net=net_00,
    title="",
    trial_idx=0,
    cell_type="L5_pyramidal",
    compartment="apical_tuft",
    sim_dpls=dpl_00_processed,
    data_dict={
        "EEG data": figure_eeg_data,
    },
    sim_dict={
        "HNN Simulation": dpl_00_processed,
    },
    colors={
        "EEG data": "blue",
        "HNN Simulation": "black",
    },
    example_L5_gid=215,
)

fig.savefig(
    analysis_dir / "grant_figure.eps",
    format="eps",
    bbox_inches="tight",
    facecolor="white",
    edgecolor="white",
    dpi=300,
)

fig.savefig(
    analysis_dir / "grant_figure.png",
    format="png",
    bbox_inches="tight",
)

fig.savefig(
    analysis_dir / "grant_figure.svg",
    format="svg",
    dpi=300,
    bbox_inches="tight",
)

# %%
# ==================================================
# TMS PULSE SIMULATIONS
# ==================================================

# --------------------------------------------------
# NOTES:
# - starting from the base network (optimized to the
#   average across all subjects), add 9 more pulses
#   spaced 50ms apart
# --------------------------------------------------

def add_tms_pulse(
    net,
    starting_drives_list,
    shift_interval_ms,
    iterations,
):

    for index in range(0, iterations):
        iteration = index + 1
        # print(iteration)
        absolute_shift = shift_interval_ms * iteration

        for drive_id in starting_drives_list:
            drive = net.external_drives[drive_id]
            # print(f"{drive_id}_{absolute_shift}")

            shifted_mu = drive["dynamics"]["mu"] + absolute_shift

            if "ngfc" in drive_id:
                net.add_ngfc_drive(
                    name=f"{drive_id}_{absolute_shift}",
                    mu=shifted_mu,
                    sigma=drive["dynamics"]["sigma"],
                    numspikes=drive["dynamics"]["numspikes"],
                    weights_gabab=drive["weights_gabab"],
                    weights_gabaa=drive["weights_gabaa"],
                    synapse_type=drive["synapse_type"],
                    n_drive_cells=drive["n_drive_cells"],
                    cell_specific=drive["cell_specific"],
                    # space_constant=drive["space_constant"],
                    synaptic_delays=drive["synaptic_delays"],
                    probability=drive["probability"],
                    event_seed=drive["event_seed"],
                    conn_seed=drive["conn_seed"],
                )

                net.add_connection(
                    src_gids=f"{drive_id}_{absolute_shift}",
                    target_gids="L2_inhibitory",
                    loc="soma",
                    receptor="gabaa",
                    weight=0.0018,
                    delay=0.1,
                    lamtha=30,
                )
                net.add_connection(
                    src_gids=f"{drive_id}_{absolute_shift}",
                    target_gids="L2_inhibitory",
                    loc="soma",
                    receptor="gabab",
                    weight=2.2,
                    delay=0.1,
                    lamtha=30,
                )

            else:
                if drive["cell_specific"] is True:
                    n_drive_cells = "n_cells"
                else:
                    n_drive_cells = drive["n_drive_cells"]

                net.add_evoked_drive(
                    name=f"{drive_id}_{absolute_shift}",
                    mu=shifted_mu,
                    sigma=drive["dynamics"]["sigma"],
                    numspikes=drive["dynamics"]["numspikes"],
                    location=drive["location"],
                    cell_specific=drive["cell_specific"],
                    n_drive_cells=n_drive_cells,
                    weights_ampa=drive["weights_ampa"],
                    weights_nmda=drive["weights_nmda"],
                    synaptic_delays=drive["synaptic_delays"],
                    probability=drive["probability"],
                    event_seed=drive["event_seed"],
                    conn_seed=drive["conn_seed"],
                )

    return


# %%

net_tms_pulses = deepcopy(base_net)
shift_interval = 50
iterations = 9

add_tms_pulse(
    net=net_tms_pulses,
    starting_drives=net_tms_pulses.external_drives,
    shift_interval_ms=shift_interval,
    iterations=iterations,
)

# %% [code]

net = net_tms_pulses
tms_sim_tstop = 750

with MPIBackend(n_procs=n_procs):
    sim_dipole = simulate_dipole(
        net=net,
        tstop=tms_sim_tstop,
        dt=dt,
        n_trials=1,
        baseline_correction=True,
        record_ca="all",
    )

# %% [code]

# process simulation dipole
tms_sim_dpl_raw = sim_dipole[0].copy()

tms_sim_dpl_processed = tms_sim_dpl_raw.copy()
tms_sim_dpl_processed.smooth(20)
tms_sim_dpl_processed.scale(1)

net = net_tms_pulses

plot_hist_sim_data(
    net=net,
    data_dict={
        # "EEG data": figure_eeg_data,
    },
    sim_dict={
        "HNN Simulation": tms_sim_dpl_processed,
    },
    colors={
        "EEG data": "blue",
        "HNN Simulation": "black",
    },
)

# %% [code]

fig, axes = plot_calcium_fig(
    net=net,
    title="",
    dpls=tms_sim_dpl_processed,
    trial_idx=0,
    cell_type="L5_pyramidal",
    compartment="apical_tuft",
    figsize=(12, 10),
    inset_colorbar=True,
    font_scale=1.2,
    tick_interval=100,
)

# %% [code]
# --------------------------------------------------
# NOTES:
# - add 4 second pause after initial 10 pulses
# - add 10 more pulses after the 4-second pause
#
# WARNING:
# - the `net_tms_pulses_01` simulation below failed
#   on my machine during the `py_gather` step
#   (ParallelContext[0]::py_gather), suggesting the
#   calcium concentration output array is too large
# --------------------------------------------------

add_second_pulse_train = False

if add_second_pulse_train:
    net_tms_pulses_01 = deepcopy(net_tms_pulses)

    net = net_tms_pulses_01

    shift_interval = 4000
    iterations = 1
    starting_drives = [
        "ngfc_450",
        "evprox_1_450",
        "evdist_1_450",
        "evdist_2_450",
        "evprox_2_450",
        "evdist_3_450",
    ]

    add_tms_pulse(
        net=net,
        starting_drives_list=starting_drives,
        shift_interval_ms=shift_interval,
        iterations=iterations,
    )

    shift_interval = 50
    iterations = 9
    starting_drives = [
        "ngfc_450_4000",
        "evprox_1_450_4000",
        "evdist_1_450_4000",
        "evdist_2_450_4000",
        "evprox_2_450_4000",
        "evdist_3_450_4000",
    ]

    add_tms_pulse(
        net=net,
        starting_drives_list=starting_drives,
        shift_interval_ms=shift_interval,
        iterations=iterations,
    )

# %% [code]

if add_second_pulse_train:
    net = net_tms_pulses_01
    tms_sim_tstop = 5200

    with MPIBackend(n_procs=n_procs):
        sim_dipole = simulate_dipole(
            net=net,
            tstop=tms_sim_tstop,
            dt=dt,
            n_trials=1,
            baseline_correction=True,
            record_ca="all",
        )

# %% [code]
# ==================================================
# HAND TUNING
# ==================================================


def hand_tuning():
    return


# --------------------------------------------------
# Add an additional proximal drive ~40ms
# --------------------------------------------------

net_01 = deepcopy(base_net)

# pointers to base evprox_1 and evprox_2 for comparison
base_prox_1 = deepcopy(net_01.external_drives["evprox_1"])
base_prox_2 = deepcopy(net_01.external_drives["evprox_2"])

base_dist_1 = deepcopy(net_01.external_drives["evdist_1"])
base_dist_2 = deepcopy(net_01.external_drives["evdist_2"])
base_dist_3 = deepcopy(net_01.external_drives["evdist_3"])
base_ngfc = deepcopy(net_01.external_drives["ngfc"])

compare_drive_params(
    ("prox1", base_prox_1),
    ("prox2", base_prox_2),
    ("dist1", base_dist_1),
    ("dist2", base_dist_2),
    ("dist3", base_dist_3),
    # ("ngfc", base_ngfc),
)

# using `evprox2` params as a starting point, add a third proximal drive

mean_arrival = 15
std_dev = 3
num_spikes = 2
cell_specific = True
drive_name = "evprox_3"
drive_loc = "proximal"

base_p2_sigma = base_prox_2["dynamics"]["sigma"]
base_p2_ampa = base_prox_2["weights_ampa"]
base_p2_nmda = base_prox_2["weights_nmda"]
base_p2_delays = base_prox_2["synaptic_delays"]

# adjust conductances
exc_ampa_gain_multiplier = 100
exc_nmda_gain_multiplier = 100

print(
    "Conductances before:",
    f"    L2 ampa: {base_p2_ampa['L2_pyramidal']}",
    f"    L5 ampa: {base_p2_ampa['L5_pyramidal']}",
    f"    L2 nmda: {base_p2_nmda['L2_pyramidal']}",
    f"    L5 nmda: {base_p2_nmda['L5_pyramidal']}",
    sep="\n",
)

base_p2_ampa["L2_pyramidal"] *= exc_ampa_gain_multiplier
base_p2_ampa["L5_pyramidal"] *= exc_ampa_gain_multiplier
base_p2_nmda["L2_pyramidal"] *= exc_nmda_gain_multiplier
base_p2_nmda["L5_pyramidal"] *= exc_nmda_gain_multiplier

print(
    "Conductances after:",
    f"    L2 ampa: {base_p2_ampa['L2_pyramidal']}",
    f"    L5 ampa: {base_p2_ampa['L5_pyramidal']}",
    f"    L2 nmda: {base_p2_nmda['L2_pyramidal']}",
    f"    L5 nmda: {base_p2_nmda['L5_pyramidal']}",
    sep="\n",
)

# %% [code]

if drive_name in net_01.external_drives.keys():
    net_01.external_drives.pop(drive_name)

net_01.add_evoked_drive(
    drive_name,
    mu=mean_arrival,
    sigma=std_dev,
    numspikes=num_spikes,
    cell_specific=cell_specific,
    weights_ampa=base_p2_ampa,
    weights_nmda=base_p2_nmda,
    location=drive_loc,
    synaptic_delays=base_p2_delays,
)

# %% [code]

tstop = 300

with MPIBackend(n_procs=n_procs):
    net_01_dipole = simulate_dipole(
        net_01,
        tstop=tstop,
        dt=dt,
        n_trials=1,
        baseline_correction=True,
        record_ca="all",
    )

set_vmax = 0.0006
set_vmax = 0.0003

dpl_net_01_raw = net_01_dipole[0].copy()
dpl_net_01_processed = dpl_net_01_raw.copy()
dpl_net_01_processed.smooth(20)
dpl_net_01_processed.scale(1)

plot_hist_sim_data(
    net=net_01,
    data_dict={
        "beta10_experimental_data": beta10_dpl,
    },
    sim_dict={
        "agg_80_optim_Final (smooth: 20, scale: 1)": dpl_00_processed,
        "additional_prox_drive (smooth: 20, scale: 1)": dpl_net_01_processed,
        # "agg_80_optim_dsd_01 (raw)": dpl_net_01_raw
    },
    colors={
        "beta10_experimental_data": "blue",
        "agg_80_optim_Final (smooth: 20, scale: 1)": "black",
        "additional_prox_drive (smooth: 20, scale: 1)": "red",
    },
)

fig, axes = plot_calcium_fig(
    net=net_01,
    title="Simulation: `additional_prox_drive`",
    dpls=dpl_net_01_processed,
    trial_idx=0,
    vmax=set_vmax,
    cell_type="L5_pyramidal",
    compartment="apical_tuft",
)

fig, ax = plot_calcium(
    net=net_01,
    dpls=dpl_net_01_processed,
    trial_idx=0,
    vmax=set_vmax,
    cell_type="L5_pyramidal",
    compartment="apical_tuft",
    print_range=True,
)

# %% [code]
# --------------------------------------------------
# BETA44 HAND TUNING SETUP
# --------------------------------------------------

if "beta44_tracker" not in locals():
    beta44_tracker = {}


# load network
# ------------------------------
subject_id = "beta44"

beta44_netfile = "optimized_beta44_03.json"
beta44_netdir = analysis_dir / "optimization_results" / subject_id / beta44_netfile

beta44_base_net = hnn_io.read_network_configuration(beta44_netdir)

# simulate network
# ------------------------------
tstop = 300

with MPIBackend(n_procs=n_procs):
    beta44_sim_dpl = simulate_dipole(
        beta44_base_net,
        tstop=tstop,
        dt=dt,
        n_trials=1,
        baseline_correction=True,
        record_ca="all",
        record_isec="all",
        record_vsec="all",
    )

# plot output
# ------------------------------
# process simulation dipole
beta44_sim_dpl_raw = deepcopy(beta44_sim_dpl[0])

beta44_sim_dpl_processed = beta44_sim_dpl_raw.copy()
beta44_sim_dpl_processed.smooth(20)

# plot simulation
plot_hist_sim_data(
    net=beta44_base_net,
    data_dict={
        "BETA44 experimental data": beta44_dpl,
    },
    sim_dict={
        "BETA44 simulation": beta44_sim_dpl_processed,
    },
    colors={
        "BETA44 experimental data": "blue",
        "BETA44 simulation": "black",
    },
)


# %% [code]
# --------------------------------------------------
# BETA44 HAND TUNING EXPERIMENT 01
# --------------------------------------------------
# Experiment:
#    -> adjust L2_inhibitory -> L5_pyramidal (distal)
#    -> adjust L5_inhibitory -> L5_pyramidal (soma)

exp_id = "01"

# copy subject's base network for hand tuning
beta44_net_01 = deepcopy(beta44_base_net)

# change 01: zero out the local
# L2 inh -> L5 pyr weights
# ------------------------------
conn_indices = pick_connection(
    net=beta44_net_01,
    src_gids="L2_inhibitory",
    target_gids="L5_pyramidal",
)

gabaa_indices = [
    idx
    for idx in conn_indices
    if beta44_net_01.connectivity[idx]["receptor"] == "gabaa_slow"
]

gabab_indices = [
    idx
    for idx in conn_indices
    if beta44_net_01.connectivity[idx]["receptor"] == "gabab"
]

gabaa_weight = beta44_net_01.connectivity[gabaa_indices[0]]["nc_dict"]["A_weight"]
gabab_weight = beta44_net_01.connectivity[gabab_indices[0]]["nc_dict"]["A_weight"]

print(
    f"GABA-A {gabaa_weight}\n",
    f"GABA-B {gabab_weight}",
    sep="",
)

# beta44_net_01.connectivity[gabaa_indices[0]]["nc_dict"]["A_weight"] = 0
# beta44_net_01.connectivity[gabab_indices[0]]["nc_dict"]["A_weight"] = 0

gabaa_weight = beta44_net_01.connectivity[gabaa_indices[0]]["nc_dict"]["A_weight"]
gabab_weight = beta44_net_01.connectivity[gabab_indices[0]]["nc_dict"]["A_weight"]

print(
    "Updated weights\n",
    f"GABA-A {gabaa_weight}\n",
    f"GABA-B {gabab_weight}",
    sep="",
)

# Change 02: increase the local
# L5 inh -> L5 pyr weights
# ------------------------------

conn_indices = pick_connection(
    net=beta44_net_01,
    src_gids="L5_inhibitory",
    target_gids="L5_pyramidal",
)

gabaa_indices = [
    idx
    for idx in conn_indices
    if beta44_net_01.connectivity[idx]["receptor"] == "gabaa"
]

gabab_indices = [
    idx
    for idx in conn_indices
    if beta44_net_01.connectivity[idx]["receptor"] == "gabab"
]

gabaa_weight = beta44_net_01.connectivity[gabaa_indices[0]]["nc_dict"]["A_weight"]
gabab_weight = beta44_net_01.connectivity[gabab_indices[0]]["nc_dict"]["A_weight"]

print(
    f"GABA-A {gabaa_weight}\n",
    f"GABA-B {gabab_weight}",
    sep="",
)

beta44_net_01.connectivity[gabaa_indices[0]]["nc_dict"]["A_weight"] = 5
beta44_net_01.connectivity[gabab_indices[0]]["nc_dict"]["A_weight"] = 5

gabaa_weight = beta44_net_01.connectivity[gabaa_indices[0]]["nc_dict"]["A_weight"]
gabab_weight = beta44_net_01.connectivity[gabab_indices[0]]["nc_dict"]["A_weight"]

print(
    "Updated weights\n",
    f"GABA-A {gabaa_weight}\n",
    f"GABA-B {gabab_weight}",
    sep="",
)

# %% [code]

# run simulation and visualize results
# ------------------------------
tstop = 300

beta44_tracker[exp_id] = {}
beta44_tracker[exp_id]["net"] = beta44_net_01

with MPIBackend(n_procs=n_procs):
    sim_dipole = simulate_dipole(
        beta44_net_01,
        tstop=tstop,
        dt=dt,
        n_trials=1,
        baseline_correction=True,
        # record_ca="all",
        # record_isec="all",
        # record_vsec="all",
    )

# process simulation dipole
beta44_tracker[exp_id]["dpl_raw"] = deepcopy(sim_dipole[0])

dpl_processed = beta44_tracker[exp_id]["dpl_raw"].copy()
dpl_processed.smooth(20)

beta44_tracker[exp_id]["dpl_processed"] = deepcopy(dpl_processed)

# plot simulation
plot_hist_sim_data(
    net=beta44_net_01,
    data_dict={
        "BETA44 experimental data": beta44_dpl,
    },
    sim_dict={
        "BETA44 simulation": beta44_sim_dpl_processed,
        f"BETA44 simulation {exp_id}": beta44_tracker[exp_id]["dpl_processed"],
    },
    colors={
        "BETA44 experimental data": "blue",
        "BETA44 simulation": "red",
        f"BETA44 simulation {exp_id}": "black",
    },
)
# %%
# --------------------------------------------------
# BETA44 HAND TUNING EXPERIMENT 01_02
# --------------------------------------------------
# Experiment
#    -> Continuing from 01, update evprox2

exp_id = "01_02"

# copy experiment 01 network for hand tuning
beta44_net_01_02 = deepcopy(beta44_net_01)

evprox2 = beta44_net_01_02.external_drives["evprox_2"]

evprox2_mu = evprox2["dynamics"]["mu"]
evprox2_numspikes = evprox2["dynamics"]["numspikes"]
print(
    evprox2_mu,
    evprox2_numspikes,
)

beta44_net_01_02.external_drives["evprox_2"]["dynamics"]["mu"] = 70
beta44_net_01_02.external_drives["evprox_2"]["dynamics"]["numspikes"] = 3

evprox2_mu = evprox2["dynamics"]["mu"]
evprox2_numspikes = evprox2["dynamics"]["numspikes"]
print(
    evprox2_mu,
    evprox2_numspikes,
)

# %%

beta44_tracker[exp_id] = {}
beta44_tracker[exp_id]["net"] = beta44_net_01_02

tstop = 300
net = beta44_tracker[exp_id]["net"]

with MPIBackend(n_procs=n_procs):
    sim_dipole = simulate_dipole(
        net=net,
        tstop=tstop,
        dt=dt,
        n_trials=1,
        baseline_correction=True,
        # record_ca="all",
        # record_isec="all",
        # record_vsec="all",
    )

# %%

# process simulation dipole
beta44_tracker[exp_id]["dpl_raw"] = deepcopy(sim_dipole[0])

dpl_processed = beta44_tracker[exp_id]["dpl_raw"].copy()
dpl_processed.smooth(20)

beta44_tracker[exp_id]["dpl_processed"] = deepcopy(dpl_processed)

# plot simulation
plot_hist_sim_data(
    net=net,
    data_dict={
        "BETA44 experimental data": beta44_dpl,
    },
    sim_dict={
        "BETA44 simulation": beta44_sim_dpl_processed,
        "BETA44 simulation 01": beta44_tracker["01"]["dpl_processed"],
        "BETA44 simulation 01_02": beta44_tracker["01_02"]["dpl_processed"],
    },
    colors={
        "BETA44 experimental data": "blue",
        "BETA44 simulation": "red",
        "BETA44 simulation 01": "black",
        "BETA44 simulation 01_02": "green",
    },
)

# %%
# --------------------------------------------------
# BETA44 HAND TUNING EXPERIMENT 01_03
# --------------------------------------------------
# Experiment
#    -> Continuing from 01, update evprox2

exp_id = "01_03"

# copy experiment 01 network for hand tuning
experiment_net = deepcopy(beta44_tracker["01"]["net"])

evprox2 = experiment_net.external_drives["evprox_2"]

evprox2_mu = evprox2["dynamics"]["mu"]
evprox2_numspikes = evprox2["dynamics"]["numspikes"]
print(
    evprox2_mu,
    evprox2_numspikes,
)

experiment_net.external_drives["evprox_2"]["dynamics"]["mu"] = 50
experiment_net.external_drives["evprox_2"]["dynamics"]["numspikes"] = 3

evprox2_mu = evprox2["dynamics"]["mu"]
evprox2_numspikes = evprox2["dynamics"]["numspikes"]
print(
    evprox2_mu,
    evprox2_numspikes,
)

# %% [code]

beta44_tracker[exp_id] = {}

beta44_tracker[exp_id]["net"] = experiment_net

tstop = 300
net = beta44_tracker[exp_id]["net"]

with MPIBackend(n_procs=n_procs):
    sim_dipole = simulate_dipole(
        net=net,
        tstop=tstop,
        dt=dt,
        n_trials=1,
        baseline_correction=True,
        # record_ca="all",
        # record_isec="all",
        # record_vsec="all",
    )

# %%

# process simulation dipole
beta44_tracker[exp_id]["dpl_raw"] = deepcopy(sim_dipole[0])

dpl_processed = beta44_tracker[exp_id]["dpl_raw"].copy()
dpl_processed.smooth(20)

beta44_tracker[exp_id]["dpl_processed"] = deepcopy(dpl_processed)

# plot simulation
plot_hist_sim_data(
    net=net,
    data_dict={
        "BETA44 experimental data": beta44_dpl,
    },
    sim_dict={
        "BETA44 simulation": beta44_sim_dpl_processed,
        "BETA44 simulation 01": beta44_tracker["01"]["dpl_processed"],
        "BETA44 simulation 01_02": beta44_tracker["01_02"]["dpl_processed"],
        f"BETA44 simulation {exp_id}": beta44_tracker[exp_id]["dpl_processed"],
    },
    colors={
        "BETA44 experimental data": "blue",
        "BETA44 simulation": "red",
        "BETA44 simulation 01": "black",
        "BETA44 simulation 01_02": "green",
    },
)

# %%
# --------------------------------------------------
# BETA44 HAND TUNING EXPERIMENT 01_04
# --------------------------------------------------
# Experiment
#    ->

exp_id = "01_04"

# copy base network for hand tuning
experiment_net = deepcopy(beta44_base_net)


# adjust local network connections
# ----------

# change 01: zero out the local
# L2 inh -> L5 pyr weights
# ------------------------------
conn_indices = pick_connection(
    net=experiment_net,
    src_gids="L2_inhibitory",
    target_gids="L5_pyramidal",
)

gabaa_indices = [
    idx
    for idx in conn_indices
    if experiment_net.connectivity[idx]["receptor"] == "gabaa_slow"
]

gabab_indices = [
    idx
    for idx in conn_indices
    if experiment_net.connectivity[idx]["receptor"] == "gabab"
]

gabaa_weight = experiment_net.connectivity[gabaa_indices[0]]["nc_dict"]["A_weight"]
gabab_weight = experiment_net.connectivity[gabab_indices[0]]["nc_dict"]["A_weight"]

print(
    f"GABA-A {gabaa_weight}\n",
    f"GABA-B {gabab_weight}",
    sep="",
)

divisor = 3

experiment_net.connectivity[gabaa_indices[0]]["nc_dict"]["A_weight"] = (
    gabaa_weight / divisor
)
experiment_net.connectivity[gabab_indices[0]]["nc_dict"]["A_weight"] = (
    gabab_weight / divisor
)

gabaa_weight = experiment_net.connectivity[gabaa_indices[0]]["nc_dict"]["A_weight"]
gabab_weight = experiment_net.connectivity[gabab_indices[0]]["nc_dict"]["A_weight"]

print(
    "Updated weights\n",
    f"GABA-A {gabaa_weight}\n",
    f"GABA-B {gabab_weight}",
    sep="",
)

# %%

# Change 02: increase the local
# L5 inh -> L5 pyr weights
# ------------------------------

conn_indices = pick_connection(
    net=experiment_net,
    src_gids="L5_inhibitory",
    target_gids="L5_pyramidal",
)

gabaa_indices = [
    idx
    for idx in conn_indices
    if experiment_net.connectivity[idx]["receptor"] == "gabaa"
]

gabab_indices = [
    idx
    for idx in conn_indices
    if experiment_net.connectivity[idx]["receptor"] == "gabab"
]

gabaa_weight = experiment_net.connectivity[gabaa_indices[0]]["nc_dict"]["A_weight"]
gabab_weight = experiment_net.connectivity[gabab_indices[0]]["nc_dict"]["A_weight"]

print(
    f"GABA-A {gabaa_weight}\n",
    f"GABA-B {gabab_weight}",
    sep="",
)

experiment_net.connectivity[gabaa_indices[0]]["nc_dict"]["A_weight"] = 0.5
experiment_net.connectivity[gabab_indices[0]]["nc_dict"]["A_weight"] = 0.5

gabaa_weight = experiment_net.connectivity[gabaa_indices[0]]["nc_dict"]["A_weight"]
gabab_weight = experiment_net.connectivity[gabab_indices[0]]["nc_dict"]["A_weight"]

print(
    "Updated weights\n",
    f"GABA-A {gabaa_weight}\n",
    f"GABA-B {gabab_weight}",
    sep="",
)

# %%

# adjust prox2
# ----------

evprox2 = experiment_net.external_drives["evprox_2"]

evprox2_mu = evprox2["dynamics"]["mu"]
evprox2_numspikes = evprox2["dynamics"]["numspikes"]
print(
    evprox2_mu,
    evprox2_numspikes,
)

experiment_net.external_drives["evprox_2"]["dynamics"]["mu"] = 50
experiment_net.external_drives["evprox_2"]["dynamics"]["numspikes"] = 3

evprox2_mu = evprox2["dynamics"]["mu"]
evprox2_numspikes = evprox2["dynamics"]["numspikes"]
print(
    evprox2_mu,
    evprox2_numspikes,
)

# %% [code]

beta44_tracker[exp_id] = {}

beta44_tracker[exp_id]["net"] = experiment_net

tstop = 300
net = beta44_tracker[exp_id]["net"]

with MPIBackend(n_procs=n_procs):
    sim_dipole = simulate_dipole(
        net=net,
        tstop=tstop,
        dt=dt,
        n_trials=1,
        baseline_correction=True,
        # record_ca="all",
        # record_isec="all",
        # record_vsec="all",
    )

# %%

# process simulation dipole
beta44_tracker[exp_id]["dpl_raw"] = deepcopy(sim_dipole[0])

dpl_processed = beta44_tracker[exp_id]["dpl_raw"].copy()
dpl_processed.smooth(20)

beta44_tracker[exp_id]["dpl_processed"] = deepcopy(dpl_processed)

previous_sim_id = list(beta44_tracker.keys())[-2]

plot_simulations = {
    "BETA44 simulation": beta44_sim_dpl_processed,
    f"BETA44 simulation {previous_sim_id}": beta44_tracker[previous_sim_id][
        "dpl_processed"
    ],
    f"BETA44 simulation {exp_id}": beta44_tracker[exp_id]["dpl_processed"],
}

line_colors = [
    "green",
    "black",
    "red",
]

plot_colors = {}

for key in plot_simulations.keys():
    plot_colors[key] = line_colors.pop()

# plot simulation
plot_hist_sim_data(
    net=net,
    data_dict={
        "BETA44 experimental data": beta44_dpl,
    },
    sim_dict=plot_simulations,
    colors=plot_colors,
)


# %%
# --------------------------------------------------
# BETA44 HAND TUNING EXPERIMENT 01_05
# --------------------------------------------------
# Experiment
#    -> adjust timing of evdist_3

exp_id = "01_05"
previous_sim_id = list(beta44_tracker.keys())[-1]

# copy previous network for hand tuning
experiment_net = deepcopy(beta44_tracker[previous_sim_id]["net"])

# adjust evdist_3
# ------------------------------

evdist3 = experiment_net.external_drives["evdist_3"]

evdist3_mu = evdist3["dynamics"]["mu"]
evdist3_numspikes = evdist3["dynamics"]["numspikes"]
print(
    evdist3_mu,
    evdist3_numspikes,
)

experiment_net.external_drives["evdist_3"]["dynamics"]["mu"] = 65
# experiment_net.external_drives["evdist_3"]["dynamics"]["numspikes"] = 2

evdist3_mu = evdist3["dynamics"]["mu"]
evdist3_numspikes = evdist3["dynamics"]["numspikes"]
print(
    evdist3_mu,
    evdist3_numspikes,
)

# %% [code]

beta44_tracker[exp_id] = {}

beta44_tracker[exp_id]["net"] = experiment_net

tstop = 300
net = beta44_tracker[exp_id]["net"]

with MPIBackend(n_procs=n_procs):
    sim_dipole = simulate_dipole(
        net=net,
        tstop=tstop,
        dt=dt,
        n_trials=1,
        baseline_correction=True,
        # record_ca="all",
        # record_isec="all",
        # record_vsec="all",
    )

# %%

# process simulation dipole
beta44_tracker[exp_id]["dpl_raw"] = deepcopy(sim_dipole[0])

dpl_processed = beta44_tracker[exp_id]["dpl_raw"].copy()
dpl_processed.smooth(20)

beta44_tracker[exp_id]["dpl_processed"] = deepcopy(dpl_processed)

# comparison_sim_id = previous_sim_id
comparison_sim_id = "01_03"

plot_simulations = {
    "BETA44 simulation": beta44_sim_dpl_processed,
    f"BETA44 simulation {comparison_sim_id}": beta44_tracker[comparison_sim_id][
        "dpl_processed"
    ],
    f"BETA44 simulation {exp_id}": beta44_tracker[exp_id]["dpl_processed"],
}

line_colors = [
    "green",
    "black",
    "red",
]

plot_colors = {}

for key in plot_simulations.keys():
    plot_colors[key] = line_colors.pop()

# plot simulation
plot_hist_sim_data(
    net=net,
    data_dict={
        "BETA44 experimental data": beta44_dpl,
    },
    sim_dict=plot_simulations,
    colors=plot_colors,
)

# %%
# --------------------------------------------------
# BETA44 HAND TUNING EXPERIMENT 01_06
# --------------------------------------------------
# Experiment
#    -> reduce local weights further

exp_id = "01_06"
previous_sim_id = list(beta44_tracker.keys())[-1]

# copy previous network for hand tuning
experiment_net = deepcopy(beta44_tracker[previous_sim_id]["net"])

# Change 02: increase the local
# L5 inh -> L5 pyr weights
# ------------------------------

conn_indices = pick_connection(
    net=experiment_net,
    src_gids="L5_inhibitory",
    target_gids="L5_pyramidal",
)

gabaa_indices = [
    idx
    for idx in conn_indices
    if experiment_net.connectivity[idx]["receptor"] == "gabaa"
]

gabab_indices = [
    idx
    for idx in conn_indices
    if experiment_net.connectivity[idx]["receptor"] == "gabab"
]

gabaa_weight = experiment_net.connectivity[gabaa_indices[0]]["nc_dict"]["A_weight"]
gabab_weight = experiment_net.connectivity[gabab_indices[0]]["nc_dict"]["A_weight"]

print(
    f"GABA-A {gabaa_weight}\n",
    f"GABA-B {gabab_weight}",
    sep="",
)

experiment_net.connectivity[gabaa_indices[0]]["nc_dict"]["A_weight"] = 0.1
experiment_net.connectivity[gabab_indices[0]]["nc_dict"]["A_weight"] = 0.1

gabaa_weight = experiment_net.connectivity[gabaa_indices[0]]["nc_dict"]["A_weight"]
gabab_weight = experiment_net.connectivity[gabab_indices[0]]["nc_dict"]["A_weight"]

print(
    "Updated weights\n",
    f"GABA-A {gabaa_weight}\n",
    f"GABA-B {gabab_weight}",
    sep="",
)


# %% [code]

beta44_tracker[exp_id] = {}

beta44_tracker[exp_id]["net"] = experiment_net

tstop = 300
net = beta44_tracker[exp_id]["net"]

with MPIBackend(n_procs=n_procs):
    sim_dipole = simulate_dipole(
        net=net,
        tstop=tstop,
        dt=dt,
        n_trials=1,
        baseline_correction=True,
        # record_ca="all",
        # record_isec="all",
        # record_vsec="all",
    )

# %%

# process simulation dipole
beta44_tracker[exp_id]["dpl_raw"] = deepcopy(sim_dipole[0])

dpl_processed = beta44_tracker[exp_id]["dpl_raw"].copy()
dpl_processed.smooth(20)

beta44_tracker[exp_id]["dpl_processed"] = deepcopy(dpl_processed)

# comparison_sim_id = previous_sim_id
comparison_sim_id = "01_03"

plot_simulations = {
    "BETA44 simulation": beta44_sim_dpl_processed,
    f"BETA44 simulation {comparison_sim_id}": beta44_tracker[comparison_sim_id][
        "dpl_processed"
    ],
    f"BETA44 simulation {exp_id}": beta44_tracker[exp_id]["dpl_processed"],
}

line_colors = [
    "green",
    "gray",
    "red",
]

plot_colors = {}

for key in plot_simulations.keys():
    plot_colors[key] = line_colors.pop()

# plot simulation
plot_hist_sim_data(
    net=net,
    data_dict={
        "BETA44 experimental data": beta44_dpl,
    },
    sim_dict=plot_simulations,
    colors=plot_colors,
)


# %%
# --------------------------------------------------
# BETA44 HAND TUNING EXPERIMENT 01_07
# --------------------------------------------------
# Experiment
#    -> reduce local weights further

exp_id = "01_07"
previous_sim_id = list(beta44_tracker.keys())[-1]

# copy previous network for hand tuning
experiment_net = deepcopy(beta44_tracker[previous_sim_id]["net"])

# adjust evdist_3
# ------------------------------

evdist3 = experiment_net.external_drives["evdist_3"]

evdist3_mu = evdist3["dynamics"]["mu"]
evdist3_numspikes = evdist3["dynamics"]["numspikes"]
print(
    evdist3_mu,
    evdist3_numspikes,
)

experiment_net.external_drives["evdist_3"]["dynamics"]["mu"] = 90
# experiment_net.external_drives["evdist_3"]["dynamics"]["numspikes"] = 2

evdist3_mu = evdist3["dynamics"]["mu"]
evdist3_numspikes = evdist3["dynamics"]["numspikes"]
print(
    evdist3_mu,
    evdist3_numspikes,
)


# %% [code]

beta44_tracker[exp_id] = {}

beta44_tracker[exp_id]["net"] = experiment_net

tstop = 300
net = beta44_tracker[exp_id]["net"]

with MPIBackend(n_procs=n_procs):
    sim_dipole = simulate_dipole(
        net=net,
        tstop=tstop,
        dt=dt,
        n_trials=1,
        baseline_correction=True,
        # record_ca="all",
        # record_isec="all",
        # record_vsec="all",
    )

# %%

# process simulation dipole
beta44_tracker[exp_id]["dpl_raw"] = deepcopy(sim_dipole[0])

dpl_processed = beta44_tracker[exp_id]["dpl_raw"].copy()
dpl_processed.smooth(20)

beta44_tracker[exp_id]["dpl_processed"] = deepcopy(dpl_processed)

# comparison_sim_id = previous_sim_id
comparison_sim_id = "01_03"

plot_simulations = {
    "BETA44 simulation": beta44_sim_dpl_processed,
    f"BETA44 simulation {comparison_sim_id}": beta44_tracker[comparison_sim_id][
        "dpl_processed"
    ],
    f"BETA44 simulation {exp_id}": beta44_tracker[exp_id]["dpl_processed"],
}

line_colors = [
    "green",
    "gray",
    "red",
]

plot_colors = {}

for key in plot_simulations.keys():
    plot_colors[key] = line_colors.pop()

# plot simulation
plot_hist_sim_data(
    net=net,
    data_dict={
        "BETA44 experimental data": beta44_dpl,
    },
    sim_dict=plot_simulations,
    colors=plot_colors,
)


# %%
