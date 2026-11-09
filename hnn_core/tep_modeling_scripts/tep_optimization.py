# %% [code]
# ==================================================
# SETUP
# ==================================================

import os
import pickle
import pprint
import time
import warnings
from copy import deepcopy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# from IPython.utils.io import capture_output
from hnn_core import Dipole, MPIBackend, hnn_io, simulate_dipole
from hnn_core.optimization import Optimizer
from hnn_core.viz import plot_dipole

dt = 0.025
n_procs = 8

beta10_datafile = "BETA10_4hnn.txt"
beta04_datafile = "BETA04_100_4hnn.txt"
beta44_datafile = "BETA44_4hnn.txt"

base_network_file = "agg_80_optim_Final.json"

analysis_dir = Path.cwd()

if analysis_dir.name != "tep_modeling_scripts":
    try:
        analysis_dir = next(
            path
            for path in analysis_dir.rglob("tep_modeling_scripts")
            if path.is_dir()
        )
    except StopIteration:
        raise FileNotFoundError(
                "Could not resolve analysis directory"
            )

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

# instantiate base network
base_net = hnn_io.read_network_configuration(
    analysis_dir / "networks" / base_network_file
)

net_00 = deepcopy(base_net)

# check external_drives keys
net_00.external_drives.keys()

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
        spike_types=["evprox", "evdist", "ngfc"],
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
    figsize=(8, 6),
    show=True,
):
    """Plot calcium concentration

    Parames
    ----------
    net : Network
    trial_idx : int
    cell_type : str
    compartment : str
    spike_types : list of str | None
        Spike types to display in the histogram. If None, use the default
    dpls : Dipole object | list of Dipole objects | None
    sort_by_peak : bool
        Whether to sort cells by their peak calcium activation time
    vmin : float
        Min calcium concentration for the color scale
    vmax : float
        Max calcium concentration for the color scale
    tick_interval : int
        Spacing between labeled ticks in ms
    figsize : tuple of float
    show : bool

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : numpy.ndarray

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


def save_optimization(
    optim,
    drive_config,
    subject,
    starting_network,
    n_procs,
    tstop,
    scale_factor,
    smooth_win,
    max_iter,
    n_trials,
    tolfun,
):
    """Save the optimized network and metadata, then return a summary DataFrame."""

    import json
    import math
    from datetime import datetime
    from pathlib import Path

    import numpy as np
    import pandas as pd

    analysis_dir = Path(__file__).resolve().parent

    if analysis_dir.name != "tep_modeling_scripts":
        analysis_dir = analysis_dir / "hnn_core" / "tep_modeling_scripts"

    output_dir = analysis_dir / "optimization_results" / subject
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = output_dir / "metadata.json"

    if metadata_path.exists():
        with metadata_path.open("r", encoding="utf-8") as f:
            subject_metadata = json.load(f)
    else:
        subject_metadata = {
            "subject": subject,
            "optimizations": [],
        }

    starting_network = Path(starting_network).resolve()

    es = optim.es_
    result = es.result
    best_evaluation = int(result.evals_best)
    population_size = es.opts["popsize"]

    optimization_metadata = {
        "starting_network": str(starting_network),
        "drive_config": drive_config,
        "optimization_setup": {
            "solver": "cma",
            "n_procs": n_procs,
            "n_jobs": n_procs,
            "tstop": tstop,
            "scale_factor": scale_factor,
            "smooth_window_len": smooth_win,
            "max_iter": max_iter,
            "n_trials": n_trials,
            "tolfun": tolfun,
        },
        "optimization_summary": {
            "termination_reason": es.stop(),
            "iterations_completed": es.countiter,
            "total_evaluations": es.countevals,
            "best_evaluation": best_evaluation,
            "best_iteration": math.ceil(best_evaluation / population_size),
            "best_objective": float(result.fbest),
        },
    }

    def json_default(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(f"Cannot serialize {type(value).__name__} to JSON")

    def normalize_for_comparison(value):
        return json.dumps(
            value,
            sort_keys=True,
            default=json_default,
        )

    optimizations = subject_metadata["optimizations"]

    if optimizations:
        previous_metadata = {
            key: value
            for key, value in optimizations[-1].items()
            if key not in {"date_time", "iteration"}
        }

        if normalize_for_comparison(optimization_metadata) == normalize_for_comparison(
            previous_metadata
        ):
            print(
                "No new optimization information. "
                "Existing metadata and network were not saved."
            )
            return pd.DataFrame(optimizations)

    iteration = (
        max(
            (record["iteration"] for record in optimizations),
            default=0,
        )
        + 1
    )

    date_time = datetime.now().strftime("%Y-%m-%d %H:%M")

    optimization_metadata["iteration"] = iteration
    optimization_metadata["date_time"] = date_time

    stem = f"optimized_{subject}_{iteration:02d}"
    network_path = output_dir / f"{stem}.json"

    optim.net_.write_configuration(network_path)

    optimizations.append(optimization_metadata)

    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(subject_metadata, f, indent=4, default=json_default)

    metadata_df = pd.DataFrame(optimizations)

    print(f"Subject: {subject}")
    print(f"Iteration: {iteration}")
    print(f"Date/time: {date_time}")
    print(f"Network saved to: {network_path}")
    print(f"Metadata saved to: {metadata_path}")

    return metadata_df


# %% [code]
# ==================================================
# SIMULATION 01 (BASE)
# ==================================================

tstop = 300
with MPIBackend(n_procs=n_procs):
    base_dipole = simulate_dipole(
        net_00,
        tstop=tstop,
        dt=dt,
        n_trials=1,
        baseline_correction=True,
        record_ca="all",
    )

# %% [code]
# plot the simulation
# --------------------------------------------------

# process simulation dipole
dpl_00_raw = base_dipole[0].copy()

dpl_00_processed = dpl_00_raw.copy()
dpl_00_processed.smooth(20)
dpl_00_processed.scale(1)


plot_hist_sim_data(
    net=net_00,
    data_dict={
        "beta10_experimental_data": beta10_dpl,
    },
    sim_dict={
        "agg_80_optim_Final (smooth: 20, scale: 1)": dpl_00_processed,
        "agg_80_optim_Final (raw)": dpl_00_raw,
    },
    colors={
        "beta10_experimental_data": "blue",
        "agg_80_optim_Final (smooth: 20, scale: 1)": "black",
        "agg_80_optim_Final (raw)": "gray",
    },
)

fig, axes = plot_calcium_fig(
    net=net_00,
    title="Simulation: `agg_80_optim_Final`",
    dpls=dpl_00_processed,
    trial_idx=0,
    cell_type="L5_pyramidal",
    compartment="apical_tuft",
)


# %% [code]
# ==================================================
# SIMULATION 02 (ADD PROX DRIVE)
# ==================================================
# add an additional proximal drive ~40ms

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

# %%

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

# [HERE]

tstop = 300
n_procs = 2

with MPIBackend(n_procs=n_procs):
    net_01_dipole = simulate_dipole(
        net_01,
        tstop=tstop,
        dt=dt,
        n_trials=1,
        baseline_correction=True,
        record_ca="all",
    )

# %%
# process simulation dipole

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

# fig, axes = plot_calcium_fig(
#     net=net_01,
#     title="Simulation: `additional_prox_drive`",
#     dpls=dpl_net_01_processed,
#     trial_idx=0,
#     vmax=set_vmax,
#     cell_type="L5_pyramidal",
#     compartment="apical_tuft",
# )

fig, ax = plot_calcium(
    net=net_01,
    dpls=dpl_net_01_processed,
    trial_idx=0,
    vmax=set_vmax,
    cell_type="L5_pyramidal",
    compartment="apical_tuft",
    print_range=True,
)

# %%


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


plot_spikes_comparison(
    {
        "base network": net_00,
        "additional proximal drive": net_01,
    }
)


# %% [code]
# ==================================================
# Optimization
# ==================================================


def make_constraints(base_net, drive_config):
    constraints = {}
    initial_conditions = {}

    for drive_name, config in drive_config.items():
        drive = base_net.external_drives[drive_name]

        for receptor, cell_type_bounds in config.get("weights", {}).items():
            for cell_type, weight_bounds in cell_type_bounds.items():
                weight = drive[f"weights_{receptor}"][cell_type]
                param_name = f"gbar_{drive_name}_{cell_type}_{receptor}"

                initial_conditions[param_name] = weight

                resolved_bounds = []

                for bound in weight_bounds:
                    if isinstance(bound, tuple) and bound[0] == "relative":
                        if weight == 0:
                            warnings.warn(
                                '\n\nCannot obtain a "relative" bound for '
                                f"{param_name}:\n   initial weight is 0, "
                                "therefore the multiplier resolves to 0.",
                                stacklevel=2,
                            )

                        resolved_bounds.append(weight * bound[1])
                    else:
                        resolved_bounds.append(bound)

                constraints[param_name] = tuple(resolved_bounds)

        for param in ("mu", "sigma"):
            initial_value = drive["dynamics"][param]
            param_name = f"{param}_{drive_name}"

            initial_conditions[param_name] = initial_value

            resolved_bounds = []

            for bound in config[f"{param}"]:
                if isinstance(bound, tuple) and bound[0] == "relative":
                    resolved_bounds.append(initial_value * bound[1])
                else:
                    resolved_bounds.append(bound)

            constraints[param_name] = tuple(resolved_bounds)

    return constraints, initial_conditions


drive_config = {
    "evprox_1": {
        "mu": (1.0, 5.0),
        "sigma": (1.0, ("relative", 3)),
        "weights": {
            "ampa": {
                "L2_inhibitory": (0.000, 0.5),
                "L2_pyramidal": (("relative", 0.2), ("relative", 1000)),
                "L5_inhibitory": (0.000, 0.5),
                "L5_pyramidal": (("relative", 0.2), ("relative", 1000)),
            },
            "nmda": {
                "L2_inhibitory": (0.000, 0.5),
                "L2_pyramidal": (("relative", 0.2), ("relative", 1000)),
                "L5_inhibitory": (0.000, 0.5),
                "L5_pyramidal": (("relative", 0.2), ("relative", 1000)),
            },
        },
    },
    "evdist_1": {
        "mu": (0.0, 5.0),
        "sigma": (1.0, ("relative", 3)),
        "weights": {
            "ampa": {
                "L2_inhibitory": (("relative", 0.5), ("relative", 1000)),
                "L2_pyramidal": (("relative", 0.2), ("relative", 1000)),
                "L5_pyramidal": (("relative", 0.2), ("relative", 1000)),
            },
            "nmda": {
                "L2_inhibitory": (("relative", 0.5), ("relative", 1000)),
                "L2_pyramidal": (("relative", 0.2), ("relative", 1000)),
                "L5_pyramidal": (("relative", 0.2), ("relative", 1000)),
            },
        },
    },
    "evprox_2": {
        "mu": (("relative", 0.60), ("relative", 1.40)),
        "sigma": (("relative", 0.33), ("relative", 3)),
        "weights": {
            "ampa": {
                "L2_inhibitory": (0.000, 0.5),
                "L5_inhibitory": (0.000, 0.5),
                "L2_pyramidal": (("relative", 0.2), ("relative", 1000)),
                "L5_pyramidal": (("relative", 0.2), ("relative", 1000)),
            },
            "nmda": {
                "L2_inhibitory": (0.000, 0.5),
                "L5_inhibitory": (0.000, 0.5),
                "L2_pyramidal": (("relative", 0.2), ("relative", 1000)),
                "L5_pyramidal": (("relative", 0.2), ("relative", 1000)),
            },
        },
    },
    "evdist_2": {
        "mu": (("relative", 0.60), ("relative", 1.40)),
        "sigma": (("relative", 0.33), ("relative", 3)),
        "weights": {
            "ampa": {
                "L2_inhibitory": (0.000, 0.5),
                "L2_pyramidal": (("relative", 0.2), ("relative", 1000)),
                "L5_pyramidal": (("relative", 0.2), ("relative", 1000)),
            },
            "nmda": {
                "L2_inhibitory": (0.000, 0.5),
                "L2_pyramidal": (("relative", 0.2), ("relative", 1000)),
                "L5_pyramidal": (("relative", 0.2), ("relative", 1000)),
            },
        },
    },
    "evdist_3": {
        "mu": (("relative", 0.85), ("relative", 1.15)),
        "sigma": (("relative", 0.75), ("relative", 1.5)),
        "weights": {
            "ampa": {
                "L2_inhibitory": (0.000, 0.5),
                "L2_pyramidal": (("relative", 0.2), ("relative", 1000)),
                "L5_pyramidal": (("relative", 0.2), ("relative", 1000)),
            },
            "nmda": {
                "L2_inhibitory": (0.000, 0.5),
                "L2_pyramidal": (("relative", 0.2), ("relative", 1000)),
                "L5_pyramidal": (("relative", 0.2), ("relative", 1000)),
            },
        },
    },
}


def set_params(net, params):
    for drive_name, config in drive_config.items():
        drive = net.external_drives[drive_name]

        drive["dynamics"]["mu"] = params[f"mu_{drive_name}"]
        drive["dynamics"]["sigma"] = params[f"sigma_{drive_name}"]

        for receptor, cell_type_bounds in config.get("weights", {}).items():
            weights = drive[f"weights_{receptor}"]

            for cell_type in cell_type_bounds:
                param_name = f"gbar_{drive_name}_{cell_type}_{receptor}"
                weights[cell_type] = params[param_name]


subject_id = "beta04"

subject_id_datafile_map = {
    "beta10": beta10_dpl,
    "beta04": beta04_dpl,
    "beta44": beta44_dpl,
}

starting_net_file = base_network_file

start_net = hnn_io.read_network_configuration(
    analysis_dir / "networks" / starting_net_file
)

constraints, initial_conditions = make_constraints(
    start_net,
    drive_config,
)

pprint.pprint(constraints)

# %%j [code]

max_iter = 30
tolfun = 1e-8

n_procs = 10

tstop = 150
scale_factor = 1
smooth_win = 20
n_trials = 3

start_time = time.perf_counter()

optim = Optimizer(
    initial_net=start_net,
    tstop=tstop,
    constraints=constraints,
    set_params=set_params,
    initial_params=initial_conditions,
    solver="cma",
)

optim.max_iter = max_iter

subject_data = subject_id_datafile_map[subject_id].copy()

if subject_data.times[-1] > tstop:
    keep = subject_data.times <= tstop
    subject_data.times = subject_data.times[keep]

    for key in subject_data.data:
        subject_data.data[key] = subject_data.data[key][keep]

# %% [code]


stdout_fd = os.dup(1)
stderr_fd = os.dup(2)
devnull_fd = os.open(os.devnull, os.O_WRONLY)

try:
    os.dup2(devnull_fd, 1)
    os.dup2(devnull_fd, 2)

    with MPIBackend(n_procs=n_procs):
        optim.fit(
            target=subject_data,
            n_trials=n_trials,
            baseline_correction=True,
            scale_factor=scale_factor,
            smooth_window_len=smooth_win,
            n_jobs=n_procs,
            max_iter=max_iter,
            tolfun=tolfun,
        )

finally:
    os.dup2(stdout_fd, 1)
    os.dup2(stderr_fd, 2)
    os.close(stdout_fd)
    os.close(stderr_fd)
    os.close(devnull_fd)

elapsed_time = time.perf_counter() - start_time
print(f"Optimization time: {elapsed_time / 60:.2f} minutes")

metadata_df = save_optimization(
    optim=optim,
    drive_config=drive_config,
    subject=subject_id,
    starting_network=starting_net_file,
    n_procs=n_procs,
    tstop=tstop,
    scale_factor=scale_factor,
    smooth_win=smooth_win,
    max_iter=max_iter,
    n_trials=n_trials,
    tolfun=tolfun,
)

# %%

subject_output_dir = Path("optimization_results") / subject_id
subject_output_dir.mkdir(
    parents=True,
    exist_ok=True,
)

iteration = int(metadata_df.iloc[-1]["iteration"])
checkpoint_file = subject_output_dir / f"cma_{iteration:02d}_checkpoint.pkl"

with checkpoint_file.open("wb") as file:
    pickle.dump(
        optim,
        file,
    )

print(f"Saved CMA-ES checkpoint to {checkpoint_file}")

with checkpoint_file.open("rb") as file:
    test_saved_optim = pickle.load(file)

test_saved_optim.plot_convergence()


# %% [code]

optimized_net = deepcopy(optim.net_)

# update_connection = optimized_net.external_drives["evdist_2"]["weights_ampa"][
#     "L5_pyramidal"
# ]
# print(update_connection)
# update_connection = update_connection * 10
# print(update_connection)

with MPIBackend(n_procs=n_procs):
    opt_dpl = simulate_dipole(
        optimized_net,
        tstop=300,
        n_trials=1,
        baseline_correction=True,
        record_ca="all",
        # record_isec="all",
        # record_vsec="all",
    )

# %% [code]

opt_dpl_processed = deepcopy(opt_dpl[0])
opt_dpl_processed.smooth(smooth_win)
opt_dpl_processed.scale(scale_factor)


plot_hist_sim_data(
    net=optimized_net,
    data_dict={
        f"{subject_id}_data": subject_id_datafile_map[subject_id],
    },
    sim_dict={
        "unoptimized_dipole (smooth: 20)": dpl_00_processed,
        "optimized_dipole (smooth: 20)": opt_dpl_processed,
        # "optimized_dipole (raw)": opt_dpl,
    },
    colors={
        "unoptimized_dipole (smooth: 20)": "red",
        "beta10_experimental_data": "blue",
        "optimized_dipole (smooth: 20, scale: 1)": "black",
        # "optimized_dipole (raw)": "gray",
    },
)

# %% [code]


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


bounds_df = get_drive_bounds(
    metadata_df["drive_config"].iloc[len(metadata_df) - 1],
    start_net,
)

bounds_df

# %% [code]


# fig, axes = plot_calcium_fig(
#     net=optim.net_,
#     title="BETA44 Optimized Simulation",
#     dpls=opt_dpl_processed,
#     trial_idx=0,
#     cell_type="L5_pyramidal",
#     compartment="apical_tuft",
# )


# def plot_spiking_custom(net):
#     ax = optim.net_.cell_response.plot_spikes_raster(
#         show=False,
#     )

#     ax.axes[0].get_legend().set_loc("lower right")

#     return


# plot_spiking_custom(optim.net_)

# %% [code]


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

        _ = cell_response.plot_spikes_raster(
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


fig, axes = plot_calcium_spiking_fig(
    net=optimized_net,
    title="BETA44 Optimized Simulation",
    trial_idx=0,
    cell_type="L5_pyramidal",
    compartment="apical_tuft",
    dpls=opt_dpl_processed,
)

# %% [code]

subject_id = "beta10"
loaded_net_file = "optimized_beta10_01.json"

loaded_net = hnn_io.read_network_configuration(
    "optimization_results/beta10/" + loaded_net_file
)

with MPIBackend(n_procs=n_procs):
    opt_dpl = simulate_dipole(
        loaded_net,
        tstop=300,
        n_trials=1,
        baseline_correction=True,
        record_ca="all",
        # record_isec="all",
        # record_vsec="all",
    )

# %% [code]

opt_dpl_processed = deepcopy(opt_dpl[0])
opt_dpl_processed.smooth(smooth_win)
opt_dpl_processed.scale(scale_factor)


plot_hist_sim_data(
    net=optimized_net,
    data_dict={
        f"{subject_id}_data": subject_id_datafile_map[subject_id],
    },
    sim_dict={
        "unoptimized_dipole (smooth: 20)": dpl_00_processed,
        "optimized_dipole (smooth: 20)": opt_dpl_processed,
        # "optimized_dipole (raw)": opt_dpl,
    },
    colors={
        "unoptimized_dipole (smooth: 20)": "red",
        f"{subject_id}_experimental_data": "blue",
        "optimized_dipole (smooth: 20, scale: 1)": "black",
        # "optimized_dipole (raw)": "gray",
    },
)

# %% [code]

fig, axes = plot_calcium_spiking_fig(
    net=loaded_net,
    title=f"{subject_id} Optimized Simulation",
    trial_idx=0,
    cell_type="L5_pyramidal",
    compartment="apical_tuft",
    dpls=opt_dpl_processed,
)

# %% [code]
# Stopping point
