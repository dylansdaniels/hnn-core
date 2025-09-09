# %% ####################
import matplotlib.pyplot as plt
import numpy as np

from hnn_core import (
    JoblibBackend,
    # MPIBackend,
    # calcium_model,
    jones_2009_model,
    simulate_dipole,
)
from hnn_core.network_models import add_erp_drives_to_jones_model
from hnn_core.viz import plot_dipole

stop_short = False

if stop_short:
    tstop = 60
else:
    tstop = 170

# %% ####################
plot_morphology = False
plot_cell_grid = False

net = jones_2009_model()

if plot_cell_grid:
    _ = net.plot_cells()

if plot_morphology:
    for type in net.cell_types.keys():
        print(f"Plotting morphology for {type}")
        net.cell_types[type].plot_morphology()

add_erp_drives_to_jones_model(net)

# %% ####################

n_trials = 1
scaling_factor = 3000

depths = list(range(-325, 2150, 100))
electrode_pos = [(135, 135, dep) for dep in depths]
net.add_electrode_array("shank1", electrode_pos)

with JoblibBackend(8):
    # with MPIBackend(n_procs=8, mpi_cmd='mpiexec'):
    dpls = simulate_dipole(
        net,
        tstop=tstop,
        n_trials=n_trials,
        record_ina="all",
        record_ik="all",
        record_inet="all",
    )

for dpl in dpls:
    dpl.scale(scaling_factor)

# %%

with JoblibBackend(8):
    # with MPIBackend(n_procs=8, mpi_cmd='mpiexec'):
    dpls2 = simulate_dipole(
        net,
        tstop=tstop,
        n_trials=n_trials,
        record_ina="all",
        record_ik="all",
        record_inet="all",
    )

window_len = 30
for dpl in dpls2:
    dpl.smooth(window_len).scale(scaling_factor)

# %% ####################
# EVALUATING CURRENT RECORDING
#########################

# %%
# get first trial only

currents = None
trial_num = 1
all_currents = net.cell_response.ionic_currents

if currents is None:
    currents = {}
    for key in all_currents.keys():
        if len(all_currents[key][trial_num - 1].keys()) > 0:
            currents[key] = all_currents[key][trial_num - 1]

# %% ####################
# GIDs by cell type
cell_gids = {}
cells_of_interest = ["L2_pyramidal", "L5_pyramidal"]

for cell_type, ranges in net.gid_ranges.items():
    if cells_of_interest is None:
        if cell_type in net.cell_types.keys():
            cell_gids[cell_type] = list(ranges)
    else:
        if cell_type in cells_of_interest:
            cell_gids[cell_type] = list(ranges)

print(f"Cell types: {cell_gids.keys()}")

# %% ####################
# get currents by section for each cell

channel_currents = {}

for channel in currents.keys():
    trial = currents[channel]

    cell_currents = {}

    for cell_type, gids in cell_gids.items():
        cell_sections = trial[gids[0]].keys()

        section_currents = {}
        section_currents["agg"] = 0

        for section in cell_sections:
            rows = []
            for gid in gids:
                if section in trial[gid]:
                    rows.append(trial[gid][section])
                else:
                    pass
            section_currents[section] = np.array(rows)

            if type(section_currents["agg"]) != np.ndarray:
                section_currents["agg"] = np.array(rows)
            else:
                section_currents["agg"] += np.array(rows)

        cell_currents[cell_type] = section_currents

    channel_currents[channel] = cell_currents

# %% ####################
cell_currents

# %% ####################

rows = len(cell_currents.keys())

fig, ax = plt.subplots(
    nrows=rows,
    ncols=1,
    figsize=(6, 6 * rows),
)

for i, cell in enumerate(cell_currents.keys()):
    ax[i].plot(
        dpls[0].times,
        np.mean(cell_currents[cell]["agg"], axis=0),
    )

    ax[i].set_xlabel("Time")
    ax[i].set_ylabel("Aggregate Na Currents (Sum Across All Sections)")
    ax[i].set_title(f"{cell}\nTransmembrane Na Currents Averaged Across Cells")

plt.tight_layout()

# %% ####################

fig, ax = plt.subplots(
    nrows=len(cell_currents.keys()),
    ncols=1,
    figsize=(6, 24),
)

for i, cell in enumerate(cell_currents.keys()):
    section_currents = cell_currents[cell]

    for section in section_currents.keys():
        if not section == "agg":
            ax[i].plot(
                dpls[0].times,
                np.mean(section_currents[section], axis=0),
                label=f"{section.replace('_', ' ').capitalize()}",
            )
    ax[i].legend()

    ax[i].set_xlabel("Time")
    ax[i].set_ylabel("Na Currents")
    ax[i].set_title(f"{cell}\nTransmembrane Na Currents Averaged Across Cells")

plt.tight_layout()

# %% ####################

# get min/max for y-axis
# all_y = []
# for channel in channel_currents.values():
#     for cell_data in channel.values():
#         all_y.append(np.mean(cell_data['agg'], axis=0))
# all_y = np.concatenate(all_y)
# ymin, ymax = np.min(all_y), np.max(all_y)

rows = len(cell_currents.keys())

fig, ax = plt.subplots(
    nrows=rows,
    ncols=1,
    figsize=(6, 6 * rows),
)

# generate plots
for channel in channel_currents.keys():
    cell_currents = channel_currents[channel]
    for i, cell in enumerate(cell_currents.keys()):
        ax[i].plot(
            dpls[0].times,
            abs(np.mean(cell_currents[cell]["agg"], axis=0)),
            label=channel,
        )

        ax[i].set_xlabel("Time")
        ax[i].set_ylabel("| Aggregate Currents |")
        ax[i].set_title(f"{cell}\nTransmembrane Currents Averaged Across Cells")

for axis in ax:
    axis.legend()
    # axis.set_ylim(ymin, ymax)

plt.tight_layout()

# %% ####################

rows = len(cell_currents.keys())

fig, ax = plt.subplots(
    nrows=rows,
    ncols=1,
    figsize=(6, 6 * rows),
)

# Define bin width (in seconds)
bin_width = 1.0
times = dpls[0].times
dt = times[1] - times[0]
samples_per_bin = int(bin_width / dt)

# Generate plots with binned data
for channel in channel_currents.keys():
    cell_currents = channel_currents[channel]
    for i, cell in enumerate(cell_currents.keys()):
        y = np.mean(cell_currents[cell]["agg"], axis=0)

        # Reshape for binning (truncate if needed)
        n_bins = len(y) // samples_per_bin
        y_binned = (
            y[: n_bins * samples_per_bin].reshape(n_bins, samples_per_bin).mean(axis=1)
        )
        t_binned = (
            times[: n_bins * samples_per_bin]
            .reshape(n_bins, samples_per_bin)
            .mean(axis=1)
        )

        ax[i].bar(
            t_binned, y_binned, width=bin_width, align="edge", label=channel, alpha=0.7
        )

        ax[i].set_xlabel("Time (s)")
        ax[i].set_ylabel("| Aggregate Currents |")
        ax[i].set_title(f"{cell}\nTransmembrane Currents Averaged Across Cells")

for axis in ax:
    axis.legend()
    # axis.set_ylim(ymin, ymax)

plt.tight_layout()

# %% ####################

bin_width = 1.0
times = dpls[0].times
dt = times[1] - times[0]
samples_per_bin = int(bin_width / dt)

channels = channel_currents.keys()
rows = len(cell_currents.keys())

fig, ax = plt.subplots(
    nrows=rows,
    ncols=2,
    figsize=(12, 6 * rows),
)

# cells = enumerate(next(iter(channel_currents.values())).keys())
cells = channel_currents[list(channels)[0]].keys()

ymax = None
ymin = None

for i, cell in enumerate(cells):
    # initialize array to hold sums
    # we will be summing the inward and outward currents to get the
    # "net" current flow across all channel for each cell type
    y_sum = np.array([])

    for channel in channels:
        # average the "agg" for all GIDs in each cell type
        y = np.mean(
            channel_currents[channel][cell]["agg"],
            axis=0,
        )
        y_sum = y if len(y_sum) == 0 else y_sum + y

    # get num of bins needed, rounding down
    n_bins = len(y_sum) // samples_per_bin

    # slice the data, discarding excess datapoints that don't
    # completely constitute a bin
    usable_data = n_bins * samples_per_bin
    y_sum = y_sum[:usable_data]

    # reshape the data so that each row is a bin and each column is a sample,
    # then average within each bin
    y_sum = y_sum.reshape(
        n_bins,
        samples_per_bin,
    )
    y_binned = y_sum.mean(axis=1)

    # reshape the time data into bins
    x_binned = (
        times[: n_bins * samples_per_bin].reshape(n_bins, samples_per_bin).mean(axis=1)
    )

    # mask for positive values
    # used for labelling / coloring data
    y_pos = y_binned >= 0

    # generate bar plot
    ax[i, 0].bar(
        x_binned[y_pos],
        y_binned[y_pos],
        width=bin_width,
        align="edge",
        alpha=0.7,
        label="inward",
        color="red",
    )

    # generate bar plot
    ax[i, 0].bar(
        x_binned[~y_pos],
        y_binned[~y_pos],
        width=bin_width,
        align="edge",
        alpha=0.7,
        label="outward",
        color="green",
    )

    # set labels
    ax[i, 0].set_xlabel("Time (s)")
    ax[i, 0].set_ylabel(" + ".join(channels))
    ax[i, 0].set_title(f"{cell}\nSum of Transmembrane Currents (Mean Across GIDs)")

    cell_max = max(y_binned)
    cell_min = min(y_binned)

    ymax = max(ymax, cell_max) if ymax is not None else cell_max
    ymin = min(ymin, cell_min) if ymin is not None else cell_min

y_lim = max(
    abs(ymax),
    abs(ymin),
)

for axis in ax[:, 0]:
    axis.set_ylim(
        y_lim * -1.05,
        y_lim * 1.05,
    )

_ = plot_dipole(
    dpls,
    ax=ax[0, 1],
    layer="L2",
    show=False,
)
_ = plot_dipole(dpls2, ax=ax[0, 1], layer="L2", show=False, color="red")

_ = plot_dipole(
    dpls,
    ax=ax[1, 1],
    layer="L5",
    show=False,
)
_ = plot_dipole(dpls2, ax=ax[1, 1], layer="L5", show=False, color="red")

plt.legend()

plt.tight_layout()

# %%

fig, axes = plt.subplots(
    nrows=2,
    ncols=1,
    figsize=(6, 12),
)

_ = plot_dipole(
    dpls,
    ax=axes[0],
    layer="L2",
    show=False,
)
_ = plot_dipole(dpls2, ax=axes[0], layer="L2", show=False, color="red")

_ = plot_dipole(
    dpls,
    ax=axes[1],
    layer="L5",
    show=False,
)
_ = plot_dipole(dpls2, ax=axes[1], layer="L5", show=False, color="red")

# %%

net.cell_response.inet[0]


# %% ####################
# NET TM CURRENT RECORDING
#########################

trial = net.cell_response.inet[0]
cell_currents = {}

for cell_type, gids in cell_gids.items():
    cell_sections = trial[gids[0]].keys()

    section_currents = {}
    section_currents["agg"] = 0

    for section in cell_sections:
        rows = []
        for gid in gids:
            if section in trial[gid]:
                rows.append(trial[gid][section])
            else:
                pass
        section_currents[section] = np.array(rows)

        if type(section_currents["agg"]) != np.ndarray:
            section_currents["agg"] = np.array(rows)
        else:
            section_currents["agg"] += np.array(rows)

    cell_currents[cell_type] = section_currents


# %%

fig, ax = plt.subplots(
    nrows=len(cell_currents.keys()),
    ncols=1,
    figsize=(6, 24),
)

for i, cell in enumerate(cell_currents.keys()):
    ax[i].plot(
        dpls[0].times,
        np.mean(cell_currents[cell]["agg"], axis=0),
    )

    ax[i].set_xlabel("Time")
    ax[i].set_ylabel("Current")
    ax[i].set_title(f"{cell}\nTransmembrane Currents Averaged Across Cells")

plt.tight_layout()

fig, ax = plt.subplots(
    nrows=len(cell_currents.keys()),
    ncols=1,
    figsize=(6, 24),
)

for i, cell in enumerate(cell_currents.keys()):
    section_currents = cell_currents[cell]

    for section in section_currents.keys():
        if not section == "agg":
            ax[i].plot(
                dpls[0].times,
                np.mean(section_currents[section], axis=0),
                label=f"{section.replace('_', ' ').capitalize()}",
            )
    ax[i].legend()

    ax[i].set_xlabel("Time")
    ax[i].set_ylabel("Currents")
    ax[i].set_title(f"{cell}\nTransmembrane Currents Averaged Across Cells")

plt.tight_layout()
