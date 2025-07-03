# %% [markdown] ##################################
"""# Set Up"""

# %% #############################################
import copy

import matplotlib.pyplot as plt
import numpy as np

from hnn_core import (
    JoblibBackend,
    # calcium_model,
    jones_2009_model,
    simulate_dipole,
)
from hnn_core.network_models import add_erp_drives_to_jones_model
from hnn_core.viz import plot_dipole

stop_short = False
check_dipole_plots = True
add_electrode = False

if stop_short:
    tstop = 60
else:
    tstop = 170

# %% [markdown] ##################################
"""## Instantiate Network"""
# %% #############################################
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

# %% [markdown] ##################################
"""## Run Simulation and Plot Dipoles"""
# %% #############################################
n_trials = 1
scaling_factor = 3000

if add_electrode:
    depths = list(range(-325, 2150, 100))
    electrode_pos = [(135, 135, dep) for dep in depths]
    net.add_electrode_array("shank1", electrode_pos)

with JoblibBackend(8):
    dpls = simulate_dipole(
        net,
        tstop=tstop,
        n_trials=n_trials,
        record_ina="all",
        record_ik="all",
        record_ik_hh2="all",
        record_ik_kca="all",
        record_ik_km="all",
        record_ica_ca="all",
        record_ica_cat="all",
        record_il_hh2="all",
        record_i_ar="all",
    )

for dpl in dpls:
    dpl.scale(scaling_factor)

dpls_smoothed = copy.deepcopy(dpls)

window_len = 30
for dpl in dpls_smoothed:
    dpl.smooth(window_len)

if check_dipole_plots:
    ymax = max(
        max(dpls[0].data["L2"]),
        max(dpls[0].data["L5"]),
    )

    ymin = min(
        min(dpls[0].data["L2"]),
        min(dpls[0].data["L5"]),
    )

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(6, 12))

    _ = plot_dipole(
        dpls,
        ax=axes[0],
        layer="L2",
        show=False,
    )

    _ = plot_dipole(
        dpls_smoothed,
        ax=axes[0],
        layer="L2",
        show=False,
    )

    _ = plot_dipole(
        dpls,
        ax=axes[1],
        layer="L5",
        show=False,
    )

    _ = plot_dipole(
        dpls_smoothed,
        ax=axes[1],
        layer="L5",
        show=False,
    )

    for ax in axes:
        ax.set_ylim(ymin * 1.05, ymax * 1.05)


# %% #############################################


def plot_layer_dipoles_in_column(
    axes,
    dpls,
    dpls_smoothed,
):
    _ = plot_dipole(
        dpls,
        ax=axes[0],
        layer="L2",
        show=False,
    )
    _ = plot_dipole(
        dpls_smoothed,
        ax=axes[0],
        layer="L2",
        show=False,
        color="red",
    )

    _ = plot_dipole(
        dpls,
        ax=axes[1],
        layer="L5",
        show=False,
    )
    _ = plot_dipole(
        dpls_smoothed,
        ax=axes[1],
        layer="L5",
        show=False,
        color="red",
    )

    return


# fig, axes = plt.subplots(
#     nrows=2,
#     ncols=2,
#     figsize=(12, 12),
# )

# plot_layer_dipoles_in_column(
#     axes[:,1],
#     dpls,
#     dpls_smoothed
# )

# %% [markdown] ##################################
"""# Analyzing Current Recording"""
# %% #############################################

net.cell_response.ionic_currents["ik_km"]

# %% ####################


def get_channel_currents(
    cell_response,
    trial_number=1,
    cells_of_interest=[
        "L2_pyramidal",
        "L5_pyramidal",
    ],
):
    """
    Helper function to get channel currents by cell_type, section for a given trial
    and for the specified cells of interest from net.cell_response.ionic_currents

    Arguments:
        cell_response: net.cell_response,
        trial_number: int,
        cells_of_interest: list,

    Returns:
        channel_currents: dict,
            {
                "channel": {
                    "cell_type": {
                        "section": np.ndarray (n_cells x n_obs)
                    }
                }
            }
    """

    # get trial data for all recorded (non-empty) channels
    # ----------------------------------------
    currents = {}
    # currents: dict,
    #     {"channel": {
    #             "gid": {
    #                 "section": list()
    #     }}}
    all_currents = cell_response.ionic_currents

    for key in all_currents.keys():
        if len(all_currents[key][trial_number - 1].keys()) > 0:
            currents[key] = all_currents[key][trial_number - 1]

    # get gids for cell types of interest
    # ----------------------------------------
    cell_gids = {}

    for cell_type, ranges in net.gid_ranges.items():
        if cells_of_interest is None:
            if cell_type in net.cell_types.keys():
                cell_gids[cell_type] = list(ranges)
        else:
            if cell_type in cells_of_interest:
                cell_gids[cell_type] = list(ranges)

    # get currents by section for each cell
    # ----------------------------------------
    channel_currents = {}

    for channel in currents.keys():
        # data for a given GID
        cell_data = currents[channel]

        cell_currents = {}

        # reshape the data to aggregate at the cell type
        # cell_currents: dict,
        #     "cell_type": {
        #         "section": np.ndarray (n_cells x n_obs)
        #     }
        for cell_type, gids in cell_gids.items():
            # get the sections for the current cell_type
            cell_sections = cell_data[gids[0]].keys()

            section_currents = {}
            section_currents["agg"] = 0

            # get the data for each section for the GIDs of the current cell type
            for section in cell_sections:
                rows = []
                for gid in gids:
                    if section in cell_data[gid]:
                        rows.append(cell_data[gid][section])
                    else:
                        pass
                section_currents[section] = np.array(rows)

                # sum all sections to get the aggregate
                if type(section_currents["agg"]) != np.ndarray:
                    section_currents["agg"] = np.array(rows)
                else:
                    section_currents["agg"] += np.array(rows)

            # add the data for the current cell_type to cell currents dictionary
            cell_currents[cell_type] = section_currents

        # add the data for all cell_types to the channel_currents dictionary
        channel_currents[channel] = cell_currents

    return channel_currents


channel_currents = get_channel_currents(net.cell_response)

print(
    channel_currents.keys(),
)

# %% ####################
# ik current sanity check


def ik_sanity(
    plot_data=True,
    flip_order=False,
    cell_type="L5_pyramidal",
    segment="agg",
    line_colors=["#1b85d1", "#f6a208"],
):
    manual_sum = (
        channel_currents["ik_kca"][cell_type][segment]
        + channel_currents["ik_km"][cell_type][segment]
        + channel_currents["ik_hh2"][cell_type][segment]
    )

    manual_sum = np.sum(manual_sum, axis=0)

    model_sum = np.sum(channel_currents["ik"][cell_type][segment], axis=0)

    if plot_data:
        order = [0, 1]

        if flip_order:
            order = [1, 0]

        plt.plot(
            dpls[0].times,
            manual_sum,
            label="manual agg",
            color=line_colors[0],
            zorder=order[0],
        )

        plt.plot(
            dpls[0].times,
            model_sum,
            label="model agg",
            color=line_colors[1],
            zorder=order[1],
        )

        plt.legend()
        plt.tight_layout()

    manual_sum = np.round(manual_sum, 2)
    model_sum = np.round(model_sum, 2)

    return sum(manual_sum == model_sum) / len(manual_sum)


ik_sanity(
    # flip_order=True,
)


# %% ####################


def plot_agg_channel_currents(channel_currents):
    fig, ax = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(6, 12),
    )

    # track min/max for setting axes bounds
    ymax = 0
    ymin = 0

    for channel, cell_currents in channel_currents.items():
        for i, cell in enumerate(cell_currents.keys()):
            agg_sum = np.sum(cell_currents[cell]["agg"], axis=0)

            if type(agg_sum) != np.ndarray:
                continue

            ax[i].plot(
                dpls[0].times,
                agg_sum,
                label=channel,
            )

            ymax = max(ymax, max(agg_sum))
            ymin = min(ymin, min(agg_sum))

            ax[i].set_xlabel("Time (s)")
            ax[i].set_ylabel("Current (mA/cm2)")
            ax[i].set_title(f"{cell}\nChannel Currents (Summed Across Cell Segments)")

    for axis in ax:
        axis.legend()
        axis.set_ylim(
            ymin * 1.1,
            ymax * 1.1,
        )

    plt.tight_layout()


plot_agg_channel_currents(channel_currents)

# %% ####################


def plot_cell_currents_by_section(
    channel_name,
    channel_currents,
):
    cell_currents = channel_currents[channel_name]

    fig, ax = plt.subplots(
        nrows=len(cell_currents.keys()),
        ncols=1,
        figsize=(6, 12),
    )

    # track min/max for setting axes bounds
    ymax = 0
    ymin = 0

    for i, cell in enumerate(cell_currents.keys()):
        section_currents = cell_currents[cell]

        for section in section_currents.keys():
            if not section == "agg":
                # for each section, sum the values for the different cells of the
                # current cell type
                section_sum = np.sum(section_currents[section], axis=0)

                ax[i].plot(
                    dpls[0].times,
                    section_sum,
                    label=f"{section.replace('_', ' ').capitalize()}",
                )

                # get min/max for y-axis
                ymax = max(ymax, max(section_sum))
                ymin = min(ymin, max(section_sum))

        ax[i].legend()

        ax[i].set_xlabel("Time")
        ax[i].set_ylabel("Current (mA/cm2)")
        ax[i].set_title(f"{cell}\n{channel_name} By Section, Summed Across Cells")

    for axis in ax:
        axis.set_ylim(
            ymin * 1.1,
            ymax * 1.1,
        )

    plt.tight_layout()

    return


plot_cell_currents_by_section(
    channel_name="ik_hh2",
    channel_currents=channel_currents,
)

# %% ####################


def plot_currents_column(
    ax,
    channel_currents,
    times,
    channels=None,
    bin_width=1.0,
):
    dt = times[1] - times[0]
    samples_per_bin = int(bin_width / dt)

    if channels is not None:
        channel_currents = {
            k: v for (k, v) in channel_currents.items() if k in channels
        }

    # generate plots with binned data
    for channel in channel_currents.keys():
        cell_currents = channel_currents[channel]

        for i, cell in enumerate(cell_currents.keys()):
            y = np.sum(cell_currents[cell]["agg"], axis=0)

            if type(y) != np.ndarray:
                continue

            # reshape data for binning
            n_bins = len(y) // samples_per_bin
            y_binned = (
                y[: n_bins * samples_per_bin]
                .reshape(n_bins, samples_per_bin)
                .mean(axis=1)
            )
            t_binned = (
                times[: n_bins * samples_per_bin]
                .reshape(n_bins, samples_per_bin)
                .mean(axis=1)
            )

            ax[i].bar(
                t_binned,
                y_binned,
                width=bin_width,
                align="edge",
                label=channel,
                alpha=0.7,
            )

            ax[i].set_xlabel("Time (s)")
            ax[i].set_ylabel("Aggregate Currents")
            ax[i].set_title(
                f"{cell}" + "\nTransmembrane Currents Averaged Across Cells",
            )


rows = 2
cols = 1
times = dpls[0].times
bin_width = 1.0

fig, ax = plt.subplots(
    nrows=rows,
    ncols=cols,
    figsize=(6, 6 * rows),
)

channels = [
    "ina",
    "ik_hh2",
    "ik_kca",
    "ik_km",
    "ica_ca",
    "ica_cat",
    "il_hh2",
    "i_ar",
]

plot_currents_column(
    ax,
    channel_currents,
    times,
    channels=channels,
    bin_width=bin_width,
)

for axis in ax:
    axis.legend()
    # axis.set_ylim(ymin, ymax)

plt.tight_layout()

# %% ####################


def plot_net_in_out_currents(
    channel_currents,
    channels=None,
):
    bin_width = 1.0
    times = dpls[0].times
    dt = times[1] - times[0]
    samples_per_bin = int(bin_width / dt)

    if channels is None:
        channels = channel_currents.keys()

    # get cell_types
    cells = []
    for channel in channel_currents:
        for cell_type in channel_currents[channel].keys():
            if cell_type not in cells:
                cells.append(cell_type)

    fig, ax = plt.subplots(
        nrows=len(cells),
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
            y = np.sum(
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
            times[: n_bins * samples_per_bin]
            .reshape(n_bins, samples_per_bin)
            .mean(axis=1)
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
            color="green",
        )

        # generate bar plot
        ax[i, 0].bar(
            x_binned[~y_pos],
            y_binned[~y_pos],
            width=bin_width,
            align="edge",
            alpha=0.7,
            label="outward",
            color="red",
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
        axis.set_ylim(axis.get_ylim()[::-1])
        axis.legend()

    plot_layer_dipoles_in_column(
        ax[:, 1],
        dpls,
        dpls_smoothed,
    )

    plt.tight_layout()

plot_net_in_out_currents(
    channel_currents,
    ["ina", "ik"],
)

# %%
