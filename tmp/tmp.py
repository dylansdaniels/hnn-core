# %% ####################
import matplotlib.pyplot as plt
import numpy as np

from hnn_core import (
    JoblibBackend,
    # MPIBackend,
    calcium_model,
    jones_2009_model,
    simulate_dipole,
)
from hnn_core.network_models import add_erp_drives_to_jones_model

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

with JoblibBackend(8):
    # with MPIBackend(n_procs=8, mpi_cmd='mpiexec'):
    dpls = simulate_dipole(
        net,
        tstop=170.0,
        n_trials=n_trials,
        record_ca="all",
        record_ina="all",
    )

# %% ####################
window_len, scaling_factor = 30, 3000
for dpl in dpls:
    dpl.smooth(window_len).scale(scaling_factor)

# %%
_ = net.cell_response.plot_spikes_raster(
    overlay_dipoles=True,
    dpl=dpls,
    show=False,
    show_legend=False,
)

# %% ####################
# EVALUATING SODIUM RECORDING
#########################

print(
    "-" * 50,
    "Inspecting net.cell_response.ina",
    type(net.cell_response.ina),
    len(net.cell_response.ina),
    "List of length 2 with each item corresponding to a trial",
    sep="\n",
)

# %%
# get first trial only
t_01 = net.cell_response.ina[0]

print(
    "-" * 50,
    "Looking at single trial",
    type(t_01),
    t_01.keys(),
    "Keys correspond to GIDs",
    sep="\n",
)

# non-empty GIDs
# instantiate "mask"
mask_na = []
for i, e in enumerate(t_01):
    if t_01[e] == {}:
        pass
    else:
        mask_na.append(e)

print(
    "-" * 50,
    "Cell GIDs with na recording",
    mask_na,
    f"Num cells: {len(mask_na)}",
    sep="\n",
)

# GIDs by cell type

cell_gids = {}

for cell_type, range in net.gid_ranges.items():
    if cell_type in net.cell_types.keys():
        cell_gids[cell_type] = list(range)

print(
    f"Cell types: {cell_gids.keys()}"
)


# %% ####################

cell_type = 'L5_pyramidal'
cell_gid = cell_gids[cell_type][0]

for key in t_01[cell_gid].keys():
    # plot values in list for each key
    plt.plot(
        dpls[0].times,
        net.cell_response.ina[0][170][key],
        label=key,
    )
plt.xlabel("time")
plt.ylabel("sodium")
plt.title(f"Sodium by section for {cell_type}, GID {cell_gid}")
plt.legend()

# %% ####################
# average across all cell gids for each section
cell_type = 'L5_pyramidal'
cell_gid = cell_gids[cell_type][0]

section_currents = {}
section_currents['agg'] = 0

for section in t_01[cell_gid].keys():
    rows = []
    for gid in cell_gids[cell_type]:
        if section in t_01[gid]:
            rows.append(t_01[gid][section])  # each is a list
        else:
            pass
    section_currents[section] = np.array(rows)

    if type(section_currents['agg']) != np.ndarray:
        section_currents['agg'] = np.array(rows)
    else:
        section_currents['agg'] += np.array(rows)

section = "soma"
print(section_currents[section].shape, sep="\n")


# %% ####################
for section in section_currents.keys():
    plt.plot(
        dpls[0].times,
        np.mean(section_currents[section], axis=0),
        label=f"{section.replace('_', ' ').capitalize()}",
    )
plt.xlabel("time")
plt.ylabel("na")
plt.title(f"Agggregate and Sectional Na Currents for {cell_type}, \nAveraged Across Cell GIDs")
plt.legend()

# %% ####################
fig, ax = plt.subplots(
    nrows=len(section_currents.keys()),
    ncols=1,
    figsize=(6, 24),
)

# Ensure ax is always iterable
if not isinstance(ax, (list, np.ndarray)):
    ax = [ax]

for i, section in enumerate(section_currents.keys()):
    ax[i].plot(
        dpls[0].times,
        np.mean(section_currents[section], axis=0),
    )

    ax[i].set_xlabel("time")
    ax[i].set_ylabel("na")
    ax[i].set_title(
        f"{cell_type} Avg Sodium for {section.replace('_', ' ').capitalize()}"
    )

plt.tight_layout()

# %%
# %% ####################
# get currents by section for each cell

trial = net.cell_response.ina[0]

cell_currents = {}

for cell_type, gids in cell_gids.items():

    cell_sections = trial[gids[0]].keys()

    section_currents = {}
    section_currents['agg'] = 0

    for section in cell_sections:
        rows = []
        for gid in gids:
            if section in trial[gid]:
                rows.append(trial[gid][section])
            else:
                pass
        section_currents[section] = np.array(rows)

        if type(section_currents['agg']) != np.ndarray:
            section_currents['agg'] = np.array(rows)
        else:
            section_currents['agg'] += np.array(rows)

    cell_currents[cell_type] = section_currents

# %% ####################

fig, ax = plt.subplots(
    nrows=len(cell_currents.keys()),
    ncols=1,
    figsize=(6, 24),
)
gs
for i, cell in enumerate(cell_currents.keys()):
    ax[i].plot(
        dpls[0].times,
        np.mean(cell_currents[cell]['agg'], axis=0),
    )

    ax[i].set_xlabel("Time")
    ax[i].set_ylabel("Aggregate Na Currents (Sum Across All Sections)")
    ax[i].set_title(
        f"{cell}\nTransmembrane Na Currents Averaged Across Cells"
    )

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
        if not section == 'agg':
            ax[i].plot(
                dpls[0].times,
                np.mean(section_currents[section], axis=0),
                label=f"{section.replace('_', ' ').capitalize()}",
            )
    ax[i].legend()

    ax[i].set_xlabel("Time")
    ax[i].set_ylabel("Na Currents")
    ax[i].set_title(
        f"{cell}\nTransmembrane Na Currents Averaged Across Cells"
    )

plt.tight_layout()

# %%
