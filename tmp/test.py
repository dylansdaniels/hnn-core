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
from hnn_core.viz import plot_dipole

# %% ####################
plot_morphology = False
plot_cell_grid = False

net = calcium_model()

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
    )

# %% ####################
window_len, scaling_factor = 30, 3000
for dpl in dpls:
    dpl.smooth(window_len).scale(scaling_factor)

# %% ####################
fig, axes = plt.subplots(
    nrows=2,
    ncols=1,
    sharex=True,
    figsize=(6, 6),
    constrained_layout=True,
)
_ = plot_dipole(
    dpls,
    ax=axes[1],
    layer="agg",
    show=False,
)
_ = net.cell_response.plot_spikes_hist(
    ax=axes[0],
    spike_types=["evprox", "evdist"],
    invert_spike_types=["evdist"],
    show=False,
)
# %% ####################
_ = plot_dipole(
    dpls,
    average=False,
    layer=["L2", "L5", "agg"],
    show=True,
)

# %% ####################

print(
    "-" * 50,
    "Inspecting net.cell_response.ca",
    type(net.cell_response.ca),
    len(net.cell_response.ca),
    "List of length 2 with each item corresponding to a trial",
    sep="\n",
)

# %% ####################
t_01 = net.cell_response.ca[0]

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
mask_ca = []
for i, e in enumerate(t_01):
    if t_01[e] == {}:
        pass
    else:
        mask_ca.append(e)

print(
    "-" * 50,
    "Cell GIDs with Ca recording",
    mask_ca,
    f"Num cells: {len(mask_ca)}",
    sep="\n",
)

# %% ####################

cell_gid = mask_ca[0]

for key in t_01[cell_gid].keys():
    # plot values in list for each key
    plt.plot(
        dpls[0].times,
        net.cell_response.ca[0][cell_gid][key],
        label=key,
    )
plt.xlabel("time")
plt.ylabel("calcium")
plt.title(f"Ca by section for GID {cell_gid}")
plt.legend()

# %% ####################
# average across all cell gids for each section
sectional_avg = {}

for section in t_01[cell_gid].keys():
    rows = []
    for gid in mask_ca:
        if section in t_01[gid]:
            rows.append(t_01[gid][section])  # each is a list
        else:
            raise ValueError(f"Section {section} not found in GID {gid}.")
    sectional_avg[section] = np.array(rows)

section = "soma"
print(sectional_avg[section].shape, sep="\n")


# %% ####################
for section in sectional_avg.keys():
    plt.plot(
        dpls[0].times,
        np.mean(sectional_avg[section], axis=0),
        label=f"{section.replace('_', ' ').capitalize()}",
    )
plt.xlabel("time")
plt.ylabel("calcium")
plt.title("Avg Ca by Section")
plt.legend()


# %% ####################
fig, ax = plt.subplots(
    nrows=len(sectional_avg.keys()),
    ncols=1,
    figsize=(6, 24),
)

for i, section in enumerate(sectional_avg.keys()):
    ax[i].plot(
        dpls[0].times,
        np.mean(sectional_avg[section], axis=0),
    )

    ax[i].set_xlabel("time")
    ax[i].set_ylabel("calcium")
    ax[i].set_title(f"Avg Ca for {section.replace('_', ' ').capitalize()}")

plt.tight_layout()

# %% ####################
fig, axes = plt.subplots(
    nrows=4,
    ncols=1,
    figsize=(6, 16),
    constrained_layout=True,
)

_ = net.cell_response.plot_spikes_hist(
    ax=axes[0],
    spike_types=["evprox", "evdist"],
    invert_spike_types=["evdist"],
    show=False,
)

for section in sectional_avg.keys():
    axes[2].plot(
        dpls[0].times,
        np.mean(sectional_avg[section], axis=0),
        label=section.replace("_", " ").capitalize(),
    )

axes[2].set_xlabel("time")
axes[2].set_ylabel("calcium")
axes[2].set_title("Avg Ca by Section")
axes[2].legend()

_ = net.cell_response.plot_spikes_raster(
    trial_idx=0,
    ax=axes[3],
    dpl=dpls[0],
    overlay_dipoles=True,
    show=False,
)

_ = plot_dipole(
    dpls[0],
    ax=axes[1],
    layer="agg",
    show=False,
)

# %% ####################
# Heatmap of calcium in apical_tuft

section = "apical_tuft"

section_data = np.array(
    [t_01[gid][section] for gid in mask_ca],
)

fig, ax = plt.subplots(figsize=(8, 6))

im = ax.pcolormesh(
    dpls[0].times,
    np.arange(len(mask_ca)),
    section_data,
    shading="gouraud",
    vmin=0,
    vmax=0.05,
)

ax.set_xlabel("Time (ms)")
ax.set_ylabel("GID index")
ax.set_title(f"Ca concentration in {section.replace('_', ' ').capitalize()}")
ax.invert_yaxis()

cb = plt.colorbar(
    im,
    ax=ax,
)
cb.set_label("Ca concentration (mM)")

# Optional overlay of dipole
ax2 = ax.twinx()
ax2.plot(
    dpls[0].times,
    dpls[0].data["agg"],
    color="white",
    alpha=0.5,
)

plt.tight_layout()

# %% ####################
from neuron import h
from hnn_core.network_builder import load_custom_mechanisms
from hnn_core.cells_default import pyramidal
from hnn_core.params import _short_name

load_custom_mechanisms()

cell = pyramidal(
    cell_name=_short_name("L5_pyramidal"),
)

# Cell.build will do the following:
#    > Create sections
#    > Create synamses
#    > Set biophysics
#    > Run _insert_dipole()
cell.build(
    sec_name_apical="apical_trunk",
)

i = 0

# Inspect the dipole point process
hoc_obj = cell.dipole_pp[i]
print(
    f"dipole_pp type: {type(hoc_obj)}",
    f"class name: {hoc_obj.__class__.__name__}",
    f"segment name: {hoc_obj.get_segment()}",
    "",
    sep="\n"
)

attributes = [att for att in dir(hoc_obj) if not att.startswith("__")]
max_len = max(len(att) for att in attributes)

for att in attributes:
    value = getattr(hoc_obj, att)
    print(f"{att:<{max_len}}   :   {value}")
print()

# cell.dipole_pp is a list of HocObject handles, each pointing
# to an inserted dipole mechanism — one per section, at a
# standard location (sec(1) i.e. end of the section).
#
# In other words, cell.dipole_pp collects one dipole per section,
# usually the one at sec(1.0) (the distal-most segment), which
# is used for calculating the total dipole moment

print("Print segments from cell.dipole_pp:")
for i, hoc_obj in enumerate(cell.dipole_pp):
    print("   ",hoc_obj.get_segment())
print()

# Iterate through each segment of each section,
# and check for the dipole attribute
for sec_name, sec in cell._nrn_sections.items():
    print(f"Section: {sec_name}, type: {type(sec)}")
    for i, seg in enumerate(sec):
        # Check if segment has dipole
        if hasattr(seg, "dipole"):
            print(f"Segment {i} in {sec_name} has dipole: {seg.dipole}")
        else:
            print(f"Segment {i} in {sec_name} has no dipole attribute")
    print()
    # break  # remove to iterate through all sections

# %% ####################

