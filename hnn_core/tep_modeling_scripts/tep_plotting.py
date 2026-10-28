"""
Originally authored by Joyce (previously `TEP_plotting.pynb`)
Updated and edited by Dylan
"""

# %%

import matplotlib.pyplot as plt
import numpy as np

from hnn_core import (
    MPIBackend,
    hnn_io,
    simulate_dipole,
)
from hnn_core.viz import plot_dipole

# %%
# Read the network parameter
net = hnn_io.read_network_configuration(
    "hnn_core/tep_modeling_scripts/networks/agg_80_optim_Final.json"
)

# %%
tstop = 300
with MPIBackend(n_procs=6):
    dpls = simulate_dipole(
        net,
        tstop=tstop,
        dt=0.025,
        n_trials=1,
        baseline_correction=True,
        record_ca="all",
        record_isec="all",
        record_vsec="all",
    )

# %%
# Smooth and scale the dipole
dpls_smoothed = [dpl.smooth(20).scale(0.8) for dpl in dpls]

# %%
# Plotting the dipole and spiking for sanity check

fig, axes = plt.subplots(
    5,
    1,
    figsize=(6, 8),
    constrained_layout=True,
    height_ratios=[1, 2, 2, 2, 2],
    sharex=True,
)

net.cell_response.plot_spikes_hist(
    ax=axes[0], spike_types=["evprox", "evdist", "ngfc"], trial_idx=0, show=False
)

plot_dipole(
    dpls_smoothed.copy(),
    ax=axes[1],
    layer="L5",
    show=False,
    average=True,
    label="Simulation",
)
plot_dipole(
    dpls_smoothed.copy(),
    ax=axes[2],
    layer="L2",
    show=False,
    average=True,
    label="Simulation",
)
plot_dipole(
    dpls_smoothed.copy(),
    ax=axes[3],
    layer="agg",
    show=False,
    average=True,
    label="Simulation",
)

net.cell_response.plot_spikes_raster(ax=axes[4], show=False, show_legend=False)

plt.show()


# %%
# get L5 pyramidal neuron id
gids = net.gid_ranges
l5_gids = np.array(gids["L5_pyramidal"])

# get calcium for each cell tuft
trial_idx = 0
ca_l5_tuft = np.zeros((l5_gids.shape[0], len(net.cell_response.times)))

for i, g in enumerate(l5_gids.astype(int)):
    ca_l5_tuft[i] = net.cell_response.ca[trial_idx][g]["apical_tuft"]

# sorting the cells by their peak calcium concentration times
max_activation_times = np.argmax(ca_l5_tuft, axis=1)
sort_idx = np.argsort(max_activation_times)[::-1]
ca_l5_tuft_sort = ca_l5_tuft[sort_idx]

# plot
fig, axes = plt.subplots(
    2, 1, sharex=False, figsize=(8, 6), constrained_layout=True, height_ratios=[1, 4]
)
timevec = np.arange(0, len(net.cell_response.times), 1)

net.cell_response.plot_spikes_hist(
    ax=axes[0], spike_types=["evprox", "evdist", "ngfc"], trial_idx=0, show=False
)

# calcium concentration
im = axes[1].pcolormesh(
    timevec,
    np.arange(len(l5_gids)),
    ca_l5_tuft_sort,
    shading="gouraud",
    vmin=0,
    vmax=0.0006,
)
axes[1].invert_yaxis()
axes[1].set_title("L5 Pyramidal Apical Tufts Calcium")
axes[1].set_xlabel("Time (ms)")
axes[1].set_ylabel("Cells")

# add colorbar
cb = plt.colorbar(im, ax=axes[1], pad=0.03)
cb.set_ticks([0, 0.0001, 0.0002, 0.0003, 0.0004])
cb.set_label("Ca concentration $mM$")

# total number of datapoints
n_points = len(timevec)

# Convert desired tick times (in ms) to indices in timevec
tick_times = np.arange(0, tstop + 1 - np.mod(tstop, 50), 50)  # 0, 50, ..., 300
tick_indices = (tick_times / tstop * n_points).astype(int)

# Set ticks and labels
axes[1].set_xticks(tick_indices)
axes[1].set_xticklabels(tick_times)
axes[1].set_xlim(0, n_points)
ax2 = axes[1].twinx()

# plot the dipole
ax2.plot(timevec, dpls_smoothed[0].data["agg"], color=(1, 1, 1), linewidth=2)
ax2.set_ylabel("Dipole Moment (nAm)")

# %%
# Plotting network calcium concentration spatial distribution through time
fig, axes = plt.subplots(
    3, 3, sharex=True, sharey=True, figsize=(8, 6), constrained_layout=True
)

# set start and end time for plotting
start, stop = 30, 300

# convert to timevec units for plotting
interval = round((stop - start) / 9)
start_vec = start * 40
stop_vec = stop * 40  # convert to timevec

# looping through each time interval to plot calcium spatial distribution
for i, ax in zip(np.arange(start_vec, stop_vec, interval * 40), axes.flat):
    ca_concentration = ca_l5_tuft[:, i : i + 1]
    grid_ca = ca_concentration.reshape((10, 10))
    c = ax.pcolormesh(
        np.arange(0, 10), np.arange(0, 10), grid_ca, shading="auto", vmin=0, vmax=0.0006
    )  # vmax may need to be adjusted to accomodate Ca concentration magnitude
    ax.set_title(f"{round(i / len(timevec) * tstop)} ms")

fig.colorbar(c, ax=axes, location="right", shrink=0.8, label="Calcium concentration")
fig.suptitle("Spatial Distribution of L5 Apical Ca Concentration")
plt.xticks([])
plt.yticks([])
plt.show()


# %%
# Plotting NMDA current, Apical tuft voltage, and Mg block opening probability


# calculating mg opening probability from membrane voltage
def calculate_mg_block(Vm):
    mgblock = 1 / (1 + np.exp(0.072 * -Vm) * (1 / 3.57))
    return mgblock


apical_tuft_v = np.zeros((len(l5_gids), len(net.cell_response.times)))
apical_tuft_mg = np.zeros((len(l5_gids), len(net.cell_response.times)))
apical_tuft_nmda = np.zeros((len(l5_gids), len(net.cell_response.times)))

trial_idx = 0

for i, cell_idx in enumerate(l5_gids):
    Vm = net.cell_response.vsec[trial_idx][cell_idx]["apical_tuft"]
    i_NMDA = net.cell_response.isec[trial_idx][cell_idx]["apical_tuft"][
        "apical_tuft_nmda"
    ]
    mgblock = [calculate_mg_block(V) for V in Vm]

    apical_tuft_v[i] = Vm
    apical_tuft_mg[i] = mgblock
    apical_tuft_nmda[i] = i_NMDA

times = net.cell_response.times

mean_v = apical_tuft_v.mean(axis=0)
std_v = apical_tuft_v.std(axis=0)

mean_mg_opening = apical_tuft_mg.mean(axis=0)
std_mg_opening = apical_tuft_mg.std(axis=0)

mean_nmda = apical_tuft_nmda.mean(axis=0)
std_nmda = apical_tuft_nmda.std(axis=0)


# plotting
fig, axes = plt.subplots(
    3, 1, sharex=False, figsize=(6, 8), constrained_layout=True, height_ratios=[1, 1, 1]
)

trial_idx = 0
cell_idx = 0

# plot NMDA curent trace
axes[0].plot(times, mean_nmda)
axes[0].fill_between(times, mean_nmda - std_nmda, mean_nmda + std_nmda, alpha=0.1)
axes[0].set_title("Apical Tuft NMDA current")

# plot membrane potential trace
axes[1].plot(times, mean_v)
axes[1].fill_between(times, mean_v - std_v, mean_v + std_v, alpha=0.1)
axes[1].set_title("Apical Tuft Membrane Voltage")

# plot mgblock status
axes[2].plot(times, mean_mg_opening)
axes[2].fill_between(
    times, mean_mg_opening - std_mg_opening, mean_mg_opening + std_mg_opening, alpha=0.1
)
axes[2].set_title("Magnesium Opening Probability")

# %%
