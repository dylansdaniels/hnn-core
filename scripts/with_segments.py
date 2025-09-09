# %% ####################
import matplotlib.pyplot as plt
import numpy as np

from hnn_core import (
    JoblibBackend,
    # MPIBackend,
    jones_2009_model,
    simulate_dipole,
)
from hnn_core.cells_default import pyramidal
from hnn_core.network_builder import load_custom_mechanisms
from hnn_core.network_models import add_erp_drives_to_jones_model
from hnn_core.params import _short_name

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

# %% [markdown]
### View Cell Properties

# %%

def view_cell_properties():
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
    # attributes = [att for att in dir(hoc_obj)]
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

    print("Print all segments from cell.dipole_pp:")
    for i, hoc_obj in enumerate(cell.dipole_pp):
        print("   ",hoc_obj.get_segment())
    print()

    # Iterate through each segment of each section,
    # and check for the dipole attribute
    for sec_name, sec in cell._nrn_sections.items():

        nrn_sec = sec
        print(nrn_sec.n3d())
        for i in range(nrn_sec.n3d()):
            print(nrn_sec.x3d(i), nrn_sec.y3d(i), nrn_sec.z3d(i))


        # print(dir(sec.x3d))
        print(f"Section: {sec_name}, type: {type(sec)}")
        for i, seg in enumerate(sec):
            # Check if segment has dipole
            print(seg)
            print(seg.x)
            if hasattr(seg, "dipole"):
                print(f"Segment {i} in {sec_name} has dipole: {seg.dipole}")
            else:
                print(f"Segment {i} in {sec_name} has no dipole attribute")
        print()
        # break  # remove to iterate through all sections


view_cell_properties()

# %%

def view_section_endpoints():
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

    # Iterate through each segment of each section,
    # and check for the dipole attribute
    for sec_name, sec in cell._nrn_sections.items():
        print(sec_name)
        nrn_sec = sec
        print(
            f"Num sections: {nrn_sec.n3d()}"
        )
        for i in range(nrn_sec.n3d()):
            print(nrn_sec.x3d(i), nrn_sec.y3d(i), nrn_sec.z3d(i))

        print()

view_section_endpoints()

# %%

# %% ####################

n_trials = 1

with JoblibBackend(8):
    # with MPIBackend(n_procs=8, mpi_cmd='mpiexec'):
    dpls = simulate_dipole(
        net,
        tstop=170.0,
        n_trials=n_trials,
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
    "Inspecting net.cell_response.ionic_currents['ina']",
    type(net.cell_response.ionic_currents['ina']),
    len(net.cell_response.ionic_currents['ina']),
    "List of length 2 with each item corresponding to a trial",
    sep="\n",
)

# %%
# get first trial only
t_01 = net.cell_response.ionic_currents['ina'][0]

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

for cell_type, _range in net.gid_ranges.items():
    if cell_type in net.cell_types.keys():
        cell_gids[cell_type] = list(_range)

print(
    f"Cell types: {cell_gids.keys()}"
)


# %% [markdown]
### Examine "na"

# %%
cell_type = 'L5_pyramidal'
cell_gid = cell_gids[cell_type][0]

for key in t_01[cell_gid].keys():
    print(f"{key}:")
    print(
        t_01[cell_gid][key].keys(),
        "\n"
    )

# for key in t_01[cell_gid].keys():
#     for subkey in t_01[cell_gid][key]["segment_info"]:
#         # if subkey not in ["start", "end"]:
#         print(
#             f"{subkey}: "
#             f"{t_01[cell_gid][key]["segment_info"][subkey]}"
#         )
#     print()

t_01[cell_gid].keys()

# %%

# %% [markdown]
### Plot L5 with segments

# %% ####################

cell_type = 'L5_pyramidal'
cell_gid = cell_gids[cell_type][0]

for key in t_01[cell_gid].keys():
    if type(t_01[cell_gid][key]) is not dict:
        # plot values in list for each key
        plt.plot(
            dpls[0].times,
            net.cell_response.ionic_currents['ina'][0][170][key],
            label=key,
        )
    else:
        for sub_key in t_01[cell_gid][key]:
            if sub_key != "segment_info":
                plt.plot(
                    dpls[0].times,
                    net.cell_response.ionic_currents['ina'][0][170][key][sub_key],
                    label=None,
                )

plt.xlabel("time")
plt.ylabel("sodium")
plt.title(f"Sodium by section for {cell_type}, GID {cell_gid}")
# plt.legend()

# %% ####################
# average across all cell gids for each section



# %%
# %% ####################
# get currents by section for each cell

# %% [markdown] #########
### Testing
# %% ####################
# get currents by section for each cell
net.cell_response.ionic_currents['ik_hh2'][0][170]