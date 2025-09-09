# %% [markdown] #########
### Setup
# %% ####################

import matplotlib.pyplot as plt
import numpy as np

from hnn_core import (
    JoblibBackend,
    jones_2009_model,
    simulate_dipole,
)
from hnn_core.cells_default import pyramidal
from hnn_core.network_builder import load_custom_mechanisms
from hnn_core.network_models import add_erp_drives_to_jones_model
from hnn_core.params import _short_name

# %%
# need to expose imem somehow
# import hnn_core.network_builder as nb
# nb._create_parallel_context(expose_imem=True)
# print(nb._CVODE)

# from hnn_core.network_builder import _create_parallel_context

# This initializes the ParallelContext and CVode
# _create_parallel_context(expose_imem=True)


# %% [markdown] #########
### Simulation
# %% ####################

net = jones_2009_model()
add_erp_drives_to_jones_model(net)

n_trials = 1

with JoblibBackend(8):
    dpls = simulate_dipole(
        net,
        tstop=170.0,
        n_trials=n_trials,
        record_isec="all",
        record_ina="all",
        record_ik_hh2="all",
        record_ik_kca="all",
        record_ik_km="all",
        record_ica_ca="all",
        record_ica_cat="all",
        record_il_hh2="all",
        record_i_ar="all",
        record_i_cap="all",
        record_i_mem="all",
    )

scaling_factor = 3000
for dpl in dpls:
    dpl.scale(scaling_factor)

# window_len, scaling_factor = 30, 3000
# for dpl in dpls:
#     dpl.smooth(window_len).scale(scaling_factor)

dpl = dpls[0]
dpl_plot = dpl.plot(
    layer=["L5"]
)


# %% [markdown] #########
### Post processing
# %% ####################

trial = 0
ion_channel = "ina"
cell_type = "L5_pyramidal"


def postproc_tm_currents(
        trial=0,
        ion_channel=None,
        cell_type="L5_pyramidal",
    ):
    dipole = None
    # --- Step 1: Build a template L5 pyramidal cell to get relative endpoints ---
    load_custom_mechanisms()
    template_cell = pyramidal(cell_name=_short_name(cell_type))
    template_cell.build(sec_name_apical="apical_trunk")

    # Collect section endpoints (relative to soma)
    rel_endpoints = {}
    for sec_name, sec in template_cell._nrn_sections.items():
        start = np.array([sec.x3d(0), sec.y3d(0), sec.z3d(0)])
        end = np.array(
            [sec.x3d(sec.n3d() - 1), sec.y3d(sec.n3d() - 1), sec.z3d(sec.n3d() - 1)]
        )
        rel_endpoints[sec_name] = (start, end)

    # --- Step 2: Compute dipole ---
    for gid in net.gid_ranges[cell_type]:
        start_index = net.gid_ranges[cell_type][0]
        soma_pos = np.array(
            net.pos_dict[cell_type][gid - start_index]
        )  # absolute soma coordinates

        cell_data = net.cell_response.ionic_currents[ion_channel][trial][gid]

        for sec_name, segs in cell_data.items():
            # shift relative endpoints to absolute coordinates
            start_rel, end_rel = rel_endpoints[sec_name]
            start = start_rel + soma_pos
            end = end_rel + soma_pos

            nseg = len(segs)
            seg_positions = [(i - 0.5) / nseg for i in range(1, nseg + 1)]

            for pos, (seg_key, vec) in zip(seg_positions, segs.items()):
                abs_pos = start + pos * (end - start)
                z_i = abs_pos[2]  # only z-coordinate contributes

                I_t = np.array(vec)  # shape: (n_timepoints,)
                contrib = I_t * z_i

                if dipole is None:
                    dipole = contrib.copy()
                else:
                    dipole += contrib
    return dipole

# %% ####################

# dipole = postproc_tm_currents(
#     ion_channel="i_cap"
# )

# plt.plot(dipole)

# %% [markdown] #########
### Post process all channels
# %% ####################

# all_tm_channels=[
#     "ina",
#     "ik_hh2",
#     "ik_kca",
#     "ik_km",
#     "ica_ca",
#     "ica_cat",
#     "il_hh2",
#     "i_ar",
#     # "i_cap",
#     # "i_mem",
# ]

# tm_dipoles = {}

# for channel in all_tm_channels:
#     tm_dipoles[channel] = postproc_tm_currents(
#         trial=0,
#         ion_channel=channel,
#         cell_type="L5_pyramidal",
#     )

# %% ####################

# for key in tm_dipoles.keys():
#     plt.plot(
#         tm_dipoles[key],
#         label=key,
#     )

# plt.legend()

# %% ####################

# fig, ax = plt.subplots(
#     nrows=2,
#     ncols=1,
#     sharex=True,
#     figsize=(8,10),
# )

# agg = None

# for key in tm_dipoles.keys():
#     data = tm_dipoles[key].copy()
#     if agg is None:
#         agg = data
#     else:
#         agg += data

# ax[0].plot(
#     dpl.times,
#     agg,
# )

# dpl_plot = dpl.plot(
#     layer=["L5"],
#     ax=ax[1]
# )

# # %% ####################

# tmp_icap = postproc_tm_currents(
#         trial=0,
#         ion_channel="i_cap",
#         cell_type="L5_pyramidal",
#     )

# plt.plot(
#     dpl.times,
#     agg-tmp_icap,
# )

# %% [markdown] #########
### segment areas
# %% ####################

def show_seg_areas():
    """
    Note: areas for segment 0 are 0... does this pose an issue?

    Section: apical_trunk
    Segment 0: area = 0.00 µm² = 0.000000 cm²
    ...
    Section: apical_1
    Segment 0: area = 0.00 µm² = 0.000000 cm²
    ...
    Section: apical_2
    Segment 0: area = 0.00 µm² = 0.000000 cm²
    ...
    Section: apical_tuft
    Segment 0: area = 0.00 µm² = 0.000000 cm²
    ...
    Section: apical_oblique
    Segment 0: area = 0.00 µm² = 0.000000 cm²
    ...
    Section: basal_1
    Segment 0: area = 0.00 µm² = 0.000000 cm²
    Section: basal_2
    Segment 0: area = 0.00 µm² = 0.000000 cm²
    ...
    Section: basal_3
    Segment 0: area = 0.00 µm² = 0.000000 cm²
    ...
    Section: soma
    Segment 0: area = 0.00 µm² = 0.000000 cm²
    """
    template_cell = pyramidal(cell_name=_short_name("L5_pyramidal"))
    template_cell.build(sec_name_apical="apical_trunk")

    for sec_name, sec in template_cell._nrn_sections.items():
        print(f"Section: {sec_name}")
        nseg = sec.nseg
        for i in range(nseg):
            seg = sec(i / nseg)
            area_um2 = seg.area()  # µm²
            area_cm2 = area_um2 * 1e-8  # cm²
            print(f"  Segment {i}: area = {area_um2:.2f} µm² = {area_cm2:.6f} cm²")
    return

show_seg_areas()



# %% [markdown] #########
### Version 2
# %% ####################

def postproc_tm_currents_v2(
        trial=0,
        cell_type="L5_pyramidal",
        scaling_factor=3000,
        tm_channels=None,
    ):

    dipole = None
    load_custom_mechanisms()

    # build a template cell to get "metadata"
    template_cell = pyramidal(cell_name=_short_name(cell_type))
    template_cell.build(sec_name_apical="apical_trunk")

    # get the relative endpoints for each section from the template cell
    rel_endpoints = {}
    for sec_name, sec in template_cell._nrn_sections.items():
        start = np.array([sec.x3d(0), sec.y3d(0), sec.z3d(0)])
        # sec.n3d() returns the number of 3D points along a section; essentially len()
        # so "sec.n3d() - 1" is the index of the last 3D point
        end = np.array(
            [sec.x3d(sec.n3d() - 1), sec.y3d(sec.n3d() - 1), sec.z3d(sec.n3d() - 1)]
        )
        rel_endpoints[sec_name] = (start, end)

    if tm_channels is None:
        all_tm_channels = [
            "ina",
            "ik_hh2",
            "ik_kca",
            "ik_km",
            "ica_ca",
            "ica_cat",
            "il_hh2",
            "i_ar",
            "i_cap",
        ]
        if cell_type == "L2_pyramidal":
            all_tm_channels = [
                "ina",
                "ik_hh2",
                "ik_km",
                "il_hh2",
                "i_cap",
            ]
    else:
        all_tm_channels = tm_channels

    # loop through GIDs for the cell_type of interest
    for gid in net.gid_ranges[cell_type]:

        # get the updated soma position for this instantiation of the cell
        # index of the first cell: e.g., 170 for the first L5Pyr cell
        start_index = net.gid_ranges[cell_type][0]
        # get soma position from position dictionary, which uses its own indexing
        # that does not match the GID, hence the "- start_index"
        soma_pos = np.array(net.pos_dict[cell_type][gid - start_index])

        # create a dictionary of all channel data for the cell
        cell_channels = {
            ch: net.cell_response.ionic_currents[ch][trial][gid]
            for ch in all_tm_channels
        }

        # get the cell sections to loop over
        # the key used shouldn't matter, but we don't want to hard code it since
        # we can pass different channels to this function, so we get it dynamically
        first_key = list(cell_channels.keys())[0]
        cell_sections = list(cell_channels[first_key].keys())

        for sec_name in cell_sections:
            # offset the start/end positions by the realized soma position for this
            # cell instantiation
            start_rel, end_rel = rel_endpoints[sec_name]
            start = start_rel + soma_pos
            end = end_rel + soma_pos

            # get the normalized segment positions along the cell section
            nseg = len(cell_channels[first_key][sec_name])
            seg_positions = [(i - 0.5) / nseg for i in range(1, nseg + 1)]

            for pos, seg_key in zip(seg_positions, cell_channels[first_key][sec_name].keys()):
                # convert the normalized position to the absolute position
                # via linear interpolation
                abs_pos = start + pos * (end - start)
                # simplification: we are using the z position only here as I've
                # artificially rotated all sections to be vertical in this
                # simulation so we don't have to worry about angles. Ergo, we only
                # need to use the z-position in the calculations below
                z_i = abs_pos[2]

                # --- sum all currents for this segment ---
                I_t = np.zeros_like(np.array(cell_channels[first_key][sec_name][seg_key]))
                for ch in all_tm_channels:
                    # get channel data
                    vec = np.array(cell_channels[ch][sec_name][seg_key])

                    # get segment area and convert from µm^2 to cm^2
                    seg = template_cell._nrn_sections[sec_name](pos)
                    area_um2 = seg.area()  # µm^2
                    area_cm2 = area_um2 * 1e-8  # cm^2

                    if ch == "i_mem":
                        # i_mem is not recorded continuously as a density; it is
                        # recorded after each timestep. Ergo, the units conversion
                        # here is not necessary as the units are already in nA
                        #
                        # multiplying the contribution by zi in um will give us fAm,
                        # so we will later need to divide by 1e6 to convert to nAm
                        I_abs = vec
                    elif ch == "i_cap":
                        # should I flip sign or not??? I'm unclear on this
                        I_abs = -vec * area_cm2 * 1e6  # old approach mA -> nA
                        I_abs = vec * area_cm2 * 1e6  # old approach mA -> nA
                        I_abs = -vec * area_cm2  # keep as mA
                        I_abs = vec * area_cm2  # keep as mA
                        I_abs = 0
                    else:
                        # convert density (mA/cm^2) -> absolute current (nA)
                        I_abs = vec * area_cm2 * 1e6  # old approach mA -> nA
                        I_abs = vec * area_cm2  # keep as mA

                    I_t += I_abs


                # multiple by r_i per Naess 2015 Ch 2 (simplified to zi in this case)
                # for ionic currents, we have 1 mA*um = 1 nAm (correct units)
                # for i_mem, we have nA rather than mA. and 1 nA*um = 1 fAm
                contrib = I_t * z_i

                # divide by 1e6 to convert fAm to nAm and apply scaling as needed to
                # compare to the dipoles simulated by hnn-core
                if ch == "i_mem":
                    contrib = contrib / 1e6 * scaling_factor
                else:
                    contrib = contrib * scaling_factor

                if dipole is None:
                    dipole = contrib.copy()
                else:
                    dipole += contrib

    return dipole

test = postproc_tm_currents_v2()

fig, ax = plt.subplots(
    nrows=3,
    ncols=1,
    sharex=True,
    figsize=(8,15),
)

ax[0].plot(
    dpl.times,
    test,
)

test_imem = postproc_tm_currents_v2(
    tm_channels=["i_mem"],
)

ax[1].plot(
    dpl.times,
    test_imem,
)

ax[1].set_ylim(-200, 100)

dpl_plot = dpl.plot(
    layer=["L5"],
    ax=ax[2]
)


# %%

# %% ####################
test = postproc_tm_currents_v2(
    cell_type="L2_pyramidal"
)

fig, ax = plt.subplots(
    nrows=2,
    ncols=1,
    sharex=True,
    figsize=(8,10),
)

ax[0].plot(
    dpl.times,
    test,
)

dpl_plot = dpl.plot(
    layer=["L2"],
    ax=ax[1]
)

# %% [markdown] #########
### View synaptic currents
# %% ####################

def plot_isec_by_section(
        gid=170,
        trial=0,
    ):

    num_secs = len(net.cell_response.isec[trial][gid].keys())
    fig_height = 5*num_secs

    fig, ax = plt.subplots(
        nrows=num_secs,
        ncols=1,
        sharex=True,
        figsize=(8,fig_height),
    )

    for i, key in enumerate(net.cell_response.isec[trial][gid].keys()):
        for channel in net.cell_response.isec[trial][gid][key].keys():
            plt_label = channel.replace(f"{key}_", "")
            ax[i].plot(
                net.cell_response.isec[trial][gid][key][channel],
                label=plt_label,
            )
            ax[i].legend()
            ax[i].set_title(f"{key}")
    plt.tight_layout()

plot_isec_by_section()


# %% [markdown] #########
### Version 3
# %% ####################


def postproc_tm_currents_v3(
        trial=0,
        cell_type="L5_pyramidal",
        scaling_factor=3000,
    ):
    """
    Evenly distribute the synaptic contributions across the segments. This is a
    simplification
    """
    dipole = None
    load_custom_mechanisms()
    template_cell = pyramidal(cell_name=_short_name(cell_type))
    template_cell.build(sec_name_apical="apical_trunk")

    # relative endpoints
    rel_endpoints = {}
    for sec_name, sec in template_cell._nrn_sections.items():
        start = np.array([sec.x3d(0), sec.y3d(0), sec.z3d(0)])
        end = np.array(
            [sec.x3d(sec.n3d() - 1), sec.y3d(sec.n3d() - 1), sec.z3d(sec.n3d() - 1)]
        )
        rel_endpoints[sec_name] = (start, end)

    all_tm_channels = [
        "ina",
        "ik_hh2",
        "ik_kca",
        "ik_km",
        "ica_ca",
        "ica_cat",
        "il_hh2",
        "i_ar",
        "i_cap",
    ]

    all_syn_channels = [
        "ampa",
        "nmda",
        "gabaa",
        "gabab",
    ]

    for gid in net.gid_ranges[cell_type]:
        start_index = net.gid_ranges[cell_type][0]
        soma_pos = np.array(net.pos_dict[cell_type][gid - start_index])

        # all channel data for this cell
        cell_channels = {
            ch: net.cell_response.ionic_currents[ch][trial][gid]
            for ch in all_tm_channels
        }

        # synaptic currents
        syn_channels = net.cell_response.isec[trial][gid]

        for sec_name in cell_channels["ina"]:
            start_rel, end_rel = rel_endpoints[sec_name]
            start = start_rel + soma_pos
            end = end_rel + soma_pos

            nseg = len(cell_channels["ina"][sec_name])
            seg_positions = [(i - 0.5) / nseg for i in range(1, nseg + 1)]

            for pos, seg_key in zip(seg_positions, cell_channels["ina"][sec_name].keys()):
                abs_pos = start + pos * (end - start)
                z_i = abs_pos[2]

                # --- sum all TM currents for this segment ---
                I_t = np.zeros_like(np.array(cell_channels["ina"][sec_name][seg_key]))
                for ch in all_tm_channels:
                    vec = np.array(cell_channels[ch][sec_name][seg_key])
                    seg = template_cell._nrn_sections[sec_name](pos)
                    area_um2 = seg.area()  # µm^2
                    area_cm2 = area_um2 * 1e-8

                    if ch == "i_cap":
                        I_abs = -vec
                    else:
                        I_abs = vec * area_cm2 * 1e3  # mA/cm^2 -> nA

                    I_t += I_abs

                # --- add synaptic contributions (Option B: uniform distribution) ---
                if sec_name in syn_channels:
                    for syn_name in syn_channels[sec_name]:
                        vec = np.array(syn_channels[sec_name][syn_name])  # nA
                        I_t += vec / nseg

                # --- dipole contribution ---
                contrib = I_t * z_i
                contrib = contrib / 1e6 * scaling_factor

                if dipole is None:
                    dipole = contrib.copy()
                else:
                    dipole += contrib

    return dipole

run_v3=False

if run_v3:
    test = postproc_tm_currents_v3()

    fig, ax = plt.subplots(
        nrows=2,
        ncols=1,
        sharex=True,
        figsize=(8,10),
    )

    ax[0].plot(
        dpl.times,
        test,
    )

    dpl_plot = dpl.plot(
        layer=["L5"],
        ax=ax[1]
    )

# %%

# %% [markdown] #########
### Version 4
# %% ####################

def postproc_tm_currents_v4(
        trial=0,
        cell_type="L5_pyramidal",
        scaling_factor=3000,
        lambda_space=3.0,  # space constant for weight scaling
    ):
    """
    Try some kind of space scaling for the isec... This doesn't really make sense
    either.
    """
    dipole = None
    load_custom_mechanisms()
    template_cell = pyramidal(cell_name=_short_name(cell_type))
    template_cell.build(sec_name_apical="apical_trunk")

    # relative endpoints
    rel_endpoints = {}
    for sec_name, sec in template_cell._nrn_sections.items():
        start = np.array([sec.x3d(0), sec.y3d(0), sec.z3d(0)])
        end = np.array(
            [sec.x3d(sec.n3d() - 1), sec.y3d(sec.n3d() - 1), sec.z3d(sec.n3d() - 1)]
        )
        rel_endpoints[sec_name] = (start, end)

    all_tm_channels = [
        "ina",
        "ik_hh2",
        "ik_kca",
        "ik_km",
        "ica_ca",
        "ica_cat",
        "il_hh2",
        "i_ar",
        "i_cap",
    ]

    all_syn_channels = [
        "ampa",
        "nmda",
        "gabaa",
        "gabab",
    ]

    for gid in net.gid_ranges[cell_type]:
        start_index = net.gid_ranges[cell_type][0]
        soma_pos = np.array(net.pos_dict[cell_type][gid - start_index])

        # TM currents
        cell_channels = {
            ch: net.cell_response.ionic_currents[ch][trial][gid]
            for ch in all_tm_channels
        }

        # synaptic currents
        syn_channels = net.cell_response.isec[trial][gid]

        # pre/post neuron positions for distance scaling
        post_pos = np.array(net.pos_dict[cell_type][gid - start_index])

        for sec_name in cell_channels["ina"]:
            start_rel, end_rel = rel_endpoints[sec_name]
            start = start_rel + soma_pos
            end = end_rel + soma_pos

            nseg = len(cell_channels["ina"][sec_name])
            seg_positions = [(i - 0.5) / nseg for i in range(1, nseg + 1)]

            for pos, seg_key in zip(seg_positions, cell_channels["ina"][sec_name].keys()):
                abs_pos = start + pos * (end - start)
                z_i = abs_pos[2]

                # sum TM currents
                I_t = np.zeros_like(np.array(cell_channels["ina"][sec_name][seg_key]))
                for ch in all_tm_channels:
                    vec = np.array(cell_channels[ch][sec_name][seg_key])
                    seg = template_cell._nrn_sections[sec_name](pos)
                    area_um2 = seg.area()  # µm^2
                    area_cm2 = area_um2 * 1e-8

                    if ch == "i_cap":
                        I_abs = -vec
                    else:
                        I_abs = vec * area_cm2 * 1e3

                    I_t += I_abs

                # add synaptic contributions (Option B: uniform distribution with weight scaling)
                if sec_name in syn_channels:
                    for syn_name in syn_channels[sec_name]:
                        vec = np.array(syn_channels[sec_name][syn_name])  # nA

                        # compute scaling based on XY distance from all sources
                        scale = 1.0  # default
                        for conn in net.connectivity:
                            if conn['target_type'] == cell_type and gid in conn['target_gids']:
                                for src_gid in conn['src_gids']:
                                    src_pos = np.array(net.pos_dict[conn['src_type']][src_gid - net.gid_ranges[conn['src_type']][0]])
                                    d_xy = np.linalg.norm(src_pos[:2] - post_pos[:2])
                                    scale += np.exp(-d_xy**2 / lambda_space**2)

                        scale = scale / max(1, len(conn['src_gids']))
                        I_t += (vec / nseg) * scale

                # dipole contribution
                contrib = I_t * z_i
                contrib = contrib / 1e6 * scaling_factor

                if dipole is None:
                    dipole = contrib.copy()
                else:
                    dipole += contrib

    return dipole

run_v4=False

if run_v4:
    test = postproc_tm_currents_v4()

    fig, ax = plt.subplots(
        nrows=2,
        ncols=1,
        sharex=True,
        figsize=(8,10),
    )

    ax[0].plot(
        dpl.times,
        test,
    )

    dpl_plot = dpl.plot(
        layer=["L5"],
        ax=ax[1]
    )

# %% [markdown] #########
### Version 5
# %% ####################


def postproc_tm_currents_v5(
        trial=0,
        cell_type="L5_pyramidal",
        scaling_factor=3000,
    ):
    """
    Add synaptic currents. This is wrong because synapses are point processes at the
    0.5 point for each *section*, and therefore the synapses are only appearing in one
    single segment for multi-segment sections. 
    """
    dipole = None
    load_custom_mechanisms()
    template_cell = pyramidal(cell_name=_short_name(cell_type))
    template_cell.build(sec_name_apical="apical_trunk")

    # relative endpoints
    rel_endpoints = {}
    for sec_name, sec in template_cell._nrn_sections.items():
        start = np.array([sec.x3d(0), sec.y3d(0), sec.z3d(0)])
        end = np.array(
            [sec.x3d(sec.n3d() - 1), sec.y3d(sec.n3d() - 1), sec.z3d(sec.n3d() - 1)]
        )
        rel_endpoints[sec_name] = (start, end)

    all_tm_channels = [
        "ina",
        "ik_hh2",
        "ik_kca",
        "ik_km",
        "ica_ca",
        "ica_cat",
        "il_hh2",
        "i_ar",
        "i_cap",
    ]

    for gid in net.gid_ranges[cell_type]:
        start_index = net.gid_ranges[cell_type][0]
        soma_pos = np.array(net.pos_dict[cell_type][gid - start_index])

        # all channel data for this cell
        cell_channels = {
            ch: net.cell_response.ionic_currents[ch][trial][gid]
            for ch in all_tm_channels
        }

        for sec_name in cell_channels["ina"]:
            start_rel, end_rel = rel_endpoints[sec_name]
            start = start_rel + soma_pos
            end = end_rel + soma_pos

            nseg = len(cell_channels["ina"][sec_name])
            seg_positions = [(i - 0.5) / nseg for i in range(1, nseg + 1)]

            for pos, seg_key in zip(seg_positions, cell_channels["ina"][sec_name].keys()):
                abs_pos = start + pos * (end - start)
                z_i = abs_pos[2]

                # --- sum all currents for this segment ---
                I_t = np.zeros_like(np.array(cell_channels["ina"][sec_name][seg_key]))
                for ch in all_tm_channels:
                    vec = np.array(cell_channels[ch][sec_name][seg_key])

                    seg = template_cell._nrn_sections[sec_name](pos)
                    area_um2 = seg.area()  # µm^2
                    area_cm2 = area_um2 * 1e-8

                    if ch == "i_cap":
                        # flip sign
                        # I_abs = -vec * area_cm2 * 1e3  # mA -> nA  # not needed; units are already correct
                        I_abs = -vec
                    else:
                        # convert density (mA/cm^2) -> absolute current (nA)
                        I_abs = vec * area_cm2 * 1e3  # mA -> nA

                    I_t += I_abs

                contrib = I_t * z_i
                contrib = contrib / 1e6 * scaling_factor

                if dipole is None:
                    dipole = contrib.copy()
                else:
                    dipole += contrib

        # ------------------------------
        # NEW: synaptic current contribs
        # ------------------------------
        cell_synapses = net.cell_response.isec[trial][gid]
        for sec_name in cell_synapses:
            start_rel, end_rel = rel_endpoints[sec_name]
            start = start_rel + soma_pos
            end = end_rel + soma_pos

            for syn_name, vec in cell_synapses[sec_name].items():
                seg = template_cell._nrn_synapses[syn_name].get_segment()
                rel_pos = seg.x
                abs_pos = start + rel_pos * (end - start)
                z_i = abs_pos[2]

                vec = np.array(vec)  # nA already
                contrib = vec * z_i
                contrib = contrib / 1e6 * scaling_factor

                if dipole is None:
                    dipole = contrib.copy()
                else:
                    dipole += contrib

    return dipole

run_v5=False

if run_v5:
    test = postproc_tm_currents_v5()

    fig, ax = plt.subplots(
        nrows=2,
        ncols=1,
        sharex=True,
        figsize=(8,10),
    )

    ax[0].plot(
        dpl.times,
        test,
    )

    dpl_plot = dpl.plot(
        layer=["L5"],
        ax=ax[1]
    )

# %% [markdown] #########
### Extras
# %% ####################

# import pickle
# with open('tmp.pkl', 'wb') as f:
#     pickle.dump(dpl.data["L5"], f)

# with open('tmp.pkl', 'rb') as f:
#     x = pickle.load(f)

# dpl.data["L5"]==x

# %%
