# %% [markdown] ###########################################################
## Disclaimer
# Note that this is **not** an actual test file and will not stay in the final
# version. I'm using this file to draft and test some features that will
# probably turn into a demo notebook for the new Textbook
# %% ######################################################################

# %% [markdown] ###########################################################
## Setup
# %% ######################################################################

import math

import bpy
import matplotlib.pyplot as plt
import numpy as np
from IPython.core.getipython import get_ipython
from matplotlib.lines import Line2D
from neuron import h

from hnn_core import (
    JoblibBackend,
    jones_2009_model,
    simulate_dipole,
)
from hnn_core.cells_default import pyramidal
from hnn_core.network_builder import load_custom_mechanisms
from hnn_core.network_models import add_erp_drives_to_jones_model

# pyright: reportOptionalMemberAccess=false

# %% [markdown] ###########################################################
## Simulations
# %% ######################################################################

if "net_base" not in locals():
    net_base = jones_2009_model()
    add_erp_drives_to_jones_model(net_base)

    with JoblibBackend(8):
        dpls = simulate_dipole(
            net_base,
            tstop=70.0,
            n_trials=1,
            record_agg_ica="all",
            record_agg_i_non_specific="all",
            record_agg_i_mem="all",
            record_agg_ina="all",
            record_agg_ik="all",
            record_agg_i_cap="all",
            record_ina_hh2="all",
            record_ik_hh2="all",
            record_ik_kca="all",
            record_ik_km="all",
            record_ica_ca="all",
            record_ica_cat="all",
            record_il_hh2="all",
            record_i_ar="all",
            record_isec="all",
        )

scaling_factor = 3000
for dpl in dpls:
    dpl.scale(scaling_factor)

dpl = dpls[0]
dpl_plot = dpl.plot(
    layer=["L5"],
)

# %% ######################################################################

if "net_rec_soma" not in locals():
    net_rec_soma = jones_2009_model()
    add_erp_drives_to_jones_model(net_rec_soma)

    with JoblibBackend(8):
        dpls2 = simulate_dipole(
            net_rec_soma,
            tstop=70.0,
            n_trials=1,
            record_agg_ica="soma",
            record_agg_i_non_specific="soma",
            record_agg_i_mem="soma",
            record_agg_ina="soma",
            record_agg_ik="soma",
            record_agg_i_cap="soma",
            record_ina_hh2="soma",
            record_ik_hh2="soma",
            record_ik_kca="soma",
            record_ik_km="soma",
            record_ica_ca="soma",
            record_ica_cat="soma",
            record_il_hh2="soma",
            record_i_ar="soma",
            record_isec="soma",
        )

# %%

scaling_factor = 3000
for dpl in dpls2:
    dpl.scale(scaling_factor)

dpl2 = dpls2[0]
dpl_plot2 = dpl2.plot(
    layer=["L5"],
)

# %% ------------------------------------------------------------
# test network with only drives
# ---------------------------------------------------------------


def init_net_no_local_one_drive(
    verbose=False,
    plot=False,
    tstop=70.0,
    record_method="all",
):
    net_no_local_one_drive = jones_2009_model()

    weights_ampa_p1 = {
        "L2_basket": 0.08831,
        "L2_pyramidal": 0.01525,
        "L5_basket": 0.19934,
        "L5_pyramidal": 0.00865,
    }
    synaptic_delays_prox = {
        "L2_basket": 0.1,
        "L2_pyramidal": 0.1,
        "L5_basket": 1.0,
        "L5_pyramidal": 1.0,
    }

    # all NMDA weights are zero; pass None explicitly
    net_no_local_one_drive.add_evoked_drive(
        "evprox1",
        mu=26.61,
        sigma=2.47,
        numspikes=1,
        weights_ampa=weights_ampa_p1,
        weights_nmda=None,
        location="proximal",
        synaptic_delays=synaptic_delays_prox,
        event_seed=544,
    )

    # print all original connections before removing connections
    if verbose:
        print("--- ORIGINAL CONNECTIVITY ---")
        for conn in net_no_local_one_drive.connectivity:
            print(
                f"Source: {conn['src_type']:<15} -> Target: {conn['target_type']:<15}"
                f" via {conn['receptor']}"
            )

    # keep only connections whose source is NOT a local cell, effectively
    # removing the entire local network
    local_types = ["L2_basket", "L2_pyramidal", "L5_basket", "L5_pyramidal"]

    stripped_connectivity = [
        conn
        for conn in net_no_local_one_drive.connectivity
        if conn["src_type"] not in local_types
    ]

    net_no_local_one_drive.connectivity = stripped_connectivity

    if verbose:
        # print updated connectivity to verify only drives remain
        print("\n--- UPDATED CONNECTIVITY ---")
        for conn in net_no_local_one_drive.connectivity:
            print(
                f"Source: {conn['src_type']:<15} -> Target: {conn['target_type']:<15}"
                f" via {conn['receptor']}"
            )

    # run simulation
    with JoblibBackend(8):
        dpls = simulate_dipole(
            net_no_local_one_drive,
            tstop=tstop,
            n_trials=1,
            record_agg_ica=f"{record_method}",
            record_agg_hh2=f"{record_method}",
            record_agg_i_non_specific=f"{record_method}",
            record_agg_i_mem=f"{record_method}",
            record_agg_ina=f"{record_method}",
            record_agg_ik=f"{record_method}",
            record_agg_i_cap=f"{record_method}",
            record_ina_hh2=f"{record_method}",
            record_ik_hh2=f"{record_method}",
            record_ik_kca=f"{record_method}",
            record_ik_km=f"{record_method}",
            record_ica_ca=f"{record_method}",
            record_ica_cat=f"{record_method}",
            record_il_hh2=f"{record_method}",
            record_i_ar=f"{record_method}",
            record_isec=f"{record_method}",
            record_prec_i_cap=f"{record_method}",
        )

    scaling_factor = 3000
    for dpl in dpls:
        dpl.scale(scaling_factor)

    dpl = dpls[0]

    if plot:
        _ = dpl.plot(
            layer=["L5"],
        )

    return net_no_local_one_drive


if "net_no_local_one_drive" not in locals():
    net_no_local_one_drive = init_net_no_local_one_drive()


# %% [markdown] ###########################################################
## Function to recreate dipole from transmembrane currents
# %% ######################################################################


def postproc_tm_currents(
    net,
    trial=0,
    cell_type="L5_pyramidal",
    scaling_factor=3000,
    from_components=False,
):
    """
    Function for processing transmembrane currents to recreate the dipole moment
    calculated from the axial currents in hnn_core. This can be done from either the
    total recorded transmembrane current, or from the constituent components.

    Note: isec (the transmembrane synaptic current) is part of the total transmembrane
    current, but is *not* aggregated with the other component channel currents
    in this function. This is due to the fact that isec contains *section-specific*
    currents (since synapses are placed at the section midpoint), as opposed to
    *segment-specfic* currents, which are required for this method of recreating
    the dipole.

    Parameters
    ----------
    net : Network object
        The network object containing the simulation data.
    trial : int
        The index of the trial to use
    cell_type : str
        The cell type to process
    scaling_factor : float
        The scaling factor to apply to the dipole
    from_components : bool
        if True, use agg_i_mem to reproduce the dipole. if False, use the component
        currents for either L5_pyramidal or L2_pyramidal

    Returns
    -------
    dipole : np.ndarray
        The reconstructed dipole moment from transmembrane currents.
    """

    load_custom_mechanisms()

    # initialize variable to hold dipole data
    dipole = None

    # build a template cell to get "metadata" for sections
    template_cell = pyramidal(cell_name=cell_type)
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

    if not from_components:
        all_tm_channels = ["agg_i_mem"]
    else:
        if cell_type == "L5_pyramidal":
            all_tm_channels = [
                "agg_i_cap",
                "ina_hh2",
                "ik_hh2",
                "ik_kca",
                "ik_km",
                "ica_ca",
                "ica_cat",
                "il_hh2",
                "i_ar",
            ]
        elif cell_type == "L2_pyramidal":
            all_tm_channels = [
                "agg_i_cap",
                "ina_hh2",
                "ik_hh2",
                "ik_km",
                "il_hh2",
            ]
        else:
            raise ValueError(
                f"Valid channels types for {cell_type} are not known.\n"
                "Please pass the channels types as a list of str to tm_channels"
            )

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
            ch: net.cell_response.transmembrane_currents[ch][trial][gid]
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

            for pos, seg_key in zip(
                seg_positions,
                cell_channels[first_key][sec_name].keys(),
            ):
                # convert the normalized position to the absolute position
                # via linear interpolation
                abs_pos = start + pos * (end - start)
                # simplification: we are using the z position only here we only
                # need the vertical component of the dipole momen
                # we do *not* need to do geometric projection (via cos_theta)
                # as we do for the dipole calculation from axial currents
                z_i = abs_pos[2]

                # sum all currents for this segment
                I_t = np.zeros_like(
                    np.array(cell_channels[first_key][sec_name][seg_key])
                )
                for ch in all_tm_channels:
                    # get channel data
                    vec = np.array(cell_channels[ch][sec_name][seg_key])

                    # get segment area and convert from µm^2 to cm^2
                    seg = template_cell._nrn_sections[sec_name](pos)
                    area_um2 = seg.area()  # µm^2
                    area_cm2 = area_um2 * 1e-8  # cm^2

                    if ch == "agg_i_mem":
                        # agg_i_mem is not recorded continuously as a density; it is
                        # recorded after each timestep. Ergo, the units conversion
                        # here is not necessary as the units are already in nA
                        #
                        # multiplying the contribution by zi in um will give us fAm,
                        # so we will later need to divide by 1e6 to convert to nAm
                        I_abs = vec
                    # convert densities (mA/cm^2) to absolute currents (mA)]
                    else:
                        I_abs = vec * area_cm2  # keep as mA

                        # [WIP]
                        # Should I flip sign for the capacitive currents? I *think*
                        # so, but I haven't found direct confirmation of this ...
                        #
                        # I ideally would want to test the sign flip empirically,
                        # and confirm that we can reproduce i_mem by summing up all
                        # of its constituent components. However, we can't get the
                        # per-segment synaptic currents since we model them as point
                        # processes at the midpoint of the section (not the segment);
                        #
                        # Ergo, we are missing the synaptic piece of the total
                        # transmembrane current needed to reproduce i_mem exactly
                        #
                        # Note: isec is the *per-synapse* current, and not the
                        # *per-segment* current that we need
                        if ch == "agg_i_cap":
                            I_abs = I_abs * -1  # flip sign (?)
                        # [end WIP]

                    I_t += I_abs

                # multiple by r_i per Naess 2015 Ch 2 (simplified to zi in this case)
                # for ionic currents, we have 1 mA*um = 1 nAm (correct units)
                # for i_mem, we have nA rather than mA. and 1 nA*um = 1 fAm
                contrib = I_t * z_i

                # for agg_i_mem, divide by 1e6 to convert fAm to nAm
                if not from_components:
                    contrib = contrib / 1e6 * scaling_factor
                else:
                    contrib = contrib * scaling_factor

                if dipole is None:
                    dipole = contrib.copy()
                else:
                    dipole += contrib

    return dipole


# %% [markdown] ###########################################################
## Compare dipoles calculated from axial vs transmembrane currents
# %% ######################################################################


# %% [markdown] ----------------------------------------
### Layer 5
# %% ---------------------------------------------------

fig, ax = plt.subplots(
    nrows=2,
    ncols=1,
    sharex=True,
    figsize=(8, 15),
)

test_imem_L5 = postproc_tm_currents(
    net=net_base,
    from_components=False,
)

ax[1].plot(
    dpl.times,
    test_imem_L5,
)

ax[1].set_ylim(-200, 100)

_ = dpl.plot(
    layer=["L5"],
    ax=ax[0],
)


# %% [markdown] ----------------------------------------
### Layers 2/3
# %% ---------------------------------------------------
test_imem_L2 = postproc_tm_currents(
    net=net_base,
    cell_type="L2_pyramidal",
    from_components=False,
)

fig, ax = plt.subplots(
    nrows=2,
    ncols=1,
    sharex=True,
    figsize=(8, 10),
)

ax[0].plot(
    dpl.times,
    test_imem_L2,
)

ax[0].set_ylim(-30, 50)

_ = dpl.plot(
    layer=["L2"],
    ax=ax[1],
)


# %% [markdown] ###########################################################
## [DEV] Process transmembrane currents for visualization
# %% ######################################################################

# %% [markdown] ----------------------------------------
# For a single trial and current, get recordings by
# section, segment for each cell
# %% ---------------------------------------------------


def agg_transmembrane_segment_recordings_by_celltype(
    net,
    trial_number=0,
    cell_type=[
        "L2_pyramidal",
        "L5_pyramidal",
    ],
    target_channel="i_mem",
):
    # get recordings for the target channel and trial number
    channel_cell_recordings = (
        net.cell_response.transmembrane_currents[target_channel][trial_number]  # noqa: E203,W503
    )

    agg_output = {}
    agg_output[target_channel] = {}

    for type in cell_type:
        # limit scope to gids for the specified cell_type
        gids = list(net.gid_ranges[type])
        channel_celltype_recordings = {
            gid: channel_cell_recordings[gid] for gid in gids
        }

        aggregate = {}

        for gid_data_dict in channel_celltype_recordings.values():
            for section_key, section_data_dict in gid_data_dict.items():
                for segment_key, segment_data in section_data_dict.items():
                    values = np.array(segment_data)
                    if section_key not in aggregate:
                        aggregate[section_key] = {}
                    if segment_key not in aggregate[section_key]:
                        aggregate[section_key][segment_key] = np.zeros_like(values)
                    aggregate[section_key][segment_key] += values

        agg_output[target_channel][type] = aggregate

    return agg_output


ina_hh2_segment_data = agg_transmembrane_segment_recordings_by_celltype(
    net_base,
    trial_number=0,
    target_channel="ina_hh2",
)


# %% [markdown] ----------------------------------------
# For a single trial and current, get recordings by
# section, adding the segments together
# %% ---------------------------------------------------


def agg_transmembrane_section_recordings_by_celltype(
    net,
    trial_number=0,
    cell_type=[
        "L2_pyramidal",
        "L5_pyramidal",
    ],
    target_channel="i_mem",
):
    # get recordings for the target channel and trial number
    channel_cell_recordings = (
        net.cell_response.transmembrane_currents[target_channel][trial_number]  # noqa: E203,W503
    )

    agg_output = {}
    agg_output[target_channel] = {}

    for type in cell_type:
        # limit scope to gids for the specified cell_type
        gids = list(net.gid_ranges[type])
        channel_celltype_recordings = {
            gid: channel_cell_recordings[gid] for gid in gids
        }

        aggregate = {}

        for gid_data_dict in channel_celltype_recordings.values():
            for section_key, section_data_dict in gid_data_dict.items():
                for segment_key, segment_data in section_data_dict.items():
                    values = np.array(segment_data)
                    if section_key not in aggregate:
                        aggregate[section_key] = np.zeros_like(values)
                    aggregate[section_key] += values

        agg_output[target_channel][type] = aggregate

    return agg_output


ina_hh2_section_data = agg_transmembrane_section_recordings_by_celltype(
    net_base,
    trial_number=0,
    target_channel="ina_hh2",
)

# %%


def get_channel_section_data(
    net,
    trial_number=0,
    channels=None,
):
    channel_section_data = {}

    if channels is None:
        channels = list(net.cell_response.transmembrane_currents.keys())

    for channel in channels:
        agg_data_dict = agg_transmembrane_section_recordings_by_celltype(
            net,
            trial_number=trial_number,
            target_channel=channel,
        )
        channel_section_data[channel] = agg_data_dict[channel]

    return channel_section_data


channel_section_data = get_channel_section_data(
    net_base,
    channels=[
        "ina_hh2",
        "ik_hh2",
        "ik_kca",
        "ik_km",
        "ica_ca",
        "ica_cat",
        "il_hh2",
        "i_ar",
    ],
)

# %% [markdown] ###########################################################
## [DEV] Visualize transmembrane currents
# %% ######################################################################

# %% [markdown] ----------------------------------------
# helper functions to plot cell morphology
# %% ---------------------------------------------------


def extract_diameters(
    section_names,
    cell_params,
):
    """
    helper to map section names to their corresponding diameters from
    params_default.py
    """
    extracted = {}
    for name in section_names:
        # adjust the section names to match the default keys
        search_key = name.replace("_", "").lower()

        # look for a key in cell_params that contains search_key and "diam"
        found = False

        for cell_param, cell_val in cell_params.items():
            clean_cell_param = cell_param.lower()

            if search_key in clean_cell_param and "diam" in clean_cell_param:
                extracted[name] = cell_val
                found = True
                break

        if not found:
            raise KeyError(
                f"Could not find diameter for section '{name}' in cell_params"
            )

    return extracted


def calculate_neuron_geometry(
    end_pts,
    gap=0,
    x_offsets=None,
    x_shift=0,
    y_shift=0,
):
    """
    calculate the 3D geometry
    """
    # handle horizontal offsets
    x_offs = (
        x_offsets  # noqa: E203,W503
        if x_offsets
        else {
            k: 0
            for k in end_pts.keys()  # noqa: E203,W503
        }
    )
    z_shift = {k: 0 for k in end_pts.keys()}

    def points_match(p1, p2, rel_tol=1e-5, abs_tol=1e-8):
        return all(
            math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol) for a, b in zip(p1, p2)
        )

    # handle vertical offsets ("gap")

    # sort sections by how close they are to soma
    # determine the order in which vertical gaps are processed, since the gaps need
    # to be accumulated as you move away from the soma towards the ends of the
    # dendrites. E.g.:
    # - "soma" is the parent of "apical_trunk", which is the parent of "apical_1" ...
    sorted_sections = sorted(
        # sections
        end_pts.keys(),
        # absolute value of distance from soma on the Z axis
        key=lambda x: abs(end_pts[x][0][2]),
    )

    for child_section in sorted_sections:
        child_section_start = end_pts[child_section][0]

        # set shift for basal_1, which shoud move in the negative direction along
        # the z axis
        if child_section != "soma" and points_match(
            child_section_start,
            end_pts["soma"][0],
        ):
            # the vertical offset for basal 2/3 should be negative
            z_shift[child_section] = -gap

        # as we move out from the soma, inherit the gap from the parent section
        for parent_section in end_pts.keys():
            parent_section_end = end_pts[parent_section][1]
            if points_match(child_section_start, parent_section_end):
                # for apical_oblique, inherit the parent_section shift, but add 0 gap
                # this keeps it in line with apical_trunk, while allowing apical_1
                # to move higher
                if "oblique" in child_section:
                    z_shift[child_section] = z_shift[parent_section]
                else:
                    current_gap = -gap if "basal" in child_section else gap
                    z_shift[child_section] = z_shift[parent_section] + current_gap
                break

    shifted_coords = {}
    for section, coords in end_pts.items():
        dx = x_offs.get(section, 0)
        dz = z_shift.get(section, 0)
        shifted_coords[section] = {
            "x": [coords[0][0] + dx + x_shift, coords[1][0] + dx + x_shift],
            "y": [coords[0][1], coords[1][1]],
            "z": [coords[0][2] + dz + y_shift, coords[1][2] + dz + y_shift],
        }

    return shifted_coords


def draw_neuron_on_axis(
    ax,
    shifted_coords,
    diameters,
    colors=None,
    width_scale=1.0,
):
    """
    helper to render the neuron from plot_flat_neuron onto a specific axis
    """
    # use normalized diameter to set line width for sections
    diam_values = list(diameters.values())
    d_min, d_max = min(diam_values), max(diam_values)

    # map section diameters to matplotlib line widths
    # scale based on width_scale parameter
    def diam_to_width(d):
        if d_max > d_min:
            # first, normalize diameters into a fixed range [1, 15]
            # - this range is arbitrary based on what looks reasonable in matplotlib
            base_width = 1 + (d - d_min) / (d_max - d_min) * 14
            # second, apply width_scale multiplier
            scaled = base_width * width_scale
            return scaled
        # fallback line width of 5 times the scales
        return 5 * width_scale

    # plot geometry
    for name, coords in shifted_coords.items():
        width = diam_to_width(
            diameters[name],
        )
        color = colors.get(name, "b") if colors else "#425896"

        # "butt" capstyle ensures sections to not overlab
        ax.plot(
            coords["x"],
            coords["z"],
            color=color,
            linewidth=width,
            solid_capstyle="butt",
        )


def plot_flat_neuron(
    end_pts,
    default_cell_params,
    x_offsets=None,
    x_shift=0,
    y_shift=0,
    gap=0,
    colors=None,
    figsize=(6, 12),
    width_scale=1.0,
    ax=None,
    show_labels=True,
    label_fontsize=8,
    legend_fontsize=8,
    label_offsets=None,
):
    # get diameters to set widths
    diameters = extract_diameters(
        end_pts.keys(),
        default_cell_params,
    )

    # get full 3D geometry
    full_coords = calculate_neuron_geometry(
        end_pts=end_pts,
        gap=gap,
        x_offsets=x_offsets,
        x_shift=x_shift,
        y_shift=y_shift,
    )

    # get 2D geometry used for plotting from the 3D geometry
    # note that "z" is the vertical component
    coords = {}
    for name, c in full_coords.items():
        coords[name] = {
            "x": c["x"],
            "z": c["z"],
        }

    # if no axis is provided, create a new figure
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    legend_elements = []

    # helper 2: drawing engine
    draw_neuron_on_axis(
        ax=ax,
        shifted_coords=coords,
        diameters=diameters,
        colors=colors,
        width_scale=width_scale,
    )

    # handle labels and legend
    for name, seg_coords in coords.items():
        x = seg_coords["x"]
        # renaming the vertical component "z" to "y" since we're in 2D
        y = seg_coords["z"]
        color = colors.get(name, "b") if colors else "b"

        # use uniform line thickness in the legend
        if colors and name in colors and show_labels:
            legend_elements.append(
                Line2D(
                    [0],
                    [0],
                    color=color,
                    lw=2,
                    label=name,
                ),
            )

        if show_labels:
            dx, dy = 0, 0
            if label_offsets and name in label_offsets:
                dx, dy = label_offsets[name]

            ax.text(
                x[1] + dx,
                y[1] + dy,
                name,
                fontsize=label_fontsize,
                va="bottom",
                ha="center",
            )

    ax.axis("off")
    if colors and show_labels:
        ax.legend(
            handles=legend_elements,
            loc="upper left",
            fontsize=legend_fontsize,
        )

    # only call if generating a new figure, not if adding to an existing one
    if ax is None:
        plt.show()


def get_celltype_plot_params(
    net,
    cell_type="L5_pyramidal",
    end_pts=None,
):
    default_cell_searchkey = net.cell_types[cell_type]["cell_object"].name
    celltype_sections = list(
        net.cell_types[cell_type]["cell_object"].sections.keys(),
    )

    # reduce _params down to current cell_type only
    default_cell_keymatches = [
        key for key in list(net._params.keys()) if default_cell_searchkey in key
    ]

    cell_default_params = {
        key: value
        for key, value in net._params.items()
        if key in default_cell_keymatches
    }

    if end_pts is None:
        procedural_end_pts = {}
        for section in celltype_sections:
            procedural_end_pts[section] = (
                net.cell_types[cell_type]["cell_object"].sections[section].end_pts
            )
        end_pts = procedural_end_pts

    return end_pts, cell_default_params


# %% ------------------------------------------------------------
# ---------------------------------------------------------------

# default cell params extracted from params_default.py for L5Pyr, but can be grabbed
# from net._params

cell_type = "L5_pyramidal"
default_cell_searchkey = net_base.cell_types[cell_type]["cell_object"].name
celltype_sections = list(
    net_base.cell_types[cell_type]["cell_object"].sections.keys(),
)

# reduce _params down to current cell_type only
default_cell_keymatches = [
    key for key in list(net_base._params.keys()) if default_cell_searchkey in key
]

default_cell_params = {
    key: value
    for key, value in net_base._params.items()
    if key in default_cell_keymatches
}


# values pulled directly from _cell_L5Pyr in cells_default
end_pts = {
    "soma": [[0, 0, 0], [0, 0, 23]],
    "apical_trunk": [[0, 0, 23], [0, 0, 83]],
    "apical_oblique": [[0, 0, 83], [-150, 0, 83]],
    "apical_1": [[0, 0, 83], [0, 0, 483]],
    "apical_2": [[0, 0, 483], [0, 0, 883]],
    "apical_tuft": [[0, 0, 883], [0, 0, 1133]],
    "basal_1": [[0, 0, 0], [0, 0, -50]],
    "basal_2": [[0, 0, -50], [-106, 0, -156]],
    "basal_3": [[0, 0, -50], [106, 0, -156]],
}

# endpoints grabbed from procedurally-generated cell
procedural_end_pts = {}
for section in celltype_sections:
    procedural_end_pts[section] = (
        net_base.cell_types[cell_type]["cell_object"].sections[section].end_pts
    )

x_offsets = {
    "apical_oblique": -5,
    "apical_1": 0,
    "apical_2": 0,
    "apical_tuft": 0,
    "basal_2": -5,
    "basal_3": 5,
}


distinct_colors = {
    "soma": "orange",
    "apical_trunk": "red",
    "apical_oblique": "purple",
    "apical_1": "magenta",
    "apical_2": "pink",
    "apical_tuft": "brown",
    "basal_1": "yellow",
    "basal_2": "olive",
    "basal_3": "darkgreen",
}

blues = {
    "soma": "#4895ef",
    "apical_trunk": "#4cc9f0",
    "apical_oblique": "#caf0f8",
    "apical_1": "#90e0ef",
    "apical_2": "#ade8f4",
    "apical_tuft": "#e0f7fa",
    "basal_1": "#4cc9f0",
    "basal_2": "#90e0ef",
    "basal_3": "#ade8f4",
}

plot_flat_neuron(
    end_pts,
    default_cell_params,
    x_offsets=x_offsets,
    gap=30,
    colors=blues,
    width_scale=2,
)

plot_flat_neuron(
    procedural_end_pts,
    default_cell_params,
    x_offsets=x_offsets,
    gap=0,
    colors=distinct_colors,
    width_scale=2,
)


# %% ------------------------------------------------------------
# ---------------------------------------------------------------

custom_label_offsets = {
    "basal_1": (60, 0),
    "soma": (80, -50),
    "apical_trunk": (90, -80),
    "basal_2": (-10, 70),
    "basal_3": (10, 70),
    "apical_oblique": (0, 20),
}

fig, ax = plt.subplots(figsize=(12, 12), facecolor="none")

l5_endpts, l5_cell_params = get_celltype_plot_params(net_base)
l2_endpts, l2_cell_params = get_celltype_plot_params(
    net_base,
    cell_type="L2_pyramidal",
)

set_gap = 70
set_width_scale = 4
set_legend_fontsize = 12
set_label_fontsize = 16
set_colors = None

x_offsets = {
    "apical_oblique": -10,
    "apical_1": 0,
    "apical_2": 0,
    "apical_tuft": 0,
    "basal_2": -10,
    "basal_3": 10,
}

plot_flat_neuron(
    l5_endpts,
    l5_cell_params,
    x_offsets=x_offsets,
    x_shift=350,
    y_shift=0,
    gap=set_gap,
    width_scale=set_width_scale,
    legend_fontsize=set_legend_fontsize,
    label_fontsize=set_label_fontsize,
    ax=ax,
    colors=set_colors,
    label_offsets=custom_label_offsets,
)

plot_flat_neuron(
    l2_endpts,
    l2_cell_params,
    x_offsets=x_offsets,
    x_shift=0,
    y_shift=1350,
    gap=set_gap,
    width_scale=set_width_scale,
    legend_fontsize=set_legend_fontsize,
    label_fontsize=set_label_fontsize,
    ax=ax,
    colors=set_colors,
    label_offsets=custom_label_offsets,
)

ax.autoscale_view()

# plt.savefig("~/Desktop/neurons.png", transparent=True, dpi=300)


# %% [markdown] ###########################################################
## [DEV] 3D Visualization
# %% ######################################################################
def draw_neuron_3d(
    ax,
    shifted_coords,
    diameters,
    colors=None,
    width_scale=1.0,
    shade=True,
):
    """
    draw the neuron in 3D using smoothed cylindrical surfaces
    """
    # number of "planes" used to form each cylinder
    # increase for a smoother object
    resolution = 25

    # iterate through segments
    for name, coords in shifted_coords.items():
        # define start and end vectors for the current segment
        start_pt = np.array(
            [
                coords["x"][0],
                coords["y"][0],
                coords["z"][0],
            ]
        )
        end_pt = np.array(
            [
                coords["x"][1],
                coords["y"][1],
                coords["z"][1],
            ]
        )

        # get the segment vector (directional) and its length (magnitude only)
        branch_vec = end_pt - start_pt
        length = np.linalg.norm(branch_vec)

        # failsafe to skip segments with no length
        if length == 0:
            continue

        # get the cylinder radius
        radius = (diameters[name] / 2.0) * width_scale

        # define the cylinder's longitudinal profile (z) and its thickness (r)
        #  - for z_steps, we start at the base, then stay at base for drawing the
        #    "cap", we move to length, then stay at the length for drawing the "cap"
        #  - for r_steps, we start at center, then expand to radius, we stay at radius
        #    while we move to length, then return to the center to create the cap
        z_steps = np.array([0, 0, length, length])
        r_steps = np.array([0, radius, radius, 0])

        # get the angles around the circle for rotation (i.e., 0 to 360 degrees)
        theta = np.linspace(0, 2 * np.pi, resolution)

        # generate coordinates for the cylinder circumference
        theta_grid, z_idx = np.meshgrid(theta, np.arange(len(z_steps)))

        # store the radius for every point on the cylinder surface
        r_grid = r_steps[z_idx]

        # transform r_grid into 3D coordinates
        x_circle = r_grid * np.cos(theta_grid)
        y_circle = r_grid * np.sin(theta_grid)

        # store the height for every point on the cylinder surface
        # does not requite 3D coordinates
        z_grid = z_steps[z_idx]

        # we need to normalize the segment vector by dividing it by its own length
        # this gives us the "pure" direction, which we will use to build our rotation
        # vectors (side_vec and up_vec)
        # if we used the un-normalized vector, the rotation math would scale the
        # cylinder's thickness or stretch the mesh in weird ways
        direction = branch_vec / length

        # rotate and translate the cylinder to match the segment vector
        if np.allclose(direction, [0, 0, 1]) or np.allclose(direction, [0, 0, -1]):
            # handle segments already aligned with the z-axis where no rotation
            # is needed
            x_final = start_pt[0] + x_circle
            y_final = start_pt[1] + y_circle
            z_final = start_pt[2] + (z_grid if direction[2] > 0 else -z_grid)
        else:
            # when rotation is needed, find a "side" vector perpendicular to the branch
            ref_vec = (
                np.array([1, 0, 0]) if abs(direction[0]) < 0.9 else np.array([0, 1, 0])
            )
            side_vec = np.cross(ref_vec, direction)
            side_vec /= np.linalg.norm(side_vec)

            # find an "up" vector perpendicular to both the branch and the side
            up_vec = np.cross(direction, side_vec)

            # transform raw coordinates into the new 3D basis oriented from the
            # start_pt toward the end_pt
            x_final = (
                start_pt[0]
                + direction[0] * z_grid
                + side_vec[0] * x_circle
                + up_vec[0] * y_circle
            )
            y_final = (
                start_pt[1]
                + direction[1] * z_grid
                + side_vec[1] * x_circle
                + up_vec[1] * y_circle
            )
            z_final = (
                start_pt[2]
                + direction[2] * z_grid
                + side_vec[2] * x_circle
                + up_vec[2] * y_circle
            )

        color = colors.get(name, "b") if colors else "b"
        ax.plot_surface(
            x_final,
            y_final,
            z_final,
            color=color,
            linewidth=0,
            antialiased=False,
            shade=shade,
            alpha=1.0,
        )


def plot_3d_neuron(
    end_pts,
    default_cell_params,
    x_offsets=None,
    gap=0,
    colors=None,
    shade=True,
    figsize=(10, 10),
    width_scale=1.0,
    ax=None,
    show_labels=True,
    show_section_labels=False,
):
    diameters = extract_diameters(
        end_pts.keys(),
        default_cell_params,
    )

    coords = calculate_neuron_geometry(
        end_pts=end_pts,
        gap=gap,
        x_offsets=x_offsets,
    )

    if ax is None:
        fig = plt.figure(
            figsize=figsize,
        )
        ax = fig.add_subplot(
            111,
            projection="3d",
        )

    draw_neuron_3d(
        ax=ax,
        shifted_coords=coords,
        diameters=diameters,
        colors=colors,
        width_scale=width_scale,
        shade=shade,
    )

    # handle legend / labels
    legend_elements = []
    for name, seg_coords in coords.items():
        color = (
            colors.get(
                name,
                "b",
            )
            if colors
            else "b"
        )

        # optionally add section labels in the 3D space
        if show_section_labels:
            ax.text(
                seg_coords["x"][1],
                seg_coords["y"][1],
                seg_coords["z"][1],
                name,
                fontsize=8,
            )

        if colors and name in colors:
            legend_elements.append(
                Line2D(
                    [0],
                    [0],
                    color=color,
                    lw=2,
                    label=name,
                ),
            )

    # force equal axis scaling to prevent neurons from looking stretched
    # ------------------------------
    # matplotlib stretches whatever ranges we give it to fill a square area in 3d
    # space. so we need to ensure that the "span" is the same for each axis,
    # otherwise axes may be stretched to fill the cube, distorting the shapes

    # get the ranges
    x_lim = ax.get_xlim3d()
    y_lim = ax.get_ylim3d()
    z_lim = ax.get_zlim3d()

    # get the widest range, divide by two since we'll center around the midpoint
    half_max_range = (
        max(
            np.diff(x_lim),
            np.diff(y_lim),
            np.diff(z_lim),
        )[0]
        / 2.0
    )

    # get the midpoint for each range for centering
    mid_x = np.median(x_lim)
    mid_y = np.median(y_lim)
    mid_z = np.median(z_lim)

    # center around the midpoint using half of the maximum range
    ax.set_xlim3d(mid_x - half_max_range, mid_x + half_max_range)
    ax.set_ylim3d(mid_y - half_max_range, mid_y + half_max_range)
    ax.set_zlim3d(mid_z - half_max_range, mid_z + half_max_range)

    ax.set_xlabel("x (µm)")
    ax.set_ylabel("y (µm)")
    ax.set_zlabel("z (µm)")
    ax.set_box_aspect((1, 1, 1))

    # adjust style of the grid
    ax.xaxis._axinfo["grid"]["linewidth"] = 0.5
    ax.yaxis._axinfo["grid"]["linewidth"] = 0.5
    ax.zaxis._axinfo["grid"]["linewidth"] = 0.5
    ax.xaxis._axinfo["grid"]["color"] = (0.5, 0.5, 0.5, 0.1)
    ax.yaxis._axinfo["grid"]["color"] = (0.5, 0.5, 0.5, 0.1)
    ax.zaxis._axinfo["grid"]["color"] = (0.5, 0.5, 0.5, 0.1)

    if colors and show_labels:
        unique_handles = []
        seen = set()
        for h in legend_elements:
            if h.get_label() not in seen:
                unique_handles.append(h)
                seen.add(h.get_label())
        ax.legend(
            handles=unique_handles,
            loc="upper left",
            fontsize=8,
        )

    return ax


# %%
# get_ipython().run_line_magic("matplotlib", "tk")

add_gap = False

if add_gap:
    gap = 20
    x_offsets = {
        "apical_oblique": -20,
        "apical_1": 0,
        "apical_2": 0,
        "apical_tuft": 0,
        "basal_2": -20,
        "basal_3": 20,
    }
else:
    gap = 0
    x_offsets = {
        "apical_oblique": 0,
        "apical_1": 0,
        "apical_2": 0,
        "apical_tuft": 0,
        "basal_2": 0,
        "basal_3": 0,
    }

ax_main = plot_3d_neuron(
    end_pts,
    default_cell_params,
    x_offsets=x_offsets,
    gap=gap,
    colors=distinct_colors,
    shade=True,
    width_scale=2,
    show_section_labels=False,
)

# plt.show()

# %%
# reset to inline
get_ipython().run_line_magic("matplotlib", "inline")

# %% [markdown] ----------------------------------------
# plotting functions for segment-specific recordings
# %% ---------------------------------------------------


def plot_segment_recordings_by_section(
    section_name,
    single_channel_data,
    cell_type="L5_pyramidal",
    overwrite_channel_name=False,
):
    channel_name = list(single_channel_data.keys())[0]
    channel_cell_data = single_channel_data[channel_name][cell_type]

    if section_name not in list(channel_cell_data.keys()):
        raise ValueError(
            f"Section '{section_name}' not in data dictionary for cell type "
            f"'{cell_type}' and channel '{channel_name}'"
        )

    segment_data_dict = channel_cell_data[section_name]

    fig, ax = plt.subplots(
        nrows=len(segment_data_dict),
        ncols=1,
        sharex=True,
        figsize=(8, 3 * len(segment_data_dict)),
    )

    # make ax always a list
    if len(segment_data_dict) == 1:
        ax = [ax]

    # np.inf guarantees the values will be replaced on the first iteration
    y_min = np.inf
    y_max = -np.inf
    for i, (segment_key, segment_data) in enumerate(segment_data_dict.items()):
        ax[i].plot(segment_data)
        ax[i].set_title(f"{segment_key.replace('seg_', 'Segment ')}")
        y_min = min(y_min, segment_data.min())
        y_max = max(y_max, segment_data.max())

    # set consistent y axes, with extra padding
    padding = 0.05 * (y_max - y_min)
    y_limits = (y_min - padding, y_max + padding)

    for axis in ax:
        axis.set_ylim(y_limits)

    if overwrite_channel_name:
        channel_name = overwrite_channel_name

    section_title = section_name.replace("_", " ").title()
    celltype_title = cell_type.replace("_", " ").title()

    fig.suptitle(
        f"Transmembrane Recordings for {celltype_title}\n"
        f"{section_title} of {channel_name}",
        fontsize=16,
    )
    plt.xlabel("Time (ms)")
    plt.tight_layout()
    plt.close(fig)

    return fig


l5_seg_fig = plot_segment_recordings_by_section(
    section_name="apical_trunk",
    single_channel_data=ina_hh2_segment_data,
    cell_type="L5_pyramidal",
    overwrite_channel_name="Na+ HH2",
)

l2_seg_fig = plot_segment_recordings_by_section(
    section_name="apical_trunk",
    single_channel_data=ina_hh2_segment_data,
    cell_type="L2_pyramidal",
    overwrite_channel_name="Na+ HH2",
)

# %% [markdown] ----------------------------------------
# plotting functions for section recordings
# %% ---------------------------------------------------


def plot_single_channel_by_section_celltype(
    single_channel_data,
    end_pts=None,
    default_cell_params=None,
    cell_type="L5_pyramidal",
    overwrite_channel_name=False,
    show_neuron_previews=False,
):
    # top to bottom section order
    section_plot_order = [
        "apical_tuft",
        "apical_2",
        "apical_1",
        "apical_trunk",
        "apical_oblique",
        "soma",
        "basal_1",
        "basal_2",
        "basal_3",
    ]

    channel_name = list(single_channel_data.keys())[0]
    channel_cell_data = single_channel_data[channel_name][cell_type]
    cell_sections = list(channel_cell_data.keys())

    # determine grid dimensions
    num_cols = 2 if show_neuron_previews else 1
    width_ratios = [1, 4] if show_neuron_previews else [1]

    fig, axes = plt.subplots(
        nrows=len(cell_sections),
        ncols=num_cols,
        sharex="col",
        figsize=(
            10 if show_neuron_previews else 8,
            3 * len(cell_sections),
        ),
        gridspec_kw={"width_ratios": width_ratios},
    )

    # make axes always 2D
    if len(cell_sections) == 1:
        axes = np.expand_dims(axes, axis=0)
    if num_cols == 1:
        axes = np.expand_dims(axes, axis=-1)

    # np.inf guarantees the values will be replaced on the first iteration
    y_min = np.inf
    y_max = -np.inf
    for section_key, section_data in channel_cell_data.items():
        y_min = min(y_min, section_data.min())
        y_max = max(y_max, section_data.max())

    # set consistent y axes, with extra padding
    padding = 0.05 * (y_max - y_min)
    y_limits = (y_min - padding, y_max + padding)

    x_offsets = {
        "apical_oblique": -10,
        "basal_2": -10,
        "basal_3": 10,
    }

    for section_key, section_data in channel_cell_data.items():
        # get index from section_plot_order
        i = section_plot_order.index(section_key)

        # optionally plot neuron morphology
        if show_neuron_previews and end_pts and default_cell_params:
            neuron_colors = {k: "lightgrey" for k in end_pts.keys()}
            neuron_colors[section_key] = "#004a9e"

            plot_flat_neuron(
                end_pts,
                default_cell_params,
                x_offsets=x_offsets,
                gap=10,
                colors=neuron_colors,
                ax=axes[i, 0],
                show_labels=False,
                width_scale=1.5,
            )

        # plot section recordings
        col_idx = 1 if show_neuron_previews else 0
        axes[i, col_idx].plot(section_data)
        axes[i, col_idx].set_title(
            f"{section_key.replace('_', ' ').title()}",
        )
        axes[i, col_idx].set_ylim(y_limits)

    if overwrite_channel_name:
        channel_name = overwrite_channel_name

    celltype_title = cell_type.replace("_", " ").title()

    fig.suptitle(
        f"Transmembrane Recordings for {celltype_title} of {channel_name}",
        fontsize=16,
    )
    plt.xlabel("Time (ms)")
    # leave space for suptitle
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.close(fig)

    return fig


plot_single_channel_by_section_celltype(
    single_channel_data=ina_hh2_section_data,
    end_pts=end_pts,
    default_cell_params=default_cell_params,
    cell_type="L5_pyramidal",
    overwrite_channel_name="Na+ HH2",
    show_neuron_previews=True,
)

# %%


def plot_channels_by_single_section(
    channel_section_data,
    times,
    section_key,
    end_pts=None,
    default_cell_params=None,
    cell_type="L5_pyramidal",
    show_neuron_preview=True,
):
    # get the channels for the selected cell_type
    channel_names = list(channel_section_data.keys())

    # get the grid dimensions
    num_channels = len(channel_names)
    num_cols = 2 if show_neuron_preview else 1
    start_col = 1 if show_neuron_preview else 0

    width_ratios = [1, 3] if show_neuron_preview else [1]

    fig, axes = plt.subplots(
        nrows=num_channels,
        ncols=num_cols,
        sharex=False,
        figsize=(
            16,
            4 * num_channels,
        ),
        dpi=300,
        gridspec_kw={"width_ratios": width_ratios},
    )

    # ensure axes is 2D for consistent indexing
    if num_channels == 1:
        axes = np.expand_dims(axes, axis=0)
    if num_cols == 1:
        axes = np.expand_dims(axes, axis=-1)

    # share x axis for the data column
    # if num_channels > 1:
    #     for r in range(num_channels - 1):
    #         axes[r, start_col].sharex(axes[num_channels - 1, start_col])
    #         plt.setp(axes[r, start_col].get_xticklabels(), visible=False)

    # calculate max absolute value for all channels in the section
    section_max = 0
    for chan in channel_names:
        section_data_tmp = channel_section_data[chan][cell_type][section_key]
        section_max = max(
            section_max,
            np.abs(section_data_tmp).max(),
        )

    if section_max == 0:
        section_max = 1e-9  # prevent divide by zero
    padding_section = 1.1 * section_max

    for i, chan in enumerate(channel_names):
        # plot neuron preview in the first column for every row
        if show_neuron_preview and end_pts and default_cell_params:
            neuron_colors = {k: "lightgrey" for k in end_pts.keys()}
            neuron_colors[section_key] = "#004a9e"

            plot_flat_neuron(
                end_pts,
                default_cell_params,
                x_offsets={
                    "apical_oblique": -10,
                    "basal_2": -10,
                    "basal_3": 10,
                },
                gap=10,
                colors=neuron_colors,
                ax=axes[i, 0],
                show_labels=False,
                width_scale=1.5,
            )
            axes[i, 0].set_ylabel(
                section_key.replace("_", " ").title(),
                rotation=0,
                ha="right",
                va="center",
                labelpad=15,
            )

        ax_local = axes[i, start_col]
        section_data = channel_section_data[chan][cell_type][section_key]

        # normally-scaled plot
        abs_max = np.abs(section_data).max()
        if abs_max == 0:
            abs_max = 1e-9  # prevent divide by zero
        padding_local = 1.1 * abs_max

        lns = []

        ln1 = ax_local.plot(
            times,
            section_data,
            color="grey",
            label="Scaled to Self",
        )
        ax_local.set_ylim(
            -padding_local,
            padding_local,
        )
        ax_local.axhline(
            0,
            color="grey",
            linestyle="--",
            alpha=0.3,
        )

        ax_local.set_title(
            f'Channel: "{chan}"',
        )
        lns += ln1

        # secondary axis for section scaling
        ax_secondary = ax_local.twinx()
        ln2 = ax_secondary.plot(
            times,
            section_data,
            color="#004a9e",
            label="Scaled to Section",
        )
        ax_secondary.set_ylim(
            -padding_section,
            padding_section,
        )
        ax_secondary.tick_params(
            axis="y",
            labelcolor="#004a9e",
        )
        lns += ln2

        # add legends
        labs = [lab.get_label() for lab in lns]
        ax_local.legend(
            lns,
            labs,
            loc="upper left",
            fontsize=12,
        )

        # only add x-label to the bottom subplot
        if i == num_channels - 1:
            ax_local.set_xlabel("Time (ms)")

    fig.suptitle(
        f"Transmembrane Current Recordings for {cell_type.replace('_', ' ').title()} "
        f"{section_key.replace('_', ' ').title()}",
        y=0.98,
        fontsize=18,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.close(fig)

    return fig


plot_channels_by_single_section(
    channel_section_data,
    dpl.times,
    section_key="soma",
    end_pts=end_pts,
    default_cell_params=default_cell_params,
    cell_type="L5_pyramidal",
    show_neuron_preview=True,
)


# %%


def plot_overlay_channels_by_section_celltype(
    channel_section_data,
    times,
    end_pts=None,
    default_cell_params=None,
    cell_type="L5_pyramidal",
    show_neuron_previews=False,
):
    # top to bottom section order
    section_plot_order = [
        "apical_tuft",
        "apical_2",
        "apical_1",
        "apical_trunk",
        "apical_oblique",
        "soma",
        "basal_1",
        "basal_2",
        "basal_3",
    ]

    # get the channels and sections for the selected cell_type
    channel_names = list(channel_section_data.keys())
    first_channel = channel_names[0]
    cell_sections = list(channel_section_data[first_channel][cell_type].keys())

    # get the grid dimensions
    num_cols = 2 if show_neuron_previews else 1
    width_ratios = [1, 4] if show_neuron_previews else [1]

    fig, axes = plt.subplots(
        nrows=len(cell_sections),
        ncols=num_cols,
        sharex="col",
        figsize=(
            10 if show_neuron_previews else 8,
            3 * len(cell_sections),
        ),
        gridspec_kw={"width_ratios": width_ratios},
    )

    # make axes always 2D
    if len(cell_sections) == 1:
        axes = np.expand_dims(axes, axis=0)
    if num_cols == 1:
        axes = np.expand_dims(axes, axis=-1)

    # get the y-limits across all channels / sections
    y_min = np.inf
    y_max = -np.inf
    for chan in channel_names:
        chan_cell_data = channel_section_data[chan][cell_type]
        for section_key, section_data in chan_cell_data.items():
            y_min = min(y_min, section_data.min())
            y_max = max(y_max, section_data.max())

    padding = 0.05 * (y_max - y_min)
    y_limits = (y_min - padding, y_max + padding)

    x_offsets = {
        "apical_oblique": -10,
        "basal_2": -10,
        "basal_3": 10,
    }

    for section_key in cell_sections:
        # get index from section_plot_order
        i = section_plot_order.index(section_key)

        # optionally plot neuron morphology
        if show_neuron_previews and end_pts and default_cell_params:
            neuron_colors = {k: "lightgrey" for k in end_pts.keys()}
            neuron_colors[section_key] = "#004a9e"

            plot_flat_neuron(
                end_pts,
                default_cell_params,
                x_offsets=x_offsets,
                gap=10,
                colors=neuron_colors,
                ax=axes[i, 0],
                show_labels=False,
                width_scale=1.5,
            )

        # plot overlaid channel recordings
        col_idx = 1 if show_neuron_previews else 0

        for chan in channel_names:
            section_data = channel_section_data[chan][cell_type][section_key]
            # use the times vector for the x-axis
            axes[i, col_idx].plot(times, section_data, label=chan)

        axes[i, col_idx].set_title(
            f"{section_key.replace('_', ' ').title()}",
        )
        axes[i, col_idx].set_ylim(y_limits)

        # place legend outside and to the right of each plot
        axes[i, col_idx].legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize=8)

    celltype_title = cell_type.replace("_", " ").title()
    fig.suptitle(
        f"Transmembrane Current Recordings for {celltype_title}",
        fontsize=16,
    )

    plt.xlabel("Time (ms)")
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.close(fig)

    return fig


plot_overlay_channels_by_section_celltype(
    channel_section_data,
    times=net_base.cell_response.times,
    end_pts=end_pts,
    default_cell_params=default_cell_params,
    cell_type="L5_pyramidal",
    show_neuron_previews=True,
)


# %%


def plot_distinct_channels_by_section_celltype(
    channel_section_data,
    times,
    end_pts=None,
    default_cell_params=None,
    cell_type="L5_pyramidal",
    show_neuron_previews=False,
    scaling_type=None,
    cell_sections=None,
):
    # top to bottom section order for full neuron plot
    section_plot_order = [
        "apical_tuft",
        "apical_2",
        "apical_1",
        "apical_trunk",
        "apical_oblique",
        "soma",
        "basal_1",
        "basal_2",
        "basal_3",
    ]

    # get the channels and sections for the selected cell_type
    channel_names = list(channel_section_data.keys())
    first_channel = channel_names[0]

    if cell_sections is None:
        cell_sections = list(channel_section_data[first_channel][cell_type].keys())

    # update section plot order based on actual cell_sections used
    section_plot_order = [sec for sec in section_plot_order if sec in cell_sections]

    # get the grid dimensions
    num_channels = len(channel_names)
    start_col = 1 if show_neuron_previews else 0
    total_cols = num_channels + start_col

    width_ratios = ([1] if show_neuron_previews else []) + [4] * num_channels

    fig, axes = plt.subplots(
        nrows=len(cell_sections),
        ncols=total_cols,
        sharex="col",
        figsize=(
            6 * total_cols,
            6 * len(cell_sections),
        ),
        dpi=300,
        gridspec_kw={"width_ratios": width_ratios},
    )

    if len(cell_sections) == 1:
        axes = np.expand_dims(axes, axis=0)
    if total_cols == 1:
        axes = np.expand_dims(axes, axis=-1)

    # get "global" y limit for scaled axes plots
    global_max = 0
    for chan in channel_names:
        chan_cell_data = channel_section_data[chan][cell_type]
        for section_data in chan_cell_data.values():
            global_max = max(
                global_max,
                np.abs(section_data).max(),
            )

    padding_global = 1.05 * global_max
    global_y_limits = (
        -padding_global,
        padding_global,
    )

    x_offsets = {
        "apical_oblique": -10,
        "basal_2": -10,
        "basal_3": 10,
    }

    for section_key in cell_sections:
        i = section_plot_order.index(section_key)

        # get y limit for all channels in the section
        section_max = 0
        for chan in channel_names:
            section_data_tmp = channel_section_data[chan][cell_type][section_key]
            section_max = max(section_max, np.abs(section_data_tmp).max())

        if section_max == 0:
            section_max = 1e-9  # prevent divide by zero
        padding_section = 1.1 * section_max

        if show_neuron_previews and end_pts and default_cell_params:
            neuron_colors = {k: "lightgrey" for k in end_pts.keys()}
            neuron_colors[section_key] = "#004a9e"
            plot_flat_neuron(
                end_pts,
                default_cell_params,
                x_offsets=x_offsets,
                gap=10,
                colors=neuron_colors,
                ax=axes[i, 0],
                show_labels=False,
                width_scale=1.5,
            )
            axes[i, 0].set_ylabel(
                section_key.replace("_", " ").title(),
                rotation=0,
                ha="right",
                va="center",
                labelpad=15,
            )

        for j, chan in enumerate(channel_names):
            ax_local = axes[i, start_col + j]
            section_data = channel_section_data[chan][cell_type][section_key]

            # normally-scaled plot
            abs_max = np.abs(section_data).max()
            if abs_max == 0:
                abs_max = 1e-9  # prevent divide by zero
            padding_local = 1.1 * abs_max

            lns = []

            ln1 = ax_local.plot(
                times,
                section_data,
                color="grey",
                label="Scaled to Self",
            )
            ax_local.set_ylim(
                -padding_local,
                padding_local,
            )
            ax_local.axhline(
                0,
                color="grey",
                linestyle="--",
                alpha=0.3,
            )
            ax_local.tick_params(
                axis="y",
                labelsize=7,
            )

            ax_local.set_title(
                f'Section: "{section_key}"; Channel: "{chan}"',
                fontsize=10,
                fontweight="bold",
            )
            lns += ln1

            if scaling_type is not None:
                # secondary axis for global/section scaling
                ax_secondary = ax_local.twinx()

                if scaling_type == "section":
                    ln2 = ax_secondary.plot(
                        times,
                        section_data,
                        color="#004a9e",
                        label="Scaled to Section Data",
                    )
                    ax_secondary.set_ylim(-padding_section, padding_section)
                    ax_secondary.tick_params(
                        axis="y",
                        labelcolor="#004a9e",
                        labelsize=7,
                    )
                    lns += ln2

                elif scaling_type == "global":
                    ln2 = ax_secondary.plot(
                        times,
                        section_data,
                        color="#004a9e",
                        label="Scaled to All Data",
                    )
                    ax_secondary.set_ylim(global_y_limits)
                    ax_secondary.tick_params(
                        axis="y",
                        labelcolor="#004a9e",
                        labelsize=7,
                    )
                    lns += ln2

            # add legends
            labs = [lab.get_label() for lab in lns]
            ax_local.legend(lns, labs, loc="upper right", fontsize=7)

            if not show_neuron_previews and j == 0:
                ax_local.set_ylabel(
                    section_key.replace("_", " ").title(),
                )

    celltype_title = cell_type.replace("_", " ").title()

    fig.suptitle(
        f"Transmembrane Current Recordings for {celltype_title}",
        fontsize=50,
        y=0.97,
    )

    plt.xlabel("Time (ms)")

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.close(fig)

    return fig


plot_distinct_channels_by_section_celltype(
    channel_section_data,
    dpl.times,
    end_pts=end_pts,
    default_cell_params=default_cell_params,
    cell_type="L5_pyramidal",
    show_neuron_previews=True,
    scaling_type="section",
    cell_sections=["soma"],
)


# %% [markdown] ###########################################################
## [WIP] Testing and Feature Development
# %% ######################################################################

# %% [markdown] ----------------------------------------
### Recreating dipole for the soma only
# %% ---------------------------------------------------


def postproc_soma_dipole(
    net,
    trial=0,
    cell_type="L5_pyramidal",
    scaling_factor=3000,
    from_components=False,
):
    """ """

    print("Running `soma` function")

    # this function will only handle the "soma", as it's composed of exactly one
    # segment where pos = 0.5
    sec_name = "soma"
    seg_key = "seg_1"
    pos = 0.5

    # load custom mechanisms
    load_custom_mechanisms()

    # initialize variable to hold dipole data
    dipole = None

    # build a template cell to get "metadata" for sections
    template_cell = pyramidal(cell_name=cell_type)
    template_cell.build(sec_name_apical="apical_trunk")

    # get the relative endpoints for the soma
    rel_endpoints = {}
    sec = template_cell._nrn_sections["soma"]
    start = np.array([sec.x3d(0), sec.y3d(0), sec.z3d(0)])
    end = np.array(
        [sec.x3d(sec.n3d() - 1), sec.y3d(sec.n3d() - 1), sec.z3d(sec.n3d() - 1)]
    )
    rel_endpoints[sec_name] = (start, end)

    if not from_components:
        all_tm_channels = ["agg_i_mem"]
    else:
        if cell_type == "L5_pyramidal":
            all_tm_channels = [
                "agg_i_cap",
                "ina_hh2",
                "ik_hh2",
                "ik_kca",
                "ik_km",
                "ica_ca",
                "ica_cat",
                "il_hh2",
                "i_ar",
            ]
        elif cell_type == "L2_pyramidal":
            all_tm_channels = [
                "agg_i_cap",
                "ina_hh2",
                "ik_hh2",
                "ik_km",
                "il_hh2",
            ]
        else:
            raise ValueError(
                f"Valid channels types for {cell_type} are not known.\n"
                "Please pass the channels types as a list of str to tm_channels"
            )

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
            ch: net.cell_response.transmembrane_currents[ch][trial][gid]
            for ch in all_tm_channels
        }

        # get the cell sections to loop over
        # the key used shouldn't matter, but we don't want to hard code it since
        # we can pass different channels to this function, so we get it dynamically
        first_key = list(cell_channels.keys())[0]

        start_rel, end_rel = rel_endpoints[sec_name]
        start = start_rel + soma_pos
        end = end_rel + soma_pos

        abs_pos = start + pos * (end - start)
        z_i = abs_pos[2]

        # sum all currents for this segment
        I_t = np.zeros_like(
            np.array(cell_channels[first_key][sec_name][seg_key]),
        )

        for ch in all_tm_channels:
            # get channel data
            vec = np.array(cell_channels[ch][sec_name][seg_key])

            # get segment area and convert from µm^2 to cm^2
            seg = template_cell._nrn_sections[sec_name](pos)
            area_um2 = seg.area()  # µm^2
            area_cm2 = area_um2 * 1e-8  # cm^2

            if ch == "agg_i_mem":
                # agg_i_mem is not recorded continuously as a density; it is
                # recorded after each timestep. Ergo, the units conversion
                # here is not necessary as the units are already in nA
                #
                # multiplying the contribution by zi in um will give us fAm,
                # so we will later need to divide by 1e6 to convert to nAm
                I_abs = vec
            # convert densities (mA/cm^2) to absolute currents (mA)]
            else:
                I_abs = vec * area_cm2  # keep as mA

            I_t += I_abs

        # arround for different structure for isec when recontructing from components
        if from_components:
            soma_isec = net.cell_response.isec[trial][gid].get(sec_name, {})
            for syn_key in soma_isec:
                # isec is measured in nA, so we need to divide by 1e6 to
                # convert nA to mA before we add to I_t
                I_t += np.array(soma_isec[syn_key]) / 1e6

        # multiple by r_i per Naess 2015 Ch 2 (simplified to zi in this case)
        # for ionic currents, we have 1 mA*um = 1 nAm (correct units)
        # for i_mem, we have nA rather than mA. and 1 nA*um = 1 fAm
        contrib = I_t * z_i

        # for agg_i_mem, divide by 1e6 to convert fAm to nAm
        if not from_components:
            contrib = contrib / 1e6 * scaling_factor
        else:
            contrib = contrib * scaling_factor

        if dipole is None:
            dipole = contrib.copy()
        else:
            dipole += contrib

    return dipole


fig, ax = plt.subplots(
    nrows=2,
    ncols=1,
    sharex=True,
    figsize=(8, 15),
)


test_imem_L5 = postproc_soma_dipole(
    net=net_base,
    from_components=False,
)

ax[0].plot(
    dpl.times[1:],
    test_imem_L5[1:],
)

test_imem_L5 = postproc_soma_dipole(
    net=net_base,
    from_components=True,
)

ax[1].plot(
    dpl.times[1:],
    test_imem_L5[1:],
)

ax[0].set_ylim(-200, 100)
ax[1].set_ylim(-200, 100)

ax[0].set_title("Soma dipole from i_mem")
ax[1].set_title("Soma dipole from components")

# %%


def check_rmse_and_residuals(
    net,
    trial=0,
    cell_type="L5_pyramidal",
):
    times = net.cell_response.times

    # get dipole from i_mem
    dpl_imem = postproc_soma_dipole(
        net,
        trial=trial,
        cell_type=cell_type,
        from_components=False,
    )
    # get the dipole reconstructed from the constituent components
    dpl_comp = postproc_soma_dipole(
        net,
        trial=trial,
        cell_type=cell_type,
        from_components=True,
    )

    if dpl_imem is None or dpl_comp is None:
        raise ValueError(
            "Dipole data could not be computed",
        )

    # calculate residual and rmse
    residual = dpl_imem - dpl_comp
    rmse = np.sqrt(np.mean(residual**2))

    # normalize by the peak-to-peak range of agg_i_mem (our "ground truth")
    dpl_range = np.max(dpl_imem) - np.min(dpl_imem)
    nrmse_pct = (rmse / dpl_range) * 100 if dpl_range != 0 else 0

    fig, ax = plt.subplots(
        2,
        1,
        figsize=(10, 10),
        sharex=True,
    )

    # overlay plot
    ax[0].plot(
        times[1:],
        dpl_imem[1:],
        label="From agg_i_mem",
        alpha=0.8,
    )
    ax[0].plot(
        times[1:],
        dpl_comp[1:],
        label="From components",
        linestyle="--",
        alpha=0.8,
    )

    # text box for rmse
    error_text = f" RMSE: {rmse:.2f}\nNRMSE:  {nrmse_pct:.2f}%"
    ax[0].text(
        x=0.05,
        y=0.95,
        s=error_text,
        transform=ax[0].transAxes,
        verticalalignment="top",
        fontsize=12,
        fontfamily="Menlo",
        bbox=dict(
            boxstyle="round",
            facecolor="white",
            alpha=0.2,
        ),
    )

    ax[0].set_title(
        f"Computed Dipole Comparison for {cell_type.replace('_', ' ').title()} Soma",
    )
    ax[0].set_ylabel("nAm")
    ax[0].grid(True, alpha=0.3)
    ax[0].legend(loc="upper right")

    ymin, ymax = ax[0].get_ylim()
    ylim = (max(abs(ymin), abs(ymax))) * 1.05
    ax[0].set_ylim(-ylim, ylim)

    # residual dipole plot
    ax[1].plot(
        times[1:],
        residual[1:],
        color="red",
        label="(i_mem - reconstructed), scaled",
    )
    ax[1].set_title(
        "Residual (Error)",
    )
    ax[1].set_ylabel("nAm")
    ax[1].set_ylim(-ylim, ylim)
    ax[1].grid(True, alpha=0.3)
    # ax[1].legend()

    rightax = ax[1].twinx()
    rightax.plot(
        times[1:],
        residual[1:],
        alpha=0.2,
        label="(i_mem - reconstructed), zoomed",
    )
    ymin, ymax = rightax.get_ylim()
    ylim = max(abs(ymin), abs(ymax))
    rightax.set_ylim(-ylim, ylim)

    lines1, labels1 = ax[1].get_legend_handles_labels()
    lines2, labels2 = rightax.get_legend_handles_labels()

    ax[1].legend(lines1 + lines2, labels1 + labels2)

    plt.tight_layout()
    plt.show()


check_rmse_and_residuals(net_base)


# %% [markdown] ----------------------------------------
### Next steps
# %% ---------------------------------------------------

"""
# For feature example (code contribution):
[x] Show recording of transmembrane currents individually; there is some baseline code
    for this that I need to migrate to this repository
[-] Likely remove the dipole reconstruction from_components for this PR; it's not
  strictly necessary and not complete
[x] Dipole reconstruction from i_mem (total transmembrane current) can be kept in the
    example notebook for this PR, as it is complete and shows off new use cases. The
    function herein should be adapted to remove the from_components section, which is
    still a work in progress

# For dipole reconstruction (science contribution):
[x] add isec to soma calculation function, which should be allowed since soma has only
    one segment, and isec should be the transmembrane synaptic current component that
    is missing
[x] run function with isec, see if dipole reproduction matches
[ ] potentially try reconstruction for one cell only to interrogate discrepancies
[ ] don't need vars exposed here:
      - net.cell_types["L5_pyramidal"]["cell_object"].to_dict()

# Notes on outcome of dipole reconstruction
- The normalized RMSE (NRMSE) in the example above is 1.21%
- Over 98.7% of the variance in the i_mem dipole is accounted for by our reconstruction
  from summing over the individual current components
- The remaining 1.21% of unexplained variance is likely due to noise from small timing
  offsets in how NEURON updates versus records variables, or from floating-point
  (rounding) differences that compound when summing over many arrays
  - Re: NEURON timing offsets: Some currents (e.g., ina) are updated at the beginning
    of a timestep, while others (e.g., agg_i_cap) depend on the voltage change across
    the timestep. agg_i_mem captures the exact timing used by the solver, whereas
    the sum in our reconstructed dipole might be slightly "out of phase" at times due
    to how the currents are updated
"""

# %% [markdown] ----------------------------------------
## Bonus content
# %% [markdown] ----------------------------------------

# %% [markdown] ----------------------------------------
### generate .glb object with blender api
# %% ---------------------------------------------------
# Notes:
# - ".glb" is a standardized format for 3D models
# - files can be openened in free model viewers such as
#   https://modelviewer.dev/editor/


def generate_blender_neuron(
    end_pts,
    default_cell_params,
    x_offsets=None,
    gap=0,
    colors=None,
    scale_factor=0.01,
    width_scale=2.0,
):
    """ """
    diameters = extract_diameters(
        end_pts.keys(),
        default_cell_params,
    )
    shifted_coords = calculate_neuron_geometry(
        end_pts=end_pts,
        gap=gap,
        x_offsets=x_offsets,
    )

    if bpy.context.object and bpy.context.object.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()

    for name, coords in shifted_coords.items():
        # raw coordinates
        p0_raw = np.array([coords["x"][0], coords["y"][0], coords["z"][0]])
        p1_raw = np.array([coords["x"][1], coords["y"][1], coords["z"][1]])

        # vectors
        vec_raw = p1_raw - p0_raw
        length_raw = np.linalg.norm(vec_raw)

        # skip iteration when possible divide-by-zero situation occurs
        if length_raw < 1e-6:
            continue

        # get radius with width scaling
        radius_raw = (diameters[name] / 2.0) * width_scale

        # apply global scale_factor for Blender
        p0 = p0_raw * scale_factor
        p1 = p1_raw * scale_factor
        length = length_raw * scale_factor
        radius = radius_raw * scale_factor
        midpoint = (p0 + p1) / 2.0

        bpy.ops.mesh.primitive_cylinder_add(
            radius=radius,
            depth=length,
            location=midpoint,
            end_fill_type="NGON",
        )

        obj = bpy.context.active_object
        obj.name = f"Section_{name}"
        obj.scale = (1.0, 1.0, 1.0)

        # align direction for vectors
        direction = vec_raw / length_raw
        v_orig = np.array([0, 0, 1])
        axis = np.cross(v_orig, direction)
        axis_norm = np.linalg.norm(axis)

        if axis_norm < 1e-6:
            if np.dot(v_orig, direction) < 0:
                obj.rotation_euler = (np.pi, 0, 0)
        else:
            angle = np.arccos(np.clip(np.dot(v_orig, direction), -1.0, 1.0))
            obj.rotation_mode = "AXIS_ANGLE"
            obj.rotation_axis_angle = (
                angle,
                axis[0],
                axis[1],
                axis[2],
            )

        if colors and name in colors:
            mat = bpy.data.materials.get(f"Mat_{name}") or bpy.data.materials.new(
                name=f"Mat_{name}",
            )
            if not obj.data.materials:
                obj.data.materials.append(mat)


generate_blender_neuron(
    end_pts=procedural_end_pts,
    default_cell_params=default_cell_params,
    x_offsets=x_offsets,
    gap=0,
    colors=distinct_colors,
    scale_factor=0.01,
    width_scale=2.0,
)

# export to .glb file
bpy.ops.export_scene.gltf(
    filepath="3D_L5_PN.glb",
    export_format="GLB",
    use_selection=False,  # save every object, not just those selected
)

# %% [markdown] ############################################################
# # [DEV] Exploring dipole discrepancy
# ##########################################################################
#
# overview:
#   - our goal is to recreate the axial dipole computed by hnn-core from the
#     individual (synaptic + ionic + capacitive) transmembrane currents
#   - we can tell NEURON to track the aggregate transmembrane currents
#     ("agg_i_mem") during simultion, and we can use this current recording
#     to recreate the axial dipole shape exactly (there is an "offset" /
#     vertical shift in the plot, but otherwise the shapes match perfectly)
#
# current issue:
#   - there is a discrepancy between the dipole computed from agg_i_mem
#     versus the dipole computed from the constituent currents
#   - since the agg_i_mem dipole matches the axial dipole computed by
#     hnn-core, we need to understand and resolve the discrepancy between
#     agg_i_mem and the constituent currents
#   - of note, the dipoles computed from agg_i_mem versus the constituent
#     currents *does* match perfectly when only looking at the soma, which
#     is composed of a single segment
#   - the fact that the some dipoles match has several possible
#     interpretations:
#       - there is a constituent current missing from the dendrites that
#         is *not*  present in the soma that causes the dipoles for the
#         dendrites to diverge
#       - the simplification of synaptic currents being at the midpoint
#         of the section does not work for multi-segment sections when
#         calculating the dipole, and thus we need segment-specific
#         synaptic currents to correctly reproduce agg_i_mem from the
#         constituent currents
#
# things to check:
#   - [x] check that the soma dipoles match for i_mem versus components
#   - [ ] check that the dendrite dipoles all match for i_mem versus components
#   - [x] check that unit match is correct
#       - notes
#   - [x] check that area math (area_cm2 calculation) is correct
#       - since the aggregates currents (agg_ina, agg_ik) exactly match
#         the sum of the mechanisms (e.g., hh2, kca, km for agg_ik), the
#         area_cm2 calculation is correct
#   - [ ] check that the residuals are zero for Na, K, Ca
#       - [x] Na residuals are 0
#       - [x] K residuals are 0
#       - [ ] Code not presently set up to record agg_ca

# %% --------------------------------------------------
# test visualization
# -----------------------------------------------------

# notes:
#   - function for visualizing dipoles computed from different
#     methods (functions) and under different conditions


def compare_imem_to_components(
    net,
    cell_type="L5_pyramidal",
    sections=None,
):
    """ """

    fig, ax = plt.subplots(
        nrows=2,
        ncols=1,
        sharex=True,
        figsize=(12, 8),
    )

    # generate data
    test_imem = postproc_dipole_from_components(
        net=net,
        from_components=False,
        cell_type=cell_type,
        sections=sections,
    )

    test_comp = postproc_dipole_from_components(
        net=net,
        from_components=True,
        cell_type=cell_type,
        sections=sections,
    )

    # plotting
    times = net.cell_response.times
    ax[0].plot(times[1:], test_imem[1:])
    ax[1].plot(times[1:], test_comp[1:])

    # synchronize scales using data from [1:]
    combined_data = np.concatenate([test_imem[1:], test_comp[1:]])
    y_min, y_max = np.min(combined_data), np.max(combined_data)
    padding = 0.05 * (y_max - y_min)

    for axis in ax:
        axis.set_ylim(y_min - padding, y_max + padding)
        axis.grid(True, alpha=0.3)
        axis.set_ylabel("nAm")

    if sections is None:
        sub_title = "\nSections: all"
    else:
        sub_title = f"\nSections: {sections}"

    ax[0].set_title(f"dipole from imem ({cell_type}){sub_title}")
    ax[1].set_title(f"dipole from components ({cell_type}){sub_title}")
    ax[1].set_xlabel("Time (ms)")

    plt.tight_layout()
    return fig


def dev_postproc_func_test(
    net,
    cell_type="L5_pyramidal",
):
    fig, ax = plt.subplots(
        nrows=4,
        ncols=1,
        sharex=True,
        figsize=(8, 15),
    )

    # net_rec_soma (record="soma") figures
    # ------------------------------

    # from imem
    test_imem_L5 = postproc_soma_dipole(
        net=net_rec_soma,
        from_components=False,
        cell_type=cell_type,
    )

    ax[0].plot(
        dpl.times[1:],
        test_imem_L5[1:],
    )

    # from components
    test_imem_L5 = postproc_dipole_from_components(
        net=net_rec_soma,
        from_components=True,
        cell_type=cell_type,
    )

    ax[1].plot(
        dpl.times,
        test_imem_L5,
    )

    ax[0].set_title('Soma dipole from imem when record="soma"')
    ax[1].set_title('Soma dipole from components when record="soma"')

    ax[0].set_ylim(-10, 80)
    ax[1].set_ylim(-10, 80)

    # net (record="all") figures
    # ------------------------------

    # from imem
    test_imem_L5 = postproc_dipole_from_components(
        net=net,
        from_components=False,
        cell_type=cell_type,
    )

    ax[2].plot(
        dpl.times[1:],
        test_imem_L5[1:],
    )

    # from components
    test_imem_L5 = postproc_dipole_from_components(
        net=net,
        from_components=True,
        cell_type=cell_type,
    )

    ax[3].plot(
        dpl.times,
        test_imem_L5,
    )

    ax[2].set_title('Agg dipole from imem when record="all"')
    ax[3].set_title('Agg dipole from components when record="all"')

    ax[2].set_ylim(-200, 200)
    ax[3].set_ylim(-200, 200)

    return fig


# %% --------------------------------------------------
# [DEV] baseline function
# -----------------------------------------------------

# notes:
#   - baseline function for computing dipole from imem OR from the
#     constituent currents
#   - this function adds isec recording at the section midpoint


def postproc_dipole_from_components(
    net,
    trial=0,
    cell_type="L5_pyramidal",
    scaling_factor=3000,
    from_components=False,
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # [new]
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    sections=None,
    # [end new]
):
    """ """

    print("Running `baseline v3` function")

    load_custom_mechanisms()

    # initialize variable to hold dipole data
    dipole = None

    # build a template cell to get "metadata" for sections
    template_cell = pyramidal(cell_name=cell_type)
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

    if not from_components:
        all_tm_channels = ["agg_i_mem"]
    else:
        if cell_type == "L5_pyramidal":
            all_tm_channels = [
                "agg_i_cap",
                # "agg_ica",
                # "agg_i_non_specific",
                # "agg_ik",
                # "agg_ina",
                "ina_hh2",
                "ik_hh2",
                "ik_kca",
                "ik_km",
                "ica_ca",
                "ica_cat",
                "il_hh2",
                "i_ar",
            ]
        elif cell_type == "L2_pyramidal":
            all_tm_channels = [
                "agg_i_cap",
                "agg_ik",
                "agg_ina",
                # "ina_hh2",
                # "ik_hh2",
                # "ik_km",
                # "il_hh2",
            ]
        else:
            raise ValueError(
                f"Valid channels types for {cell_type} are not known.\n"
                "Please pass the channels types as a list of str to tm_channels"
            )

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
            ch: net.cell_response.transmembrane_currents[ch][trial][gid]
            for ch in all_tm_channels
        }

        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # [new]
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # first_key = list(cell_channels.keys())[0]
        first_key = next(iter(cell_channels))

        if isinstance(sections, list):
            cell_sections = sections
        else:
            # get the cell sections to loop over
            # the key used shouldn't matter, but we don't want to hard code it since
            # we can pass different channels to this function, so we get it dynamically
            cell_sections = list(cell_channels[first_key].keys())
        # [end new]

        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # [new] Access synaptic data for this specific GID
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        cell_syn_data = net.cell_response.isec[trial][gid]
        # [end new]

        for sec_name in cell_sections:
            # offset the start/end positions by the realized soma position for this
            # cell instantiation
            start_rel, end_rel = rel_endpoints[sec_name]
            start = start_rel + soma_pos
            end = end_rel + soma_pos

            # get the normalized segment positions along the cell section
            nseg = len(cell_channels[first_key][sec_name])

            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            # [new]
            # notes:
            #   - no effect
            # template_cell._nrn_sections[sec_name].nseg = nseg
            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

            seg_positions = [(i - 0.5) / nseg for i in range(1, nseg + 1)]

            for pos, seg_key in zip(
                seg_positions,
                cell_channels[first_key][sec_name].keys(),
            ):
                # convert the normalized position to the absolute position
                # via linear interpolation
                abs_pos = start + pos * (end - start)
                # simplification: we are using the z position only here we only
                # need the vertical component of the dipole momen
                # we do *not* need to do geometric projection (via cos_theta)
                # as we do for the dipole calculation from axial currents
                z_i = abs_pos[2]

                # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                # [new]
                # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                # notes:
                #   - no effect compared to z_i = abs_pos[2]
                # z_i = (start_rel + pos * (end_rel - start_rel))[2]
                # [end new]

                # sum all currents for this segment
                I_t = np.zeros_like(
                    np.array(cell_channels[first_key][sec_name][seg_key])
                )
                for ch in all_tm_channels:
                    # get channel data
                    vec = np.array(cell_channels[ch][sec_name][seg_key])

                    # get segment area and convert from µm^2 to cm^2
                    seg = template_cell._nrn_sections[sec_name](pos)
                    area_um2 = seg.area()  # µm^2
                    area_cm2 = area_um2 * 1e-8  # cm^2

                    if ch == "agg_i_mem":
                        # agg_i_mem is not recorded continuously as a density; it is
                        # recorded after each timestep. Ergo, the units conversion
                        # here is not necessary as the units are already in nA
                        #
                        # multiplying the contribution by zi in um will give us fAm,
                        # so we will later need to divide by 1e6 to convert to nAm
                        I_abs = vec
                    # convert densities (mA/cm^2) to absolute currents (mA)]
                    elif ch == "agg_i_cap":
                        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                        # [new]
                        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                        # notes:
                        #   - FIX: i_cap is uA/cm2, others are mA/cm2.
                        #   - Divide by 1000 to convert uA to mA.
                        #   - this does NOT work
                        # I_abs = (vec / 1000.0) * area_cm2
                        I_abs = -vec * area_cm2  # keep as mA

                    else:
                        I_abs = vec * area_cm2  # keep as mA

                    I_t += I_abs

                # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                # [new]
                # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                # implement isec recording only at the section midpoint (0.5)
                # to prevent overcounting as synapses are point-processes [2]
                if from_components and np.isclose(pos, 0.5):
                    if sec_name in cell_syn_data:
                        for receptor_vec in cell_syn_data[sec_name].values():
                            # convert nA to mA to match ionic units
                            I_t += np.array(receptor_vec) * 1e-6
                # [end new]

                # multiple by r_i per Naess 2015 Ch 2 (simplified to zi in this case)
                # for ionic currents, we have 1 mA*um = 1 nAm (correct units)
                # for i_mem, we have nA rather than mA. and 1 nA*um = 1 fAm
                contrib = I_t * z_i

                # for agg_i_mem, divide by 1e6 to convert fAm to nAm
                if not from_components:
                    contrib = contrib / 1e6 * scaling_factor
                else:
                    contrib = contrib * scaling_factor

                if dipole is None:
                    dipole = contrib.copy()
                else:
                    dipole += contrib

    return dipole


net = net_base
baseline_fig = dev_postproc_func_test(
    net,
    cell_type="L5_pyramidal",
)

for net in [net_base, net_no_local_one_drive]:
    _ = compare_imem_to_components(
        net,
        sections=["soma"],
    )


# [DEV] end baseline
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# %% --------------------------------------------------
# [DEV] test_01
# -----------------------------------------------------

# notes:
#   - this is essentially a Google Gemini refactor of the "baseline" function
#     above used for testing / iteration
#   - this version yields the same results, but it is a bit more organazied and thus
#     more readable; consider adapting this version in vF


def postproc_dipole_from_components_v1(  # noqa
    net,
    trial=0,
    cell_type="L5_pyramidal",
    scaling_factor=3000,
    from_components=False,
):
    """ """

    print("Running `test_01` function")

    load_custom_mechanisms()

    # Build a template cell
    template_cell = pyramidal(cell_name=cell_type)
    template_cell.build(sec_name_apical="apical_trunk")

    # Get relative endpoints for each section
    rel_endpoints = {}
    for sec_name, sec in template_cell._nrn_sections.items():
        start = np.array([sec.x3d(0), sec.y3d(0), sec.z3d(0)])
        end = np.array(
            [sec.x3d(sec.n3d() - 1), sec.y3d(sec.n3d() - 1), sec.z3d(sec.n3d() - 1)],
        )
        rel_endpoints[sec_name] = (start, end)

    # Define which channels to aggregate
    if not from_components:
        all_tm_channels = ["agg_i_mem"]
    else:
        if cell_type == "L5_pyramidal":
            all_tm_channels = [
                "agg_i_cap",
                "ina_hh2",
                "ik_hh2",
                "ik_kca",
                "ik_km",
                "ica_ca",
                "ica_cat",
                "il_hh2",
                "i_ar",
            ]
        elif cell_type == "L2_pyramidal":
            all_tm_channels = ["agg_i_cap", "ina_hh2", "ik_hh2", "ik_km", "il_hh2"]
        else:
            raise ValueError(f"Unknown channel list for {cell_type}")

    dipole = None

    # Loop through GIDs for the specific cell type
    for gid in net.gid_ranges[cell_type]:
        start_index = net.gid_ranges[cell_type][0]
        soma_pos = np.array(net.pos_dict[cell_type][gid - start_index])

        # Access transmembrane and synaptic recordings for this GID
        cell_tm_data = {
            ch: net.cell_response.transmembrane_currents[ch][trial][gid]
            for ch in all_tm_channels
        }
        cell_syn_data = net.cell_response.isec[trial][gid]

        # Use the first recorded channel to determine the sections present
        first_ch = all_tm_channels[0]
        for sec_name in cell_tm_data[first_ch].keys():
            start_rel, end_rel = rel_endpoints[sec_name]
            start = start_rel + soma_pos
            end = end_rel + soma_pos

            # --- 1. Process Segment-based Currents (Ionic, Capacitive, or agg_i_mem)
            nseg = len(cell_tm_data[first_ch][sec_name])
            seg_indices = list(cell_tm_data[first_ch][sec_name].keys())

            # Segment centers are at (i - 0.5) / nseg
            for i, seg_key in enumerate(seg_indices):
                pos = (i + 0.5) / nseg
                abs_pos = start + pos * (end - start)
                z_i = abs_pos[2]  # Vertical component

                for ch in all_tm_channels:
                    vec = np.array(cell_tm_data[ch][sec_name][seg_key])

                    if dipole is None:
                        dipole = np.zeros_like(vec)

                    if ch == "agg_i_mem":
                        # agg_i_mem is recorded in nA.
                        # (nA * um) / 1e6 = nAm
                        dipole += (vec * z_i) / 1e6 * scaling_factor
                    else:
                        # Densities (mA/cm^2) must be converted to absolute current (mA)
                        # mA * um = nAm (no 1e6 division needed)
                        seg_nrn = template_cell._nrn_sections[sec_name](pos)
                        area_cm2 = seg_nrn.area() * 1e-8

                        I_abs = vec * area_cm2
                        dipole += (I_abs * z_i) * scaling_factor

            # --- 2. Process Section-based Point Currents (isec) ---
            # Synapses are modeled as point processes at the section midpoint (0.5)
            if from_components and sec_name in cell_syn_data:
                mid_pos = start + 0.5 * (end - start)
                z_mid = mid_pos[2]

                for receptor_vec in cell_syn_data[sec_name].values():
                    # isec is recorded in nA.
                    # (nA * um) / 1e6 = nAm
                    I_syn_nA = np.array(receptor_vec)
                    dipole += (I_syn_nA * z_mid) / 1e6 * scaling_factor

    return dipole


# test_01_fig = dev_postproc_func_test()

# [DEV] end test_01
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# %% --------------------------------------------------
# [DEV] test_02
# -----------------------------------------------------

# notes:
#   - simulation scaling applied to the template
#   - synced nseg so area() is segment-specific
#   - add isec
#   - this version yields the same results as `baseline`


def postproc_dipole_from_components_v2(  # noqa
    net,
    trial=0,
    cell_type="L5_pyramidal",
    scaling_factor=3000,
    from_components=False,
):
    print("Running `test_02` function")

    load_custom_mechanisms()

    # Build a template cell
    template_cell = pyramidal(cell_name=cell_type)
    template_cell.build(sec_name_apical="apical_trunk")

    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # [new] apply simulation scaling to the template
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    p_prefix = "L5Pyr" if "L5" in cell_type else "L2Pyr"
    rel_endpoints = {}
    for sec_name, sec in template_cell._nrn_sections.items():
        # Get scaled dimensions from net._params
        new_L = net._params.get(f"{p_prefix}_{sec_name}_L", sec.L)
        sec.diam = net._params.get(f"{p_prefix}_{sec_name}_diam", sec.diam)

        start = np.array([sec.x3d(0), sec.y3d(0), sec.z3d(0)])
        end_orig = np.array(
            [sec.x3d(sec.n3d() - 1), sec.y3d(sec.n3d() - 1), sec.z3d(sec.n3d() - 1)]
        )

        # Adjust endpoint and internal Length to match simulation scaling
        if sec.L > 0:
            end = start + (end_orig - start) * (new_L / sec.L)
            sec.L = new_L
        else:
            end = end_orig
        rel_endpoints[sec_name] = (start, end)
    # [end new]

    # Define which channels to aggregate
    if not from_components:
        all_tm_channels = ["agg_i_mem"]
    else:
        if cell_type == "L5_pyramidal":
            all_tm_channels = [
                "agg_i_cap",
                "ina_hh2",
                "ik_hh2",
                "ik_kca",
                "ik_km",
                "ica_ca",
                "ica_cat",
                "il_hh2",
                "i_ar",
            ]
        elif cell_type == "L2_pyramidal":
            all_tm_channels = ["agg_i_cap", "ina_hh2", "ik_hh2", "ik_km", "il_hh2"]
        else:
            raise ValueError(f"Unknown channel list for {cell_type}")

    dipole = None

    for gid in net.gid_ranges[cell_type]:
        start_index = net.gid_ranges[cell_type][0]
        soma_pos = np.array(net.pos_dict[cell_type][gid - start_index])

        cell_tm_data = {
            ch: net.cell_response.transmembrane_currents[ch][trial][gid]
            for ch in all_tm_channels
        }
        cell_syn_data = net.cell_response.isec[trial][gid]

        first_ch = all_tm_channels[0]
        for sec_name in cell_tm_data[first_ch].keys():
            start_rel, end_rel = rel_endpoints[sec_name]
            start = start_rel + soma_pos
            end = end_rel + soma_pos

            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            # [new] Sync nseg so area() is segment-specific
            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            nseg = len(cell_tm_data[first_ch][sec_name])
            template_cell._nrn_sections[sec_name].nseg = nseg
            # [end new]

            seg_indices = list(cell_tm_data[first_ch][sec_name].keys())
            for i, seg_key in enumerate(seg_indices):
                pos = (i + 0.5) / nseg
                abs_pos = start + pos * (end - start)
                z_i = abs_pos[2]

                for ch in all_tm_channels:
                    vec = np.array(cell_tm_data[ch][sec_name][seg_key])

                    if dipole is None:
                        dipole = np.zeros_like(vec)

                    if ch == "agg_i_mem":
                        dipole += (vec * z_i) / 1e6 * scaling_factor
                    else:
                        seg_nrn = template_cell._nrn_sections[sec_name](pos)
                        area_cm2 = seg_nrn.area() * 1e-8
                        I_abs = vec * area_cm2
                        dipole += (I_abs * z_i) * scaling_factor

            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            # [new] add isec once per section
            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            # This ensures synaptic currents are included even if nseg is even
            if from_components and sec_name in cell_syn_data:
                mid_pos = start + 0.5 * (end - start)
                z_mid = mid_pos[2]
                for receptor_vec in cell_syn_data[sec_name].values():
                    I_syn_nA = np.array(receptor_vec)
                    dipole += (I_syn_nA * z_mid) / 1e6 * scaling_factor
            # [end new]

    return dipole


# test_02_fig = dev_postproc_func_test()


# %% [markdown] --------------------------------------------------------
# # Diagnostics
# ----------------------------------------------------------------------


# %% ------------------------------------------------------------
# validate correct area
# ---------------------------------------------------------------


def val_area(net, channel="agg_i_mem"):
    # access simulation metadata for a dendrite
    cell_meta = net.cell_types["L5_pyramidal"]["cell_object"]
    sec_meta = cell_meta.sections["apical_trunk"]

    # build current template

    template_cell = pyramidal(cell_name="L5_pyramidal")
    template_cell.build(sec_name_apical="apical_trunk")
    sec_temp = template_cell._nrn_sections["apical_trunk"]

    print("--- GEOMETRY CHECK ---")
    print(f"Simulation 'apical_trunk': L={sec_meta.L}, diam={sec_meta.diam}")
    print(f"Template   'apical_trunk': L={sec_temp.L}, diam={sec_temp.diam}")

    print("\n--- AREA CALCULATION CHECK ---")
    # get nseg from recorded data
    # first_ch = list(net.cell_response.transmembrane_currents.keys())[0]
    nseg_sim = len(
        net.cell_response.transmembrane_currents[channel][0][
            net.gid_ranges["L5_pyramidal"][0]
        ]["apical_trunk"]
    )

    # what the current function calculates as the area for one segment:
    area_reconstructed = sec_temp(0.5).area()

    # what the area of one segment should be (total area / nseg):
    total_area_sim = np.pi * sec_meta.diam * sec_meta.L
    area_expected = total_area_sim / nseg_sim

    print(f"Simulation nseg for this section: {nseg_sim}")
    print(f" Area used in the reconstruction: {area_reconstructed:.2f} um2")
    print(f"       Expected area per segment: {area_expected:.2f} um2")
    print(
        f"               Difference Factor: {area_reconstructed / area_expected:.2f}x"
    )


net = net_base

val_area(net)


# %% ------------------------------------------------------------
# check for i_pass
# ---------------------------------------------------------------


"""
It is documented in the foundational science for HNN, specifically the Jones et al.
2009 paper (Quantitative analysis and biophysically realistic neural modeling of the
MEG mu rhythm).[1]
In that paper, the methodology states:
"The dendrites were passive with a membrane resistance of 30,000
Ω
Ω
 cm²..."
In the NEURON simulator (which hnn-core uses), "passive" means the section has the pas
mechanism inserted. This mechanism produces the i_pas current variable. By contrast,
the soma is "active" because it has the hh2 mechanism inserted to generate action
potentials. The hh2 mechanism includes its own internal leak term, which is recorded as
il_hh2.
Why Row 4 was failing (The Missing Leak)
This distinction is the "smoking gun" for why your Row 4 diverged from Row 3:
agg_i_mem (Row 3): This is a "catch-all" variable. It records every single charge
crossing the membrane, including the current from pas mechanisms and hh2 mechanisms. It
is correct by default.
Components (Row 4): You were only summing il_hh2.
In the Soma, hh2 is present, so il_hh2 represents the leak. Row 2 matched Row 1.
In the Dendrites, hh2 is not present, so il_hh2 is zero. The leak in the dendrites is
actually i_pas. Because you weren't including i_pas in your sum, you were effectively
modeling a cell with "non-leaky" dendrites.
"""

for sec_name in ["soma", "apical_1", "apical_2", "basal_1"]:
    # Find the HOC section
    hoc_sec = next((s for s in h.allsec() if sec_name in s.name()), None)
    if hoc_sec:
        # ismembrane checks if the mechanism is inserted
        has_pas = h.ismembrane("pas", sec=hoc_sec)
        has_hh2 = h.ismembrane("hh2", sec=hoc_sec)
        print(
            f"Section {sec_name:<10} | Has pas: {bool(has_pas):<5}"
            f" | Has hh2: {bool(has_hh2)}"
        )

# %% ------------------------------------------------------------
# check the ionic current residuals
# ---------------------------------------------------------------


def check_ion_residuals(net):
    gid = net.gid_ranges["L5_pyramidal"][0]
    sec = "apical_2"
    seg = "seg_1"
    trial = 0

    # get the agg currents (our "ground truth" in this case)
    agg_ina = np.array(
        net.cell_response.transmembrane_currents["agg_ina"][trial][gid][sec][seg]
    )
    agg_ik = np.array(
        net.cell_response.transmembrane_currents["agg_ik"][trial][gid][sec][seg]
    )
    imem = np.array(
        net.cell_response.transmembrane_currents["agg_i_mem"][trial][gid][sec][seg]
    )
    icap_dens = np.array(
        net.cell_response.transmembrane_currents["agg_i_cap"][trial][gid][sec][seg]
    )

    # get the component (mechanism-specific) currents
    ina_hh2 = np.array(
        net.cell_response.transmembrane_currents["ina_hh2"][trial][gid][sec][seg]
    )
    ik_hh2 = np.array(
        net.cell_response.transmembrane_currents["ik_hh2"][trial][gid][sec][seg]
    )
    ik_kca = np.array(
        net.cell_response.transmembrane_currents["ik_kca"][trial][gid][sec][seg]
    )
    ik_km = np.array(
        net.cell_response.transmembrane_currents["ik_km"][trial][gid][sec][seg]
    )

    # calcute the differentials
    #   - if these are ~not~ zero, a mechanism is writing to out agg currents that
    #     isn't accounted for
    na_residual = agg_ina - ina_hh2
    k_residual = agg_ik - (ik_hh2 + ik_kca + ik_km)

    # calculate total currents in nA
    #   - agg_i_mem = i_cap + i_na + i_k + i_ca + i_non_specific
    template_cell = pyramidal(cell_name=cell_type)
    template_cell.build(sec_name_apical="apical_trunk")
    area_cm2 = template_cell._nrn_sections[sec](0.5).area() * 1e-8

    imem_nA = imem
    ina_nA = agg_ina * area_cm2 * 1e6
    ik_nA = agg_ik * area_cm2 * 1e6
    icap_nA = icap_dens * area_cm2 * 1e6

    # calculate the remainder
    #   - if the Na and K residuals are zero, then the remainder must be
    #     accounted for by one of the other currents (ica_ca, ica_cat, il_hh2,
    #     and i_ar)
    remainder_nA = imem_nA - (icap_nA + ina_nA + ik_nA)

    print(f"--- Residuals Analysis for {sec} ---")
    print(f"   Sodium Residual:  {np.mean(na_residual):.8e} mA/cm2")
    print(f"Potassium Residual:  {np.mean(k_residual):.8e} mA/cm2")
    print(f"   Total Remainder:  {np.mean(remainder_nA):.8e} nA")


net = net_base

check_ion_residuals(net)

# %% ------------------------------------------------------------
# check all point processes for L5_pyramidal
# ---------------------------------------------------------------

print(f"{'HOC Section Name':<30} | {'Object Type':<15} | {'Current (i)':<10}")
print("-" * 60)

# Iterate through every segment of every section in NEURON
for sec in h.allsec():
    # Only look at the first L5 pyramidal cell
    if "L5Pyr" in sec.name():
        for seg in sec:
            # Look at all point processes on this segment
            for pp in seg.point_processes():
                # We are looking for things that have a current 'i'
                # (Synapses, TonicBias, etc.)
                if hasattr(pp, "i"):
                    # Get the class name (e.g., 'NMDA', 'Exp2Syn')
                    obj_type = pp.hname().split("[")[0]
                    print(f"{sec.name():<30} | {obj_type:<15} | {pp.i:.4e}")

# %% ------------------------------------------------------------
# check point processes in a section
# ---------------------------------------------------------------


def check_sectin_pp(net):
    # Fixed NameError: access via net metadata
    cell_type = "L5_pyramidal"
    cell_obj = net.cell_types[cell_type]["cell_object"]
    sec = "apical_2"

    print(f"--- Point Processes on {sec} ---")
    # This shows every point process (Synapses, IClamps, TonicBias)
    for syn in cell_obj.sections[sec].syns:
        # Check the name of the HOC object (e.g., 'Exp2Syn', 'TonicBias')
        print(f" - {syn}")

    # Now check what is actually inside your isec recording for that section
    gid = net.gid_ranges[cell_type][0]
    trial = 0
    recorded_syns = list(net.cell_response.isec[trial][gid].get(sec, {}).keys())
    print(f"\nReceptors recorded in isec for {sec}:")
    print(recorded_syns)

    # Find the apical_2 section
    target_sec = next((s for s in h.allsec() if sec in s.name()), None)

    if target_sec:
        print(f"\n--- Synapse Placement on {target_sec.name()} ---")
        # Iterate through all segments to see where point processes are attached
        for seg in target_sec:
            pps = seg.point_processes()
            if pps:
                print(f"\nSegment at x={seg.x}:")
                for pp in pps:
                    # Get the class name and any current variable
                    obj_name = pp.hname().split("[")[0]
                    current_val = getattr(pp, "i", "No i attribute")
                    print(f"  - {obj_name} (Current: {current_val})")
    else:
        print("Dendrite section not found.")


net = net_base

check_sectin_pp(net)


# %%
# %% --------------------------------------------------
# [DEV] residual analysis function
# -----------------------------------------------------

# notes:
#   - function for validating the reconstruction of the dipole moment
#     from individual components vs the 'agg_i_mem' ground truth
#   - goal is to prove: Sum(Components) == agg_i_mem
#   - dipole math magnifies segment-level errors via the vertical lever arm (zi)
#
# testing for:
#   - sign conventions: check for inverted synaptic pulses
#   - missing mechanisms: check for DC offsets (leaks, h-currents)
#   - synaptic collisions: check for shape changes from overwriting dictionary keys
#   - capacitive jitter: check for high-frequency noise during spikes (agg_i_cap)


def dipole_residual_analysis(
    net,
    cell_type="L5_pyramidal",
    sec="apical_2",
    seg="seg_1",
):
    """
    Performs the segment-level balance check exactly as used in previous
    diagnostic steps to compare agg_i_mem against reconstructed components.
    """
    import numpy as np

    # 1. Setup identifiers
    gid = net.gid_ranges[cell_type][0]
    trial = 0

    # 2. Get the actual segment area from metadata
    cell_obj = net.cell_types[cell_type]["cell_object"]
    sec_meta = cell_obj.sections[sec]

    # Formula for segment area: (PI * diam * L) / nseg
    # We use nseg from the recorded data length to be 100% sure
    nseg_sim = len(
        net.cell_response.transmembrane_currents["agg_i_mem"][trial][gid][sec]
    )
    area_um2 = (np.pi * sec_meta.diam * sec_meta.L) / nseg_sim
    scale = area_um2 * 0.01  # Converts mA/cm2 to nA

    # 3. Ground Truth Total Membrane Current (nA)
    imem_nA = np.array(
        net.cell_response.transmembrane_currents["agg_i_mem"][trial][gid][sec][seg]
    )

    # 4. Aggregates (Truth for these species)
    icap_nA = (
        np.array(
            net.cell_response.transmembrane_currents["agg_i_cap"][trial][gid][sec][seg]
        )
        * scale
    )
    ina_nA = (
        np.array(
            net.cell_response.transmembrane_currents["agg_ina"][trial][gid][sec][seg]
        )
        * scale
    )
    ik_nA = (
        np.array(
            net.cell_response.transmembrane_currents["agg_ik"][trial][gid][sec][seg]
        )
        * scale
    )
    ica_nA = (
        np.array(
            net.cell_response.transmembrane_currents["agg_ica"][trial][gid][sec][seg]
        )
        * scale
    )

    # 5. Nonspecific Mechanism Components (nA)
    # Summing the ones we know write to "i" or "il"
    inonspec_mechs_nA = (
        np.array(
            net.cell_response.transmembrane_currents["il_hh2"][trial][gid][sec][seg]
        )
        + np.array(
            net.cell_response.transmembrane_currents["i_ar"][trial][gid][sec][seg]
        )
        + np.array(
            net.cell_response.transmembrane_currents["ica_cat"][trial][gid][sec][seg]
        )
    ) * scale

    # 6. Synaptic Components (nA)
    isec_data = net.cell_response.isec[trial][gid].get(sec, {})
    isec_nA = (
        np.sum([np.array(v) for v in isec_data.values()], axis=0) if isec_data else 0.0
    )

    # --- THE FINAL EQUATION ---
    reconstructed_total_nA = (
        icap_nA + ina_nA + ik_nA + ica_nA + inonspec_mechs_nA + isec_nA
    )
    residual = imem_nA - reconstructed_total_nA

    # Printing means from [1:] to ignore initialization transients
    print(f"--- Full Balance Check for {sec} {seg} ---")
    print(f"agg_i_mem (Truth):      {np.mean(imem_nA[1:]):.8e} nA")
    print(f"Reconstructed Sum:      {np.mean(reconstructed_total_nA[1:]):.8e} nA")
    print(f"Residual Error:         {np.mean(residual[1:]):.8e} nA")

    return residual


def plot_dipole_residual_analysis(
    net,
    cell_type="L5_pyramidal",
):
    """
    Diagnostic to compare agg_i_mem dipole against component reconstruction
    and plot the residual against recorded synaptic currents.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    # assume trial 0 for diagnostics
    trial = 0
    times = net.cell_response.times

    # 1. Recreate the whole-cell dipoles
    # truth: calculated from the 'catch-all' agg_i_mem variable
    dpl_truth = postproc_dipole_from_components(
        net,
        trial=trial,
        cell_type=cell_type,
        from_components=False,
    )
    # sum: calculated by manually adding ions, leak, and synapses
    dpl_sum = postproc_dipole_from_components(
        net,
        trial=trial,
        cell_type=cell_type,
        from_components=True,
    )

    # 2. Calculate the "Missing Waveform" (the residual)
    missing_waveform = dpl_truth - dpl_sum

    # 3. Sum all synaptic recordings from isec for the specified cell_type
    total_isec = np.zeros_like(missing_waveform)
    for gid in net.gid_ranges[cell_type]:
        gid_isec = net.cell_response.isec[trial].get(gid, {})
        for sec in gid_isec:
            for rec_vec in gid_isec[sec].values():
                total_isec += np.array(rec_vec)

    # 4. Plotting
    fig, (ax1, ax2) = plt.subplots(
        nrows=2,
        ncols=1,
        sharex=True,
        figsize=(12, 8),
    )

    # top plot: the error (nAm)
    ax1.plot(
        times[1:],
        missing_waveform[1:],
        label='The "Missing" Part of the Waveform (Truth - Sum)',
        color="tab:blue",
    )
    ax1.set_title(f"Dipole Residual Analysis: {cell_type}")
    ax1.set_ylabel("nAm")
    ax1.axhline(0, color="black", lw=1, alpha=0.5, linestyle="--")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # bottom plot: total synaptic current (nA)
    ax2.plot(
        times,
        total_isec,
        label="Total Recorded Synaptic Current (isec)",
        color="orange",
    )
    ax2.set_xlabel("Time (ms)")
    ax2.set_ylabel("nA")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    return fig


net = net_base
net = net_no_local_one_drive

_ = plot_dipole_residual_analysis(net)


# %%

# %% [markdown] -------------------------------------------------
# notes re: residuals analysis
# ---------------------------------------------------------------

# The fact that the **Sodium and Potassium residuals are exactly 0.0** is an incredibly important result. It definitively proves several things:
# 1.  **Unit Math is Correct:** Your $mA/cm^2 \to nA$ conversion and the $1e6$ scaling are perfect.
# 2.  **Area Math is Correct:** Since the aggregates (`agg_ina/ik`) exactly match the sum of your specific mechanisms (`hh2`, `kca`, `km`), your `area_cm2` calculation is correct for that segment.
# 3.  **Mechanical Coverage is Correct (for Na/K):** There are no "hidden" sodium or potassium currents coming from GHK-based calcium channels or other mechanisms.

# We have successfully narrowed the entire discrepancy down to the **"Other" category** (Calcium, Leak, H-current, and Synapses).

# The residual you found in the "Total Remainder" ($Ca + Leak + Ar$) is **`-3.798e-2 nA`**. You previously noted that after accounting for your recorded `ica_ca`, `ica_cat`, `il_hh2`, and `i_ar`, you still have a **residual error of `-0.000315 nA` (0.3 pA)**.

### Why this 0.3 pA residual matters
# While 0.3 pA is tiny, in a whole-cell dipole calculation, it is multiplied by the **Z-coordinate (the lever arm)**.
# *   In the **Soma**, $Z \approx 0$. Even if there's a 0.3 pA error, $0.3 \times 0 = 0$. This is why your Soma plots match **PERFECTLY**.
# *   In the **Dendrites**, $Z$ can be $1000$ $\mu$m. $0.3$ pA $\times 1000$ $\mu$m = $0.3$ fAm. Across hundreds of segments, these "phantom" currents accumulate into a visible DC offset or drift in the dipole (Row 4).

### To Validate `agg_i_mem` exactly, we must find that 0.3 pA.
# Since $Na$ and $K$ are perfect, the error is either in the Calcium sum or the Non-specific sum. To bisect this, you need to record **`agg_ica`** (the NEURON `ica` pointer).

# **Please add `agg_ica` to your recording logic.**

# Once you have it, we can run this definitive diagnostic:

# ```python
# # Check Calcium Balance
# # If this is 0, then Ca is not the source of the 0.3 pA residual
# agg_ica = np.array(net.cell_response.transmembrane_currents['agg_ica'][trial][gid][sec][seg])
# ica_sum = (np.array(net.cell_response.transmembrane_currents['ica_ca'][trial][gid][sec][seg]) +
#            np.array(net.cell_response.transmembrane_currents['ica_cat'][trial][gid][sec][seg]))

# ca_residual = np.mean(agg_ica - ica_sum)
# print(f"Ca2+ Residual (agg_ica - components): {ca_residual:.8e} mA/cm2")
# ```

### Two other things to check right now:

# **1. Point Processes (Non-Synaptic)**
# You mentioned there are no synapses in that segment. However, HNN-core uses other point processes, like **`TonicBias`**. Even if it's not a "synapse," it is a point process that writes to the membrane current.
# Run this to see if any point process is "hiding" on that segment:
# ```python
# # You'll need to do this while the NEURON objects are alive,
# # or check the cell_obj.sections['apical_2'].syns list in your metadata.
# print(f"Point Processes on {sec}:", cell_obj.sections[sec].syns)
# ```

# **2. The `dipole` Mechanism**
# Your `psection()` showed a mechanism called **`dipole`** inserted in the dendrite.
# Does that mechanism produce a current? Even a tiny maintenance current in a `dipole.mod` would contribute to `agg_i_mem` but be missing from your manual sum. Check the `.mod` file for any `i`, `ina`, `ik`, or `ica` assignments.

# **Summary:** Record `agg_ica`. If the Calcium balance is 0.0, we know the 0.3 pA residual is in the non-specific currents (`il_hh2`, `i_ar`) or a hidden point process. Finding this is the final step to making Row 4 match Row 3.


# %% [markdown] ###########################################################
## Debugging
# %% ######################################################################


# %% --------------------------------------------------
# [DEV] check for agg_i_non_specific
# -----------------------------------------------------

net = net_no_local_one_drive

sections = net.cell_response.transmembrane_currents["agg_hh2"][0][170]
for key, val in sections.items():
    print(key)
    print(val.keys())


# %% --------------------------------------------------
# [DEV] missing charge waveform analysis
# -----------------------------------------------------

# notes:
#   - subtracts every known component from agg_i_mem
#   - the resulting 'missing_charge' waveform is the objective shape of what remains
#   - we can visualize if the error tracks with voltage (spikes), drives (synapses),
#     or something else


def analyze_missing_charge_waveform(
    net,
    gid=183,
    cell_type="L5_pyramidal",
    sec_name="soma",
    trial=0,
):
    # import numpy as np
    # import matplotlib.pyplot as plt

    seg_key = "seg_1"
    times = net.cell_response.times

    # 1. get area scaling
    cell_obj = net.cell_types[cell_type]["cell_object"]
    sec_meta = cell_obj.sections[sec_name]
    nseg = len(
        net.cell_response.transmembrane_currents["agg_i_mem"][trial][gid][sec_name]
    )
    area_um2 = (np.pi * sec_meta.diam * sec_meta.L) / nseg
    scale = area_um2 * 0.01

    # 2. retrieve the balanced "Truths" (mA/cm2 -> nA)
    # we use aggregates because we proved they match the mechanisms
    ina_nA = (
        np.array(
            net.cell_response.transmembrane_currents["agg_ina"][trial][gid][sec_name][
                seg_key
            ]
        )
        * scale
    )
    ik_nA = (
        np.array(
            net.cell_response.transmembrane_currents["agg_ik"][trial][gid][sec_name][
                seg_key
            ]
        )
        * scale
    )
    ica_nA = (
        np.array(
            net.cell_response.transmembrane_currents["agg_ica"][trial][gid][sec_name][
                seg_key
            ]
        )
        * scale
    )
    icap_nA = (
        np.array(
            net.cell_response.transmembrane_currents["agg_i_cap"][trial][gid][sec_name][
                seg_key
            ]
        )
        * scale
    )

    # 3. retrieve non-specific mechanisms (mA/cm2 -> nA)
    # il_hh2 is technically non-specific but has its own pointer
    ins_nA = (
        np.array(
            net.cell_response.transmembrane_currents["il_hh2"][trial][gid][sec_name][
                seg_key
            ]
        )
        + np.array(
            net.cell_response.transmembrane_currents["i_ar"][trial][gid][sec_name][
                seg_key
            ]
        )
        + np.array(
            net.cell_response.transmembrane_currents["ica_cat"][trial][gid][sec_name][
                seg_key
            ]
        )
    ) * scale

    # 4. retrieve synapses (nA)
    isec_data = net.cell_response.isec[trial][gid].get(sec_name, {})
    isec_nA = (
        np.sum([np.array(v) for v in isec_data.values()], axis=0) if isec_data else 0.0
    )

    # 5. calculate the missing waveform
    imem_truth = np.array(
        net.cell_response.transmembrane_currents["agg_i_mem"][trial][gid][sec_name][
            seg_key
        ]
    )

    # Missing = Truth - (Cap + Na + K + Ca + Nonspec_Mechs + Synapses)
    missing_charge = imem_truth - (icap_nA + ina_nA + ik_nA + ica_nA + ins_nA + isec_nA)

    # 6. plotting
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(times[1:], missing_charge[1:], label="The Missing Current (Residual)")
    ax.plot(
        times[1:], imem_truth[1:], label="agg_i_mem (Truth)", alpha=0.3, color="black"
    )
    ax.set_title(f"Waveform of the 6.0 nA Error (GID {gid} Soma)")
    ax.set_ylabel("nA")
    ax.set_xlabel("Time (ms)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    return missing_charge


# execute analysis
net = net_no_local_one_drive

missing_wave = analyze_missing_charge_waveform(net)

# %% --------------------------------------------------
# [DEV] missing charge waveform analysis
# -----------------------------------------------------

# notes:
#   - subtracts every known component from agg_i_mem
#   - the resulting 'missing_charge' waveform is the objective shape of what remains
#   - includes icap_shift to test temporal alignment between capacitance and ions


def analyze_missing_charge_waveform_shift(
    net,
    gid=183,
    cell_type="L5_pyramidal",
    sec_name="soma",
    trial=0,
    icap_shift=0,
):
    """
    Calculates the residual error waveform with an optional time-shift
    for the capacitive current.
    """
    # import numpy as np
    # import matplotlib.pyplot as plt

    seg_key = "seg_1"
    times = net.cell_response.times

    # 1. get area scaling
    cell_obj = net.cell_types[cell_type]["cell_object"]
    sec_meta = cell_obj.sections[sec_name]
    nseg = len(
        net.cell_response.transmembrane_currents["agg_i_mem"][trial][gid][sec_name]
    )
    area_um2 = (np.pi * sec_meta.diam * sec_meta.L) / nseg
    scale = area_um2 * 0.01

    # 2. retrieve the balanced "Truths" (nA)
    ina_nA = (
        np.array(
            net.cell_response.transmembrane_currents["agg_ina"][trial][gid][sec_name][
                seg_key
            ]
        )
        * scale
    )
    ik_nA = (
        np.array(
            net.cell_response.transmembrane_currents["agg_ik"][trial][gid][sec_name][
                seg_key
            ]
        )
        * scale
    )
    ica_nA = (
        np.array(
            net.cell_response.transmembrane_currents["agg_ica"][trial][gid][sec_name][
                seg_key
            ]
        )
        * scale
    )

    # 3. retrieve non-specific mechanisms (nA)
    ins_nA = (
        np.array(
            net.cell_response.transmembrane_currents["il_hh2"][trial][gid][sec_name][
                seg_key
            ]
        )
        + np.array(
            net.cell_response.transmembrane_currents["i_ar"][trial][gid][sec_name][
                seg_key
            ]
        )
        + np.array(
            net.cell_response.transmembrane_currents["ica_cat"][trial][gid][sec_name][
                seg_key
            ]
        )
    ) * scale

    # 4. retrieve synapses (nA)
    isec_data = net.cell_response.isec[trial][gid].get(sec_name, {})
    isec_nA = (
        np.sum([np.array(v) for v in isec_data.values()], axis=0) if isec_data else 0.0
    )

    # 5. handle capacitive current and optional shift
    icap_raw = (
        np.array(
            net.cell_response.transmembrane_currents["agg_i_cap"][trial][gid][sec_name][
                seg_key
            ]
        )
        * scale
    )

    if icap_shift != 0:
        # np.roll shifts elements; we zero out the wrapped-around values for rigor
        icap_nA = np.roll(icap_raw, icap_shift)
        if icap_shift > 0:
            icap_nA[:icap_shift] = 0
        else:
            icap_nA[icap_shift:] = 0
    else:
        icap_nA = icap_raw

    # 6. calculate the missing waveform
    imem_truth = np.array(
        net.cell_response.transmembrane_currents["agg_i_mem"][trial][gid][sec_name][
            seg_key
        ]
    )

    # Residual = Truth - (Cap_shifted + Sum_of_Ions + Synapses)
    missing_charge = imem_truth - (icap_nA + ina_nA + ik_nA + ica_nA + ins_nA + isec_nA)

    # 7. plotting
    fig, ax = plt.subplots(figsize=(10, 5))
    # slice [1:] to avoid t=0 initialization artifact
    ax.plot(
        times[1:], missing_charge[1:], label=f"Missing Current (Shift={icap_shift})"
    )
    ax.plot(
        times[1:], imem_truth[1:], label="agg_i_mem (Truth)", alpha=0.3, color="black"
    )

    ax.set_title(f"Residual Waveform Analysis (GID {gid} {sec_name})")
    ax.set_ylabel("nA")
    ax.set_xlabel("Time (ms)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    return missing_charge


# execute analysis
net = net_no_local_one_drive

missing_wave = analyze_missing_charge_waveform_shift(
    net,
    icap_shift=1,
)
