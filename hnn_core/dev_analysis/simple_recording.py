# %% [markdown] ###########################################################
## Setup
# #########################################################################

# %%
import matplotlib.pyplot as plt
import numpy as np
from IPython import get_ipython

from hnn_core import (
    JoblibBackend,
    jones_2009_model,
    simulate_dipole,
)
from hnn_core.network_models import add_erp_drives_to_jones_model

ipython = get_ipython()
if ipython is not None:
    ipython.run_line_magic("load_ext", "autoreload")
    ipython.run_line_magic("autoreload", "2")

# %% [code]
# vars for cell_response indexing

trial=0
cell_id=170
section="soma"
current="agg_i_mem"
segment="seg_1"

# %% [code]
# simulate base network

tstop = 70.0
dt = 0.1
n_trials = 1
record_method = "soma"

net = jones_2009_model()
add_erp_drives_to_jones_model(net)

tm_currents={
    "agg_ina": {
        "type": "ionic",
        "mech": None,
        "ref": "_ref_ina",
        "per_segment": True,
        "recorded_sections": record_method,
    }
}

with JoblibBackend(n_jobs=1):
    dpls = simulate_dipole(
        net,
        tstop=tstop,
        dt=dt,
        n_trials=1,
        tm_currents=tm_currents,
    )

# %% [code]
# diagnostic check for base simulation *before* updates
pre_update_diagnostics = False
if pre_update_diagnostics:
    for current, values in net.cell_response.tm_currents.items():
        print(
            f"Current: {current}",
        )
        sections = list(values[trial][cell_id].keys())
        if not sections:
            print("    Not recorded")
        else:
            print(
                f"    {section.capitalize()} value check: ~"
                f"{round(values[trial][cell_id][section][segment][0], 5)}"
            )


# diagnostic check for base simulation *after* updates
post_update_diagnostics = True
gid = 170

channel = "agg_ina"
if post_update_diagnostics:
    try:
        tm_currents = net.cell_response.tm_currents
        gids = list(tm_currents[channel]["data"][0].keys())
        print(
            "GIDs: ",
            f"{min(gids)} -> {max(gids)}",
        )
        print(
            f'Checking GID "{gid}" for channel "{channel}"'
        )
        sections = list(
            tm_currents[channel]["data"][0][gid].keys()
        )
        print(
            "Sections: ",
            sections,
        )
        sect = sections[0]
        segments = list(
            tm_currents[channel]["data"][0][gid][sect].keys()
        )
        for section in sections:
            print(
                f"Segments on {sect}: ",
                segments,
            )
        seg_0 = tm_currents[channel]["data"][0][gid][section][segments[0]]
        print(
            f"Data check on {sect}, {segments[0]}: ",
            seg_0[0:5]
        )
    except Exception:
        pass
