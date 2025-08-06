# %%
import os
import matplotlib.pyplot as plt
from hnn_core import jones_2009_model
from hnn_core.viz import NetworkPlotter

net = jones_2009_model(mesh_shape=(10, 10))
net.set_cell_positions(inplane_distance=150)


# %%
net_plot = NetworkPlotter(
    net,
    bg_color="none",
    colorbar=False,
    cell_type_colors={
        "default": "#3A599A",
        "L5_basket": "#FFAE1E",
        "L2_basket": "#FFAE1E",
        # "L5_pyramidal": "#5589DD",
        # "L2_pyramidal": "#4F9756"
    },
    azim=-75,
)


# %%
net_plot.fig.savefig(
    os.path.join('tmp','file.png'),
    transparent=True,
)

# %%
net_plot = NetworkPlotter(
    net,
    bg_color="none",
    colorbar=False,
    cell_type_colors={
        "default": "#3A599A",
        "L5_basket": "#FFAE1E",
        "L2_basket": "#FFAE1E",
        "L5_pyramidal": "#5589DD",
        "L2_pyramidal": "#4F9756"
    },
    azim=-110,
    elev=20,
)

# %%
