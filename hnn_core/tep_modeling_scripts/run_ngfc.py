"""
Originally authored by Joyce (previously named `run_ngfc_weights_sweep.py`)
Updated and edited by Dylan

Description:
Run the Duecker ET model with custom NGFC GABAA and GABAB weights.
"""

# %% [code]
# ==================================================
# SETUP
# ==================================================

from hnn_core import MPIBackend, duecker_ET_model, simulate_dipole

# custom NGFC weights
# --------------------------------------------------
# adjust these weights to explore different inhibitory strengths
# NOTE:
#   - keys must match pyramidal cell types in `duecker_ET_model` that
#     have `apical_tuft`
ngfc_weights_gabab = {
    "L2_pyramidal": 0.005,
    "L5_pyramidal": 0.005,
}
ngfc_weights_gabaa = {
    "L2_pyramidal": 0.002,
    "L5_pyramidal": 0.002,
}

# instantiate duecker_ET_model
# --------------------------------------------------
# NOTE:
net = duecker_ET_model(
    ngfc_weights_gabab=ngfc_weights_gabab,
    ngfc_weights_gabaa=ngfc_weights_gabaa,
)
net.set_cell_positions(inplane_distance=30.0)

# add drives
# --------------------------------------------------
weights_ampa_p1 = {
    "L2_inhibitory": 0.01,
    "L2_pyramidal": 0.015,
    "L5_inhibitory": 0.0,
    "L5_pyramidal": 0.03,
}
weights_nmda_p1 = {
    "L2_inhibitory": 0.01,
    "L2_pyramidal": 0.05,
    "L5_inhibitory": 0.0,
    "L5_pyramidal": 0.025,
}
synaptic_delays_prox = {
    "L2_inhibitory": 0.1,
    "L2_pyramidal": 0.1,
    "L5_inhibitory": 1,
    "L5_pyramidal": 1,
}
net.add_evoked_drive(
    "prox1",
    mu=18,
    sigma=2.5,
    numspikes=1,
    weights_ampa=weights_ampa_p1,
    weights_nmda=weights_nmda_p1,
    location="proximal",
    synaptic_delays=synaptic_delays_prox,
)

weights_ampa_d1 = {
    "L2_inhibitory": 0.005,
    "L2_pyramidal": 0.01,
    "L5_pyramidal": 1.0,
}
weights_nmda_d1 = {
    "L2_inhibitory": 0.0,
    "L2_pyramidal": 0.01,
    "L5_pyramidal": 1.0,
}
net.add_evoked_drive(
    "dist1",
    mu=62,
    sigma=5,
    numspikes=2,
    weights_ampa=weights_ampa_d1,
    weights_nmda=weights_nmda_d1,
    location="distal",
    synaptic_delays={"L2_inhibitory": 0.1, "L2_pyramidal": 0.1, "L5_pyramidal": 0.1},
)

weights_ampa_p2 = {
    "L2_inhibitory": 0.01,
    "L2_pyramidal": 0.3,
    "L5_inhibitory": 0.001,
    "L5_pyramidal": 0.3,
}
weights_nmda_p2 = {
    "L2_inhibitory": 0.01,
    "L2_pyramidal": 0.2,
    "L5_inhibitory": 0.001,
    "L5_pyramidal": 0.2,
}
net.add_evoked_drive(
    "prox2",
    mu=100,
    sigma=15,
    numspikes=1,
    weights_ampa=weights_ampa_p2,
    weights_nmda=weights_nmda_p2,
    location="proximal",
    synaptic_delays={
        "L2_inhibitory": 0.1,
        "L2_pyramidal": 0.1,
        "L5_inhibitory": 1.0,
        "L5_pyramidal": 1.0,
    },
)

print(f"NGFC weights_gabab: {ngfc_weights_gabab}")
print(f"NGFC weights_gabaa: {ngfc_weights_gabaa}")

# %% [code]
# ==================================================
# SIMULATE DIPOLE
# ==================================================

with MPIBackend():
    dpls = simulate_dipole(
        net,
        tstop=170.0,
        baseline_correction=True,
    )

print("Simulation complete.")
print(f"Dipole shape: {dpls[0].data['agg'].shape}")

# %%
