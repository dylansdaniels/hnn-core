

def test_optimizer_outputs(base_net, target):
    """Run a minimal one-parameter CMA-ES optimization test."""

    drive_config = {
        "evprox_1": {
            "mu": (("relative", 0.85), ("relative", 1.15)),
            "sigma": (("relative", 1.0), ("relative", 1.0)),
            "weights": {},
        },
    }

    constraints, initial_conditions = make_constraints(
        base_net,
        drive_config,
    )

    def set_test_params(net, params):
        drive = net.external_drives["evprox_1"]
        drive["dynamics"]["mu"] = params["mu_evprox_1"]

    test_optim = Optimizer(
        initial_net=base_net.copy(),
        tstop=300,
        constraints=constraints,
        set_params=set_test_params,
        initial_params=initial_conditions,
        solver="cma",
    )

    with MPIBackend(n_procs=10):
        test_optim.fit(
            target=target,
            n_trials=1,
            baseline_correction=True,
            scale_factor=1,
            smooth_window_len=20,
            n_jobs=10,
            max_iter=1,
            popsize=4,
        )

    print(f"raw_obj_ exists: {hasattr(test_optim, 'raw_obj_')}")
    print(f"es_ exists: {hasattr(test_optim, 'es_')}")

    if hasattr(test_optim, "raw_obj_"):
        print(f"Raw objective evaluations: {len(test_optim.raw_obj_)}")

    if hasattr(test_optim, "es_"):
        print(f"CMA-ES iterations: {test_optim.es_.countiter}")
        print(f"CMA-ES evaluations: {test_optim.es_.countevals}")

    return test_optim


test_optim = test_optimizer_outputs(
    base_net=base_net,
    target=beta10_dpl,
)

# %%

test_file = (
    subject_output_dir / "test_checkpoint.pkl"

)

with test_file.open("wb") as file:
    pickle.dump(
        optim,
        file
    )


with test_file.open("rb") as file:
    test_save_optim = pickle.load(file)

fig = test_save_optim.plot_convergence()