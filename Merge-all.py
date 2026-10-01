import os
import importlib
import torch
import numpy as np
import matplotlib.pyplot as plt

# =====================================================================
# 1. IMPORT DIRECTLY FROM YOUR HYPHENATED FILES IN Temp-VV
# =====================================================================
pid_mod = importlib.import_module("PID-controller")
lqr_mod = importlib.import_module("LQR-controller")
rl_mod = importlib.import_module("RL-controller")

PlantParams = pid_mod.PlantParams
CartPolePlant = pid_mod.CartPolePlant
compute_metrics = pid_mod.compute_metrics

CascadedPID = pid_mod.CascadedPID
LQRController = lqr_mod.LQRController
RLController = rl_mod.RLController


# =====================================================================
# 2. UNIFIED SCENARIO RUNNER & SMART RL WEIGHT LOADER
# =====================================================================
def run_unified_scenario(plant_params, controller, init_state, T=15.0, dt=0.005, dist=(0.0, 0.0, 0.0)):
    plant = CartPolePlant(params=plant_params, dt=dt)
    if hasattr(controller, "reset"):
        controller.reset()

    steps = int(T / dt)
    t_hist = np.zeros(steps)
    s_hist = np.zeros((steps, 4))
    u_hist = np.zeros(steps)

    s = plant.reset(np.array(init_state, dtype=float))
    d_start, d_end, d_force = dist

    for k in range(steps):
        t = k * dt
        d_u = d_force if (d_start <= t <= d_end) else 0.0

        u_cmd = controller.act(s, t)
        s_next, u_applied = plant.step(u_cmd, disturbance_force=d_u)

        t_hist[k] = t
        s_hist[k] = s
        u_hist[k] = u_applied
        s = s_next

    x_ref = getattr(controller, "x_ref", 0.0)
    metrics = compute_metrics(t_hist, s_hist, u_hist, x_ref=x_ref)
    return t_hist, s_hist, u_hist, metrics


def load_or_train_rl(nominal_params, dt, weights_path="rl_policy.pt"):
    """
    Loads 'rl_policy.pt' and ensures the fast NumPy weight cache `w` used by
    RLController._forward_np() is properly populated before returning.
    """
    # 1. Instantiate RLController with whichever constructor signature it uses
    rl = None
    for kwargs in [
        {"nominal_params": nominal_params, "dt": dt, "u_max": nominal_params.u_max},
        {"params": nominal_params, "dt": dt, "u_max": nominal_params.u_max},
        {"dt": dt, "u_max": nominal_params.u_max},
        {},
    ]:
        try:
            rl = RLController(**kwargs)
            break
        except TypeError:
            continue

    # Detect which attribute on `rl` holds the `w` dict accessed inside `_forward_np`
    w_attr = None
    if hasattr(rl, "_forward_np"):
        for name in rl._forward_np.__code__.co_names:
            if hasattr(rl, name) and getattr(rl, name) is None:
                w_attr = name
                break

    def is_rl_ready(ctrl):
        try:
            if hasattr(ctrl, "reset"):
                ctrl.reset()
            _ = ctrl.act(np.zeros(4, dtype=float), 0.0)
            return True
        except Exception:
            return False

    # 2. Try RLController's own load methods FIRST
    if os.path.exists(weights_path):
        for load_m in ["load", "load_weights", "load_policy", "load_model"]:
            if hasattr(rl, load_m):
                try:
                    getattr(rl, load_m)(weights_path)
                    if is_rl_ready(rl):
                        print(f"Successfully loaded '{weights_path}' via rl.{load_m}()!")
                        return rl
                except Exception as e:
                    print(f"Note: rl.{load_m}() raised: {e}")

        # 3. Inspect checkpoint contents directly
        try:
            ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
            print(f"Inspecting '{weights_path}' -> type: {type(ckpt).__name__}", end="")
            if isinstance(ckpt, dict):
                print(f", keys: {list(ckpt.keys())}")
            else:
                print()

            def to_np_dict(d):
                return {
                    k: (v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else np.asarray(v))
                    for k, v in d.items()
                }

            # Case A: Checkpoint directly IS the `w` dict (contains "sc")
            if isinstance(ckpt, dict) and "sc" in ckpt and w_attr:
                setattr(rl, w_attr, to_np_dict(ckpt))
                if is_rl_ready(rl):
                    print(f"Loaded numpy weights dict directly into rl.{w_attr}!")
                    return rl

            # Case B: Checkpoint wraps the `w` dict under a sub-key
            if isinstance(ckpt, dict) and w_attr:
                for k, v in ckpt.items():
                    if isinstance(v, dict) and "sc" in v:
                        setattr(rl, w_attr, to_np_dict(v))
                        if is_rl_ready(rl):
                            print(f"Loaded ckpt['{k}'] directly into rl.{w_attr}!")
                            return rl

            # Case C: Checkpoint is/has a PyTorch state_dict -> load into nn.Module & sync numpy cache
            state_dict = None
            if isinstance(ckpt, dict):
                for key in ["model_state_dict", "state_dict", "policy_state_dict", "actor_state_dict", "policy", "net"]:
                    if key in ckpt and isinstance(ckpt[key], dict):
                        state_dict = ckpt[key]
                        break
                if state_dict is None and any(isinstance(v, torch.Tensor) for v in ckpt.values()):
                    state_dict = ckpt

            if state_dict is not None:
                for attr_name, attr_val in rl.__dict__.items():
                    if isinstance(attr_val, torch.nn.Module):
                        try:
                            attr_val.load_state_dict(state_dict)
                            print(f"Loaded state_dict into rl.{attr_name}")
                        except Exception:
                            pass

            # Call any 0-arg helper method on `rl` that populates `w_attr` (e.g. _sync_np, _cache_weights)
            for m_name in dir(rl):
                if m_name.startswith("__") or m_name in ["reset", "act", "_forward_np", "train", "train_policy"]:
                    continue
                m_obj = getattr(rl, m_name)
                if callable(m_obj) and hasattr(m_obj, "__code__"):
                    n_req = m_obj.__code__.co_argcount - (len(m_obj.__defaults__) if m_obj.__defaults__ else 0)
                    if n_req <= 1 and (w_attr is None or w_attr in m_obj.__code__.co_names):
                        try:
                            m_obj()
                            if is_rl_ready(rl):
                                print(f"Synced numpy weights via rl.{m_name}()!")
                                return rl
                        except Exception:
                            pass
        except Exception as e:
            print(f"Checkpoint inspection note: {e}")

    # 4. Fallback: Run RL training method (which populates `w` and saves weights)
    print("Populating RL controller via training method...")
    for train_m in ["train_policy", "train", "fit", "learn"]:
        if hasattr(rl, train_m):
            getattr(rl, train_m)()
            if is_rl_ready(rl):
                return rl

    return rl


# =====================================================================
# 3. MAIN 3-WAY OVERLAY & BENCHMARK TABLE
# =====================================================================
def main():
    dt = 0.005
    T = 15.0
    nominal = PlantParams()
    perturbed = PlantParams(M=1.1, m=0.26, l=0.60, b_c=0.07, b_p=0.003)

    # 1. Initialize all 3 controllers from your files
    pid = CascadedPID(dt=dt, u_max=nominal.u_max)
    lqr = LQRController(nominal_params=nominal, dt=dt, u_max=nominal.u_max)
    rl = load_or_train_rl(nominal, dt, weights_path="rl_policy.pt")

    controllers = {
        "Cascaded PID": pid,
        "Optimal LQR": lqr,
        "RL Policy": rl,
    }
    colors = {
        "Cascaded PID": "#1f77b4",  # Blue
        "Optimal LQR": "#ff7f0e",   # Orange
        "RL Policy": "#2ca02c",     # Green
    }
    linestyles = {
        "Cascaded PID": "-",
        "Optimal LQR": "--",
        "RL Policy": "-.",
    }

    scenarios = {
        "1. Initial Offset (x0=0.5m, th0=15 deg)": {
            "params": nominal,
            "init": [0.5, 0.0, np.deg2rad(15.0), 0.0],
            "dist": (0.0, 0.0, 0.0),
        },
        "2. Impulse Disturbance (8N hit @ t=3.0s)": {
            "params": nominal,
            "init": [0.0, 0.0, 0.0, 0.0],
            "dist": (3.0, 3.10, 8.0),
        },
        "3. Parameter Uncertainty (+30% m, +20% l)": {
            "params": perturbed,
            "init": [0.5, 0.0, np.deg2rad(15.0), 0.0],
            "dist": (0.0, 0.0, 0.0),
        },
    }

    fig, axes = plt.subplots(3, 3, figsize=(16, 9), sharex=True)
    fig.suptitle(
        "Head-to-Head Benchmark: Cascaded PID vs. Optimal LQR vs. Reinforcement Learning",
        fontsize=14,
        fontweight="bold",
    )

    print("\n" + "=" * 105)
    print("UNIFIED 3-CONTROLLER BENCHMARK SUMMARY (PID vs. LQR vs. RL)")
    print("=" * 105)

    for col, (scen_name, cfg) in enumerate(scenarios.items()):
        print(f"\n### {scen_name}")
        print(
            "| Controller | Stabilised | Settling [1°,5cm] (s) | Settling [0.5°,2cm] (s) | "
            "Rise Time (s) | Overshoot / Peak | SS Err |x| (m) | Energy ∫u²dt | Peak |u| (N) |"
        )
        print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")

        for c_name, ctrl in controllers.items():
            t, s, u, m = run_unified_scenario(
                cfg["params"], ctrl, cfg["init"], T=T, dt=dt, dist=cfg["dist"]
            )

            print(
                f"| **{c_name}** | {m['Stabilised']} | {m['Settling Time [1deg, 5cm] (s)']} | "
                f"{m['Settling Time [0.5deg, 2cm] (s)']} | {m['Rise Time (s)']} | "
                f"{m['Overshoot (%) / Peak (deg)']} | {m['SS Error |x| (m)']} | "
                f"{m['Control Energy int(u^2 dt)']} | {m['Peak |u| (N)']} |"
            )

            x = s[:, 0]
            th_deg = np.rad2deg(s[:, 2])

            axes[0, col].plot(t, x, label=c_name, color=colors[c_name], ls=linestyles[c_name], lw=2.0)
            axes[1, col].plot(t, th_deg, label=c_name, color=colors[c_name], ls=linestyles[c_name], lw=2.0)
            axes[2, col].plot(t, u, label=c_name, color=colors[c_name], ls=linestyles[c_name], lw=1.8)

        axes[0, col].set_title(scen_name, fontsize=10, fontweight="bold")
        axes[0, col].axhline(0.0, color="black", ls=":", lw=1, alpha=0.6)
        axes[1, col].axhline(0.0, color="black", ls=":", lw=1, alpha=0.6)
        axes[2, col].axhline(nominal.u_max, color="red", ls=":", lw=1, alpha=0.4)
        axes[2, col].axhline(-nominal.u_max, color="red", ls=":", lw=1, alpha=0.4)
        axes[2, col].set_xlabel("Time (s)")

        for row in range(3):
            axes[row, col].grid(True, alpha=0.3)

    axes[0, 0].set_ylabel("Cart Position x (m)")
    axes[1, 0].set_ylabel("Pole Angle theta (deg)")
    axes[2, 0].set_ylabel("Control Force u (N)")
    axes[0, 0].legend(loc="upper right", framealpha=0.9)
    axes[1, 0].legend(loc="upper right", framealpha=0.9)
    axes[2, 0].legend(loc="upper right", framealpha=0.9)

    plt.tight_layout()
    out_img = "PID_vs_LQR_vs_RL_Overlay.png"
    plt.savefig(out_img, dpi=300)
    print(f"\nSaved high-resolution comparison figure to '{out_img}'!")
    plt.show()


if __name__ == "__main__":
    main()