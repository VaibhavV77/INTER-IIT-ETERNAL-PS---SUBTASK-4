import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Dict, List, Tuple


# =====================================================================
# 1. PLANT PARAMETERS & NONLINEAR RK4 ENVIRONMENT
# =====================================================================
@dataclass
class PlantParams:
    M: float = 1.0        # Cart mass (kg)
    m: float = 0.2        # Pole mass (kg)
    l: float = 0.5        # Half-length to pole CoM (m) -> total length = 1.0m
    b_c: float = 0.1      # Cart viscous friction (N*s/m)
    b_p: float = 0.005    # Pole pivot damping (N*m*s/rad)
    g: float = 9.81       # Gravity (m/s^2)
    u_max: float = 20.0   # Max actuator force (N)
    du_max: float = 500.0 # Max actuator slew rate (N/s)

    @property
    def I(self) -> float:
        return (1.0 / 3.0) * self.m * (self.l ** 2)


class CartPolePlant:
    """
    Nonlinear Cart-Pole Benchmark Environment integrated via RK4.
    State vector s = [x, x_dot, theta, theta_dot]
    """
    def __init__(self, params: PlantParams = PlantParams(), dt: float = 0.005):
        self.p = params
        self.dt = dt
        self.state = np.zeros(4, dtype=float)
        self.prev_u = 0.0

    def reset(self, init_state: np.ndarray) -> np.ndarray:
        self.state = np.array(init_state, dtype=float).copy()
        self.prev_u = 0.0
        return self.state.copy()

    def _derivatives(self, s: np.ndarray, u_total: float) -> np.ndarray:
        x, x_dot, th, th_dot = s
        p = self.p

        sin_th = np.sin(th)
        cos_th = np.cos(th)
        I_total = p.I + p.m * (p.l ** 2)

        denom = (p.M + p.m) * I_total - (p.m * p.l * cos_th) ** 2

        F_eff = u_total - p.b_c * x_dot + p.m * p.l * (th_dot ** 2) * sin_th
        T_eff = p.m * p.g * p.l * sin_th - p.b_p * th_dot

        x_ddot = (I_total * F_eff - p.m * p.l * cos_th * T_eff) / denom
        th_ddot = ((p.M + p.m) * T_eff - p.m * p.l * cos_th * F_eff) / denom

        return np.array([x_dot, x_ddot, th_dot, th_ddot], dtype=float)

    def step(self, u_cmd: float, disturbance_force: float = 0.0) -> Tuple[np.ndarray, float]:
        max_delta = self.p.du_max * self.dt
        u_rate_limited = np.clip(u_cmd, self.prev_u - max_delta, self.prev_u + max_delta)
        u_applied = float(np.clip(u_rate_limited, -self.p.u_max, self.p.u_max))
        self.prev_u = u_applied

        u_total = u_applied + disturbance_force

        s = self.state
        dt = self.dt
        k1 = self._derivatives(s, u_total)
        k2 = self._derivatives(s + 0.5 * dt * k1, u_total)
        k3 = self._derivatives(s + 0.5 * dt * k2, u_total)
        k4 = self._derivatives(s + dt * k3, u_total)

        self.state = s + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        self.state[2] = (self.state[2] + np.pi) % (2.0 * np.pi) - np.pi

        return self.state.copy(), u_applied


# =====================================================================
# 2. FIXED CASCADED PID CONTROLLER
# =====================================================================
class CascadedPID:
    """
    Fixed Cascaded PID Controller for Cart-Pole:
    1. Slow outer loop (kp_x=0.04, kd_x=0.10, ki_x=0.002, tau_x=0.10s, th_ref_max=8 deg)
       to respect non-minimum-phase bandwidth separation.
    2. Low-pass filter on lean command th_ref (tau_r=0.15s) to prevent relay switching.
    3. Inner loop uses derivative-on-measurement (d(th_meas)/dt) to eliminate
       derivative kick and algebraic velocity cross-coupling.
    """
    def __init__(
        self,
        dt: float = 0.005,
        u_max: float = 20.0,
        x_ref: float = 0.0,
        gains: Dict[str, float] = None,
    ):
        self.dt = dt
        self.u_max = u_max
        self.x_ref = x_ref

        default_gains = {
            # Outer Loop (Cart Position -> Target Lean Angle)
            "kp_x": 0.04,
            "ki_x": 0.002,
            "kd_x": 0.10,
            "tau_x": 0.10,                        # Outer derivative filter time constant (s)
            "th_ref_max": np.deg2rad(8.0),        # Clamp lean command to +/- 8 deg
            "i_x_max": 1.0,
            "tau_r": 0.15,                        # Reference smoothing filter time constant (s)
            # Inner Loop (Pole Lean Angle -> Horizontal Cart Force)
            "kp_a": 68.0,
            "ki_a": 0.0,                          # Zero inner integral prevents phase-lag fighting
            "kd_a": 14.5,
            "tau_a": 0.015,                       # Inner derivative filter time constant (s)
            "i_a_max": 2.0,
        }
        self.g = default_gains if gains is None else {**default_gains, **gains}
        self.reset()

    def reset(self):
        self.i_x = 0.0
        self.i_a = 0.0
        self.d_x_filt = 0.0
        self.d_a_filt = 0.0
        self.th_ref_f = 0.0
        self.prev_x = None
        self.prev_th = None
        self.last_th_ref = 0.0

    def act(self, s: np.ndarray, t: float = 0.0) -> float:
        g = self.g
        x_meas = s[0]
        th_meas = s[2]

        # -------------------------------------------------------------
        # OUTER LOOP: Cart Position -> Target Lean Angle (th_ref)
        # -------------------------------------------------------------
        if self.prev_x is None:
            self.prev_x = x_meas

        ex = x_meas - self.x_ref
        raw_dx = (x_meas - self.prev_x) / self.dt
        alpha_x = self.dt / (g["tau_x"] + self.dt)
        self.d_x_filt = (1.0 - alpha_x) * self.d_x_filt + alpha_x * raw_dx
        self.prev_x = x_meas

        # Cart too far right (ex > 0) -> ask pole to lean left (th_ref < 0)
        th_ref_unsat = -(g["kp_x"] * ex + g["ki_x"] * self.i_x + g["kd_x"] * self.d_x_filt)
        th_ref = float(np.clip(th_ref_unsat, -g["th_ref_max"], g["th_ref_max"]))

        # Outer-loop conditional anti-windup
        if (th_ref == th_ref_unsat) or (ex * (-th_ref_unsat) < 0):
            self.i_x = float(np.clip(self.i_x + ex * self.dt, -g["i_x_max"], g["i_x_max"]))

        # -------------------------------------------------------------
        # INNER LOOP: Smooth Reference & Derivative on Measurement
        # -------------------------------------------------------------
        alpha_r = self.dt / (g["tau_r"] + self.dt)             # tau_r ~ 0.15 s
        self.th_ref_f += alpha_r * (th_ref - self.th_ref_f)    # smooth the lean command
        self.last_th_ref = self.th_ref_f

        e_th = th_meas - self.th_ref_f
        if self.prev_th is None:
            self.prev_th = th_meas

        # Derivative on measurement (th_meas) to avoid derivative kick from th_ref changes
        raw_dth = (th_meas - self.prev_th) / self.dt
        alpha_a = self.dt / (g["tau_a"] + self.dt)
        self.d_a_filt = (1.0 - alpha_a) * self.d_a_filt + alpha_a * raw_dth
        self.prev_th = th_meas

        u_unsat = g["kp_a"] * e_th + g["ki_a"] * self.i_a + g["kd_a"] * self.d_a_filt
        u = float(np.clip(u_unsat, -self.u_max, self.u_max))

        # Inner-loop anti-windup
        if (u == u_unsat) or (e_th * u_unsat < 0):
            self.i_a = float(np.clip(self.i_a + e_th * self.dt, -g["i_a_max"], g["i_a_max"]))

        return u


# =====================================================================
# 3. QUANTITATIVE BENCHMARK METRICS EVALUATOR
# =====================================================================
def compute_metrics(
    t: np.ndarray,
    states: np.ndarray,
    u_hist: np.ndarray,
    x_ref: float = 0.0,
) -> Dict[str, float]:
    dt = t[1] - t[0]
    x = states[:, 0]
    th_deg = np.rad2deg(states[:, 2])

    stabilised = bool(np.max(np.abs(th_deg)) < 45.0 and np.abs(th_deg[-1]) < 1.0)

    # Settling band: |theta| < 1.0 deg AND |x - x_ref| < 0.05 m (practical band)
    # plus strict 0.5 deg / 0.02 m band for comparison
    in_band_practical = (np.abs(th_deg) < 1.0) & (np.abs(x - x_ref) < 0.05)
    out_prac = np.where(~in_band_practical)[0]
    ts_prac = 0.0 if len(out_prac) == 0 else (float(t[-1]) if out_prac[-1] == len(t) - 1 else float(t[out_prac[-1] + 1]))

    in_band_strict = (np.abs(th_deg) < 0.5) & (np.abs(x - x_ref) < 0.02)
    out_strict = np.where(~in_band_strict)[0]
    ts_strict = 0.0 if len(out_strict) == 0 else (float(t[-1]) if out_strict[-1] == len(t) - 1 else float(t[out_strict[-1] + 1]))

    th0 = np.abs(th_deg[0])
    if th0 > 1.0:
        idx_90 = np.where(np.abs(th_deg) <= 0.10 * th0)[0]
        idx_10 = np.where(np.abs(th_deg) <= 0.90 * th0)[0]
        rise_time = float(t[idx_90[0]] - t[idx_10[0]]) if (len(idx_90) > 0 and len(idx_10) > 0) else np.nan
        opposite_dir = -np.sign(th_deg[0]) * th_deg
        overshoot_pct = (max(0.0, float(np.max(opposite_dir))) / th0) * 100.0
    else:
        rise_time = 0.0
        overshoot_pct = float(np.max(np.abs(th_deg)))

    n_ss = int(1.0 / dt)
    ss_error_x = float(np.mean(np.abs(x[-n_ss:] - x_ref)))
    ss_error_th = float(np.mean(np.abs(th_deg[-n_ss:])))
    energy_l2 = float(np.sum(u_hist ** 2) * dt)
    max_u = float(np.max(np.abs(u_hist)))

    return {
        "Stabilised": stabilised,
        "Settling Time [1deg, 5cm] (s)": round(ts_prac, 3),
        "Settling Time [0.5deg, 2cm] (s)": round(ts_strict, 3),
        "Rise Time (s)": round(rise_time, 3) if not np.isnan(rise_time) else -1.0,
        "Overshoot (%) / Peak (deg)": round(overshoot_pct, 2),
        "SS Error |x| (m)": round(ss_error_x, 4),
        "SS Error |th| (deg)": round(ss_error_th, 4),
        "Control Energy int(u^2 dt)": round(energy_l2, 2),
        "Peak |u| (N)": round(max_u, 2),
    }


# =====================================================================
# 4. SIMULATION RUNNER & MATPLOTLIB EVALUATION SUITE
# =====================================================================
def run_scenario(
    plant_params: PlantParams,
    controller: CascadedPID,
    init_state: List[float],
    T: float = 15.0,
    dt: float = 0.005,
    disturbance_window: Tuple[float, float, float] = (0.0, 0.0, 0.0),
):
    plant = CartPolePlant(params=plant_params, dt=dt)
    controller.reset()

    steps = int(T / dt)
    t_hist = np.zeros(steps)
    s_hist = np.zeros((steps, 4))
    u_hist = np.zeros(steps)
    th_ref_hist = np.zeros(steps)

    s = plant.reset(np.array(init_state, dtype=float))
    d_start, d_end, d_force = disturbance_window

    for k in range(steps):
        t = k * dt
        d_u = d_force if (d_start <= t <= d_end) else 0.0

        u_cmd = controller.act(s, t)
        s_next, u_applied = plant.step(u_cmd, disturbance_force=d_u)

        t_hist[k] = t
        s_hist[k] = s
        u_hist[k] = u_applied
        th_ref_hist[k] = controller.last_th_ref
        s = s_next

    metrics = compute_metrics(t_hist, s_hist, u_hist, x_ref=controller.x_ref)
    return t_hist, s_hist, u_hist, th_ref_hist, metrics


def main():
    dt = 0.005
    T = 15.0  # 15s horizon to show full position recovery of the slower outer loop
    nominal_params = PlantParams()
    perturbed_params = PlantParams(M=1.1, m=0.26, l=0.60, b_c=0.07, b_p=0.003)

    pid = CascadedPID(dt=dt, u_max=nominal_params.u_max)

    scenarios = {
        "1. Initial Offset (x0=0.5m, th0=15 deg)": {
            "params": nominal_params,
            "init": [0.5, 0.0, np.deg2rad(15.0), 0.0],
            "dist": (0.0, 0.0, 0.0),
        },
        "2. Impulse Disturbance (8N hit @ t=3.0s)": {
            "params": nominal_params,
            "init": [0.0, 0.0, 0.0, 0.0],
            "dist": (3.0, 3.10, 8.0),
        },
        "3. Parameter Uncertainty (+30% m, +20% l)": {
            "params": perturbed_params,
            "init": [0.5, 0.0, np.deg2rad(15.0), 0.0],
            "dist": (0.0, 0.0, 0.0),
        },
    }

    results = {}
    print("\n" + "=" * 95)
    print("FIXED CASCADED PID BENCHMARK EVALUATION SUMMARY")
    print("=" * 95)

    for name, cfg in scenarios.items():
        t, s, u, th_ref, m = run_scenario(
            cfg["params"], pid, cfg["init"], T=T, dt=dt, disturbance_window=cfg["dist"]
        )
        results[name] = (t, s, u, th_ref, m)
        print(f"\nScenario: {name}")
        for k, v in m.items():
            print(f"  {k:<32}: {v}")

    # -----------------------------------------------------------------
    # Figure 1: 3-Scenario Benchmark Dashboard
    # -----------------------------------------------------------------
    fig, axes = plt.subplots(3, 3, figsize=(15, 9), sharex=True)
    fig.suptitle(
        "Fixed Cascaded PID Benchmark (Inner Measurement Derivative + Reference Smoothing + Slow Outer Loop)",
        fontsize=13,
        fontweight="bold",
    )

    for col, (name, (t, s, u, th_ref, m)) in enumerate(results.items()):
        x = s[:, 0]
        th_deg = np.rad2deg(s[:, 2])
        th_ref_deg = np.rad2deg(th_ref)

        # Row 0: Cart Position x(t)
        ax_x = axes[0, col]
        ax_x.plot(t, x, color="#1f77b4", lw=2, label="Cart x (m)")
        ax_x.axhline(0.0, color="black", ls="--", lw=1, alpha=0.6, label="x_ref")
        ax_x.set_title(name, fontsize=10, fontweight="bold")
        ax_x.grid(True, alpha=0.3)
        if col == 0:
            ax_x.set_ylabel("Position x (m)")
            ax_x.legend(loc="upper right")

        # Row 1: Pole Angle theta(t) vs Smoothed th_ref_f(t)
        ax_th = axes[1, col]
        ax_th.plot(t, th_deg, color="#d62728", lw=2, label="Pole theta (deg)")
        ax_th.plot(t, th_ref_deg, color="#2ca02c", ls="--", lw=1.5, label="Smoothed th_ref_f")
        ax_th.axhline(0.0, color="black", ls=":", lw=1, alpha=0.5)
        ax_th.grid(True, alpha=0.3)
        if col == 0:
            ax_th.set_ylabel("Angle (deg)")
            ax_th.legend(loc="upper right")

        # Row 2: Control Effort u(t)
        ax_u = axes[2, col]
        ax_u.plot(t, u, color="#9467bd", lw=1.8, label="Force u (N)")
        ax_u.axhline(nominal_params.u_max, color="red", ls="--", lw=1, alpha=0.5, label="Saturation")
        ax_u.axhline(-nominal_params.u_max, color="red", ls="--", lw=1, alpha=0.5)
        ax_u.set_xlabel("Time (s)")
        ax_u.grid(True, alpha=0.3)
        if col == 0:
            ax_u.set_ylabel("Control Force u (N)")
            ax_u.legend(loc="upper right")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
