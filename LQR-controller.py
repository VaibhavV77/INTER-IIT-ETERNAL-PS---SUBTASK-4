import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Dict, List, Tuple

try:
    from scipy.linalg import solve_discrete_are
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


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
# 2. STANDALONE LQR CONTROLLER
# =====================================================================
class LQRController:
    """
    Standard Linear Quadratic Regulator (LQR) for Cart-Pole.
    1. Linearizes the nominal plant around upright equilibrium s = [0, 0, 0, 0].
    2. Discretizes (A, B) at control timestep dt.
    3. Solves the Discrete Algebraic Riccati Equation (DARE) for optimal K.
    """
    def __init__(
        self,
        nominal_params: PlantParams,
        dt: float = 0.005,
        u_max: float = 20.0,
        x_ref: float = 0.0,
        Q: np.ndarray = None,
        R: np.ndarray = None,
    ):
        self.p = nominal_params
        self.dt = dt
        self.u_max = u_max
        self.x_ref = x_ref

        # Well-damped weights keeping initial control effort u(0) < u_max (20N)
        # for x0 = 0.5m, th0 = 15 deg:
        # q_x=20, q_xdot=8, q_th=300, q_thdot=30, R=1.0
        self.Q = np.diag([20.0, 8.0, 300.0, 30.0]) if Q is None else Q
        self.R = np.array([[1.0]]) if R is None else R

        # 1. Linearize continuous-time state-space matrices A and B
        self.A, self.B = self._linearize_plant(nominal_params)

        # 2. Verify Kalman Controllability Rank == 4
        C_mat = np.hstack([
            self.B,
            self.A @ self.B,
            np.linalg.matrix_power(self.A, 2) @ self.B,
            np.linalg.matrix_power(self.A, 3) @ self.B,
        ])
        self.controllability_rank = int(np.linalg.matrix_rank(C_mat))
        if self.controllability_rank < 4:
            raise ValueError(f"Plant is not controllable! Rank = {self.controllability_rank}")

        # 3. Synthesize optimal discrete-time feedback gain K
        self.K, self.P, self.eigvals = self._synthesize_dlqr(self.A, self.B, self.Q, self.R, self.dt)
        self.last_th_eq = 0.0

    @staticmethod
    def _linearize_plant(p: PlantParams) -> Tuple[np.ndarray, np.ndarray]:
        I_t = p.I + p.m * (p.l ** 2)
        D0 = (p.M + p.m) * I_t - (p.m * p.l) ** 2

        A = np.array([
            [0.0, 1.0, 0.0, 0.0],
            [0.0, -I_t * p.b_c / D0, -(p.m ** 2) * (p.l ** 2) * p.g / D0, p.m * p.l * p.b_p / D0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, p.m * p.l * p.b_c / D0, (p.M + p.m) * p.m * p.g * p.l / D0, -(p.M + p.m) * p.b_p / D0],
        ], dtype=float)

        B = np.array([
            [0.0],
            [I_t / D0],
            [0.0],
            [-p.m * p.l / D0],
        ], dtype=float)

        return A, B

    @staticmethod
    def _synthesize_dlqr(
        A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray, dt: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # 4th-order Taylor matrix exponential for Zero-Order Hold discretization
        I4 = np.eye(4)
        A2 = A @ A
        A3 = A2 @ A
        A4 = A3 @ A
        Ad = I4 + A * dt + 0.5 * A2 * (dt ** 2) + (1.0 / 6.0) * A3 * (dt ** 3) + (1.0 / 24.0) * A4 * (dt ** 4)
        Bd = (I4 * dt + 0.5 * A * (dt ** 2) + (1.0 / 6.0) * A2 * (dt ** 3) + (1.0 / 24.0) * A3 * (dt ** 4)) @ B

        if SCIPY_AVAILABLE:
            P = solve_discrete_are(Ad, Bd, Q * dt, R * dt)
        else:
            Qd, Rd = Q * dt, R * dt
            P = Qd.copy()
            for _ in range(15000):
                BtP = Bd.T @ P
                K_step = np.linalg.solve(Rd + BtP @ Bd, BtP @ Ad)
                P_next = Ad.T @ P @ (Ad - Bd @ K_step) + Qd
                if np.max(np.abs(P_next - P)) < 1e-12:
                    P = P_next
                    break
                P = P_next

        Rd = R * dt
        K = np.linalg.solve(Rd + Bd.T @ P @ Bd, Bd.T @ P @ Ad).flatten()

        # Compute continuous-time equivalent closed-loop eigenvalues of (A - B*K)
        cl_eigvals = np.linalg.eigvals(A - B @ K.reshape(1, -1))
        return K, P, cl_eigvals

    def reset(self):
        self.last_th_eq = 0.0

    def act(self, s: np.ndarray, t: float = 0.0) -> float:
        """
        Full-state optimal feedback control law: u = -K * (s - s_ref)
        """
        ex = s[0] - self.x_ref
        xdot = s[1]
        th = s[2]
        thdot = s[3]

        s_err = np.array([ex, xdot, th, thdot], dtype=float)

        # Equivalent target lean angle implied by LQR position & velocity gains
        # (for visualization comparison with outer-loop th_ref)
        self.last_th_eq = float(-(self.K[0] * ex + self.K[1] * xdot) / self.K[2])

        u_unsat = float(-np.dot(self.K, s_err))
        return float(np.clip(u_unsat, -self.u_max, self.u_max))


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

    # Practical settling band: |theta| < 1.0 deg AND |x - x_ref| < 0.05 m
    in_band_practical = (np.abs(th_deg) < 1.0) & (np.abs(x - x_ref) < 0.05)
    out_prac = np.where(~in_band_practical)[0]
    ts_prac = 0.0 if len(out_prac) == 0 else (float(t[-1]) if out_prac[-1] == len(t) - 1 else float(t[out_prac[-1] + 1]))

    # Strict settling band: |theta| < 0.5 deg AND |x - x_ref| < 0.02 m
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
    controller: LQRController,
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
    th_eq_hist = np.zeros(steps)

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
        th_eq_hist[k] = controller.last_th_eq
        s = s_next

    metrics = compute_metrics(t_hist, s_hist, u_hist, x_ref=controller.x_ref)
    return t_hist, s_hist, u_hist, th_eq_hist, metrics


def main():
    dt = 0.005
    T = 15.0
    nominal_params = PlantParams()
    perturbed_params = PlantParams(M=1.1, m=0.26, l=0.60, b_c=0.07, b_p=0.003)

    lqr = LQRController(nominal_params=nominal_params, dt=dt, u_max=nominal_params.u_max)

    print("\n" + "=" * 95)
    print("STANDALONE LQR CONTROLLER BENCHMARK EVALUATION SUMMARY")
    print("=" * 95)
    print(f"Controllability Rank      : {lqr.controllability_rank}/4")
    print(f"Synthesized LQR Gain K    : {np.round(lqr.K, 4)}")
    print(f"Closed-Loop Eigenvalues   : {np.round(lqr.eigvals, 3)}")

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
    for name, cfg in scenarios.items():
        t, s, u, th_eq, m = run_scenario(
            cfg["params"], lqr, cfg["init"], T=T, dt=dt, disturbance_window=cfg["dist"]
        )
        results[name] = (t, s, u, th_eq, m)
        print(f"\nScenario: {name}")
        for k, v in m.items():
            print(f"  {k:<32}: {v}")

    # -----------------------------------------------------------------
    # Plot 3-Scenario LQR Benchmark Dashboard
    # -----------------------------------------------------------------
    fig, axes = plt.subplots(3, 3, figsize=(15, 9), sharex=True)
    fig.suptitle(
        "Optimal LQR Controller Benchmark Across Test Scenarios (Q = diag(20, 8, 300, 30), R = 1.0)",
        fontsize=13,
        fontweight="bold",
    )

    for col, (name, (t, s, u, th_eq, m)) in enumerate(results.items()):
        x = s[:, 0]
        th_deg = np.rad2deg(s[:, 2])
        th_eq_deg = np.rad2deg(th_eq)

        # Row 0: Cart Position x(t)
        ax_x = axes[0, col]
        ax_x.plot(t, x, color="#1f77b4", lw=2, label="Cart x (m)")
        ax_x.axhline(0.0, color="black", ls="--", lw=1, alpha=0.6, label="x_ref")
        ax_x.set_title(name, fontsize=10, fontweight="bold")
        ax_x.grid(True, alpha=0.3)
        if col == 0:
            ax_x.set_ylabel("Position x (m)")
            ax_x.legend(loc="upper right")

        # Row 1: Pole Angle theta(t) vs Implicit LQR Equilibrium Angle th_eq(t)
        ax_th = axes[1, col]
        ax_th.plot(t, th_deg, color="#d62728", lw=2, label="Pole theta (deg)")
        ax_th.plot(t, th_eq_deg, color="#2ca02c", ls="--", lw=1.5, label="Implicit LQR th_eq")
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