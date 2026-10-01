"""
Optimised RL controller for the cart-pole benchmark (PID / LQR / RL comparison).

Changes vs. the previous version
--------------------------------
1. Training dynamics match deployment: the 500 N/s slew limit and 20 N saturation
   are modelled inside the differentiable rollout (applied per 5 ms plant sub-step,
   same order as CartPolePlant.step), with a straight-through gradient so learning
   does not stall when the limiter is active.
2. Control rate matches: training uses dt_ctrl = 10 ms (2 x 5 ms RK4 sub-steps) and
   the deployed controller holds each action for 2 plant steps (zero-order hold).
3. Longer, vectorised training: per-sample random disturbance timing, 12 N hits,
   domain randomisation, 6 s final horizon, more epochs.
4. Validation on a fixed 10 s domain-randomised set; the best checkpoint is kept.
5. Policy has a linear skip path (sign-correct init) + small tanh MLP, still
   bias-free and odd-symmetric so pi(0) = 0.
6. Fast NumPy inference at evaluation time.
7. Benchmark is identical to the PID run: T = 10 s, 12 N hit, both settling bands.

Usage:
    python rl_cartpole_optimised.py              # full training (~10-20 min CPU)
    python rl_cartpole_optimised.py --quick      # short smoke-test training
    python rl_cartpole_optimised.py --load rl_policy.pt   # skip training
"""
import argparse
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# Shared benchmark settings (keep identical across PID / LQR / RL)
BENCH_T = 10.0
BENCH_DT = 0.005
BENCH_DIST_FORCE = 12.0
BENCH_DIST_WINDOW = (3.0, 3.10)


# =====================================================================
# 1. PLANT (identical to PID/LQR files)
# =====================================================================
@dataclass
class PlantParams:
    M: float = 1.0
    m: float = 0.2
    l: float = 0.5
    b_c: float = 0.1
    b_p: float = 0.005
    g: float = 9.81
    u_max: float = 20.0
    du_max: float = 500.0

    @property
    def I(self) -> float:
        return (1.0 / 3.0) * self.m * (self.l ** 2)


class CartPolePlant:
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
        sin_th, cos_th = np.sin(th), np.cos(th)
        I_total = p.I + p.m * (p.l ** 2)
        denom = (p.M + p.m) * I_total - (p.m * p.l * cos_th) ** 2
        F_eff = u_total - p.b_c * x_dot + p.m * p.l * (th_dot ** 2) * sin_th
        T_eff = p.m * p.g * p.l * sin_th - p.b_p * th_dot
        x_ddot = (I_total * F_eff - p.m * p.l * cos_th * T_eff) / denom
        th_ddot = ((p.M + p.m) * T_eff - p.m * p.l * cos_th * F_eff) / denom
        return np.array([x_dot, x_ddot, th_dot, th_ddot], dtype=float)

    def step(self, u_cmd: float, disturbance_force: float = 0.0) -> Tuple[np.ndarray, float]:
        max_delta = self.p.du_max * self.dt
        u_rl = np.clip(u_cmd, self.prev_u - max_delta, self.prev_u + max_delta)
        u_applied = float(np.clip(u_rl, -self.p.u_max, self.p.u_max))
        self.prev_u = u_applied
        u_total = u_applied + disturbance_force

        s, dt = self.state, self.dt
        k1 = self._derivatives(s, u_total)
        k2 = self._derivatives(s + 0.5 * dt * k1, u_total)
        k3 = self._derivatives(s + 0.5 * dt * k2, u_total)
        k4 = self._derivatives(s + dt * k3, u_total)
        self.state = s + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        self.state[2] = (self.state[2] + np.pi) % (2.0 * np.pi) - np.pi
        return self.state.copy(), u_applied


# =====================================================================
# 2. BATCHED TORCH ENVIRONMENT (slew limit + saturation included)
# =====================================================================
class BatchedCartPole:
    """
    N parallel cart-poles. One call to `control_interval` advances one control
    period (n_sub plant sub-steps of dt_plant) while holding the commanded force,
    applying the same slew-rate limit and saturation as CartPolePlant.step.
    """
    def __init__(self, nominal: PlantParams, dt_plant: float, n_sub: int, device: torch.device):
        self.nom = nominal
        self.dt_plant = dt_plant
        self.n_sub = n_sub
        self.dt_ctrl = dt_plant * n_sub
        self.device = device

    def sample_params(self, n: int, randomize: bool) -> Dict[str, torch.Tensor]:
        nom, dev = self.nom, self.device

        def u(lo, hi):
            return torch.empty(n, device=dev).uniform_(lo, hi)

        if randomize:
            M = nom.M * u(0.80, 1.25)
            m = nom.m * u(0.75, 1.35)
            l = nom.l * u(0.80, 1.25)
            b_c = nom.b_c * u(0.60, 1.40)
            b_p = nom.b_p * u(0.60, 1.40)
        else:
            full = lambda v: torch.full((n,), v, device=dev)
            M, m, l, b_c, b_p = full(nom.M), full(nom.m), full(nom.l), full(nom.b_c), full(nom.b_p)
        return {"M": M, "m": m, "l": l, "b_c": b_c, "b_p": b_p, "I": (1.0 / 3.0) * m * l ** 2}

    def _deriv(self, s, u_total, p):
        x_dot, th, th_dot = s[:, 1], s[:, 2], s[:, 3]
        M, m, l, b_c, b_p, I = p["M"], p["m"], p["l"], p["b_c"], p["b_p"], p["I"]
        sin_th, cos_th = torch.sin(th), torch.cos(th)
        I_total = I + m * l ** 2
        denom = (M + m) * I_total - (m * l * cos_th) ** 2
        F_eff = u_total - b_c * x_dot + m * l * th_dot ** 2 * sin_th
        T_eff = m * self.nom.g * l * sin_th - b_p * th_dot
        x_ddot = (I_total * F_eff - m * l * cos_th * T_eff) / denom
        th_ddot = ((M + m) * T_eff - m * l * cos_th * F_eff) / denom
        return torch.stack([x_dot, x_ddot, th_dot, th_ddot], dim=-1)

    def _rk4(self, s, u_total, p):
        dt = self.dt_plant
        k1 = self._deriv(s, u_total, p)
        k2 = self._deriv(s + 0.5 * dt * k1, u_total, p)
        k3 = self._deriv(s + 0.5 * dt * k2, u_total, p)
        k4 = self._deriv(s + dt * k3, u_total, p)
        return s + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    def control_interval(self, s, u_cmd, prev_u, p, d_u):
        """Returns (next_state, last applied force). Straight-through gradient on limiter."""
        dl = self.nom.du_max * self.dt_plant
        u_hold = u_cmd.detach()
        u_prev = prev_u.detach()
        u_sub = u_prev
        for _ in range(self.n_sub):
            u_sub = torch.clamp(u_hold, u_sub - dl, u_sub + dl)
            u_sub = torch.clamp(u_sub, -self.nom.u_max, self.nom.u_max)
            # forward value = limited force, backward gradient = identity wrt u_cmd
            u_eff = u_sub + (u_cmd - u_hold)
            s = self._rk4(s, u_eff + d_u, p)
        return s, u_eff


# =====================================================================
# 3. POLICY
# =====================================================================
class PolicyNetwork(nn.Module):
    """Bias-free, odd-symmetric:  u = u_max * tanh( W_lin s_n + MLP(s_n) )."""
    def __init__(self, u_max: float = 20.0):
        super().__init__()
        self.u_max = u_max
        self.register_buffer("s_scale", torch.tensor([1.0, 2.0, 0.35, 3.0]))
        self.lin = nn.Linear(4, 1, bias=False)
        self.net = nn.Sequential(
            nn.Linear(4, 64, bias=False), nn.Tanh(),
            nn.Linear(64, 64, bias=False), nn.Tanh(),
            nn.Linear(64, 1, bias=False),
        )
        for mod in self.net:
            if isinstance(mod, nn.Linear):
                nn.init.orthogonal_(mod.weight, gain=0.6)
        nn.init.orthogonal_(self.net[-1].weight, gain=0.05)
        # Sign-correct prior only (push toward the lean, same sign on x / xdot): NOT an LQR solution
        with torch.no_grad():
            self.lin.weight.copy_(torch.tensor([[0.2, 0.3, 1.0, 0.3]]))

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        sn = s / self.s_scale
        return self.u_max * torch.tanh(self.lin(sn) + self.net(sn)).squeeze(-1)


# =====================================================================
# 4. RL CONTROLLER (training + deployment)
# =====================================================================
class RLController:
    DIST_LEN = 10  # control steps (=0.1 s) for training disturbance pulses

    def __init__(self, nominal_params: PlantParams, dt: float = 0.005, u_max: float = 20.0,
                 x_ref: float = 0.0, device: str = "cpu", n_sub: int = 2):
        self.p = nominal_params
        self.dt = dt
        self.u_max = u_max
        self.x_ref = x_ref
        self.device = torch.device(device)
        self.n_sub = n_sub  # plant steps per control update (ZOH)
        self.env = BatchedCartPole(nominal_params, dt, n_sub, self.device)
        self.policy = PolicyNetwork(u_max).to(self.device)
        self.loss_history: List[float] = []
        self.val_history: List[Tuple[int, float]] = []
        self.equivalent_gains = np.zeros(4)
        self._np_w = None
        self.reset()

    # ----------------------------- training ---------------------------
    def _rollout(self, s, p, steps, w, dist_start=None, dist_mag=None):
        env, dtc = self.env, self.env.dt_ctrl
        n = s.shape[0]
        prev_u = torch.zeros(n, device=self.device)
        cost = torch.zeros(n, device=self.device)
        fall_lim = np.deg2rad(40.0)
        for k in range(steps):
            u_cmd = self.policy(s)
            if dist_mag is not None:
                active = (k >= dist_start) & (k < dist_start + self.DIST_LEN)
                d_u = torch.where(active, dist_mag, torch.zeros_like(dist_mag))
            else:
                d_u = 0.0
            s_next, u_app = env.control_interval(s, u_cmd, prev_u, p, d_u)
            du = u_app - prev_u
            cost = cost + (
                w["qx"] * s[:, 0] ** 2 + w["qxd"] * s[:, 1] ** 2
                + w["qth"] * s[:, 2] ** 2 + w["qthd"] * s[:, 3] ** 2
                + w["ru"] * u_app ** 2 + w["rdu"] * du ** 2
            ) * dtc + 500.0 * torch.relu(s_next[:, 2].abs() - fall_lim) ** 2
            prev_u, s = u_app, s_next
        terminal = 2.5 * (w["qx"] * s[:, 0] ** 2 + w["qxd"] * s[:, 1] ** 2
                          + w["qth"] * s[:, 2] ** 2 + w["qthd"] * s[:, 3] ** 2)
        return cost + terminal

    def _sample_init(self, n):
        dev = self.device
        u = lambda a: torch.empty(n, device=dev).uniform_(-a, a)
        return torch.stack([u(0.75), u(0.8), u(np.deg2rad(22.0)), u(1.2)], dim=-1)

    def _make_val_set(self, n=128, steps=1000):
        g_state = torch.get_rng_state()
        torch.manual_seed(1234)
        s0 = self._sample_init(n)
        p = self.env.sample_params(n, randomize=True)
        dstart = torch.randint(30, 400, (n,), device=self.device)
        dmag = torch.where(torch.rand(n, device=self.device) < 0.5,
                           torch.empty(n, device=self.device).uniform_(-12.0, 12.0),
                           torch.zeros(n, device=self.device))
        torch.set_rng_state(g_state)
        return s0, p, dstart, dmag, steps

    @torch.no_grad()
    def _validate(self, val, w):
        s0, p, ds, dm, steps = val
        cost = self._rollout(s0, p, steps, w, ds, dm)
        return float(cost.mean().item())

    def train_policy(self, scale: float = 1.0, verbose: bool = True):
        base = dict(qx=22.0, qxd=9.0, qth=300.0, qthd=32.0, ru=1.0, rdu=0.05)
        stages = [
            dict(name="Stage 1 (Angle catch)", epochs=100, steps=80, lr=4e-3, dr=False, dist=False,
                 w=dict(base, qx=2.0, qxd=1.0, qth=320.0, qthd=32.0, ru=0.6, rdu=0.02)),
            dict(name="Stage 2 (Position recovery)", epochs=150, steps=250, lr=3e-3, dr=False, dist=False,
                 w=dict(base, rdu=0.05)),
            dict(name="Stage 3 (DR + disturbances)", epochs=250, steps=600, lr=2e-3, dr=True, dist=True,
                 w=dict(base, rdu=0.08)),
        ]
        val_w = stages[-1]["w"]
        val = self._make_val_set()
        batch = 256
        best_val, best_state = float("inf"), None
        t0 = time.time()
        epoch_global = 0

        if verbose:
            print("\n" + "=" * 95)
            print("TRAINING RL POLICY (slew-limited rollouts, 10 ms control, DR + disturbances)")
            print("=" * 95)

        for st in stages:
            epochs = max(5, int(st["epochs"] * scale))
            opt = torch.optim.AdamW(self.policy.parameters(), lr=st["lr"], weight_decay=1e-4)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=2e-4)
            for ep in range(1, epochs + 1):
                opt.zero_grad()
                s0 = self._sample_init(batch)
                p = self.env.sample_params(batch, randomize=st["dr"])
                if st["dist"]:
                    dstart = torch.randint(10, max(11, st["steps"] // 2), (batch,), device=self.device)
                    dmag = torch.where(torch.rand(batch, device=self.device) < 0.30,
                                       torch.empty(batch, device=self.device).uniform_(-12.0, 12.0),
                                       torch.zeros(batch, device=self.device))
                else:
                    dstart = dmag = None
                loss = self._rollout(s0, p, st["steps"], st["w"], dstart, dmag).mean()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), 5.0)
                opt.step()
                sched.step()
                self.loss_history.append(float(loss.item()))
                epoch_global += 1

                if st["dr"] and (ep % 10 == 0 or ep == epochs):
                    v = self._validate(val, val_w)
                    self.val_history.append((epoch_global, v))
                    if v < best_val:
                        best_val = v
                        best_state = {k: t.clone() for k, t in self.policy.state_dict().items()}
                    if verbose and ep % 50 == 0:
                        print(f"    ep {ep:4d}/{epochs} | train {loss.item():9.2f} | "
                              f"val(10s) {v:9.2f} | best {best_val:9.2f}")
            if verbose:
                print(f"  {st['name']:<30} | horizon {st['steps'] * self.env.dt_ctrl:4.1f}s | "
                      f"final train cost {loss.item():9.2f} | elapsed {time.time() - t0:6.1f}s")

        if best_state is not None:
            self.policy.load_state_dict(best_state)
            if verbose:
                print(f"Restored best validation checkpoint (val cost {best_val:.2f})")
        self._finalise(verbose)

    # ------------------------- deployment helpers ---------------------
    def _finalise(self, verbose=True):
        self.policy.eval()
        s0 = torch.zeros(1, 4, device=self.device, requires_grad=True)
        self.policy(s0).sum().backward()
        self.equivalent_gains = s0.grad.detach().cpu().numpy().flatten()
        sd = {k: v.detach().cpu().numpy().astype(np.float64) for k, v in self.policy.state_dict().items()}
        self._np_w = dict(
            sc=sd["s_scale"], lin=sd["lin.weight"], w1=sd["net.0.weight"],
            w2=sd["net.2.weight"], w3=sd["net.4.weight"],
        )
        if verbose:
            print(f"Local gains du/ds at 0 [x, xd, th, thd]: {np.round(self.equivalent_gains, 3)}")
            if not (self.equivalent_gains[2] > 0 and self.equivalent_gains[3] > 0):
                print("  WARNING: angle / rate gains are not positive -> policy is likely not stabilising.")

    def save(self, path: str):
        torch.save(self.policy.state_dict(), path)

    def load(self, path: str):
        self.policy.load_state_dict(torch.load(path, map_location=self.device))
        self._finalise()

    def reset(self):
        self.last_th_eq = 0.0
        self._hold = 0
        self._u_hold = 0.0

    def _forward_np(self, s_err: np.ndarray) -> float:
        w = self._np_w
        sn = s_err / w["sc"]
        z = (w["lin"] @ sn) + w["w3"] @ np.tanh(w["w2"] @ np.tanh(w["w1"] @ sn))
        return float(self.u_max * np.tanh(z[0]))

    def act(self, s: np.ndarray, t: float = 0.0) -> float:
        """Zero-order hold: recompute every n_sub plant steps (matches training)."""
        ex = s[0] - self.x_ref
        if self._hold == 0:
            self._u_hold = self._forward_np(np.array([ex, s[1], s[2], s[3]]))
            g_x, g_xd, g_th, _ = self.equivalent_gains
            if abs(g_th) > 1e-3:
                self.last_th_eq = float(-(g_x * ex + g_xd * s[1]) / g_th)
        self._hold = (self._hold + 1) % self.n_sub
        return float(np.clip(self._u_hold, -self.u_max, self.u_max))


# =====================================================================
# 5. METRICS (same definitions as PID file, both settling bands)
# =====================================================================
def _settle(t, mask_in):
    out = np.where(~mask_in)[0]
    if len(out) == 0:
        return 0.0
    return float(t[-1]) if out[-1] == len(t) - 1 else float(t[out[-1] + 1])


def compute_metrics(t, states, u_hist, x_ref=0.0) -> Dict[str, float]:
    dt = t[1] - t[0]
    x = states[:, 0]
    th_deg = np.rad2deg(states[:, 2])

    stabilised = bool(np.max(np.abs(th_deg)) < 45.0 and np.abs(th_deg[-1]) < 1.0)
    ts_prac = _settle(t, (np.abs(th_deg) < 1.0) & (np.abs(x - x_ref) < 0.05))
    ts_strict = _settle(t, (np.abs(th_deg) < 0.5) & (np.abs(x - x_ref) < 0.02))

    th0 = np.abs(th_deg[0])
    if th0 > 1.0:
        i90 = np.where(np.abs(th_deg) <= 0.10 * th0)[0]
        i10 = np.where(np.abs(th_deg) <= 0.90 * th0)[0]
        rise = float(t[i90[0]] - t[i10[0]]) if (len(i90) and len(i10)) else np.nan
        opp = -np.sign(th_deg[0]) * th_deg
        os_pct = max(0.0, float(np.max(opp))) / th0 * 100.0
    else:
        rise, os_pct = 0.0, float(np.max(np.abs(th_deg)))

    n_ss = int(1.0 / dt)
    return {
        "Stabilised": stabilised,
        "Settling Time [1deg, 5cm] (s)": round(ts_prac, 3),
        "Settling Time [0.5deg, 2cm] (s)": round(ts_strict, 3),
        "Rise Time (s)": round(rise, 3) if not np.isnan(rise) else -1.0,
        "Overshoot (%) / Peak (deg)": round(os_pct, 2),
        "SS Error |x| (m)": round(float(np.mean(np.abs(x[-n_ss:] - x_ref))), 4),
        "SS Error |th| (deg)": round(float(np.mean(np.abs(th_deg[-n_ss:]))), 4),
        "Control Energy int(u^2 dt)": round(float(np.sum(u_hist ** 2) * dt), 2),
        "Peak |u| (N)": round(float(np.max(np.abs(u_hist))), 2),
    }


# =====================================================================
# 6. SIMULATION + PLOTS
# =====================================================================
def run_scenario(plant_params, controller, init_state, T=BENCH_T, dt=BENCH_DT,
                 disturbance_window=(0.0, 0.0, 0.0)):
    plant = CartPolePlant(params=plant_params, dt=dt)
    controller.reset()
    steps = int(T / dt)
    t_h, s_h = np.zeros(steps), np.zeros((steps, 4))
    u_h, th_eq_h = np.zeros(steps), np.zeros(steps)
    s = plant.reset(np.array(init_state, dtype=float))
    d0, d1, df = disturbance_window
    for k in range(steps):
        t = k * dt
        d_u = df if (d0 <= t <= d1) else 0.0
        u_cmd = controller.act(s, t)
        s_next, u_app = plant.step(u_cmd, disturbance_force=d_u)
        t_h[k], s_h[k], u_h[k], th_eq_h[k] = t, s, u_app, controller.last_th_eq
        s = s_next
    return t_h, s_h, u_h, th_eq_h, compute_metrics(t_h, s_h, u_h, controller.x_ref)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="short training for a smoke test")
    ap.add_argument("--load", type=str, default=None, help="load saved policy, skip training")
    ap.add_argument("--save", type=str, default="rl_policy.pt")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    nominal = PlantParams()
    perturbed = PlantParams(M=1.1, m=0.26, l=0.60, b_c=0.07, b_p=0.003)

    rl = RLController(nominal, dt=BENCH_DT, u_max=nominal.u_max, device=args.device, n_sub=2)
    if args.load:
        rl.load(args.load)
    else:
        rl.train_policy(scale=0.1 if args.quick else 1.0)
        rl.save(args.save)
        print(f"Saved policy to {args.save}")

    d0, d1 = BENCH_DIST_WINDOW
    scenarios = {
        "1. Initial Offset (x0=0.5m, th0=15 deg)": dict(
            params=nominal, init=[0.5, 0.0, np.deg2rad(15.0), 0.0], dist=(0.0, 0.0, 0.0)),
        f"2. Impulse Disturbance ({BENCH_DIST_FORCE:.0f}N hit @ t={d0:.1f}s)": dict(
            params=nominal, init=[0.0, 0.0, 0.0, 0.0], dist=(d0, d1, BENCH_DIST_FORCE)),
        "3. Perturbed Plant (M+10%, m+30%, l+20%, b_c-30%)": dict(
            params=perturbed, init=[0.5, 0.0, np.deg2rad(15.0), 0.0], dist=(0.0, 0.0, 0.0)),
    }

    print("\n" + "=" * 95)
    print("RL CONTROLLER BENCHMARK EVALUATION SUMMARY")
    print("=" * 95)
    results = {}
    for name, cfg in scenarios.items():
        out = run_scenario(cfg["params"], rl, cfg["init"], disturbance_window=cfg["dist"])
        results[name] = out
        print(f"\nScenario: {name}")
        for k, v in out[4].items():
            print(f"  {k:<32}: {v}")

    if args.no_plot:
        return

    fig, axes = plt.subplots(3, 3, figsize=(15, 9), sharex=True)
    fig.suptitle("Optimised RL Controller Benchmark Across Test Scenarios", fontsize=14, fontweight="bold")
    for col, (name, (t, s, u, th_eq, _)) in enumerate(results.items()):
        ax = axes[0, col]
        ax.plot(t, s[:, 0], color="#1f77b4", lw=2, label="Cart x (m)")
        ax.axhline(0.0, color="black", ls="--", lw=1, alpha=0.6, label="x_ref")
        ax.set_title(name, fontsize=10, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax = axes[1, col]
        ax.plot(t, np.rad2deg(s[:, 2]), color="#d62728", lw=2, label="Pole theta (deg)")
        ax.plot(t, np.rad2deg(th_eq), color="#2ca02c", ls="--", lw=1.2, label="Implicit th_eq (linearised)")
        ax.axhline(0.0, color="black", ls=":", lw=1, alpha=0.5)
        ax.grid(True, alpha=0.3)
        ax = axes[2, col]
        ax.plot(t, u, color="#9467bd", lw=1.8, label="Force u (N)")
        ax.axhline(nominal.u_max, color="red", ls="--", lw=1, alpha=0.5, label="Saturation")
        ax.axhline(-nominal.u_max, color="red", ls="--", lw=1, alpha=0.5)
        ax.set_xlabel("Time (s)")
        ax.grid(True, alpha=0.3)
        if col == 0:
            axes[0, 0].set_ylabel("Position x (m)")
            axes[1, 0].set_ylabel("Angle (deg)")
            axes[2, 0].set_ylabel("Control Force u (N)")
            for r in range(3):
                axes[r, 0].legend(loc="upper right")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()