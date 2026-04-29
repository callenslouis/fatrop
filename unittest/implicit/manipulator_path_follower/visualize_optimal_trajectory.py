import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import casadi as ca

from visualize_robot_beam import show_robot_with_beam, show_robot_with_surface

# ── Configuration ──────────────────────────────────────────────────────────────
# scenario = "beam"
scenario = "contact"
SOLUTION_FILE  = f"manipulator_path_follower_{scenario}_solution.json"
PATH_FUNC_FILE = f"path_func_{scenario}.casadi"
L              = 0.3
FOLDER         = "panda/"
FPS            = 25          # animation frames per second
PLAYBACK_SPEED = 1.0         # set < 1 to slow down, > 1 to speed up

# ── Load solution ──────────────────────────────────────────────────────────────
with open(SOLUTION_FILE, "r") as f:
    sol = json.load(f)

N   = sol["N"]
T   = sol["T"]
q   = np.array(sol["q"])    # (7, N+1)
if scenario == "beam":
    th  = np.array(sol["th"])   # (1, N+1)
else:
    qd = np.array(sol["qd"])  # (7, N+1)
dp  = np.array(sol["dp"])   # (1, N)  — real-time duration of each interval
p   = np.array(sol["p"]) if sol["p"] is not None else np.linspace(0, T, N + 1).reshape(1, -1)

# ── Build the real-time knot vector from dp ────────────────────────────────────
# dp[:,k] is the real-time length of interval k  →  cumsum gives knot times
dp_flat = dp.flatten()                     # length N
t_knots = np.concatenate([[0.0], np.cumsum(dp_flat)])   # length N+1
T_total = t_knots[-1]                      # should equal T

# ── Uniform animation time grid ────────────────────────────────────────────────
n_frames   = max(2, int(np.ceil(T_total * FPS / PLAYBACK_SPEED)))
t_uniform  = np.linspace(0.0, T_total, n_frames)
interval_ms = int(1000 / FPS)

# ── Helper: linearly interpolate q and th onto a scalar time value ─────────────
def interp_state(t_query):
    """Return (q_interp [7], th_interp [scalar]) at real time t_query."""
    # find surrounding knot indices
    idx = np.searchsorted(t_knots, t_query, side="right") - 1
    idx = np.clip(idx, 0, N - 1)
    t0, t1 = t_knots[idx], t_knots[idx + 1]
    alpha  = (t_query - t0) / (t1 - t0) if (t1 > t0) else 0.0
    alpha  = float(np.clip(alpha, 0.0, 1.0))

    q_interp  = (1 - alpha) * q[:, idx]  + alpha * q[:, idx + 1]
    if scenario == "beam":
        th_interp = (1 - alpha) * float(th[idx]) + alpha * float(th[idx + 1])
        return q_interp, th_interp
    else:
        qd_interp = (1 - alpha) * qd[:, idx] + alpha * qd[:, idx + 1]
        return q_interp, qd_interp

# ── Desired trajectory (evaluate at uniform time grid) ────────────────────────
path_func = ca.Function.load(PATH_FUNC_FILE)
if scenario == "contact":
    contact_force_func = ca.Function.load(f"{FOLDER}panda_normal_force.casadi")  # for visualizing contact forces

# The path_func takes the progress variable p, not real time.
# Interpolate p onto the uniform time grid first.
p_flat = p.flatten()   # length N+1, indexed by knot

def interp_p(t_query):
    idx   = np.searchsorted(t_knots, t_query, side="right") - 1
    idx   = np.clip(idx, 0, N - 1)
    t0, t1 = t_knots[idx], t_knots[idx + 1]
    alpha  = (t_query - t0) / (t1 - t0) if (t1 > t0) else 0.0
    alpha  = float(np.clip(alpha, 0.0, 1.0))
    return (1 - alpha) * p_flat[idx] + alpha * p_flat[idx + 1]

desired_positions = np.zeros((3, n_frames))
for i, t in enumerate(t_uniform):
    pk = interp_p(t)
    desired_positions[:, i] = np.array(path_func(pk)).flatten()

# ── Actual beam-tip trace (at uniform time grid) ──────────────────────────────
relevant_end_pos = ca.Function.load(f"{FOLDER}panda_beam_end_pos.casadi") if scenario == "beam" else ca.Function.load(f"{FOLDER}panda_fkpos_ee.casadi")

actual_tip = np.zeros((3, n_frames))
for i, t in enumerate(t_uniform):
    if scenario == "contact":
        qk, qdk = interp_state(t)
        actual_tip[:, i] = np.array(relevant_end_pos(qk)).flatten()  # th=0 for contact scenario
    else:
        qk, thk = interp_state(t)
        actual_tip[:, i] = np.array(relevant_end_pos(qk, thk, L)).flatten()

# ── Axis limits ────────────────────────────────────────────────────────────────
all_pts = np.hstack([desired_positions, actual_tip])
margin  = 0.15
xlim = (all_pts[0].min() - margin, all_pts[0].max() + margin)
ylim = (all_pts[1].min() - margin, all_pts[1].max() + margin)
zlim = (max(0.0, all_pts[2].min() - margin), all_pts[2].max() + margin)

# ── Figure ─────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(10, 8))
ax  = fig.add_subplot(111, projection="3d")

# ── Update function ────────────────────────────────────────────────────────────
def update(frame):
    ax.cla()

    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    ax.set_title("Panda + Beam – Path Following Animation")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_zlim(*zlim)

    # 3. Robot + beam at interpolated configuration
    t_now = t_uniform[frame]
    if scenario == "beam":
        qk, thk = interp_state(t_now)
        show_robot_with_beam(ax, qk, thk, L)
    else:
        qk, qdk = interp_state(t_now)
        show_robot_with_surface(ax, qk, sol["floor_height"])


    # 1. Full desired path (static)
    ax.plot(
        desired_positions[0], desired_positions[1], desired_positions[2],
        "g--", linewidth=1.5, label="Desired path", zorder=2,
    )

    # 2. Beam-tip trace up to current frame
    ax.plot(
        actual_tip[0, :frame + 1],
        actual_tip[1, :frame + 1],
        actual_tip[2, :frame + 1],
        "b-", linewidth=1.5, label="Beam tip trace", zorder=3,
    )

    # 4. Time stamp
    ax.text2D(0.02, 0.95, f"t = {t_now:.2f} s  (frame {frame}/{n_frames - 1})",
              transform=ax.transAxes, fontsize=10)
    if scenario == "beam":
        # show theta value as well
        thk_degree = np.degrees(thk)
        ax.text2D(0.02, 0.90, f"theta = {thk_degree:.2f} deg",
              transform=ax.transAxes, fontsize=10)
    else:
        force = contact_force_func(qk, qdk, sol["floor_height"], sol['k'], sol['alpha'], sol['d']).full().flatten()
        ax.text2D(0.02, 0.90, f"contact force = {float(force[0]):.2f} N",
              transform=ax.transAxes, fontsize=10)

    ax.legend(loc="upper right", fontsize=8)


# ── Animate ────────────────────────────────────────────────────────────────────
ani = animation.FuncAnimation(
    fig,
    update,
    frames=n_frames,
    interval=interval_ms,
    blit=False,
    repeat=True,
)

plt.tight_layout()

# ── Optional: save ─────────────────────────────────────────────────────────────
ani.save("path_follower_animation.mp4", writer="ffmpeg", fps=FPS, dpi=150)
# ani.save("path_follower_animation.gif", writer="pillow", fps=FPS)

# plt.show()