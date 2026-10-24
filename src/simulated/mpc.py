import asyncio
from math import fabs
import time
import simulated.camera_simulated as my_estimatePoseSingleMarkers
import casadi as ca
import simulated.fc_simulated as fc
from mavsdk_grpc import System
from mavsdk_grpc.offboard import AccelerationNed, PositionNedYaw, VelocityNedYaw
import numpy as np
import ruckig
import simulated.utilities_simulated as utilities


def _ned_to_aligned(vec: np.ndarray, heading_deg: float) -> np.ndarray:
    """Rotates NED vectors into the Drone's Aligned Body Frame (Front-Right-Down)"""
    xy = utilities.rotate_vector(vec[0], vec[1], -heading_deg)
    return np.array([xy[0], xy[1], vec[2]])


def _aligned_to_ned(vec: np.ndarray, heading_deg: float) -> np.ndarray:
    """Rotates Aligned Body Frame vectors back to Earth NED"""
    xy = utilities.rotate_vector(vec[0], vec[1], heading_deg)
    return np.array([xy[0], xy[1], vec[2]])


class RuckigReference:
    """Generates time-optimal feedforward references."""

    def __init__(self):
        self._traj = None
        self._t_start = None
        self.duration = 0.0

    def compute(
        self,
        pos_a: np.ndarray,
        vel_a: np.ndarray,
        acc_a: np.ndarray,
        target_a: np.ndarray,
        j_max: float,
        v_max: float,
        a_max: float,
    ) -> bool:
        otg = ruckig.Ruckig(3)
        inp = ruckig.InputParameter(3)
        traj = ruckig.Trajectory(3)

        inp.current_position = pos_a.tolist()
        inp.current_velocity = vel_a.tolist()
        inp.current_acceleration = acc_a.tolist()
        inp.target_position = target_a.tolist()
        inp.target_velocity = [0.0, 0.0, 0.0]
        inp.target_acceleration = [0.0, 0.0, 0.0]
        inp.max_velocity = [v_max, v_max, v_max]
        inp.max_acceleration = [a_max, a_max, a_max]
        inp.max_jerk = [j_max, j_max, j_max]

        result = otg.calculate(inp, traj)
        if result not in (ruckig.Result.Working, ruckig.Result.Finished):
            return False

        self._traj = traj
        self.duration = traj.duration
        self._t_start = time.perf_counter()
        return True

    @property
    def elapsed(self) -> float:
        return (
            time.perf_counter() - self._t_start if self._t_start else 0.0
        )

    def at_elapsed(self, t: float):
        """Wrapper for Ruckig trajectory sampling."""
        p, v, a = self._traj.at_time(np.clip(t, 0.0, self.duration))
        return np.array(p), np.array(v), np.array(a)

    def horizon(self, t_elapsed: float, N: int, dt: float):
        """Returns P, V, A, J over the prediction horizon."""
        P = np.empty((3, N + 1))
        V = np.empty((3, N + 1))
        A = np.empty((3, N + 1))

        for k in range(N + 1):
            t = float(np.clip(t_elapsed + k * dt, 0.0, self.duration))
            p, v, a = self._traj.at_time(t)
            P[:, k], V[:, k], A[:, k] = p, v, a

        # Jerk from finite difference of acceleration
        J = (A[:, 1:] - A[:, :-1]) / dt
        return P, V, A, J


class ArucoTrackingMPC:
    """Absolute-State Tracking MPC around a Ruckig reference.

    - Constraint: Monotonic descent (vel[2] <= 0).
    """

    G = 9.81

    def __init__(
        self,
        fx: float,
        fy: float,
        frame_width: int,
        frame_height: int,
        dt: float = 0.05,
        horizon: int = 15,
        fov_margin: int = 50,
        j_max: float = 2.0,
        v_max: float = 3.0,
        a_max: float = 3.0,
        w_pos: float = 100.0,
        w_vel: float = 10.0,
        w_acc: float = 5.0,
        w_jerk: float = 0.5,
    ):
        self.fx = fx
        self.fy = fy
        self.W = frame_width
        self.H = frame_height
        self.dt = dt
        self.N = horizon
        self.margin = fov_margin

        self._X_prev = None
        self._U_prev = None

        self._build(
            j_max, v_max, a_max, w_pos, w_vel, w_acc, w_jerk
        )

    def _build(
        self, j_max, v_max, a_max, w_pos, w_vel, w_acc, w_jerk
    ):
        N = self.N
        dt = self.dt
        dt2 = dt**2
        dt3 = dt**3
        g = self.G

        opti = ca.Opti()

        # Parameters
        x0_p = opti.parameter(9)
        ref_pos_p = opti.parameter(3, N + 1)
        ref_vel_p = opti.parameter(3, N + 1)
        ref_acc_p = opti.parameter(3, N + 1)
        ref_jrk_p = opti.parameter(3, N)
        marker_p = opti.parameter(3)

        X = opti.variable(9, N + 1)
        U = opti.variable(3, N)
        slack_x = opti.variable(1, N)
        slack_y = opti.variable(1, N)

        lim_x = self.W / 2.0 - self.margin
        lim_y = self.H / 2.0 - self.margin
        cost = 0.0
        w_slack = 1e6

        for k in range(N):
            pos = X[0:3, k]
            vel = X[3:6, k]
            acc = X[6:9, k]
            jrk = U[:, k]

            opti.subject_to(
                X[:, k + 1]
                == ca.vertcat(
                    pos + vel * dt + 0.5 * acc * dt2 + (1 / 6) * jrk * dt3,
                    vel + acc * dt + 0.5 * jrk * dt2,
                    acc + jrk * dt,
                )
            )

            # --- DESCEND ONLY CONSTRAINT ---
            # FIXED: Apply only for k > 0 to prevent initial measurement noise at k=0 
            # from causing immediate solver infeasibility.
            if k > 0:
                opti.subject_to(vel[2] >= 0.0)

            # Camera Projection
            rel_front = marker_p[0] - pos[0]
            rel_right = marker_p[1] - pos[1]
            altitude = ca.fmax(marker_p[2] - pos[2], 0.3)

            px_trans = +self.fx * rel_right / altitude
            py_trans = -self.fy * rel_front / altitude

            acc_front = acc[0]
            acc_right = acc[1]
            acc_down = acc[2]
            g_eff = ca.fmax(g - acc_down, 2.0)

            px_tilt = +self.fx * (acc_right / g_eff)
            py_tilt = -self.fy * (acc_front / g_eff)

            px = px_trans + px_tilt
            py = py_trans + py_tilt

            opti.subject_to(slack_x[0, k] >= 0)
            opti.subject_to(slack_y[0, k] >= 0)
            opti.subject_to(
                opti.bounded(
                    -lim_x - slack_x[0, k], px, lim_x + slack_x[0, k]
                )
            )
            opti.subject_to(
                opti.bounded(
                    -lim_y - slack_y[0, k], py, lim_y + slack_y[0, k]
                )
            )

            opti.subject_to(opti.bounded(-j_max, jrk, j_max))
            opti.subject_to(opti.bounded(-v_max, vel, v_max))
            opti.subject_to(opti.bounded(-a_max, acc, a_max))

            cost += w_pos * ca.sumsqr(pos - ref_pos_p[:, k])
            cost += w_vel * ca.sumsqr(vel - ref_vel_p[:, k])
            cost += w_acc * ca.sumsqr(acc - ref_acc_p[:, k])
            cost += w_jerk * ca.sumsqr(jrk - ref_jrk_p[:, k])
            cost += w_slack * (
                slack_x[0, k] ** 2 + slack_y[0, k] ** 2
            )

        opti.minimize(cost)
        opti.subject_to(X[:, 0] == x0_p)

        opti.solver(
            "ipopt",
            {},
            {
                "max_iter": 100,
                "print_level": 0,
                "warm_start_init_point": "yes",
                "tol": 1e-3,
                "acceptable_tol": 1e-2,
                "linear_solver": "mumps",
            },
        )

        self._opti = opti
        self._x0_p = x0_p
        self._ref_pos_p = ref_pos_p
        self._ref_vel_p = ref_vel_p
        self._ref_acc_p = ref_acc_p
        self._ref_jrk_p = ref_jrk_p
        self._marker_p = marker_p
        self._X = X
        self._U = U

    def solve(
        self, pos_a, vel_a, acc_a, ref: RuckigReference, marker_a: np.ndarray
    ):
        t_el = ref.elapsed
        x0 = np.concatenate([pos_a, vel_a, acc_a])
        P, V, A, J = ref.horizon(t_el, self.N, self.dt)
        self._opti.set_value(self._x0_p, x0)
        self._opti.set_value(self._ref_pos_p, P)
        self._opti.set_value(self._ref_vel_p, V)
        self._opti.set_value(self._ref_acc_p, A)
        self._opti.set_value(self._ref_jrk_p, J)
        self._opti.set_value(self._marker_p, marker_a)

        if self._X_prev is not None:
            X_ws = np.hstack([self._X_prev[:, 1:], self._X_prev[:, -1:]])
            U_ws = np.hstack([self._U_prev[:, 1:], self._U_prev[:, -1:]])
        else:
            X_ws = np.zeros((9, self.N + 1))
            X_ws[0:3, :] = P
            X_ws[3:6, :] = V
            X_ws[6:9, :] = A
            U_ws = J

        self._opti.set_initial(self._X, X_ws)
        self._opti.set_initial(self._U, U_ws)
        
        lim_x = self.W / 2.0 - self.margin
        lim_y = self.H / 2.0 - self.margin

        try:
            sol = self._opti.solve()
            self._X_prev = sol.value(self._X)
            self._U_prev = sol.value(self._U)

            # ── DUAL PATH DIAGNOSTIC PRINTING ──
            # This evaluates camera projection coordinates on both horizons side-by-side.
            print("\n" + "="*95)
            print("  DIAGNOSTIC HORIZON LOG (RUCKIG REFERENCE vs MPC OPTIMAL PATH)")
            print("="*95)
            print(f"Active Camera Boundaries -> Limit X: ±{lim_x:.1f}px | Limit Y: ±{lim_y:.1f}px")
            print(f"Format: [k] | Ruckig Pixel (px, py) vs MPC Pixel (px, py) | [Alt] [Acc Fwd, Acc Right]")
            print("-"*95)
            
            for k in range(self.N):
                # 1. Evaluate unconstrained Ruckig projection
                pos_r = P[:, k]
                acc_r = A[:, k]
                alt_r = max(marker_a[2] - pos_r[2], 0.3)
                
                px_trans_r = self.fx * (marker_a[1] - pos_r[1]) / alt_r
                py_trans_r = -self.fy * (marker_a[0] - pos_r[0]) / alt_r
                
                g_eff_r = max(self.G - acc_r[2], 2.0)
                px_tilt_r = self.fx * (acc_r[1] / g_eff_r)
                py_tilt_r = -self.fy * (acc_r[0] / g_eff_r)
                
                px_r = px_trans_r + px_tilt_r
                py_r = py_trans_r + py_tilt_r
                
                # 2. Evaluate actual MPC solved projection
                pos_m = self._X_prev[0:3, k]
                acc_m = self._X_prev[6:9, k]
                alt_m = max(marker_a[2] - pos_m[2], 0.3)
                
                px_trans_m = self.fx * (marker_a[1] - pos_m[1]) / alt_m
                py_trans_m = -self.fy * (marker_a[0] - pos_m[0]) / alt_m
                
                g_eff_m = max(self.G - acc_m[2], 2.0)
                px_tilt_m = self.fx * (acc_m[1] / g_eff_m)
                py_tilt_m = -self.fy * (acc_m[0] / g_eff_m)
                
                px_m = px_trans_m + px_tilt_m
                py_m = py_trans_m + py_tilt_m
                
                # Flag if Ruckig would violate the camera boundary limits
                viol_flag = "⚠️ OUT OF FOV!" if (abs(px_r) > lim_x or abs(py_r) > lim_y) else "  OK"
                
                print(
                    f"k={k:2d} | Ruckig: ({px_r:6.1f}, {py_r:6.1f}) {viol_flag} vs MPC: ({px_m:6.1f}, {py_m:6.1f}) | "
                    f"Alt: {alt_m:4.2f}m | MPC Acc: [{acc_m[0]:5.2f}, {acc_m[1]:5.2f}]"
                )
            print("="*95 + "\n")

            return (
                self._X_prev[0:3, 1],
                self._X_prev[3:6, 1],
                self._X_prev[6:9, 1],
                True,
            )
        except Exception as e:
            print(f"MPC failed: {e}")
            self._X_prev = None
            self._U_prev = None
            print("ruckig fallback")
            return P[:, 1], V[:, 1], A[:, 1], False


async def align_to_aruco_mpc(
    drone: System, video, detector, mtx: np.ndarray, dist: np.ndarray, mpc: ArucoTrackingMPC
) -> bool:
    TARGET_ALTITUDE = 1.0
    ref = RuckigReference()

    for _ in range(20):
        frame = video.frame()
        if frame is not None:
            corners, ids, _ = detector.detectMarkers(frame)
            if ids is not None:
                break
        await asyncio.sleep(0.05)
    else:
        print("Cannot start: no marker visible")
        return False

    pv = await fc.get_drone_ned_position_velocity(drone)
    heading_start = await fc.get_drone_heading(drone)
    pos_ned = np.array(
        [pv.position.north_m, pv.position.east_m, pv.position.down_m]
    )
    vel_ned = np.array(
        [pv.velocity.north_m_s, pv.velocity.east_m_s, pv.velocity.down_m_s]
    )

    aruco_dist = utilities.get_aruco_distances_and_yaw(
        video, detector, mtx, dist
    )
    if aruco_dist is None:
        return False

    pos_a = _ned_to_aligned(pos_ned, heading_start)
    vel_a = _ned_to_aligned(vel_ned, heading_start)
    marker_a = pos_a + np.array(
        [aruco_dist[0], aruco_dist[1], aruco_dist[2]]
    )
    target_a = marker_a - np.array([0.0, 0.0, TARGET_ALTITUDE])

    # Correct signature pass to RuckigReference.compute (removed heading parameter)
    if not ref.compute(pos_a, vel_a, np.zeros(3), target_a,
                       j_max=2.0, v_max=3.0, a_max=3.0):
        return False

    # Initialize trackers with the initial velocity and zero acceleration
    last_v_a = vel_a.copy()
    last_a_a = np.zeros(3)

    while True:
        pv  = await fc.get_drone_ned_position_velocity(drone)
        pos_ned = np.array([pv.position.north_m, pv.position.east_m, pv.position.down_m])

        pos_a = _ned_to_aligned(pos_ned, heading_start)
        
        # Assume current velocity and acceleration are exactly what we commanded last time
        vel_a = last_v_a
        acc_a = last_a_a
        
        frame = video.frame()
        if frame is None:
            continue

        corners, ids, _ = detector.detectMarkers(frame)
        aruco_visible = ids is not None

        if aruco_visible:
            aruco_dist = utilities.get_aruco_distances_and_yaw(
                video, detector, mtx, dist
            )
            if aruco_dist is None:
                await asyncio.sleep(0.01)
                continue

            px0 = float(corners[0][0].mean(axis=0)[0])
            py0 = float(corners[0][0].mean(axis=0)[1])
            marker_a = pos_a + np.array(
                [aruco_dist[0], aruco_dist[1], aruco_dist[2]]
            )

            position_ok = (
                fabs(aruco_dist[0]) < 0.15
                and fabs(aruco_dist[1]) < 0.15
                and fabs(aruco_dist[2] - TARGET_ALTITUDE) < 0.15
            )

            if position_ok:
                await drone.offboard.set_position_ned(
                    PositionNedYaw(
                        pv.position.north_m,
                        pv.position.east_m,
                        pv.position.down_m,
                        float("nan"),
                    )
                )
                await asyncio.sleep(3)
                await drone.offboard.set_position_ned(
                    PositionNedYaw(
                        pv.position.north_m,
                        pv.position.east_m,
                        pv.position.down_m - 2,
                        float("nan"),
                    )
                )
                await asyncio.sleep(7)
                return True

            p_a, v_a, a_a, _ = mpc.solve(pos_a, vel_a, acc_a, ref, marker_a)

        else:
            print("Aruco not visible, Ruckig fallback")
            p_a, v_a, a_a = ref.at_elapsed(ref.elapsed)

        # Store the computed setpoints to be used as starting state in the next iteration
        last_v_a = v_a.copy()
        last_a_a = a_a.copy()

        p_ned = _aligned_to_ned(p_a, heading_start)
        v_ned = _aligned_to_ned(v_a, heading_start)
        a_ned = _aligned_to_ned(a_a, heading_start)

        await drone.offboard.set_position_velocity_acceleration_ned(
            PositionNedYaw(p_ned[0], p_ned[1], p_ned[2], float("nan")),
            VelocityNedYaw(v_ned[0], v_ned[1], v_ned[2], float("nan")),
            AccelerationNed(a_ned[0], a_ned[1], a_ned[2]),
        )
        #await asyncio.sleep(0.01)