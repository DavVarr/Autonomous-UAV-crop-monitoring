"""One-shot perception-aware jerk-limited trajectory optimization.

CasADi/IPOPT optimizes a complete trajectory once. A straight Ruckig
trajectory is used only as the initial guess. The optimized trajectory is then
followed blindly using wall-clock elapsed time.

State (NED): [p_n, p_e, p_d, v_n, v_e, v_d, a_n, a_e, a_d]
Control:     [j_n, j_e, j_d]
"""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, radians, sin
import time
from typing import Sequence

import casadi as ca
import cv2
import numpy as np
import ruckig
from mavsdk import System
from mavsdk.offboard import AccelerationNed, PositionNedYaw, VelocityNedYaw

import fc_simulated as fc
from camera_simulated import Video, my_estimatePoseSingleMarkers


G = 9.81
CAMERA_TO_BODY = np.array(
    [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
    dtype=float,
)


def _vec3(value: Sequence[float], name: str) -> np.ndarray:
    result = np.asarray(value, dtype=float).reshape(-1)
    if result.size != 3 or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain three finite values")
    return result


def _body_to_ned(attitude) -> np.ndarray:
    roll, pitch, yaw = map(
        radians, (attitude.roll_deg, attitude.pitch_deg, attitude.yaw_deg)
    )
    cr, sr = cos(roll), sin(roll)
    cp, sp = cos(pitch), sin(pitch)
    cy, sy = cos(yaw), sin(yaw)
    return np.array(
        [
            [cp * cy, sr * sp * cy - cr * sy, cr * sp * cy + sr * sy],
            [cp * sy, sr * sp * sy + cr * cy, cr * sp * sy - sr * cy],
            [-sp, sr * cp, cr * cp],
        ],
        dtype=float,
    )


def _marker_points(marker_size: float) -> np.ndarray:
    half = marker_size / 2.0
    return np.array(
        [
            [-half, -half, 0.0],
            [+half, -half, 0.0],
            [+half, +half, 0.0],
            [-half, +half, 0.0],
        ],
        dtype=float,
    )


@dataclass(frozen=True)
class ArucoObservation:
    corners_px: np.ndarray
    rvec: np.ndarray
    tvec: np.ndarray


def _observe(
    video: Video,
    detector,
    mtx: np.ndarray,
    dist: np.ndarray,
    marker_size: float,
    marker_id: int | None,
) -> ArucoObservation | None:
    if not video.frame_available():
        return None
    frame = video.frame()
    if frame is None:
        return None

    corners, ids, _ = detector.detectMarkers(frame)
    if ids is None or len(ids) == 0:
        return None

    ids_flat = np.asarray(ids, dtype=int).reshape(-1)
    if marker_id is None:
        index = 0
    else:
        matches = np.flatnonzero(ids_flat == int(marker_id))
        if len(matches) == 0:
            return None
        index = int(matches[0])

    selected = corners[index]
    rvecs, tvecs, _ = my_estimatePoseSingleMarkers(
        [selected], marker_size, mtx, dist
    )
    if not rvecs or not tvecs:
        return None

    return ArucoObservation(
        corners_px=np.asarray(selected, dtype=float).reshape(4, 2),
        rvec=np.asarray(rvecs[0], dtype=float).reshape(3),
        tvec=np.asarray(tvecs[0], dtype=float).reshape(3),
    )


def _marker_geometry_ned(
    observation: ArucoObservation,
    drone_position: np.ndarray,
    body_to_ned: np.ndarray,
    marker_size: float,
) -> tuple[np.ndarray, np.ndarray]:
    marker_to_camera, _ = cv2.Rodrigues(observation.rvec.reshape(3, 1))
    corners_camera = (
        marker_to_camera @ _marker_points(marker_size).T
    ).T + observation.tvec
    corners_body = (CAMERA_TO_BODY @ corners_camera.T).T
    corners_ned = drone_position + (body_to_ned @ corners_body.T).T

    centre_body = CAMERA_TO_BODY @ observation.tvec
    centre_ned = drone_position + body_to_ned @ centre_body
    return centre_ned, corners_ned


def _attitude_from_acceleration_symbolic(
    acceleration: ca.MX, yaw_rad: float
) -> ca.MX:
    # For body FRD in NED: T*b3 = g*e3 - a.
    body_down = ca.vertcat(
        -acceleration[0],
        -acceleration[1],
        G - acceleration[2],
    )
    b3 = body_down / ca.sqrt(ca.dot(body_down, body_down) + 1e-12)
    heading = ca.DM([cos(yaw_rad), sin(yaw_rad), 0.0])
    b2_raw = ca.cross(b3, heading)
    b2 = b2_raw / ca.sqrt(ca.dot(b2_raw, b2_raw) + 1e-12)
    b1 = ca.cross(b2, b3)
    return ca.horzcat(b1, b2, b3)


def _attitude_from_acceleration_numeric(
    acceleration: np.ndarray, yaw_rad: float
) -> np.ndarray:
    body_down = np.array(
        [-acceleration[0], -acceleration[1], G - acceleration[2]],
        dtype=float,
    )
    b3 = body_down / np.linalg.norm(body_down)
    heading = np.array([cos(yaw_rad), sin(yaw_rad), 0.0])
    b2_raw = np.cross(b3, heading)
    b2 = b2_raw / np.linalg.norm(b2_raw)
    b1 = np.cross(b2, b3)
    return np.column_stack((b1, b2, b3))


def _detected_margin(corners: np.ndarray, width: int, height: int) -> float:
    u = corners[:, 0]
    v = corners[:, 1]
    return float(np.min(np.concatenate((u, width - u, v, height - v))))


@dataclass
class JerkTrajectory:
    duration: float
    states: np.ndarray
    jerks: np.ndarray
    solve_time_s: float
    minimum_margin_px: float

    @property
    def intervals(self) -> int:
        return int(self.jerks.shape[0])

    @property
    def dt(self) -> float:
        return self.duration / self.intervals

    def at_time(self, t: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        t = float(np.clip(t, 0.0, self.duration))
        if t >= self.duration:
            final = self.states[-1]
            return final[:3].copy(), final[3:6].copy(), final[6:9].copy()

        index = min(int(t / self.dt), self.intervals - 1)
        tau = t - index * self.dt
        x = self.states[index]
        jerk = self.jerks[index]
        position = x[:3] + x[3:6] * tau + 0.5 * x[6:9] * tau**2
        position += jerk * tau**3 / 6.0
        velocity = x[3:6] + x[6:9] * tau + 0.5 * jerk * tau**2
        acceleration = x[6:9] + jerk * tau
        return position, velocity, acceleration


class CasadiArucoTrajectoryPlanner:
    """Complete nonlinear trajectory optimization with robust FOV reserve."""

    def __init__(
        self,
        mtx: np.ndarray,
        frame_width: int = 1280,
        frame_height: int = 960,
        intervals: int = 40,
        minimum_duration: float = 1.0,
        maximum_duration: float = 12.0,
        max_velocity: Sequence[float] = (3.0, 3.0, 1.5),
        min_velocity: Sequence[float] = (-3.0, -3.0, -1.0),
        max_acceleration: Sequence[float] = (3.0, 3.0, 2.0),
        min_acceleration: Sequence[float] = (-3.0, -3.0, -2.0),
        max_jerk: Sequence[float] = (2.0, 2.0, 2.0),
        robust_margin_px: float = 40.0,
        margin_ramp_fraction: float = 0.20,
        minimum_camera_depth_m: float = 0.10,
        fov_substeps: Sequence[float] = (0.25, 0.5, 0.75, 1.0),
        validation_dt: float = 0.01,
        time_weight: float = 10.0,
        jerk_weight: float = 0.01,
        acceleration_weight: float = 0.002,
        image_centre_weight: float = 1.0,
        ipopt_max_iterations: int = 2500,
        ipopt_print_level: int = 5,
    ):
        self.mtx = np.asarray(mtx, dtype=float)
        if self.mtx.shape != (3, 3):
            raise ValueError("mtx must be 3x3")
        self.fx, self.fy = float(self.mtx[0, 0]), float(self.mtx[1, 1])
        self.cx, self.cy = float(self.mtx[0, 2]), float(self.mtx[1, 2])
        self.width, self.height = int(frame_width), int(frame_height)
        self.intervals = int(intervals)
        self.minimum_duration = float(minimum_duration)
        self.maximum_duration = float(maximum_duration)
        self.max_velocity = _vec3(max_velocity, "max_velocity")
        self.min_velocity = _vec3(min_velocity, "min_velocity")
        self.max_acceleration = _vec3(max_acceleration, "max_acceleration")
        self.min_acceleration = _vec3(min_acceleration, "min_acceleration")
        self.max_jerk = _vec3(max_jerk, "max_jerk")
        self.robust_margin_px = float(robust_margin_px)
        self.margin_ramp_fraction = float(margin_ramp_fraction)
        self.minimum_camera_depth_m = float(minimum_camera_depth_m)
        self.fov_substeps = tuple(float(x) for x in fov_substeps)
        self.validation_dt = float(validation_dt)
        self.time_weight = float(time_weight)
        self.jerk_weight = float(jerk_weight)
        self.acceleration_weight = float(acceleration_weight)
        self.image_centre_weight = float(image_centre_weight)
        self.ipopt_max_iterations = int(ipopt_max_iterations)
        self.ipopt_print_level = int(ipopt_print_level)

        if self.intervals < 10:
            raise ValueError("intervals must be at least 10")
        if not 0.0 < self.margin_ramp_fraction <= 1.0:
            raise ValueError("margin_ramp_fraction must be in (0, 1]")
        if any(x <= 0.0 or x > 1.0 for x in self.fov_substeps):
            raise ValueError("fov_substeps must lie in (0, 1]")

    def _ruckig_guess(
        self, initial_state: np.ndarray, target: np.ndarray
    ) -> tuple[float, np.ndarray, np.ndarray]:
        inp = ruckig.InputParameter(3)
        out = ruckig.Trajectory(3)
        inp.current_position = initial_state[:3].tolist()
        inp.current_velocity = initial_state[3:6].tolist()
        inp.current_acceleration = initial_state[6:9].tolist()
        inp.target_position = target.tolist()
        inp.target_velocity = [0.0, 0.0, 0.0]
        inp.target_acceleration = [0.0, 0.0, 0.0]
        inp.max_velocity = np.maximum(
            np.abs(self.min_velocity), np.abs(self.max_velocity)
        ).tolist()
        inp.max_acceleration = np.maximum(
            np.abs(self.min_acceleration), np.abs(self.max_acceleration)
        ).tolist()
        inp.max_jerk = self.max_jerk.tolist()
        inp.synchronization = ruckig.Synchronization.Phase

        result = ruckig.Ruckig(3).calculate(inp, out)
        if result not in (ruckig.Result.Working, ruckig.Result.Finished):
            raise RuntimeError(f"Ruckig initial guess failed: {result}")

        duration = float(out.duration)
        states = np.empty((self.intervals + 1, 9), dtype=float)
        for k in range(self.intervals + 1):
            p, v, a = out.at_time(duration * k / self.intervals)
            states[k] = np.concatenate((p, v, a))
        jerks = np.diff(states[:, 6:9], axis=0) / (duration / self.intervals)
        return duration, states, np.clip(jerks, -self.max_jerk, self.max_jerk)

    def _project(
        self,
        position: ca.MX,
        acceleration: ca.MX,
        point_ned: np.ndarray,
        yaw_rad: float,
    ) -> tuple[ca.MX, ca.MX, ca.MX]:
        body_to_ned = _attitude_from_acceleration_symbolic(acceleration, yaw_rad)
        point_body = body_to_ned.T @ (ca.DM(point_ned) - position)
        point_camera = ca.DM(CAMERA_TO_BODY.T) @ point_body
        depth = point_camera[2]
        u = self.fx * point_camera[0] / depth + self.cx
        v = self.fy * point_camera[1] / depth + self.cy
        return u, v, depth

    def _margin(self, initial_margin: float, progress: float) -> float:
        start = float(np.clip(initial_margin, 0.0, self.robust_margin_px))
        alpha = min(1.0, progress / self.margin_ramp_fraction)
        return start + alpha * (self.robust_margin_px - start)

    def solve(
        self,
        initial_position: Sequence[float],
        initial_velocity: Sequence[float],
        initial_acceleration: Sequence[float],
        target_position: Sequence[float],
        marker_centre_ned: Sequence[float],
        marker_corners_ned: np.ndarray,
        initial_margin_px: float,
        yaw_deg: float,
    ) -> JerkTrajectory | None:
        initial_state = np.concatenate(
            (
                _vec3(initial_position, "initial_position"),
                _vec3(initial_velocity, "initial_velocity"),
                _vec3(initial_acceleration, "initial_acceleration"),
            )
        )
        target = _vec3(target_position, "target_position")
        marker_centre = _vec3(marker_centre_ned, "marker_centre_ned")
        marker_corners = np.asarray(marker_corners_ned, dtype=float)
        if marker_corners.shape != (4, 3):
            raise ValueError("marker_corners_ned must have shape (4, 3)")

        guess_t, guess_x, guess_j = self._ruckig_guess(initial_state, target)
        guess_t = float(np.clip(
            guess_t, self.minimum_duration, self.maximum_duration
        ))

        N = self.intervals
        yaw_rad = radians(float(yaw_deg))
        opti = ca.Opti()
        X = opti.variable(9, N + 1)
        J = opti.variable(3, N)
        total_time = opti.variable()
        dt = total_time / N

        opti.subject_to(opti.bounded(
            self.minimum_duration, total_time, self.maximum_duration
        ))
        opti.subject_to(X[:, 0] == ca.DM(initial_state))
        opti.subject_to(X[:3, N] == ca.DM(target))
        opti.subject_to(X[3:6, N] == 0)
        opti.subject_to(X[6:9, N] == 0)

        z_min = min(initial_state[2], target[2])
        z_max = max(initial_state[2], target[2])
        objective = self.time_weight * total_time

        for k in range(N):
            p, v, a, jerk = X[:3, k], X[3:6, k], X[6:9, k], J[:, k]
            opti.subject_to(X[:3, k + 1] == (
                p + v * dt + 0.5 * a * dt**2 + jerk * dt**3 / 6.0
            ))
            opti.subject_to(X[3:6, k + 1] == (
                v + a * dt + 0.5 * jerk * dt**2
            ))
            opti.subject_to(X[6:9, k + 1] == a + jerk * dt)
            opti.subject_to(opti.bounded(
                ca.DM(-self.max_jerk), jerk, ca.DM(self.max_jerk)
            ))

            for fraction in self.fov_substeps:
                tau = fraction * dt
                p_tau = p + v * tau + 0.5 * a * tau**2 + jerk * tau**3 / 6.0
                v_tau = v + a * tau + 0.5 * jerk * tau**2
                a_tau = a + jerk * tau
                opti.subject_to(opti.bounded(
                    ca.DM(self.min_velocity), v_tau, ca.DM(self.max_velocity)
                ))
                opti.subject_to(opti.bounded(
                    ca.DM(self.min_acceleration),
                    a_tau,
                    ca.DM(self.max_acceleration),
                ))
                opti.subject_to(opti.bounded(z_min, p_tau[2], z_max))

                progress = (k + fraction) / N
                margin = self._margin(initial_margin_px, progress)
                for corner in marker_corners:
                    u, pixel_v, depth = self._project(
                        p_tau, a_tau, corner, yaw_rad
                    )
                    opti.subject_to(depth >= self.minimum_camera_depth_m)
                    opti.subject_to(opti.bounded(
                        margin, u, self.width - margin
                    ))
                    opti.subject_to(opti.bounded(
                        margin, pixel_v, self.height - margin
                    ))

            u_c, v_c, depth_c = self._project(p, a, marker_centre, yaw_rad)
            opti.subject_to(depth_c >= self.minimum_camera_depth_m)
            centre_error = ca.vertcat(
                (u_c - self.cx) / (self.width / 2.0),
                (v_c - self.cy) / (self.height / 2.0),
            )
            objective += dt * (
                self.jerk_weight * ca.dot(jerk, jerk)
                + self.acceleration_weight * ca.dot(a, a)
                + self.image_centre_weight * ca.dot(centre_error, centre_error)
            )

        opti.minimize(objective)
        opti.set_initial(total_time, guess_t)
        opti.set_initial(X, guess_x.T)
        opti.set_initial(J, guess_j.T)
        opti.solver(
            "ipopt",
            {"expand": True, "print_time": False},
            {
                "max_iter": self.ipopt_max_iterations,
                "print_level": self.ipopt_print_level,
                "tol": 1e-6,
                "acceptable_tol": 1e-4,
                "mu_strategy": "adaptive",
                "sb": "yes",
            },
        )

        started = time.perf_counter()
        try:
            solution = opti.solve()
        except RuntimeError as error:
            status = opti.stats().get("return_status", "unknown")
            print(f"IPOPT failed: {status}: {error}")
            return None

        trajectory = JerkTrajectory(
            duration=float(solution.value(total_time)),
            states=np.asarray(solution.value(X), dtype=float).T,
            jerks=np.asarray(solution.value(J), dtype=float).T,
            solve_time_s=time.perf_counter() - started,
            minimum_margin_px=float("inf"),
        )
        safe, minimum_margin = self.validate(
            trajectory,
            marker_corners,
            initial_margin_px,
            yaw_deg,
        )
        trajectory.minimum_margin_px = minimum_margin
        if not safe:
            print(f"Dense validation failed: margin={minimum_margin:.2f} px")
            return None
        return trajectory

    def validate(
        self,
        trajectory: JerkTrajectory,
        marker_corners_ned: np.ndarray,
        initial_margin_px: float,
        yaw_deg: float,
    ) -> tuple[bool, float]:
        yaw_rad = radians(float(yaw_deg))
        minimum_margin = max(0.0, float(initial_margin_px))
        times = np.arange(
            self.validation_dt,
            trajectory.duration + self.validation_dt,
            self.validation_dt,
        )
        for t in times:
            t = min(float(t), trajectory.duration)
            p, v, a = trajectory.at_time(t)
            if np.any(v < self.min_velocity - 1e-5) or np.any(
                v > self.max_velocity + 1e-5
            ):
                return False, minimum_margin
            if np.any(a < self.min_acceleration - 1e-5) or np.any(
                a > self.max_acceleration + 1e-5
            ):
                return False, minimum_margin

            body_to_ned = _attitude_from_acceleration_numeric(a, yaw_rad)
            ned_to_camera = CAMERA_TO_BODY.T @ body_to_ned.T
            required = self._margin(initial_margin_px, t / trajectory.duration)
            for corner in marker_corners_ned:
                camera = ned_to_camera @ (corner - p)
                if camera[2] < self.minimum_camera_depth_m:
                    return False, minimum_margin
                u = self.fx * camera[0] / camera[2] + self.cx
                pixel_v = self.fy * camera[1] / camera[2] + self.cy
                margin = min(u, self.width - u, pixel_v, self.height - pixel_v)
                minimum_margin = min(minimum_margin, float(margin))
                if margin < required - 0.5:
                    return False, minimum_margin
        return True, minimum_margin


async def follow_jerk_trajectory(
    drone: System,
    trajectory: JerkTrajectory,
    yaw_deg: float,
) -> bool:
    start = time.perf_counter()
    while True:
        elapsed = time.perf_counter() - start
        p, v, a = trajectory.at_time(min(elapsed, trajectory.duration))
        await drone.offboard.set_position_velocity_acceleration_ned(
            PositionNedYaw(*map(float, p), float(yaw_deg)),
            VelocityNedYaw(*map(float, v), float(yaw_deg)),
            AccelerationNed(*map(float, a)),
        )
        if elapsed >= trajectory.duration:
            return True


async def align_to_aruco_casadi(
    drone: System,
    video: Video,
    detector,
    mtx: np.ndarray,
    dist: np.ndarray,
    planner: CasadiArucoTrajectoryPlanner,
    marker_size: float = 0.5,
    marker_id: int | None = None,
    target_altitude: float = 1.0,
    initial_acceleration_ned: Sequence[float] = (0.0, 0.0, 0.0),
) -> bool:
    observation = _observe(
        video, detector, mtx, dist, marker_size, marker_id
    )
    if observation is None:
        print("ArUco not detected")
        return False

    pv = await fc.get_drone_ned_position_velocity(drone)
    attitude = await fc.get_drone_attitude_euler(drone)
    yaw_deg = float(await fc.get_drone_heading(drone))
    position = np.array(
        [pv.position.north_m, pv.position.east_m, pv.position.down_m],
        dtype=float,
    )
    velocity = np.array(
        [
            pv.velocity.north_m_s,
            pv.velocity.east_m_s,
            pv.velocity.down_m_s,
        ],
        dtype=float,
    )
    acceleration = _vec3(initial_acceleration_ned, "initial_acceleration_ned")

    marker_centre, marker_corners = _marker_geometry_ned(
        observation, position, _body_to_ned(attitude), marker_size
    )
    target = marker_centre.copy()
    target[2] -= float(target_altitude)
    initial_margin = _detected_margin(
        observation.corners_px, planner.width, planner.height
    )

    print(
        f"Planning: error={target - position}, "
        f"initial_margin={initial_margin:.1f} px"
    )
    trajectory = planner.solve(
        initial_position=position,
        initial_velocity=velocity,
        initial_acceleration=acceleration,
        target_position=target,
        marker_centre_ned=marker_centre,
        marker_corners_ned=marker_corners,
        initial_margin_px=initial_margin,
        yaw_deg=yaw_deg,
    )
    if trajectory is None:
        return False

    print(
        f"IPOPT solve={trajectory.solve_time_s:.2f} s, "
        f"flight={trajectory.duration:.2f} s, "
        f"predicted minimum margin={trajectory.minimum_margin_px:.1f} px"
    )
    return await follow_jerk_trajectory(drone, trajectory, yaw_deg)