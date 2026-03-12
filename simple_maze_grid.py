import math
import random
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib import patches
import numpy as np
import pygame
import gymnasium as gym
from gymnasium import spaces

from path_planner import dubins_path


class SimpleMazeGrid(gym.Env):
    """
    EN 좌표계 (E:+x, N:+y) 기반 Gymnasium 환경

    주요 기능
    - Dubins 전역 경로 생성
    - LOS 기반 path tracking
    - 원형 장애물 / hard zone / safety zone 관리
    - LIDAR 기반 local observation 생성
    - 논문용 figure 저장 함수 제공
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        global_map_size: int,
        local_map_size: int,
        v: float,
        w: Tuple[float, float],
        dt: float = 0.05,
        render_option: bool = False,
        random_seed: Optional[int] = None,
        spec=None,
        obstacle_count: int = 0,
        obstacle_min_radius: float = 2.0,
        obstacle_max_radius: float = 10.0,
        sensor_range: Optional[float] = None,
        use_lidar_edges: bool = True,
        lidar_num_rays: int = 360,
        lidar_fov: float = 2 * math.pi,
        reference_L: Optional[float] = None,
        hard_zone: float = 2.0,
        safety_zone: float = 4.0,
        obstacle_layout: str = "auto",
        corridor_half_width: float = 2.5,
        random_path_clearance: float = 1.5,
        show_safety_ring: bool = True,
    ):
        super().__init__()

        self.global_map_size = int(global_map_size)
        self.local_map_size = int(local_map_size)
        self.dt = float(dt)
        self.render_option = bool(render_option)
        self.spec = spec

        self.v = float(v)
        self.w = 0.0
        self.min_w = float(w[0])
        self.max_w = float(w[1])

        self.R = self.v / max(self.max_w, 1e-9)
        self.path_planner = dubins_path(1 / self.R, 0.1)
        self.extension_len = 80

        self.max_steps = 3000
        self.terminated_radius = 3.0
        self.radius = 150

        self.hist_len = 1
        self.w_hist: List[float] = [0.0] * self.hist_len
        self.eta_hist: List[float] = [0.0] * self.hist_len

        self.sensor_range = float(sensor_range) if sensor_range is not None else (self.local_map_size / 2.0)
        if self.sensor_range <= 0:
            raise ValueError("sensor_range must be > 0")
        self.reference_L = float(reference_L) if reference_L is not None else (0.8 * self.sensor_range)

        self.use_lidar_edges = bool(use_lidar_edges)
        self.lidar_num_rays = int(lidar_num_rays)
        self.lidar_fov = float(lidar_fov)
        self.lidar_max_range = float(self.sensor_range)

        self.obstacle_count = int(obstacle_count)
        self.obstacle_min_r = float(obstacle_min_radius)
        self.obstacle_max_r = float(obstacle_max_radius)
        self.agent_radius = 0.0
        self.hard_zone = float(hard_zone)
        self.safety_zone = float(safety_zone)
        self.obstacle_layout = str(obstacle_layout)
        self.corridor_half_width = float(corridor_half_width)
        self.random_path_clearance = float(random_path_clearance)
        self.show_safety_ring = bool(show_safety_ring)

        self.terminated = False
        self.goal = False
        self.steps = 0
        self.cumulative_reward = 0.0

        self.visited_path: List[np.ndarray] = []

        self.global_path = np.zeros((0, 2), dtype=np.float32)
        self.path_end_state = None
        self.reference_point = None
        self.closest_path_idx = None
        self.closest_path_point = None

        self.a_cmd = 0.0
        self.w_cmd = 0.0
        self.eta = 0.0

        self.obstacles: List[Tuple[float, float, float]] = []
        self.latest_layout_name = "none"

        self.reward_cfg = {
            "w_smooth": 0.0,
            "w_obs": 0.75,
            "w_time": 0.5,
            "w_track": 0.5,
            "w_eta": 0.5,
            "bonus_goal": 100.0,
            "penalty_collision": -100.0,
            "penalty_timeout": -100.0,
            "clip_per_step": 1.0,
        }

        if self.spec is not None:
            self._reset_core_spec(random_seed)
        else:
            self._reset_core(random_seed)

        self._build_observation_space()

        if self.render_option:
            self._init_render()

    # =========================
    # Gym spaces
    # =========================
    def _build_observation_space(self):
        L = self.local_map_size
        self.observation_space = spaces.Dict(
            {
                "w_hist": spaces.Box(
                    low=np.full((self.hist_len,), self.min_w, dtype=np.float32),
                    high=np.full((self.hist_len,), self.max_w, dtype=np.float32),
                    dtype=np.float32,
                ),
                "eta_hist": spaces.Box(
                    low=-1.0,
                    high=1.0,
                    shape=(2 * self.hist_len,),
                    dtype=np.float32,
                ),
                "obstacle_pos": spaces.Box(
                    low=0,
                    high=255,
                    shape=(1, L, L),
                    dtype=np.uint8,
                ),
            }
        )
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

    # =========================
    # Reset
    # =========================
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if self.spec is not None:
            obs = self._reset_core_spec(random_seed=seed)
        else:
            obs = self._reset_core(random_seed=seed)
        return obs, {}

    def _reset_common(self, random_seed=None):
        self.terminated = False
        self.goal = False
        self.steps = 0
        self.cumulative_reward = 0.0
        self.w = 0.0
        self.visited_path = []
        self.w_hist = [0.0] * self.hist_len
        self.eta_hist = [0.0] * self.hist_len
        self.time_table, self.v_table, self.w_table = [], [], []

        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)

    def _reset_core(self, random_seed=None):
        self._reset_common(random_seed)

        center_E = self.global_map_size // 2
        center_N = self.global_map_size // 2
        psi0 = random.uniform(-math.pi, math.pi)

        self.initial_player_pos = np.array([center_E, center_N, psi0], dtype=np.float32)
        self.player_pos = self.initial_player_pos.copy()

        phi = random.uniform(-math.pi, math.pi)
        ge = float(center_E + self.radius * math.cos(phi))
        gn = float(center_N + self.radius * math.sin(phi))
        ge = float(np.clip(ge, 0.0, self.global_map_size - 1))
        gn = float(np.clip(gn, 0.0, self.global_map_size - 1))
        gpsi = random.uniform(-math.pi, math.pi)
        self.goal_pos = np.array([ge, gn, gpsi], dtype=np.float32)

        self._build_global_dubins_path()
        self.update_reference_point()

        self.obstacles = []
        self.generate_obstacles(random_seed=random_seed)
        self.obs_grid = self.compute_local_grids()
        return self.get_state()

    def _reset_core_spec(self, random_seed=None):
        self._reset_common(random_seed)
        initial_player_pos, goal_pos, obs_spec = self.spec

        self.initial_player_pos = np.array(initial_player_pos[0:3], dtype=np.float32)
        self.player_pos = self.initial_player_pos.copy()
        self.goal_pos = np.array(goal_pos[0:3], dtype=np.float32)

        self._build_global_dubins_path()
        self.update_reference_point()

        self.obstacles = []
        if obs_spec is not None:
            obs_arr = np.array(obs_spec, dtype=float)
            if obs_arr.ndim == 1:
                if obs_arr.size != 3:
                    raise ValueError(f"obs_spec 1D인데 길이가 3이 아님: got {obs_arr.size}")
                obs_arr = obs_arr.reshape(1, 3)
            for i in range(obs_arr.shape[0]):
                E, N, r = float(obs_arr[i, 0]), float(obs_arr[i, 1]), float(obs_arr[i, 2])
                self.obstacles.append((E, N, r))
            self.latest_layout_name = "spec"
        else:
            self.generate_obstacles(random_seed=random_seed)

        self.obs_grid = self.compute_local_grids()
        return self.get_state()

    def retry(self):
        self.player_pos = self.initial_player_pos.copy()
        self.terminated = False
        self.goal = False
        self.cumulative_reward = 0.0
        self.steps = 0
        self.w = 0.0
        self.w_hist = [0.0] * self.hist_len
        self.eta_hist = [0.0] * self.hist_len
        self.time_table, self.v_table, self.w_table = [], [], []
        self.visited_path = []
        self.update_reference_point()
        self.obs_grid = self.compute_local_grids()
        return self.get_state(), {}

    # =========================
    # Angle utils
    # =========================
    @staticmethod
    def normalize_angle(angle):
        return (angle + np.pi) % (2 * np.pi) - np.pi

    @staticmethod
    def dir_from_heading(psi):
        return math.cos(psi), math.sin(psi)

    # =========================
    # State builder
    # =========================
    def _push_w(self, w):
        self.w_hist = (self.w_hist + [float(w)])[-self.hist_len:]

    def _push_eta(self, e):
        self.eta_hist = (self.eta_hist + [float(e)])[-self.hist_len:]

    def _build_state(self):
        w_hist_vec = np.array(self.w_hist, dtype=np.float32)
        eta_pairs = []
        for e in self.eta_hist:
            eta_pairs.extend([math.cos(e), math.sin(e)])
        eta_hist_vec = np.array(eta_pairs, dtype=np.float32)
        obs_img = (self.obs_grid * 255).astype(np.uint8)[np.newaxis, :, :]
        return {
            "w_hist": w_hist_vec,
            "eta_hist": eta_hist_vec,
            "obstacle_pos": obs_img,
        }

    def get_state(self):
        return self._build_state()

    # =========================
    # Step
    # =========================
    def step(self, action):
        if self.terminated:
            return self.get_state(), 0.0, True, False, {}

        cfg = self.reward_cfg
        terminated = False
        truncated = False
        old_w = self.w

        self.update_reference_point()
        self.obs_grid = self.compute_local_grids()
        self.path_tracking()

        raw = float(action[0])
        self.w = float(np.clip(raw, -1.0, 1.0) * self.max_w)

        new_pos = self.player_pos.copy()
        new_pos[2] = self.normalize_angle(new_pos[2] + self.w * self.dt)
        dE = self.v * math.cos(new_pos[2]) * self.dt
        dN = self.v * math.sin(new_pos[2]) * self.dt
        new_pos[0] = float(np.clip(new_pos[0] + dE, 0.0, self.global_map_size - 1))
        new_pos[1] = float(np.clip(new_pos[1] + dN, 0.0, self.global_map_size - 1))

        self.player_pos = new_pos
        self.update_reference_point()
        self.obs_grid = self.compute_local_grids()
        self.path_tracking()

        self._push_w(self.w)
        self._push_eta(self.eta)

        path_EN = np.asarray(self.global_path[:, :2], dtype=float)
        pos_xy = np.asarray(new_pos[:2], dtype=float)
        if path_EN.shape[0] >= 1:
            diffs = path_EN - pos_xy
            dists = np.hypot(diffs[:, 0], diffs[:, 1])
            idx = int(np.argmin(dists))
            closest_point = path_EN[idx]
            psi = float(new_pos[2])
            e = closest_point - pos_xy
            y_body = -np.sin(psi) * e[0] + np.cos(psi) * e[1]
            path_dist = float(abs(y_body))
        else:
            path_dist = 0.0

        dw = self.w - old_w
        w_range = max(2 * self.max_w, 1e-6)
        dw_n = dw / w_range
        smooth = float(dw_n ** 2)
        R_smooth = -float(np.clip(smooth, 0.0, 1.0))

        pen = self.compute_obstacle_penalties(new_pos)
        p_new = np.clip((pen["clear"] + pen["hard"]) / 3.0, 0.0, 1.0)
        R_obs = -float(p_new)
        R_time = -5.0 * float(self.dt)

        if path_dist <= 0.1:
            path_dist = 0.0
        R_track = float(np.clip((0.1) ** path_dist, -1.0, 1.0))

        new_eta = abs(self.eta)
        if new_eta <= math.radians(1):
            new_eta = 0.0
        R_eta = float(np.clip((0.1) ** new_eta, -1.0, 1.0))

        reward = (
            cfg["w_smooth"] * R_smooth
            + cfg["w_obs"] * R_obs
            + cfg["w_time"] * R_time
            + cfg["w_track"] * R_track
            + cfg["w_eta"] * R_eta
        )

        if self.check_collision(new_pos[:2]):
            reward += cfg["penalty_collision"]
            terminated = True

        target_xy, target_psi = self._target_goal_state()
        cur_goal_dist = float(np.linalg.norm(new_pos[:2] - target_xy))
        heading_ok = math.cos(new_pos[2] - target_psi) >= math.cos(math.radians(30))
        if (cur_goal_dist < self.terminated_radius) and heading_ok:
            reward += cfg["bonus_goal"]
            self.goal = True
            terminated = True

        if self.steps > self.max_steps:
            reward += cfg["penalty_timeout"]
            truncated = True

        self.cumulative_reward += reward
        self.steps += 1
        self.visited_path.append(self.player_pos.copy())
        self.time_table.append(self.steps * self.dt)
        self.w_table.append(self.w)
        self.terminated = terminated or truncated

        return self.get_state(), reward, terminated, truncated, {}

    def _target_goal_state(self):
        if getattr(self, "path_end_state", None) is not None:
            return self.path_end_state[:2], float(self.path_end_state[2])
        return self.goal_pos[:2], float(self.goal_pos[2])

    # =========================
    # Obstacle helpers
    # =========================
    def _distance_point_to_segment(self, p, a, b):
        p = np.asarray(p, dtype=float)
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        ab = b - a
        denom = float(np.dot(ab, ab))
        if denom <= 1e-12:
            return float(np.linalg.norm(p - a))
        t = float(np.clip(np.dot(p - a, ab) / denom, 0.0, 1.0))
        proj = a + t * ab
        return float(np.linalg.norm(p - proj))

    def _min_distance_point_to_polyline(self, point_EN):
        path_EN = np.asarray(self.global_path[:, :2], dtype=float)
        if path_EN.shape[0] == 0:
            return float("inf")
        if path_EN.shape[0] == 1:
            return float(np.linalg.norm(path_EN[0] - np.asarray(point_EN[:2], dtype=float)))
        d_min = float("inf")
        for i in range(path_EN.shape[0] - 1):
            d = self._distance_point_to_segment(point_EN, path_EN[i], path_EN[i + 1])
            if d < d_min:
                d_min = d
        return d_min

    def min_surface_distance_to_obstacles(self, pos_EN):
        E, N = float(pos_EN[0]), float(pos_EN[1])
        if not self.obstacles:
            return float("inf")
        d_min = float("inf")
        for (oE, oN, oR) in self.obstacles:
            center_dist = math.hypot(E - oE, N - oN)
            surface_dist = center_dist - (oR + self.agent_radius)
            if surface_dist < d_min:
                d_min = surface_dist
        return d_min

    def _can_place_obstacle(
        self,
        E: float,
        N: float,
        r: float,
        start_EN,
        end_EN,
        keep_path_clear: bool = False,
        path_clearance: float = 0.0,
    ):
        margin = r + self.safety_zone + 2.0
        if E < margin or E > (self.global_map_size - margin):
            return False
        if N < margin or N > (self.global_map_size - margin):
            return False

        hard_buffer = r + self.hard_zone + self.agent_radius
        if math.hypot(E - start_EN[0], N - start_EN[1]) <= hard_buffer:
            return False
        if math.hypot(E - end_EN[0], N - end_EN[1]) <= hard_buffer:
            return False

        for (oE, oN, oR) in self.obstacles:
            min_center_sep = (r + self.hard_zone) + (oR + self.hard_zone) + 0.75
            if math.hypot(E - oE, N - oN) <= min_center_sep:
                return False

        if keep_path_clear and self.global_path.shape[0] >= 2:
            d_path = self._min_distance_point_to_polyline((E, N))
            if d_path <= (r + self.hard_zone + path_clearance):
                return False

        return True

    def _generate_corridor_pair_obstacles(self, max_attempts=5000):
        if self.global_path.shape[0] < 10:
            return False

        path_EN = np.asarray(self.global_path[:, :2], dtype=float)
        start_EN = np.asarray(self.initial_player_pos[:2], dtype=float)
        if getattr(self, "path_end_state", None) is not None:
            end_EN = np.asarray(self.path_end_state[:2], dtype=float)
        else:
            end_EN = np.asarray(self.goal_pos[:2], dtype=float)

        n_pts = path_EN.shape[0]
        idx_lo = max(5, int(0.30 * n_pts))
        idx_hi = min(n_pts - 6, int(0.70 * n_pts))
        if idx_hi <= idx_lo:
            idx_lo = 1
            idx_hi = n_pts - 2
        if idx_hi <= idx_lo:
            return False

        for _ in range(max_attempts):
            idx = random.randint(idx_lo, idx_hi)
            p_prev = path_EN[max(0, idx - 3)]
            p_cur = path_EN[idx]
            p_next = path_EN[min(n_pts - 1, idx + 3)]

            tangent = p_next - p_prev
            tangent_norm = float(np.linalg.norm(tangent))
            if tangent_norm <= 1e-9:
                continue
            tangent /= tangent_norm
            normal = np.array([-tangent[1], tangent[0]], dtype=float)

            r1 = random.uniform(self.obstacle_min_r, self.obstacle_max_r)
            r2 = random.uniform(self.obstacle_min_r, self.obstacle_max_r)
            corridor_half_width = random.uniform(
                self.corridor_half_width,
                self.corridor_half_width + 1.5,
            )
            tangent_jitter = random.uniform(-0.15 * self.sensor_range, 0.15 * self.sensor_range)
            anchor = p_cur + tangent * tangent_jitter

            offset1 = r1 + self.hard_zone + corridor_half_width
            offset2 = r2 + self.hard_zone + corridor_half_width
            c1 = anchor + normal * offset1
            c2 = anchor - normal * offset2

            if not self._can_place_obstacle(
                c1[0], c1[1], r1, start_EN, end_EN,
                keep_path_clear=True,
                path_clearance=corridor_half_width,
            ):
                continue
            if not self._can_place_obstacle(
                c2[0], c2[1], r2, start_EN, end_EN,
                keep_path_clear=True,
                path_clearance=corridor_half_width,
            ):
                continue

            hard_sep = np.linalg.norm(c1 - c2) - ((r1 + self.hard_zone) + (r2 + self.hard_zone))
            if hard_sep <= 2.0 * corridor_half_width - 1e-6:
                continue

            self.obstacles.append((float(c1[0]), float(c1[1]), float(r1)))
            self.obstacles.append((float(c2[0]), float(c2[1]), float(r2)))
            self.latest_layout_name = "corridor_pair"
            return True

        return False

    def _generate_remaining_random_obstacles(self, remaining_count: int, max_attempts=5000, keep_path_clear=False):
        if remaining_count <= 0:
            return

        start_EN = np.asarray(self.initial_player_pos[:2], dtype=float)
        if getattr(self, "path_end_state", None) is not None:
            end_EN = np.asarray(self.path_end_state[:2], dtype=float)
        else:
            end_EN = np.asarray(self.goal_pos[:2], dtype=float)

        for _ in range(remaining_count):
            placed = False
            for _ in range(max_attempts):
                r = random.uniform(self.obstacle_min_r, self.obstacle_max_r)
                margin = r + self.safety_zone + 2.0
                E = random.uniform(margin, self.global_map_size - margin)
                N = random.uniform(margin, self.global_map_size - margin)
                if not self._can_place_obstacle(
                    E, N, r, start_EN, end_EN,
                    keep_path_clear=keep_path_clear,
                    path_clearance=self.random_path_clearance,
                ):
                    continue
                self.obstacles.append((float(E), float(N), float(r)))
                placed = True
                break
            if not placed:
                break

    def generate_obstacles(self, random_seed=None, max_attempts=5000):
        self.obstacles = []
        self.latest_layout_name = "none"
        if self.obstacle_count <= 0:
            return

        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)

        layout = self.obstacle_layout.lower().strip()
        if layout == "auto":
            layout = "corridor_pair" if self.obstacle_count >= 2 else "midline"

        if layout in {"corridor_pair", "two_between", "pair_gap"} and self.obstacle_count >= 2:
            ok = self._generate_corridor_pair_obstacles(max_attempts=max_attempts)
            if ok:
                self._generate_remaining_random_obstacles(
                    self.obstacle_count - 2,
                    max_attempts=max_attempts,
                    keep_path_clear=True,
                )
                return

        # fallback: single midline obstacle or random layout
        start_EN = np.asarray(self.initial_player_pos[:2], dtype=float)
        if getattr(self, "path_end_state", None) is not None:
            end_EN = np.asarray(self.path_end_state[:2], dtype=float)
        else:
            end_EN = np.asarray(self.goal_pos[:2], dtype=float)

        path_EN = np.asarray(self.global_path[:, :2], dtype=float)
        if layout in {"midline", "single_midline"} and self.obstacle_count >= 1 and path_EN.shape[0] >= 2:
            for _ in range(max_attempts):
                idx = random.randint(max(1, int(0.35 * len(path_EN))), max(1, int(0.65 * len(path_EN))))
                p_prev = path_EN[max(0, idx - 3)]
                p_cur = path_EN[idx]
                p_next = path_EN[min(len(path_EN) - 1, idx + 3)]
                tangent = p_next - p_prev
                tangent_norm = float(np.linalg.norm(tangent))
                if tangent_norm <= 1e-9:
                    continue
                tangent /= tangent_norm
                normal = np.array([-tangent[1], tangent[0]], dtype=float)
                r = random.uniform(self.obstacle_min_r, self.obstacle_max_r)
                offset = random.uniform(-0.1 * self.sensor_range, 0.1 * self.sensor_range)
                side = random.choice([-1.0, 1.0])
                lateral = r + self.hard_zone + self.corridor_half_width
                center = p_cur + tangent * offset + side * normal * lateral
                if self._can_place_obstacle(
                    center[0], center[1], r, start_EN, end_EN,
                    keep_path_clear=True,
                    path_clearance=self.corridor_half_width,
                ):
                    self.obstacles.append((float(center[0]), float(center[1]), float(r)))
                    self.latest_layout_name = "midline"
                    break

        remaining = self.obstacle_count - len(self.obstacles)
        if remaining > 0:
            self._generate_remaining_random_obstacles(
                remaining,
                max_attempts=max_attempts,
                keep_path_clear=(layout in {"midline", "single_midline"}),
            )
            if self.latest_layout_name == "none":
                self.latest_layout_name = "random"

    def check_collision(self, pos_EN):
        return self.min_surface_distance_to_obstacles(pos_EN) <= 0.0

    # =========================
    # LIDAR
    # =========================
    def _ray_circle_first_hit_t(self, cx, cy, r, ray_cos, ray_sin):
        B = -(cx * ray_cos + cy * ray_sin) * 2.0
        C = (cx * cx + cy * cy - r * r)
        disc = B * B - 4.0 * C
        if disc < 0.0:
            return None
        sqrt_disc = math.sqrt(disc)
        t1 = (-B - sqrt_disc) / 2.0
        t2 = (-B + sqrt_disc) / 2.0
        hits = [t for t in (t1, t2) if t >= 0.0]
        return min(hits) if hits else None

    def lidar_scan_hits_body(self, agent_ENpsi=None):
        if agent_ENpsi is None:
            agent_ENpsi = self.player_pos

        circles_b = []
        for (oE, oN, oR) in self.obstacles:
            xb, yb = self.world_to_body((oE, oN), agent_ENpsi=agent_ENpsi)
            if (
                xb < -self.lidar_max_range - oR
                or xb > self.lidar_max_range + oR
                or yb < -self.lidar_max_range - oR
                or yb > self.lidar_max_range + oR
            ):
                continue
            circles_b.append((xb, yb, oR))

        if self.lidar_num_rays <= 0 or self.lidar_fov <= 0.0 or len(circles_b) == 0:
            return np.zeros((0, 2), dtype=float)

        hits = []
        start_ang = -0.5 * self.lidar_fov
        d_ang = self.lidar_fov / float(self.lidar_num_rays)
        for i in range(self.lidar_num_rays):
            ang = start_ang + i * d_ang
            c, s = math.cos(ang), math.sin(ang)
            nearest_t = None
            for (cx, cy, r) in circles_b:
                t = self._ray_circle_first_hit_t(cx, cy, r, c, s)
                if t is None:
                    continue
                if t < 1e-6:
                    t = 0.0
                if t <= self.lidar_max_range:
                    if nearest_t is None or t < nearest_t:
                        nearest_t = t
            if nearest_t is not None:
                xb_hit = nearest_t * c
                yb_hit = nearest_t * s
                if abs(xb_hit) <= self.sensor_range and abs(yb_hit) <= self.sensor_range:
                    hits.append((xb_hit, yb_hit))

        return np.asarray(hits, dtype=float) if hits else np.zeros((0, 2), dtype=float)

    # =========================
    # Local grids
    # =========================
    def _rect_circle_intersects(self, xmin, xmax, ymin, ymax, cx, cy, r):
        closest_x = min(max(cx, xmin), xmax)
        closest_y = min(max(cy, ymin), ymax)
        dx = cx - closest_x
        dy = cy - closest_y
        return (dx * dx + dy * dy) <= (r * r + 1e-12)

    def compute_local_grids(self):
        L = self.local_map_size
        S = self.sensor_range
        obs_grid = np.zeros((L, L), dtype=np.float32)

        if self.use_lidar_edges:
            hits = self.lidar_scan_hits_body(self.player_pos)
            return self._obs_grid_from_lidar_hits(hits)

        cell_world = (2.0 * S) / L
        x_tops = S - np.arange(0, L) * cell_world
        x_bottoms = x_tops - cell_world
        y_lefts = -S + np.arange(0, L) * cell_world
        y_rights = y_lefts + cell_world

        for (oE, oN, oR) in self.obstacles:
            xb_o, yb_o = self.world_to_body((oE, oN))
            if (xb_o < -S - oR) or (xb_o > S + oR) or (yb_o < -S - oR) or (yb_o > S + oR):
                continue
            for rr in range(L):
                xmin = x_bottoms[rr]
                xmax = x_tops[rr]
                if xb_o < (xmin - oR) or xb_o > (xmax + oR):
                    continue
                for cc in range(L):
                    ymin = y_lefts[cc]
                    ymax = y_rights[cc]
                    if yb_o < (ymin - oR) or yb_o > (ymax + oR):
                        continue
                    if self._rect_circle_intersects(xmin, xmax, ymin, ymax, xb_o, yb_o, oR):
                        obs_grid[rr, cc] = 1.0
        return obs_grid

    def _obs_grid_from_lidar_hits(self, hits_body):
        L = self.local_map_size
        S = self.sensor_range
        cell_world = (2.0 * S) / L
        obs_grid = np.zeros((L, L), dtype=np.float32)
        if hits_body.shape[0] == 0:
            return obs_grid
        for xb, yb in hits_body:
            r_float = (S - xb) / cell_world
            c_float = (yb + S) / cell_world
            r_idx = int(np.clip(np.round(r_float - 0.5), 0, L - 1))
            c_idx = int(np.clip(np.round(c_float - 0.5), 0, L - 1))
            obs_grid[r_idx, c_idx] = 1.0
        return obs_grid

    # =========================
    # Coord transforms
    # =========================
    def world_to_body(self, point_EN, agent_ENpsi=None):
        if agent_ENpsi is None:
            agent_ENpsi = self.player_pos
        E_a, N_a, psi = float(agent_ENpsi[0]), float(agent_ENpsi[1]), float(agent_ENpsi[2])
        dE = float(point_EN[0]) - E_a
        dN = float(point_EN[1]) - N_a
        x_b = dE * math.cos(psi) + dN * math.sin(psi)
        y_b = dE * math.sin(psi) - dN * math.cos(psi)
        return x_b, y_b

    def body_to_world(self, xb, yb, agent_ENpsi=None):
        if agent_ENpsi is None:
            agent_ENpsi = self.player_pos
        E0, N0, psi = float(agent_ENpsi[0]), float(agent_ENpsi[1]), float(agent_ENpsi[2])
        dE = xb * math.cos(psi) + yb * math.sin(psi)
        dN = xb * math.sin(psi) - yb * math.cos(psi)
        return E0 + dE, N0 + dN

    # =========================
    # Dubins / reference
    # =========================
    def _build_global_dubins_path(self):
        try:
            path_x, path_y, path_yaw, modes, lengths = self.path_planner.plan(
                self.initial_player_pos[0],
                self.initial_player_pos[1],
                self.initial_player_pos[2],
                self.goal_pos[0],
                self.goal_pos[1],
                self.goal_pos[2],
            )

            path_x = list(path_x)
            path_y = list(path_y)
            path_yaw = list(path_yaw)

            if len(path_x) > 0:
                last_x = path_x[-1]
                last_y = path_y[-1]
                last_yaw = path_yaw[-1]
                step_size = 0.1
                num_points = int(self.extension_len / step_size)
                for i in range(1, num_points + 1):
                    dist = i * step_size
                    path_x.append(last_x + dist * math.cos(last_yaw))
                    path_y.append(last_y + dist * math.sin(last_yaw))

            if len(path_x) >= 2:
                last_x, last_y = path_x[-1], path_y[-1]
                prev_x, prev_y = path_x[-2], path_y[-2]
                psi_end = math.atan2(last_y - prev_y, last_x - prev_x)
                self.path_end_state = np.array([last_x, last_y, psi_end], dtype=np.float32)
            else:
                self.path_end_state = None

            self.global_path = np.stack([path_x, path_y], axis=1).astype(np.float32)
        except Exception as e:
            print("[Dubins] path planning failed:", e)
            self.global_path = np.zeros((0, 2), dtype=np.float32)
            self.path_end_state = None

    def compute_reference_point_on_global(self, L=None):
        if L is None:
            L = self.reference_L
        if self.global_path is None or self.global_path.shape[0] < 2:
            self.closest_path_idx = None
            self.closest_path_point = None
            return None

        path_EN = np.asarray(self.global_path[:, :2], dtype=float)
        agent = np.array(self.player_pos[:2], dtype=float)
        diffs = path_EN - agent
        dists = np.hypot(diffs[:, 0], diffs[:, 1])
        cur_idx = int(np.argmin(dists))
        self.closest_path_idx = cur_idx
        self.closest_path_point = path_EN[cur_idx].copy()

        if L <= 0.0:
            return self.closest_path_point.copy()

        remaining_len = 0.0
        for i in range(cur_idx, len(path_EN) - 1):
            p0, p1 = path_EN[i], path_EN[i + 1]
            remaining_len += float(math.hypot(p1[0] - p0[0], p1[1] - p0[1]))
        if remaining_len <= L:
            return path_EN[-1].copy()

        acc = 0.0
        for i in range(cur_idx, len(path_EN) - 1):
            p0, p1 = path_EN[i], path_EN[i + 1]
            seg_len = float(math.hypot(p1[0] - p0[0], p1[1] - p0[1]))
            acc += seg_len
            if acc >= L:
                return p1.copy()
        return path_EN[-1].copy()

    def update_reference_point(self):
        self.reference_point = self.compute_reference_point_on_global(L=self.reference_L)

    def path_tracking(self):
        V = float(self.v)
        L = float(self.reference_L)
        if V <= 1e-6 or L <= 1e-6 or self.reference_point is None:
            self.eta = 0.0
            self.a_cmd = 0.0
            self.w_cmd = 0.0
            return

        pos = np.array(self.player_pos[:2], dtype=float)
        ref = np.array(self.reference_point[:2], dtype=float)
        dEN = ref - pos
        R = float(np.linalg.norm(dEN))
        if R < 1e-6:
            self.eta = 0.0
            self.a_cmd = 0.0
            self.w_cmd = 0.0
            return

        chi_ref = math.atan2(dEN[1], dEN[0])
        psi = float(self.player_pos[2])
        eta = self.normalize_angle(chi_ref - psi)
        a_cmd = 2.0 * (V * V / L) * math.sin(eta)
        w_cmd = a_cmd / V
        self.eta = eta
        self.a_cmd = a_cmd
        self.w_cmd = w_cmd

    # =========================
    # Reward / penalties
    # =========================
    def obstacle_cell_distances_at(self, agent_ENpsi):
        L = self.local_map_size
        S = self.sensor_range
        cell_world = (2.0 * S) / L
        obs_grid = np.zeros((L, L), dtype=np.float32)
        x_tops = S - np.arange(0, L) * cell_world
        x_bottoms = x_tops - cell_world
        y_lefts = -S + np.arange(0, L) * cell_world
        y_rights = y_lefts + cell_world

        for (oE, oN, oR) in self.obstacles:
            xb_o, yb_o = self.world_to_body((oE, oN), agent_ENpsi=agent_ENpsi)
            if (xb_o < -S - oR) or (xb_o > S + oR) or (yb_o < -S - oR) or (yb_o > S + oR):
                continue
            for rr in range(L):
                xmin = x_bottoms[rr]
                xmax = x_tops[rr]
                if xb_o < (xmin - oR) or xb_o > (xmax + oR):
                    continue
                for cc in range(L):
                    ymin = y_lefts[cc]
                    ymax = y_rights[cc]
                    if yb_o < (ymin - oR) or yb_o > (ymax + oR):
                        continue
                    if self._rect_circle_intersects(xmin, xmax, ymin, ymax, xb_o, yb_o, oR):
                        obs_grid[rr, cc] = 1.0

        rr, cc = np.where(obs_grid > 0.5)
        if rr.size == 0:
            return np.zeros((0,), dtype=float), [], np.zeros((0, 2), dtype=float)

        xb_centers = 0.5 * (x_tops[rr] + x_bottoms[rr])
        yb_centers = 0.5 * (y_rights[cc] + y_lefts[cc])
        centers_body = np.vstack([xb_centers, yb_centers]).T
        distances = np.hypot(xb_centers, yb_centers)
        order = np.argsort(distances)
        distances = distances[order]
        centers_body = centers_body[order]
        rc_indices = [(int(rr[i]), int(cc[i])) for i in order]
        return distances, rc_indices, centers_body

    def compute_obstacle_penalties(self, new_pos):
        d_surface = self.min_surface_distance_to_obstacles(new_pos)
        if not np.isfinite(d_surface):
            return {"clear": 0.0, "hard": 0.0}

        eps = 1e-6
        clear = float(np.clip(1.0 - d_surface / max(self.safety_zone, eps), 0.0, 1.0))
        hard = 2.0 if d_surface < float(self.hard_zone) else 0.0
        return {"clear": clear, "hard": hard}

    # =========================
    # Rendering
    # =========================
    def _init_render(self):
        pygame.init()

        self.global_screen_width = 1100
        self.global_screen_height = 1100
        self.info_width = 280
        self.local_screen_width = 620
        self.local_screen_height = 620
        self.total_width = self.global_screen_width + self.info_width + self.local_screen_width
        self.total_height = self.global_screen_height

        self.screen = pygame.display.set_mode((self.total_width, self.total_height))
        pygame.display.set_caption("Paper-ready Maze Viewer")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 48)
        self.medium_font = pygame.font.Font(None, 30)
        self.small_font = pygame.font.Font(None, 24)

        self.rect_global = pygame.Rect(0, 0, self.global_screen_width, self.global_screen_height)
        self.rect_info = pygame.Rect(self.global_screen_width, 0, self.info_width, self.global_screen_height)
        self.rect_local = pygame.Rect(self.global_screen_width + self.info_width, 0, self.local_screen_width, self.local_screen_height)

        self.COLOR_BG = (250, 251, 253)
        self.COLOR_GLOBAL = (245, 246, 248)
        self.COLOR_INFO = (255, 255, 255)
        self.COLOR_LOCAL = (245, 247, 250)
        self.COLOR_GRID = (222, 228, 236)

        self.COLOR_AGENT = (31, 78, 255)
        self.COLOR_START = (31, 78, 255)
        self.COLOR_GOAL = (0, 160, 90)
        self.COLOR_PATH = (166, 33, 185)
        self.COLOR_DUBINS = (0, 170, 190)
        self.COLOR_REF_POINT = (180, 0, 180)
        self.COLOR_ARROW = (30, 30, 30)
        self.COLOR_GOAL_ARROW = (0, 120, 70)
        self.COLOR_OBS_FILL = (90, 96, 110)
        self.COLOR_OBS_EDGE = (45, 45, 55)
        self.COLOR_HARD_ZONE = (225, 40, 45)
        self.COLOR_HARD_FILL = (225, 40, 45, 45)
        self.COLOR_SAFE_ZONE = (245, 140, 40)
        self.COLOR_SAFE_FILL = (245, 140, 40, 24)
        self.COLOR_LOCAL_BOX = (20, 155, 90)
        self.COLOR_CLOSEST = (255, 80, 80)
        self.COLOR_TEXT = (20, 22, 28)
        self.COLOR_SUBTEXT = (70, 76, 88)

    def world_to_screen(self, e, n, target_rect, cell_size=None):
        if cell_size is None:
            cell_size = target_rect.width / self.global_map_size
        sx = target_rect.x + e * cell_size + cell_size / 2.0
        sy = target_rect.y + target_rect.height - (n * cell_size + cell_size / 2.0)
        return int(round(sx)), int(round(sy))

    def draw_arrow(self, start_xy, end_xy, color, width=3, head_len=12, head_angle=math.pi / 6):
        sx, sy = start_xy
        ex, ey = end_xy
        pygame.draw.line(self.screen, color, (sx, sy), (ex, ey), width)
        theta = math.atan2(-(ey - sy), (ex - sx))
        left_theta = theta + head_angle
        right_theta = theta - head_angle
        left_x = ex - head_len * math.cos(left_theta)
        left_y = ey + head_len * math.sin(left_theta)
        right_x = ex - head_len * math.cos(right_theta)
        right_y = ey + head_len * math.sin(right_theta)
        pygame.draw.polygon(self.screen, color, [(ex, ey), (left_x, left_y), (right_x, right_y)])

    def draw_x(self, center_xy, size, color, width=2):
        cx, cy = int(center_xy[0]), int(center_xy[1])
        half = size / 2.0
        pygame.draw.line(self.screen, color, (cx - half, cy - half), (cx + half, cy + half), width)
        pygame.draw.line(self.screen, color, (cx - half, cy + half), (cx + half, cy - half), width)

    def _draw_legend(self, cell_size):
        legend_x = self.rect_global.x + 20
        legend_y = self.rect_global.y + 20
        line_gap = 26
        entries = [
            (self.COLOR_DUBINS, "Planned path"),
            (self.COLOR_PATH, "Executed trajectory"),
            (self.COLOR_HARD_ZONE, "Hard zone"),
            (self.COLOR_SAFE_ZONE, "Safety zone"),
        ]
        panel = pygame.Rect(legend_x - 12, legend_y - 10, 250, 120)
        pygame.draw.rect(self.screen, (255, 255, 255), panel, border_radius=10)
        pygame.draw.rect(self.screen, (210, 214, 220), panel, width=1, border_radius=10)
        for i, (color, label) in enumerate(entries):
            y = legend_y + i * line_gap
            pygame.draw.line(self.screen, color, (legend_x, y), (legend_x + 34, y), 4)
            text = self.small_font.render(label, True, self.COLOR_TEXT)
            self.screen.blit(text, (legend_x + 46, y - 10))

    def draw_obstacles_on_global(self, cell_size):
        overlay = pygame.Surface((self.rect_global.width, self.rect_global.height), pygame.SRCALPHA)
        for (oE, oN, oR) in self.obstacles:
            sx, sy = self.world_to_screen(oE, oN, self.rect_global, cell_size)
            lx = sx - self.rect_global.x
            ly = sy - self.rect_global.y

            safe_px = max(1, int(round((oR + self.safety_zone) * cell_size)))
            hard_px = max(1, int(round((oR + self.hard_zone) * cell_size)))
            obs_px = max(2, int(round(oR * cell_size)))

            if self.show_safety_ring:
                pygame.draw.circle(overlay, self.COLOR_SAFE_FILL, (lx, ly), safe_px, 0)
                pygame.draw.circle(overlay, self.COLOR_SAFE_ZONE, (lx, ly), safe_px, 2)

            pygame.draw.circle(overlay, self.COLOR_HARD_FILL, (lx, ly), hard_px, 0)
            pygame.draw.circle(overlay, self.COLOR_HARD_ZONE, (lx, ly), hard_px, 4)

            pygame.draw.circle(self.screen, self.COLOR_OBS_FILL, (sx, sy), obs_px, 0)
            pygame.draw.circle(self.screen, self.COLOR_OBS_EDGE, (sx, sy), obs_px, 2)

        self.screen.blit(overlay, self.rect_global.topleft)

    def draw_local_range_on_global(self, cell_size):
        px, py = self.world_to_screen(self.player_pos[0], self.player_pos[1], self.rect_global, cell_size)
        radius_px = max(1, int(round(self.sensor_range * cell_size)))
        pygame.draw.circle(self.screen, self.COLOR_LOCAL_BOX, (px, py), radius_px, 3)

    def _body_frame_curve_pixels(self):
        if self.global_path is None or self.global_path.shape[0] < 2:
            return []
        pts = []
        S = float(self.sensor_range)
        cx = self.rect_local.x + self.rect_local.width / 2.0
        cy = self.rect_local.y + self.rect_local.height / 2.0
        for p in self.global_path:
            xb, yb = self.world_to_body((float(p[0]), float(p[1])), agent_ENpsi=self.player_pos)
            if abs(xb) > S or abs(yb) > S:
                continue
            sx = cx + (yb / S) * (self.rect_local.width / 2.0)
            sy = cy - (xb / S) * (self.rect_local.height / 2.0)
            pts.append((int(round(sx)), int(round(sy))))
        return pts

    def _draw_local_obstacles(self):
        overlay = pygame.Surface((self.rect_local.width, self.rect_local.height), pygame.SRCALPHA)
        cx = self.rect_local.width / 2.0
        cy = self.rect_local.height / 2.0
        S = float(self.sensor_range)
        scale = (self.rect_local.width / 2.0) / S

        for (oE, oN, oR) in self.obstacles:
            xb, yb = self.world_to_body((oE, oN), agent_ENpsi=self.player_pos)
            if abs(xb) > (S + oR + self.safety_zone) or abs(yb) > (S + oR + self.safety_zone):
                continue

            x_px = cx + yb * scale
            y_px = cy - xb * scale
            obs_px = max(2, int(round(oR * scale)))
            hard_px = max(2, int(round((oR + self.hard_zone) * scale)))
            safe_px = max(2, int(round((oR + self.safety_zone) * scale)))

            if self.show_safety_ring:
                pygame.draw.circle(overlay, self.COLOR_SAFE_FILL, (int(round(x_px)), int(round(y_px))), safe_px, 0)
                pygame.draw.circle(overlay, self.COLOR_SAFE_ZONE, (int(round(x_px)), int(round(y_px))), safe_px, 2)
            pygame.draw.circle(overlay, self.COLOR_HARD_FILL, (int(round(x_px)), int(round(y_px))), hard_px, 0)
            pygame.draw.circle(overlay, self.COLOR_HARD_ZONE, (int(round(x_px)), int(round(y_px))), hard_px, 3)
            pygame.draw.circle(self.screen, self.COLOR_OBS_FILL, (int(round(self.rect_local.x + x_px)), int(round(self.rect_local.y + y_px))), obs_px, 0)
            pygame.draw.circle(self.screen, self.COLOR_OBS_EDGE, (int(round(self.rect_local.x + x_px)), int(round(self.rect_local.y + y_px))), obs_px, 2)

        self.screen.blit(overlay, self.rect_local.topleft)

    def render(self, fps=30):
        if not self.render_option:
            return

        self.screen.fill(self.COLOR_BG)
        pygame.draw.rect(self.screen, self.COLOR_GLOBAL, self.rect_global)
        pygame.draw.rect(self.screen, self.COLOR_INFO, self.rect_info)
        pygame.draw.rect(self.screen, self.COLOR_LOCAL, self.rect_local)
        pygame.draw.rect(self.screen, (220, 224, 231), self.rect_info, width=1)
        pygame.draw.rect(self.screen, (220, 224, 231), self.rect_local, width=1)

        cell_size = self.rect_global.width / self.global_map_size
        e, n = float(self.player_pos[0]), float(self.player_pos[1])
        ge, gn = float(self.goal_pos[0]), float(self.goal_pos[1])
        se, sn = float(self.initial_player_pos[0]), float(self.initial_player_pos[1])

        self.draw_obstacles_on_global(cell_size)

        # planned path
        if isinstance(self.global_path, np.ndarray) and self.global_path.shape[0] >= 2:
            dubins_pts = [self.world_to_screen(float(p[0]), float(p[1]), self.rect_global, cell_size) for p in self.global_path]
            pygame.draw.lines(self.screen, self.COLOR_DUBINS, False, dubins_pts, 5)

        # executed path
        if len(self.visited_path) >= 2:
            pts = [self.world_to_screen(p[0], p[1], self.rect_global, cell_size) for p in self.visited_path]
            pygame.draw.lines(self.screen, self.COLOR_PATH, False, pts, 4)

        # start
        sx, sy = self.world_to_screen(se, sn, self.rect_global, cell_size)
        pygame.draw.circle(self.screen, self.COLOR_START, (sx, sy), 8)

        # goal
        gx, gy = self.world_to_screen(ge, gn, self.rect_global, cell_size)
        pygame.draw.circle(self.screen, self.COLOR_GOAL, (gx, gy), 9)
        dE_g, dN_g = self.dir_from_heading(self.goal_pos[2])
        self.draw_arrow(
            (gx, gy),
            (int(gx + 24 * dE_g), int(gy - 24 * dN_g)),
            self.COLOR_GOAL_ARROW,
            width=4,
            head_len=12,
        )

        # agent
        px, py = self.world_to_screen(e, n, self.rect_global, cell_size)
        pygame.draw.circle(self.screen, self.COLOR_AGENT, (px, py), 10)
        dE_a, dN_a = self.dir_from_heading(self.player_pos[2])
        self.draw_arrow(
            (px, py),
            (int(px + 28 * dE_a), int(py - 28 * dN_a)),
            self.COLOR_ARROW,
            width=4,
            head_len=12,
        )

        if self.reference_point is not None:
            rx, ry = self.world_to_screen(self.reference_point[0], self.reference_point[1], self.rect_global, cell_size)
            self.draw_x((rx, ry), size=14, color=self.COLOR_REF_POINT, width=3)

        if self.closest_path_point is not None:
            cxp, cyp = self.world_to_screen(
                float(self.closest_path_point[0]),
                float(self.closest_path_point[1]),
                self.rect_global,
                cell_size,
            )
            pygame.draw.circle(self.screen, self.COLOR_CLOSEST, (cxp, cyp), 5)

        self.draw_local_range_on_global(cell_size)
        self._draw_legend(cell_size)

        # info panel
        info_x = self.rect_info.x + 18
        y = 22
        blocks = [
            (self.font, f"Return: {float(self.cumulative_reward):.2f}"),
            (self.font, f"Steps: {self.steps}"),
            (self.medium_font, f"v = {self.v:.2f} m/s"),
            (self.medium_font, f"w = {self.w:.3f} rad/s"),
            (self.medium_font, f"a_cmd = {self.a_cmd:.2f}"),
            (self.medium_font, f"eta = {math.degrees(self.eta):.2f} deg"),
            (self.medium_font, f"Sensor R = {self.sensor_range:.1f} m"),
            (self.medium_font, f"hard / safe = {self.hard_zone:.1f} / {self.safety_zone:.1f} m"),
            (self.medium_font, f"layout = {self.latest_layout_name}"),
        ]
        for font, txt in blocks:
            surf = font.render(txt, True, self.COLOR_TEXT)
            self.screen.blit(surf, (info_x, y))
            y += surf.get_height() + 12

        self.render_local_body_view()

        if self.terminated:
            tag = "SUCCESS" if self.goal else "FINISHED"
            panel = pygame.Rect(self.rect_global.x + 20, self.rect_global.bottom - 74, 180, 50)
            pygame.draw.rect(self.screen, (255, 255, 255), panel, border_radius=10)
            pygame.draw.rect(self.screen, (220, 224, 231), panel, width=1, border_radius=10)
            finished_text = self.medium_font.render(tag, True, self.COLOR_TEXT)
            self.screen.blit(finished_text, (panel.x + 18, panel.y + 12))

        pygame.display.flip()
        self.clock.tick(fps)

    def render_local_body_view(self):
        r = self.rect_local
        self.screen.fill(self.COLOR_LOCAL, r)
        pygame.draw.rect(self.screen, (220, 224, 231), r, width=1)

        L = self.local_map_size
        cell_px = r.width / L
        obs_grid = self.obs_grid

        for rr in range(L):
            for cc in range(L):
                if obs_grid[rr, cc] > 0.5:
                    x0 = r.x + cc * cell_px
                    y0 = r.y + rr * cell_px
                    rect = pygame.Rect(x0, y0, cell_px, cell_px)
                    pygame.draw.rect(self.screen, (125, 132, 146), rect)

        for i in range(L + 1):
            x_pix = r.x + i * cell_px
            pygame.draw.line(self.screen, self.COLOR_GRID, (x_pix, r.y), (x_pix, r.y + r.height), 1)
        for j in range(L + 1):
            y_pix = r.y + j * cell_px
            pygame.draw.line(self.screen, self.COLOR_GRID, (r.x, y_pix), (r.x + r.width, y_pix), 1)

        self._draw_local_obstacles()

        cx = r.x + r.width / 2.0
        cy = r.y + r.height / 2.0
        pygame.draw.circle(self.screen, self.COLOR_AGENT, (int(cx), int(cy)), 10)
        axis_len_px = 0.35 * L * cell_px
        self.draw_arrow((cx, cy), (cx, cy - axis_len_px), self.COLOR_ARROW, width=4, head_len=12)
        self.draw_arrow((cx, cy), (cx - axis_len_px, cy), self.COLOR_ARROW, width=4, head_len=12)

        pts_body_px = self._body_frame_curve_pixels()
        if len(pts_body_px) >= 2:
            pygame.draw.lines(self.screen, self.COLOR_DUBINS, False, pts_body_px, 4)

        if self.reference_point is not None:
            xb, yb = self.world_to_body((self.reference_point[0], self.reference_point[1]), agent_ENpsi=self.player_pos)
            S = float(self.sensor_range)
            if abs(xb) <= S and abs(yb) <= S:
                sx = cx + yb / S * (r.width / 2.0)
                sy = cy - xb / S * (r.height / 2.0)
                self.draw_x((sx, sy), size=14, color=self.COLOR_REF_POINT, width=3)

        if self.closest_path_point is not None:
            xb, yb = self.world_to_body((float(self.closest_path_point[0]), float(self.closest_path_point[1])), agent_ENpsi=self.player_pos)
            S = float(self.sensor_range)
            if abs(xb) <= S and abs(yb) <= S:
                sx = cx + yb / S * (r.width / 2.0)
                sy = cy - xb / S * (r.height / 2.0)
                pygame.draw.circle(self.screen, self.COLOR_CLOSEST, (int(round(sx)), int(round(sy))), 6)

        title = self.medium_font.render("Body-frame view", True, self.COLOR_TEXT)
        self.screen.blit(title, (r.x + 16, r.y + 14))

    # =========================
    # Paper figure export
    # =========================
    def save_publication_figure(self, save_path="paper_figure.png", dpi=300, show_local=True):
        if show_local:
            fig = plt.figure(figsize=(14, 7), dpi=dpi)
            gs = fig.add_gridspec(1, 2, width_ratios=[1.7, 1.0])
            ax_g = fig.add_subplot(gs[0, 0])
            ax_l = fig.add_subplot(gs[0, 1])
        else:
            fig, ax_g = plt.subplots(figsize=(8.5, 8.0), dpi=dpi)
            ax_l = None

        # global view
        ax_g.set_facecolor("#f7f8fa")
        for (oE, oN, oR) in self.obstacles:
            if self.show_safety_ring:
                ax_g.add_patch(patches.Circle((oE, oN), oR + self.safety_zone, fill=True, facecolor="#f59e0b", alpha=0.10, edgecolor="#f59e0b", linestyle="--", linewidth=1.5))
            ax_g.add_patch(patches.Circle((oE, oN), oR + self.hard_zone, fill=True, facecolor="#ef4444", alpha=0.12, edgecolor="#dc2626", linewidth=2.0))
            ax_g.add_patch(patches.Circle((oE, oN), oR, fill=True, facecolor="#6b7280", edgecolor="#111827", linewidth=1.5))

        if isinstance(self.global_path, np.ndarray) and self.global_path.shape[0] >= 2:
            ax_g.plot(self.global_path[:, 0], self.global_path[:, 1], color="#06b6d4", linewidth=2.8, label="Planned path")
        if len(self.visited_path) >= 2:
            vp = np.asarray(self.visited_path, dtype=float)
            ax_g.plot(vp[:, 0], vp[:, 1], color="#c026d3", linewidth=2.8, label="Executed trajectory")

        start = np.asarray(self.initial_player_pos[:2], dtype=float)
        goal = np.asarray(self.goal_pos[:2], dtype=float)
        ax_g.scatter([start[0]], [start[1]], s=80, c="#2563eb", label="Start", zorder=5)
        ax_g.scatter([goal[0]], [goal[1]], s=80, c="#16a34a", label="Goal", zorder=5)

        ax_g.arrow(
            start[0], start[1],
            12 * math.cos(float(self.initial_player_pos[2])),
            12 * math.sin(float(self.initial_player_pos[2])),
            width=0.6, head_width=4.0, head_length=5.0, color="#2563eb", length_includes_head=True,
        )
        ax_g.arrow(
            goal[0], goal[1],
            12 * math.cos(float(self.goal_pos[2])),
            12 * math.sin(float(self.goal_pos[2])),
            width=0.6, head_width=4.0, head_length=5.0, color="#16a34a", length_includes_head=True,
        )

        if self.reference_point is not None:
            ax_g.scatter([self.reference_point[0]], [self.reference_point[1]], s=70, marker="x", c="#a21caf", linewidths=2.0, label="Reference")
        if self.closest_path_point is not None:
            ax_g.scatter([self.closest_path_point[0]], [self.closest_path_point[1]], s=50, c="#ef4444", edgecolors="white", linewidths=0.8, label="Closest path point")

        sensor_circle = patches.Circle((self.player_pos[0], self.player_pos[1]), self.sensor_range, fill=False, edgecolor="#10b981", linewidth=1.8, linestyle="--", alpha=0.9)
        ax_g.add_patch(sensor_circle)

        ax_g.set_xlim(0, self.global_map_size)
        ax_g.set_ylim(0, self.global_map_size)
        ax_g.set_aspect("equal", adjustable="box")
        ax_g.set_xlabel("E [m]")
        ax_g.set_ylabel("N [m]")
        ax_g.set_title("Global trajectory and obstacle corridor", fontweight="bold")
        ax_g.grid(True, linestyle=":", alpha=0.35)
        ax_g.legend(loc="upper right", fontsize=9, frameon=True)

        # local view
        if ax_l is not None:
            ax_l.set_facecolor("#f7f8fa")
            S = float(self.sensor_range)
            ax_l.imshow(self.obs_grid, cmap="Greys", origin="upper", extent=[-S, S, -S, S], alpha=0.30)

            for (oE, oN, oR) in self.obstacles:
                xb, yb = self.world_to_body((oE, oN), agent_ENpsi=self.player_pos)
                center_xy = (yb, xb)
                if self.show_safety_ring:
                    ax_l.add_patch(patches.Circle(center_xy, oR + self.safety_zone, fill=True, facecolor="#f59e0b", alpha=0.10, edgecolor="#f59e0b", linestyle="--", linewidth=1.2))
                ax_l.add_patch(patches.Circle(center_xy, oR + self.hard_zone, fill=True, facecolor="#ef4444", alpha=0.12, edgecolor="#dc2626", linewidth=1.6))
                ax_l.add_patch(patches.Circle(center_xy, oR, fill=True, facecolor="#6b7280", edgecolor="#111827", linewidth=1.2))

            if isinstance(self.global_path, np.ndarray) and self.global_path.shape[0] >= 2:
                body_xy = np.array([self.world_to_body((float(p[0]), float(p[1])), agent_ENpsi=self.player_pos) for p in self.global_path], dtype=float)
                keep = (np.abs(body_xy[:, 0]) <= S) & (np.abs(body_xy[:, 1]) <= S)
                body_xy = body_xy[keep]
                if len(body_xy) >= 2:
                    ax_l.plot(body_xy[:, 1], body_xy[:, 0], color="#06b6d4", linewidth=2.5)

            ax_l.scatter([0.0], [0.0], s=80, c="#2563eb", zorder=5)
            ax_l.arrow(0.0, 0.0, 0.0, 10.0, width=0.25, head_width=2.0, head_length=2.5, color="#111827", length_includes_head=True)
            ax_l.arrow(0.0, 0.0, -10.0, 0.0, width=0.25, head_width=2.0, head_length=2.5, color="#111827", length_includes_head=True)

            if self.reference_point is not None:
                xb, yb = self.world_to_body((self.reference_point[0], self.reference_point[1]), agent_ENpsi=self.player_pos)
                ax_l.scatter([yb], [xb], s=70, marker="x", c="#a21caf", linewidths=2.0)

            ax_l.set_xlim(-S, S)
            ax_l.set_ylim(-S, S)
            ax_l.set_aspect("equal", adjustable="box")
            ax_l.set_xlabel(r"$y_b$ [m]")
            ax_l.set_ylabel(r"$x_b$ [m]")
            ax_l.set_title("Body-frame local view", fontweight="bold")
            ax_l.grid(True, linestyle=":", alpha=0.35)

        fig.tight_layout()
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
        return save_path

    # =========================
    # Misc
    # =========================
    def close(self):
        if self.render_option:
            pygame.quit()

    def handle_keyboard_input(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if not self.terminated:
                        target_heading = None
                        if event.key == pygame.K_UP:
                            target_heading = math.pi / 2
                        elif event.key == pygame.K_RIGHT:
                            target_heading = 0.0
                        elif event.key == pygame.K_DOWN:
                            target_heading = -math.pi / 2
                        elif event.key == pygame.K_LEFT:
                            target_heading = math.pi
                        elif event.key == pygame.K_w:
                            target_heading = 3 * math.pi / 4
                        elif event.key == pygame.K_e:
                            target_heading = math.pi / 4
                        elif event.key == pygame.K_z:
                            target_heading = -3 * math.pi / 4
                        elif event.key == pygame.K_c:
                            target_heading = -math.pi / 4

                        if target_heading is not None:
                            dpsi = self.normalize_angle(target_heading - self.player_pos[2])
                            w = np.clip(dpsi / self.dt, -self.max_w, self.max_w)
                            raw = w / self.max_w
                            self.step(np.array([raw], dtype=np.float32))

                    if event.key == pygame.K_q or event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_r:
                        self.reset()

            self.render()
            self.clock.tick(30)

    def set_player_pos(self, player_pos):
        self.player_pos = np.array(player_pos[:3], dtype=np.float32)

    def get_player_pos(self):
        return self.player_pos.copy()

    def get_all_states(self):
        return [[j, i] for i in range(self.global_map_size) for j in range(self.global_map_size)]

    def simulate_action(self, player_pos, action):
        self.retry()
        self.set_player_pos(player_pos)
        next_state, reward, terminated, truncated, info = self.step(action)
        return next_state, reward, terminated

    def plot_w_history(self, show=True, save_path=None, title="Yaw rate (w) vs Time"):
        if len(self.time_table) == 0 or len(self.w_table) == 0:
            print("[plot_w_history] No data to plot. Run steps first.")
            return

        t = np.asarray(self.time_table, dtype=float)
        w = np.asarray(self.w_table, dtype=float)

        plt.figure(figsize=(8, 4.5))
        plt.plot(t, w, linewidth=2.0)
        plt.xlabel("Time [s]")
        plt.ylabel("Yaw rate w [rad/s]")
        plt.title(title)
        plt.grid(True, linestyle=":", alpha=0.4)

        if save_path is not None:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        if show:
            plt.show()
        else:
            plt.close()


if __name__ == "__main__":
    env = SimpleMazeGrid(
        global_map_size=400,
        local_map_size=125,
        v=5.0,
        w=[-0.46, 0.46],
        dt=0.1,
        render_option=True,
        random_seed=0,
        spec=None,
        obstacle_count=2,
        obstacle_min_radius=2.0,
        obstacle_max_radius=10.0,
        sensor_range=35.0,
        use_lidar_edges=True,
        lidar_num_rays=360,
        lidar_fov=math.pi * 2,
        reference_L=15.0,
        hard_zone=2.0,
        safety_zone=4.0,
        obstacle_layout="corridor_pair",
    )

    env.render()
    env.handle_keyboard_input()
    env.close()
