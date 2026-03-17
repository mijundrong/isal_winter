import math
import random
from typing import List, Optional, Tuple, Dict

import numpy as np
import pygame
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt

from path_planner import dubins_path


class SimpleMazeGrid(gym.Env):
    """
    EN 좌표계 (E: 동(+x), N: 북(+y))
    Gymnasium 환경

    - Dubins 전역 경로 생성
    - 경로 추종(LOS 기반 eta) + (옵션) 장애물 관측(로컬 grid / LIDAR)
    - action: yaw rate(w) 직접 제어 [-1,1] -> [-max_w, max_w]
    """

    metadata = {"render_modes": ["human"]}

    # =========================
    # Init
    # =========================
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
        # obstacles
        obstacle_count: int = 0,
        obstacle_min_radius: float = 1.0,
        obstacle_max_radius: float = 3.0,
        # sensor range
        sensor_range: Optional[float] = None,
        # lidar
        use_lidar_edges: bool = True,
        lidar_num_rays: int = 360,
        lidar_fov: float = 2 * math.pi,
        reference_L: Optional[float] = None,
    ):
        super().__init__()

        # ---- Core params ----
        self.global_map_size = int(global_map_size)
        self.local_map_size = int(local_map_size)
        self.dt = float(dt)
        self.render_option = bool(render_option)
        self.spec = spec

        # ---- Motion ----
        self.v = float(v)
        self.w = 0.0
        self.min_w = float(w[0])
        self.max_w = float(w[1])

        # Dubins planner
        self.R = self.v / max(self.max_w, 1e-9)
        #self.path_planner = dubins_path(1 / self.R, 0.1)
        self.extension_len = 400

        self.max_steps = 3000
        self.terminated_radius = 3.0

        # ---- History ----
        self.hist_len = 1
        self.w_hist: List[float] = [0.0] * self.hist_len
        self.eta_hist: List[float] = [0.0] * self.hist_len

        # ---- Sensor range ----
        self.sensor_range = float(sensor_range) if sensor_range is not None else (self.local_map_size / 2.0)
        if self.sensor_range <= 0:
            raise ValueError("sensor_range must be > 0")

        # Reference lookahead distance
        self.reference_L = float(reference_L) if reference_L is not None else (0.8 * self.sensor_range)

        # ---- LIDAR ----
        self.use_lidar_edges = bool(use_lidar_edges)
        self.lidar_num_rays = int(lidar_num_rays)
        self.lidar_fov = float(lidar_fov)
        self.lidar_max_range = float(self.sensor_range)

        # ---- Obstacles / safety ----
        self.obstacle_count = int(obstacle_count)
        self.obstacle_min_r = float(obstacle_min_radius)
        self.obstacle_max_r = float(obstacle_max_radius)

        self.radius = 400
        self.agent_radius = 0.0 #로봇의 물리적인 크기, 논문에 넣을거면 넣어도 괜찮음
        self.safety_zone = 4.0  # 충돌하진 않았지만 위험할 정도로 가까운 상태를 평가
        self.hard_zone = 2.0    # 도달하면 아예 안되는 구간

        # ---- Runtime states ----
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

        # ---- Reward config ----
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

        # ---- Reset core ----
        if self.spec is not None:
            self._reset_core_spec(random_seed)
        else:
            self._reset_core(random_seed)

        # ---- Observation space ----
        self._build_observation_space()

        # ---- Render init ----
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

        # action: [-1,1] -> [-max_w, max_w]
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

    def _reset_core(self, random_seed=None):
        self._reset_common(random_seed)

        # agent at center + random heading
        center_E = self.global_map_size // 2
        center_N = self.global_map_size // 2
        psi0 = random.uniform(-math.pi, math.pi)

        self.initial_player_pos = np.array([center_E, center_N, psi0], dtype=np.float32)
        self.player_pos = self.initial_player_pos.copy()

        # goal: radius away from center + random heading

        phi = random.uniform(-math.pi, math.pi)

        ge = float(center_E + self.radius * math.cos(phi))
        gn = float(center_N + self.radius * math.sin(phi))
        ge = float(np.clip(ge, 0.0, self.global_map_size - 1))
        gn = float(np.clip(gn, 0.0, self.global_map_size - 1))
        gpsi = random.uniform(-math.pi, math.pi)

        self.goal_pos = np.array([ge, gn, gpsi], dtype=np.float32)

        self._build_global_dubins_path()
        self.update_reference_point()

        # obstacles
        self.obstacles = []
        self.generate_obstacles(random_seed, midline_offset_ratio=0.01)

        # local obs grid
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

        # obstacles from spec or random
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
        else:
            self.generate_obstacles(random_seed, midline_offset_ratio=0.01)

        self.obs_grid = self.compute_local_grids()
        return self.get_state()

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

    def retry(self):
        self.player_pos = self.initial_player_pos.copy()
        self.terminated = False
        self.goal = False
        self.cumulative_reward = 0.0
        self.steps = 0
        self.w = 0.0

        self.w_hist = [0.0] * self.hist_len
        self.eta_hist = [0.0] * self.hist_len

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

        old_pos = self.player_pos.copy()
        old_w = self.w

        # pre-update (old state 기준)
        self.update_reference_point()
        self.obs_grid = self.compute_local_grids()
        self.path_tracking()

        # action -> w
        raw = float(action[0])  # [-1,1]
        self.w = float(np.clip(raw, -1.0, 1.0) * self.max_w)

        # integrate
        new_pos = self.player_pos.copy()
        new_pos[2] = self.normalize_angle(new_pos[2] + self.w * self.dt)
        dE = self.v * math.cos(new_pos[2]) * self.dt
        dN = self.v * math.sin(new_pos[2]) * self.dt
        new_pos[0] = float(np.clip(new_pos[0] + dE, 0.0, self.global_map_size - 1))
        new_pos[1] = float(np.clip(new_pos[1] + dN, 0.0, self.global_map_size - 1))

        # new state 기준 reference/eta 갱신
        self.player_pos = new_pos
        self.update_reference_point()
        self.obs_grid = self.compute_local_grids()
        self.path_tracking()

        self._push_w(self.w)
        self._push_eta(self.eta)

        # path lateral distance (body y)
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

        # reward parts
        dw = (self.w - old_w)
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


        R_track = float((0.1)**path_dist)
        R_track = float(np.clip(R_track, -1.0, 1.0))
        # print("1", R_track)

        new_eta = abs(self.eta)
        if new_eta <= math.radians(1):
            new_eta = 0.0
        R_eta = float((0.1)**new_eta)
        R_eta = float(np.clip(R_eta, -1.0, 1.0))
        # print("2", R_eta)

        reward = (
            cfg["w_smooth"] * R_smooth
            + cfg["w_obs"] * R_obs
            + cfg["w_time"] * R_time
            + cfg["w_track"] * R_track
            + cfg["w_eta"] * R_eta
        )
        # reward = float(np.clip(reward, -cfg["clip_per_step"], cfg["clip_per_step"]))

        # termination
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

        # logs
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
    # Obstacles
    # =========================
    def generate_obstacles(
            self,
            random_seed=None,
            max_attempts=5000,
            ensure_midline=True,  
            midline_offset_ratio=0.2, 
    ):
        """
        수정된 방식: 생성된 Dubins 전역 경로(self.global_path) 위에 직접 장애물을 배치합니다.
        (기존의 midline 및 랜덤 배치 코드는 완전히 삭제됨)
        """
        self.obstacles = []
        if self.obstacle_count <= 0:
            return

        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)

        # 1. 생성된 전역 경로가 없거나 너무 짧으면 배치 취소
        if not hasattr(self, 'global_path') or self.global_path is None or len(self.global_path) < 20:
            return

        # 시작점과 끝점 설정
        E0, N0 = float(self.initial_player_pos[0]), float(self.initial_player_pos[1])
        if getattr(self, "path_end_state", None) is not None:
            Eg_end, Ng_end = float(self.path_end_state[0]), float(self.path_end_state[1])
        else:
            Eg_end, Ng_end = float(self.goal_pos[0]), float(self.goal_pos[1])

        def min_sep_with_hardzone(r_obs: float) -> float:
            return float(r_obs + self.hard_zone + self.agent_radius)

        path_len = len(self.global_path)

        for _ in range(self.obstacle_count):
            placed = False
            for _ in range(max_attempts):
                # 2. 경로의 35% ~ 75% 구간 중에서 무작위로 점(인덱스)을 하나 고름
                idx = random.randint(int(path_len * 0.35), int(path_len * 0.75))
                target_pt = self.global_path[idx]

                # 3. 해당 경로 위의 좌표에 장애물 중심을 설정
                E, N = float(target_pt[0]), float(target_pt[1])
                r = random.uniform(self.obstacle_min_r, self.obstacle_max_r)

                # 4. 시작점/종료점 하드존(절대 침범 불가 구역)과 너무 가까운지 확인
                if math.hypot(E - E0, N - N0) < min_sep_with_hardzone(r):
                    continue
                if math.hypot(E - Eg_end, N - Ng_end) < min_sep_with_hardzone(r):
                    continue

                # 5. 기존에 배치된 다른 장애물과 겹치는지 확인
                overlap = False
                for (oE, oN, oR) in self.obstacles:
                    if math.hypot(E - oE, N - oN) < ((r + self.hard_zone) + (oR + self.hard_zone) + 1.0):
                        overlap = True
                        break

                if overlap:
                    continue

                # 모든 조건을 통과하면 경로 위에 장애물 추가 완료!
                self.obstacles.append((E, N, r))
                placed = True
                break

            if not placed:
                print(f"[Warning] Could not place obstacle {len(self.obstacles)+1} on path.")


    def check_collision(self, pos_EN):
        E, N = float(pos_EN[0]), float(pos_EN[1])
        for (oE, oN, oR) in self.obstacles:
            if math.hypot(E - oE, N - oN) <= (oR + self.agent_radius):
                return True
        return False

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
        """
        obstacle grid만 반환 (L,L) float32 {0,1}
        """
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
    # Dubins path & reference
    # =========================
    def _build_global_dubins_path(self):
        """
        initial -> goal Dubins 경로 + 직선 연장.
        self.global_path: (N,2)
        self.path_end_state: (E,N,psi_end)
        """
        # --- 수정된 부분: 회전 반경(R)을 1~1.8배 랜덤하게 증가시켜 완만하게 만듦 ---
        random_factor = random.uniform(1.0, 1.8)
        adjusted_R = self.R * random_factor
        
        # 새로운 곡률(1/adjusted_R)을 적용한 임시 플래너 생성
        current_planner = dubins_path(1 / adjusted_R, 0.1)
        # -------------------------------------------------------------------------

        try:
            # 기존 self.path_planner.plan(...) 대신 current_planner.plan(...) 사용
            path_x, path_y, path_yaw, modes, lengths = current_planner.plan(
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

            # extend straight segment
            if len(path_x) > 0:
                last_x = path_x[-1]
                last_y = path_y[-1]
                last_yaw = path_yaw[-1]

                step_size = 0.1
                num_points = int(self.extension_len / step_size)

                for i in range(1, num_points + 1):
                    dist = i * step_size
                    nx = last_x + dist * math.cos(last_yaw)
                    ny = last_y + dist * math.sin(last_yaw)
                    
                    # --- 수정 1: 연장되는 선이 맵 경계 밖으로 나가면 즉시 선 긋기를 멈춥니다 ---
                    if 0 <= nx <= self.global_map_size and 0 <= ny <= self.global_map_size:
                        path_x.append(nx)
                        path_y.append(ny)
                    else:
                        break

            # end state
            if len(path_x) >= 2:
                last_x, last_y = path_x[-1], path_y[-1]
                prev_x, prev_y = path_x[-2], path_y[-2]
                psi_end = math.atan2(last_y - prev_y, last_x - prev_x)
                self.path_end_state = np.array([last_x, last_y, psi_end], dtype=np.float32)
            else:
                self.path_end_state = None

            # --- 수정 2: Dubins 곡선 자체가 경계를 살짝 벗어나는 경우 경계선에 딱 맞게 잘라줍니다 ---
            path_x = np.clip(path_x, 0.0, self.global_map_size)
            path_y = np.clip(path_y, 0.0, self.global_map_size)

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

        # remaining length
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

    # =========================
    # Path tracking (LOS)
    # =========================
    def path_tracking(self):
        """
        LOS 기반 heading error eta:
        eta = atan2(ref-pos) - psi

        a_cmd = 2 * V^2/L * sin(eta)
        w_cmd = a_cmd / V
        """
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
    # Obstacle penalties
    # =========================
    def obstacle_cell_distances_at(self, agent_ENpsi):
        """
        (LIDAR OFF일 때 기준) 로컬 그리드에 채워지는 obstacle 셀 중심까지 거리
        """
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
        """
        clear: 최소 안전거리 위반 정도 [0,1]
        hard : hard-zone 안이면 2.0 아니면 0.0 (원 코드 유지)
        """
        # LIDAR ON이면 obstacle_cell_distances_at이 "grid 기반"이라 의미가 애매하지만,
        # 원 코드 흐름 유지(여기서는 여전히 grid 기반으로 페널티 산출).
        dists, _, _ = self.obstacle_cell_distances_at(new_pos)
        if dists.size == 0:
            return {"clear": 0.0, "hard": 0.0}

        eps = 1e-6
        d_min = float(np.min(dists))

        d_safe = float(self.agent_radius + self.safety_zone)
        clear = float(np.clip(1.0 - d_min / max(d_safe, eps), 0.0, 1.0))

        hard = 2.0 if d_min < float(self.hard_zone) else 0.0
        return {"clear": clear, "hard": hard}

    # =========================
    # Rendering
    # =========================
    def _init_render(self):
        pygame.init()

        self.global_screen_width = 1500
        self.global_screen_height = 1500
        self.info_width = 220
        self.local_screen_width = 500
        self.local_screen_height = 500

        self.total_width = self.global_screen_width + self.info_width + self.local_screen_width
        self.total_height = self.global_screen_height

        self.screen = pygame.display.set_mode((self.total_width, self.total_height))
        pygame.display.set_caption("(E-N World + Info + Body-Frame)")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)
        self.small_font = pygame.font.Font(None, 20)

        self.rect_global = pygame.Rect(0, 0, self.global_screen_width, self.global_screen_height)
        self.rect_info = pygame.Rect(self.global_screen_width, 0, self.info_width, self.global_screen_height)
        self.rect_local = pygame.Rect(
            self.global_screen_width + self.info_width, 0, self.local_screen_width, self.local_screen_height
        )

        # colors (사용되는 것만)
        self.COLOR_BG = (255, 255, 255)
        self.COLOR_GLOBAL = (230, 230, 230)
        self.COLOR_INFO = (245, 245, 245)
        self.COLOR_LOCAL = (230, 230, 255)
        self.COLOR_GRID = (210, 210, 240)

        self.COLOR_AGENT = (0, 0, 255)
        self.COLOR_GOAL = (0, 200, 0)
        self.COLOR_PATH = (150, 0, 150)
        self.COLOR_ARROW = (0, 0, 0)
        self.COLOR_GOAL_ARROW = (255, 0, 0)

        self.COLOR_OBS_FILL = (90, 90, 90)
        self.COLOR_OBS_EDGE = (0, 0, 0)
        self.COLOR_OBS_CELL = (110, 110, 110)
        self.COLOR_HARD_ZONE = (255, 0, 0)

        self.COLOR_DUBINS = (0, 200, 200)
        self.COLOR_REF_POINT = (200, 0, 200)
        self.COLOR_LOCAL_BOX = (0, 255, 0)

        self.COLOR_CLOSEST = (255, 0, 0)  # closest path point

    def world_to_screen(self, e, n, target_rect, cell_size=None):
        if cell_size is None:
            cell_size = target_rect.width / self.global_map_size
        sx = target_rect.x + e * cell_size + cell_size / 2.0
        sy = target_rect.y + target_rect.height - (n * cell_size + cell_size / 2.0)
        return int(sx), int(sy)

    def draw_arrow(self, start_xy, end_xy, color, width=3, head_len=10, head_angle=math.pi / 6):
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

    def draw_obstacles_on_global(self, cell_size):
        for (oE, oN, oR) in self.obstacles:
            sx, sy = self.world_to_screen(oE, oN, self.rect_global, cell_size)

            pr_obs = max(1, int(oR * cell_size))
            pygame.draw.circle(self.screen, self.COLOR_OBS_FILL, (sx, sy), pr_obs, 0)
            pygame.draw.circle(self.screen, self.COLOR_OBS_EDGE, (sx, sy), pr_obs, 2)

            hard_r = oR + self.hard_zone
            hard_px = max(1, int(hard_r * cell_size))
            pygame.draw.circle(self.screen, self.COLOR_HARD_ZONE, (sx, sy), hard_px, 2)

    def draw_local_range_on_global(self, cell_size):
        px, py = self.world_to_screen(self.player_pos[0], self.player_pos[1], self.rect_global, cell_size)
        radius_px = max(1, int(self.sensor_range * cell_size))
        pygame.draw.circle(self.screen, self.COLOR_LOCAL_BOX, (px, py), radius_px, 3)

    def render(self, fps=30):
        if not self.render_option:
            return

        self.screen.fill(self.COLOR_BG)
        pygame.draw.rect(self.screen, self.COLOR_GLOBAL, self.rect_global)
        pygame.draw.rect(self.screen, self.COLOR_INFO, self.rect_info)
        pygame.draw.rect(self.screen, self.COLOR_LOCAL, self.rect_local)

        cell_size = self.rect_global.width / self.global_map_size
        e, n = float(self.player_pos[0]), float(self.player_pos[1])
        ge, gn = float(self.goal_pos[0]), float(self.goal_pos[1])

        # obstacles
        self.draw_obstacles_on_global(cell_size)

        # goal
        gx, gy = self.world_to_screen(ge, gn, self.rect_global, cell_size)
        pygame.draw.circle(self.screen, self.COLOR_GOAL, (gx, gy), int(cell_size / 3))
        dE_g, dN_g = self.dir_from_heading(self.goal_pos[2])
        self.draw_arrow(
            (gx, gy),
            (int(gx + 1.5 * cell_size * dE_g), int(gy - 1.5 * cell_size * dN_g)),
            self.COLOR_GOAL_ARROW,
            width=3,
            head_len=10,
        )

        # dubins path
        if isinstance(self.global_path, np.ndarray) and self.global_path.shape[0] >= 2:
            dubins_pts = [self.world_to_screen(float(p[0]), float(p[1]), self.rect_global, cell_size) for p in self.global_path]
            pygame.draw.lines(self.screen, self.COLOR_DUBINS, False, dubins_pts, 2)

        # agent
        px, py = self.world_to_screen(e, n, self.rect_global, cell_size)
        pygame.draw.circle(self.screen, self.COLOR_AGENT, (px, py), max(2, int(self.agent_radius * cell_size)))
        dE_a, dN_a = self.dir_from_heading(self.player_pos[2])
        self.draw_arrow(
            (px, py),
            (int(px + 1.5 * cell_size * dE_a), int(py - 1.5 * cell_size * dN_a)),
            self.COLOR_ARROW,
            width=3,
            head_len=10,
        )

        # visited path
        if len(self.visited_path) >= 2:
            pts = [self.world_to_screen(p[0], p[1], self.rect_global, cell_size) for p in self.visited_path]
            pygame.draw.lines(self.screen, self.COLOR_PATH, False, pts, 2)

        # reference point
        if self.reference_point is not None:
            rx, ry = self.world_to_screen(self.reference_point[0], self.reference_point[1], self.rect_global, cell_size)
            self.draw_x((rx, ry), size=10, color=self.COLOR_REF_POINT, width=2)

        # closest path point (global에서 빨간 점)
        if self.closest_path_point is not None:
            cxp, cyp = self.world_to_screen(
                float(self.closest_path_point[0]),
                float(self.closest_path_point[1]),
                self.rect_global,
                cell_size,
            )
            pygame.draw.circle(self.screen, self.COLOR_CLOSEST, (cxp, cyp), 3)

        # sensor range
        self.draw_local_range_on_global(cell_size)

        # info
        info_x = self.rect_info.x + 10
        cell_world = (2 * self.sensor_range) / self.local_map_size

        self.screen.blit(self.font.render(f"Return: {float(self.cumulative_reward):.2f}", True, (0, 0, 0)), (info_x, 10))
        self.screen.blit(self.font.render(f"Steps: {self.steps}", True, (0, 0, 0)), (info_x, 50))
        self.screen.blit(self.small_font.render(f"v: {self.v:.2f}", True, (0, 0, 0)), (info_x, 90))
        self.screen.blit(self.small_font.render(f"w: {self.w:.3f}", True, (0, 0, 0)), (info_x, 110))
        self.screen.blit(self.small_font.render(f"a_cmd: {self.a_cmd:.2f}", True, (0, 0, 0)), (info_x, 130))
        self.screen.blit(self.small_font.render(f"Sensor R: {self.sensor_range:.2f}", True, (0, 0, 0)), (info_x, 150))
        self.screen.blit(self.small_font.render(f"Cell = {cell_world:.2f} units", True, (0, 0, 0)), (info_x, 175))

        # local panel
        self.render_local_body_grid()

        if self.terminated:
            finished_text = self.font.render("FINISHED", True, (0, 0, 0))
            self.screen.blit(
                finished_text,
                (self.rect_global.x + self.rect_global.width // 2 - 70, self.rect_global.y + self.rect_global.height // 2 - 20),
            )

        pygame.display.flip()
        self.clock.tick(fps)

    def render_local_body_grid(self):
        r = self.rect_local
        pygame.draw.rect(self.screen, (0, 0, 0), r, width=2)
        self.screen.fill(self.COLOR_LOCAL, r)

        L = self.local_map_size
        cell_px = r.width / L
        obs_grid = self.obs_grid

        # 장애물 셀
        for rr in range(L):
            for cc in range(L):
                if obs_grid[rr, cc] > 0.5:
                    x0 = r.x + cc * cell_px
                    y0 = r.y + rr * cell_px
                    rect = pygame.Rect(x0, y0, cell_px, cell_px)
                    pygame.draw.rect(self.screen, self.COLOR_OBS_CELL, rect)

        # 격자선
        for i in range(L + 1):
            x_pix = r.x + i * cell_px
            pygame.draw.line(
                self.screen,
                self.COLOR_GRID,
                (x_pix, r.y),
                (x_pix, r.y + r.height),
                1,
            )
        for j in range(L + 1):
            y_pix = r.y + j * cell_px
            pygame.draw.line(
                self.screen,
                self.COLOR_GRID,
                (r.x, y_pix),
                (r.x + r.width, y_pix),
                1,
            )

        # 로컬 프레임 중심 + 축
        cx = r.x + r.width / 2.0
        cy = r.y + r.height / 2.0
        pygame.draw.circle(self.screen, self.COLOR_AGENT, (int(cx), int(cy)), 8)
        axis_len_px = 0.35 * L * cell_px
        # x_b: 위쪽, y_b: 왼쪽 (원하는 방향)
        self.draw_arrow(
            (cx, cy),
            (cx, cy - axis_len_px),
            self.COLOR_ARROW,
            width=3,
            head_len=10,
        )
        self.draw_arrow(
            (cx, cy),
            (cx - axis_len_px, cy),  # <-- 여기만 변경
            self.COLOR_ARROW,
            width=3,
            head_len=10,
        )

        S = float(self.sensor_range)

        # Dubins 경로 (바디 프레임에서 렌더)
        if (
            hasattr(self, "global_path")
            and self.global_path is not None
            and isinstance(self.global_path, np.ndarray)
            and self.global_path.shape[0] >= 2
        ):
            pts_body_px = []
            for p in self.global_path:
                E, N = float(p[0]), float(p[1])
                xb, yb = self.world_to_body((E, N), agent_ENpsi=self.player_pos)
                if abs(xb) > S or abs(yb) > S:
                    continue
                nx = xb / S
                ny = yb / S
                sx = cx + ny * (r.width / 2.0)
                sy = cy - nx * (r.height / 2.0)
                pts_body_px.append((int(sx), int(sy)))

            if len(pts_body_px) >= 2:
                pygame.draw.lines(
                    self.screen, self.COLOR_DUBINS, False, pts_body_px, 2
                )

        # reference point (바디에서 X표시)
        if self.reference_point is not None:
            xb, yb = self.world_to_body(
                (self.reference_point[0], self.reference_point[1]),
                agent_ENpsi=self.player_pos,
            )
            if abs(xb) <= S and abs(yb) <= S:
                nx = xb / S
                ny = yb / S
                sx = cx + ny * (r.width / 2.0)
                sy = cy - nx * (r.height / 2.0)
                self.draw_x((sx, sy), size=10, color=self.COLOR_REF_POINT, width=2)

        # closest path point (body에서 빨간 점)
        if self.closest_path_point is not None:
            xb, yb = self.world_to_body(
                (float(self.closest_path_point[0]), float(self.closest_path_point[1])),
                agent_ENpsi=self.player_pos,
            )
            if abs(xb) <= S and abs(yb) <= S:
                nx = xb / S
                ny = yb / S
                sx = cx + ny * (r.width / 2.0)
                sy = cy - nx * (r.height / 2.0)
                pygame.draw.circle(self.screen, self.COLOR_CLOSEST, (int(sx), int(sy)), 6)

    def draw_local_range_on_global(self, cell_size):
        px, py = self.world_to_screen(
            self.player_pos[0],
            self.player_pos[1],
            self.rect_global,
            cell_size,
        )
        radius_px = max(1, int(self.sensor_range * cell_size))
        pygame.draw.circle(
            self.screen, self.COLOR_LOCAL_BOX, (px, py), radius_px, 3
        )

    def draw_obstacles_on_global(self, cell_size):
        safety_zone = float(getattr(self, "safety_zone", 0.0))

        for (oE, oN, oR) in self.obstacles:
            sx, sy = self.world_to_screen(oE, oN, self.rect_global, cell_size)

            pr_obs = max(1, int(oR * cell_size))
            pygame.draw.circle(self.screen, self.COLOR_OBS_FILL, (sx, sy), pr_obs, 0)
            pygame.draw.circle(self.screen, self.COLOR_OBS_EDGE, (sx, sy), pr_obs, 2)

            if safety_zone > 0.0:
                hard_r = oR + self.hard_zone
                safe_r = oR + safety_zone

                hard_px = max(1, int(hard_r * cell_size))
                safe_px = max(1, int(safe_r * cell_size))

                pygame.draw.circle(
                    self.screen, self.COLOR_HARD_ZONE, (sx, sy), hard_px, 2
                )
                # pygame.draw.circle(
                #     self.screen, self.COLOR_SAFE_ZONE, (sx, sy), safe_px, 2
                # )

    # =========================
    # 기타
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
                            dpsi = self.normalize_angle(
                                target_heading - self.player_pos[2]
                            )
                            w = np.clip(
                                dpsi / self.dt, -self.max_w, self.max_w
                            )
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
        return [
            [j, i]
            for i in range(self.global_map_size)
            for j in range(self.global_map_size)
        ]

    def simulate_action(self, player_pos, action):
        self.retry()
        self.set_player_pos(player_pos)
        next_state, reward, terminated, truncated, info = self.step(action)
        return next_state, reward, terminated

    def plot_w_history(self, show=True, save_path=None, title="Yaw rate (w) vs Time"):
        """
        누적된 time_table, w_table을 사용해서 각속도(w) 그래프를 그림.
        - show=True  : 창 띄움
        - save_path  : 경로 지정 시 이미지 저장
        """
        if len(self.time_table) == 0 or len(self.w_table) == 0:
            print("[plot_w_history] No data to plot. Run steps first.")
            return

        t = np.asarray(self.time_table, dtype=float)
        w = np.asarray(self.w_table, dtype=float)

        plt.figure()
        plt.plot(t, w)
        plt.xlabel("Time [s]")
        plt.ylabel("Yaw rate w [rad/s]")
        plt.title(title)
        plt.grid(True)

        if save_path is not None:
            plt.savefig(save_path, dpi=150)

        if show:
            plt.show()
        else:
            plt.close()


# =========================
# 메인 테스트
# =========================
if __name__ == "__main__":
    spec = (
        [90, 90, -2 * math.pi / 4],
        [20, 35, -2 * math.pi / 4],
        [69, 72, 5],
    )

    env = SimpleMazeGrid(
        global_map_size=1500,
        local_map_size=100,
        v=15.0,
        w=[-0.46, 0.46],
        dt=0.1,
        render_option=True,
        random_seed=None,
        spec=None,  # 고정 시나리오
        obstacle_count=1,
        obstacle_min_radius=30.0,
        obstacle_max_radius=50.0,
        sensor_range=100.0, 
        use_lidar_edges=True,
        lidar_num_rays=360,
        lidar_fov=math.pi * 2,
        reference_L=50.0,  
    )

    env.render()
    env.handle_keyboard_input()
    env.close()
