from math import atan2

import numpy as np
import random
import pygame
import gymnasium as gym
from gymnasium import spaces
import math
from path_planner import dubins_path
from stable_baselines3 import SAC
import matplotlib
matplotlib.use("TkAgg")  # 또는 "Qt5Agg"
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class SimpleMazeGrid(gym.Env):
    """
    EN 좌표계 (E: 동(+x), N: 북(+y))
    시뮬레이션 전용 환경 (학습 X)
    - Dubins 전역 경로 + 경로 추종(path tracking)
    - APF 기반 장애물 회피 (참조점이 장애물 근처일 때만)
    - 학습이 끝난 SAC 정책 vs APF 제어 성능 비교용
    """

    metadata = {"render.modes": ["human"]}

    # =========================
    # Init & Config
    # =========================
    def __init__(
        self,
        global_map_size=None,
        local_map_size=None,
        v=None,
        w=None,
        dt=0.1,  # simple_maze_grid.py 기준 0.1로 변경
        render_option=False,
        random_seed=None,
        spec=None,
        # Waypoints (전역 Dubins 경로가 순서대로 통과할 점들)
        waypoints=None,
        # sensor range
        sensor_range=None,
        # ---- LIDAR 옵션 ----
        use_lidar_edges=True,
        lidar_num_rays=360,
        lidar_fov=2 * math.pi,  # 360 deg
        reference_L=None,
        reference_L1=None,

            # ---- Fly-by 옵션 ----
        fly_by=False,
    ):
        super(SimpleMazeGrid, self).__init__()

        # 기본 파라미터
        self.global_map_size = int(global_map_size)
        self.local_map_size = int(local_map_size)
        self.dt = float(dt)
        self.render_option = render_option
        self.spec = spec

        self.terminated = False
        self.terminated_radius =5.0  # simple_maze_grid 기준, 5.0
        self.goal = False
        self.visited_path = []

        # ===== Waypoints / Dubins 경로 =====
        self.raw_waypoints = waypoints
        self.waypoints = waypoints
        self.waypoint_states = []
        self.current_wp_index = 0
        self.current_goal_pos = None
        self.final_goal_pos = None

        # Fly-by 옵션
        self.fly_by = bool(fly_by)

        # 이동 파라미터
        self.v = float(v)
        self.w = 0.0
        self.min_w = float(w[0])
        self.max_w = float(w[1])
        self.R = self.v / max(self.max_w, 1e-9) # 0으로 나누기 방지 안전장치 추가
        self.path_planner = dubins_path(1 / self.R, 0.1)
        self.max_steps = 4000 # simple_maze_grid 기준

        # waypoint 도달 판정 반경 (맵 크기에 맞춰 키움)
        self.wp_reach_radius = 20.0

        # 히스토리 길이
        self.hist_len = 1

        # sensor range (월드 단위)
        if sensor_range is None:
            self.sensor_range = self.local_map_size / 2.0
        else:
            self.sensor_range = float(sensor_range)
            if self.sensor_range <= 0:
                raise ValueError("sensor_range must be > 0")

        # reference point까지의 거리 L (월드 거리)
        if reference_L is None:
            self.reference_L = 0.8 * self.sensor_range
        else:
            self.reference_L = float(reference_L)

        if reference_L1 is None:
            self.reference_L1 = self.reference_L
        else:
            self.reference_L1 = float(reference_L1)

        self.reference_point = None
        self.closest_path_point = None
        self.closest_path_idx = None
        self.a_cmd = 0.0
        self.w_cmd = 0.0
        self.eta = 0.0

        # LIDAR 설정
        self.use_lidar_edges = bool(use_lidar_edges)
        self.lidar_num_rays = int(lidar_num_rays)
        self.lidar_fov = float(lidar_fov)
        self.lidar_max_range = float(self.sensor_range)

        # [변경] simple_maze_grid.py 환경 설정 동기화
        self.agent_radius = 5.0      # 기존 1.0 -> 5.0
        self.safety_zone = 50.0      # 기존 10.0 -> 50.0
        self.hard_zone = 25.0        # 기존 3.5 -> 25.0

        # [변경] APF 파라미터 (스케일 변화에 따른 조정)
        # 거리가 15배 늘어났으므로 감쇠계수(l_rep)는 줄이고, 영향거리(d0)는 늘림
        self.apf_d0 = 100.0          # APF 영향 거리 (센서 범위와 유사하게)
        self.apf_K_rep = 30.0        # 반발력 게인
        self.apf_K_att = 0.0
        self.l_rep = 0.0008          # 기존 1/24.5(~0.04) -> 0.0008 (거리 제곱에 반비례하므로 대폭 축소)

        # 색상 (렌더링)
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
        self.COLOR_LOCAL_BOX = (0, 255, 0)
        self.COLOR_OBS_FILL = (90, 90, 90)
        self.COLOR_OBS_EDGE = (0, 0, 0)
        self.COLOR_OBS_CELL = (110, 110, 110)
        self.COLOR_SAFE_ZONE = (255, 165, 0)
        self.COLOR_HARD_ZONE = (255, 0, 0)
        self.COLOR_DUBINS = (0, 200, 200)
        self.COLOR_REF_POINT = (0, 0, 0)
        self.COLOR_WAYPOINT = (255, 140, 0)
        self.COLOR_WAYPOINT_FLYBY = (0, 128, 255)

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )

        self.reward_cfg = {
            "w_smooth": 0.0,
            "w_obs": 0.75,
            "w_time": 0.5,
            "w_track": 0.5,
            "bonus_goal": 100.0,
            "penalty_collision": -100.0,
            "penalty_timeout": -100.0,
            "clip_per_step": 1.0,
        }

        if self.spec is None:
            center = self.global_map_size * 0.5
            self.spec = ([center, center, 0.0], None)

        self._reset_core_spec(random_seed)

        if self.render_option:
            pygame.init()
            self.global_screen_width = 1000
            self.global_screen_height = 1000
            self.info_width = 220
            self.local_screen_width = 500
            self.local_screen_height = 500
            self.total_width = (
                self.global_screen_width + self.info_width + self.local_screen_width
            )
            self.total_height = self.global_screen_height
            self.screen = pygame.display.set_mode((self.total_width, self.total_height))
            pygame.display.set_caption("(E-N World + Info + Body-Frame)")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.Font(None, 36)
            self.small_font = pygame.font.Font(None, 20)

            self.rect_global = pygame.Rect(
                0, 0, self.global_screen_width, self.global_screen_height
            )
            self.rect_info = pygame.Rect(
                self.global_screen_width, 0, self.info_width, self.global_screen_height
            )
            self.rect_local = pygame.Rect(
                self.global_screen_width + self.info_width,
                0,
                self.local_screen_width,
                self.local_screen_height,
            )

    # =========================
    # Observation space builder
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

    # =========================
    # Core reset (spec 기반)
    # =========================
    def _reset_core_spec(self, random_seed=None):
        self.terminated = False
        self.goal = False
        self.visited_path = []

        self.current_wp_index = 0
        self.current_goal_pos = None
        self.final_goal_pos = None

        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)

        if self.spec is None:
            raise ValueError("spec이 None 입니다. (초기 상태/장애물 정보를 넣어주세요)")

        if len(self.spec) == 2:
            initial_player_pos, obs_spec = self.spec
            goal_pos = None
        elif len(self.spec) == 3:
            initial_player_pos, goal_pos, obs_spec = self.spec
        else:
            raise ValueError("spec은 (init, obs) 또는 (init, goal, obs) 형태여야 합니다.")

        self.initial_player_pos = np.array(initial_player_pos[0:3], dtype=np.float32)
        self.player_pos = self.initial_player_pos.copy()

        if goal_pos is not None:
            self.goal_pos = np.array(goal_pos[0:3], dtype=np.float32)
        else:
            self.goal_pos = None

        self._build_global_dubins_path()
        self.update_reference_point()

        self.cumulative_reward = 0.0
        self.steps = 0
        self.visited_path.append(self.player_pos.copy())

        self.w_hist = [0.0] * self.hist_len
        self.eta_hist = [0.0] * self.hist_len

        self.obstacles = []
        if obs_spec is not None:
            obs_arr = np.array(obs_spec, dtype=float)
            if obs_arr.size == 0:
                self.obstacles = []
            else:
                if obs_arr.ndim == 1:
                    if obs_arr.size != 3:
                        raise ValueError(f"obs_spec 1D인데 길이가 3이 아님: got {obs_arr.size}")
                    obs_arr = obs_arr.reshape(1, 3)
                for i in range(obs_arr.shape[0]):
                    E, N, r = float(obs_arr[i, 0]), float(obs_arr[i, 1]), float(obs_arr[i, 2])
                    self.obstacles.append((E, N, r))

        self.obs_grid = self.compute_local_grids()
        self.time_table, self.v_table, self.w_table, self.dist_error_table = [], [], [], []
        self._build_observation_space()

        return self.get_state()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        obs = self._reset_core_spec(random_seed=seed)
        info = {}
        return obs, info

    def retry(self):
        self.player_pos = self.initial_player_pos.copy()
        self.terminated = False
        self.cumulative_reward = 0.0
        self.steps = 0
        self.w = 0.0
        self.w_hist = [0.0] * self.hist_len
        self.eta_hist = [0.0] * self.hist_len
        self.current_wp_index = 0
        if self.waypoint_states:
            self.current_goal_pos = self.waypoint_states[0].copy()
        else:
            self.current_goal_pos = self.final_goal_pos
        self.update_reference_point()
        return self.get_state(), {}

    # =========================
    # Util: angle / heading
    # =========================
    @staticmethod
    def normalize_angle(angle):
        return (angle + np.pi) % (2 * np.pi) - np.pi

    @staticmethod
    def angle_between(v1, v2):
        v1 = np.asarray(v1, dtype=float)
        v2 = np.asarray(v2, dtype=float)
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        if n1 < 1e-9 or n2 < 1e-9:
            return 0.0
        c = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
        return math.acos(c)

    @staticmethod
    def dir_from_heading(psi):
        return math.cos(psi), math.sin(psi)

    # =========================
    # State build & history
    # =========================
    def _build_state(self):
        obs_grid = self.obs_grid
        w_hist_vec = np.array(self.w_hist, dtype=np.float32)
        err_pairs = []
        for e in self.eta_hist:
            err_pairs.extend([math.cos(e), math.sin(e)])
        eta_hist_vec = np.array(err_pairs, dtype=np.float32)
        obs_img = (obs_grid * 255).astype(np.uint8)[np.newaxis, :, :]

        state = {
            "w_hist": w_hist_vec,
            "eta_hist": eta_hist_vec,
            "obstacle_pos": obs_img,
        }
        return state

    def get_state(self):
        return self._build_state()

    def _push_w(self, w):
        self.w_hist = (self.w_hist + [float(w)])[-self.hist_len:]

    def _push_err(self, e):
        self.eta_hist = (self.eta_hist + [float(e)])[-self.hist_len:]

    # =========================
    # Step (RL 인터페이스)
    # =========================
    def step(self, action):
        if self.terminated:
            obs = self.get_state()
            return obs, 0.0, True, False, {}

        terminated = False
        truncated = False
        cfg = self.reward_cfg

        old_pos = self.player_pos.copy()

        self.update_reference_point()
        self.obs_grid = self.compute_local_grids()
        self.path_tracking()
        self._update_current_goal_waypoint()

        old_path_dist = None
        if hasattr(self, "closest_path_point") and self.closest_path_point is not None:
            old_path_dist = float(
                np.linalg.norm(old_pos[:2] - self.closest_path_point)
            )

        raw = float(action[0])
        raw = np.clip(raw, -1.0, 1.0)
        self.w = raw * self.max_w

        self._push_w(self.w)
        self._push_err(self.eta)

        new_pos = self.player_pos.copy()
        new_pos[2] = self.normalize_angle(new_pos[2] + self.w * self.dt)
        dE = self.v * math.cos(new_pos[2]) * self.dt
        dN = self.v * math.sin(new_pos[2]) * self.dt
        new_pos[0] = np.clip(new_pos[0] + dE, 0.0, self.global_map_size - 1)
        new_pos[1] = np.clip(new_pos[1] + dN, 0.0, self.global_map_size - 1)
        self.player_pos = new_pos

        new_path_dist = None
        if (
            hasattr(self, "global_path")
            and self.global_path is not None
            and isinstance(self.global_path, np.ndarray)
            and self.global_path.shape[0] >= 1
        ):
            path_EN = np.asarray(self.global_path[:, :2], dtype=float)
            diffs_new = path_EN - np.asarray(new_pos[:2], dtype=float)
            dists_new = np.hypot(diffs_new[:, 0], diffs_new[:, 1])
            idx_new = int(np.argmin(dists_new))
            closest_point_new = path_EN[idx_new]
            new_path_dist = float(
                np.linalg.norm(new_pos[:2] - closest_point_new)
            )

        w_arr = np.array(self.w_hist, dtype=np.float32)
        dw = np.diff(w_arr)
        w_range = max(2.0 * self.max_w, 1e-6)
        dw_n = dw / w_range
        smooth = float(np.mean(dw_n ** 2)) if dw_n.size > 0 else 0.0
        R_smooth = -float(np.clip(smooth, 0.0, 1.0))

        pen_old = self.compute_obstacle_penalties(old_pos)
        pen_new = self.compute_obstacle_penalties(new_pos)
        p_old = np.clip((pen_old["clear"] + pen_old["hard"]) / 2.0, 0.0, 1.0)
        p_new = np.clip((pen_new["clear"] + pen_new["hard"]) / 2.0, 0.0, 1.0)
        R_obs = float(np.clip(p_old - p_new, -1.0, 1.0))

        R_time = -5.0 * float(self.dt)

        if (old_path_dist is not None) and (new_path_dist is not None):
            delta_d = (old_path_dist - new_path_dist) / (self.v * self.dt)
            R_track = float(np.clip(delta_d, -1.0, 1.0))
        else:
            R_track = 0.0

        reward = (
            cfg["w_smooth"] * R_smooth
            + cfg["w_obs"] * R_obs
            + cfg["w_time"] * R_time
            + cfg["w_track"] * R_track
        )
        reward = float(np.clip(reward, -cfg["clip_per_step"], cfg["clip_per_step"]))

        if self.check_collision(new_pos[:2]):
            reward += cfg["penalty_collision"]
            terminated = True

        if self.final_goal_pos is not None:
            cur_goal_dist = float(np.linalg.norm(new_pos[:2] - self.final_goal_pos[:2]))
            heading_ok = math.cos(new_pos[2] - self.final_goal_pos[2]) >= math.cos(math.radians(30))
            # [수정할 부분] 목표 도달 조건 확인 블록
            if (cur_goal_dist < self.terminated_radius) and self.current_wp_index != 0:
                reward += cfg["bonus_goal"]
                self.goal = True
                terminated = True
                
                # === [추가] 도착 시간 출력 코드 시작 ===
                arrival_time = self.steps * self.dt
                print(f"Goal Reached! Total Time: {arrival_time:.2f} sec")

        next_state = self._build_state()

        self.cumulative_reward += reward
        self.steps += 1
        self.visited_path.append(self.player_pos.copy())
        self.time_table.append(self.steps * self.dt)
        self.w_table.append(self.w)
        self.dist_error_table.append(new_path_dist)

        self.terminated = terminated or truncated
        info = {}
        return next_state, reward, terminated, truncated, info

    def auto_step_path_tracking(self):
        if self.terminated:
            return self.player_pos.copy()

        self.update_reference_point()
        self.path_tracking()
        self.w = float(np.clip(self.w_cmd, self.min_w, self.max_w))

        new_pos = self.player_pos.copy()
        new_pos[2] = self.normalize_angle(new_pos[2] + self.w * self.dt)
        dE = self.v * math.cos(new_pos[2]) * self.dt
        dN = self.v * math.sin(new_pos[2]) * self.dt
        new_pos[0] = np.clip(new_pos[0] + dE, 0.0, self.global_map_size - 1)
        new_pos[1] = np.clip(new_pos[1] + dN, 0.0, self.global_map_size - 1)
        self.player_pos = new_pos
        self.steps += 1
        self.visited_path.append(self.player_pos.copy())
        self.obs_grid = self.compute_local_grids()

        if self.check_collision(self.player_pos[:2]):
            self.terminated = True

        if self.final_goal_pos is not None:
            cur_dist = float(np.linalg.norm(new_pos[:2] - self.final_goal_pos[:2]))
            heading_ok = math.cos(
                new_pos[2] - self.normalize_angle(self.final_goal_pos[2])
            ) >= math.cos(math.radians(30))
            if (cur_dist < self.terminated_radius) and heading_ok:
                self.terminated = True

        return self.player_pos.copy()

    def check_collision(self, pos_EN):
        E, N = float(pos_EN[0]), float(pos_EN[1])
        for (oE, oN, oR) in self.obstacles:
            if math.hypot(E - oE, N - oN) <= (oR + self.agent_radius):
                return True
        return False

    def is_reference_point_in_obstacle_zone(self, margin=None):
        margin = self.hard_zone if margin is None else float(margin)
        L_check = getattr(self, "reference_L1", None)
        ref_pt = self.compute_reference_point_on_global(L=L_check)

        if ref_pt is None:
            return False

        E_r = float(ref_pt[0])
        N_r = float(ref_pt[1])

        for (oE, oN, oR) in self.obstacles:
            d = math.hypot(E_r - oE, N_r - oN)
            if d <= (oR + margin):
                return True
        return False

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
        if not hits:
            return None
        return min(hits)

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

        if (
            self.lidar_num_rays <= 0
            or self.lidar_fov <= 0.0
            or len(circles_b) == 0
        ):
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
                if (
                    abs(xb_hit) <= self.sensor_range
                    and abs(yb_hit) <= self.sensor_range
                ):
                    hits.append((xb_hit, yb_hit))

        if len(hits) == 0:
            return np.zeros((0, 2), dtype=float)
        return np.asarray(hits, dtype=float)

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

        if getattr(self, "use_lidar_edges", False):
            hits = self.lidar_scan_hits_body(self.player_pos)
            return self._obs_grid_from_lidar_hits(hits)

        cell_world = (2.0 * S) / L
        x_tops = S - np.arange(0, L) * cell_world
        x_bottoms = x_tops - cell_world
        y_lefts = -S + np.arange(0, L) * cell_world
        y_rights = y_lefts + cell_world

        for (oE, oN, oR) in self.obstacles:
            xb_o, yb_o = self.world_to_body((oE, oN), agent_ENpsi=self.player_pos)
            if (xb_o < -S - oR) or (xb_o > S + oR) or (yb_o < -S - oR) or (yb_o > S + oR):
                continue
            for r in range(L):
                xmin = x_bottoms[r]
                xmax = x_tops[r]
                if xb_o < (xmin - oR) or xb_o > (xmax + oR):
                    continue
                for c in range(L):
                    ymin = y_lefts[c]
                    ymax = y_rights[c]
                    if yb_o < (ymin - oR) or yb_o > (ymax + oR):
                        continue
                    if self._rect_circle_intersects(
                            xmin, xmax, ymin, ymax, xb_o, yb_o, oR
                    ):
                        obs_grid[r, c] = 1.0

        return obs_grid

    def world_to_body(self, point_EN, agent_ENpsi=None):
        if agent_ENpsi is None:
            agent_ENpsi = self.player_pos
        E_a, N_a, psi = (
            float(agent_ENpsi[0]),
            float(agent_ENpsi[1]),
            float(agent_ENpsi[2]),
        )
        dE = float(point_EN[0]) - E_a
        dN = float(point_EN[1]) - N_a
        x_b = dE * math.cos(psi) + dN * math.sin(psi)
        y_b = dE * math.sin(psi) - dN * math.cos(psi)
        return x_b, y_b

    def body_to_world(self, xb, yb, agent_ENpsi=None):
        if agent_ENpsi is None:
            agent_ENpsi = self.player_pos
        E0, N0, psi = (
            float(agent_ENpsi[0]),
            float(agent_ENpsi[1]),
            float(agent_ENpsi[2]),
        )
        dE = xb * math.cos(psi) + yb * math.sin(psi)
        dN = xb * math.sin(psi) - yb * math.cos(psi)
        return E0 + dE, N0 + dN

    def _build_flyby_waypoints(self, E, N, R):
        E = np.asarray(E, dtype=float)
        N = np.asarray(N, dtype=float)
        n = len(E)
        eps = 1e-9

        if n == 0:
            self.E_fly, self.N_fly, self.psi_fly = [], [], []
            return np.array([]), np.array([]), np.array([])
        if n == 1:
            self.E_fly, self.N_fly, self.psi_fly = [E[0]], [N[0]], [0.0]
            return np.array([E[0]]), np.array([N[0]]), np.array([0.0])

        Eo, No, Psio = [], [], []

        for k in range(1, n - 1):
            dE_prev = E[k] - E[k - 1]
            dN_prev = N[k] - N[k - 1]
            L_prev = math.hypot(dE_prev, dN_prev)

            dE_next = E[k + 1] - E[k]
            dN_next = N[k + 1] - N[k]
            L_next = math.hypot(dE_next, dN_next)

            if L_prev < eps or L_next < eps:
                continue

            u_prev = np.array([dE_prev, dN_prev], dtype=float) / L_prev
            u_next = np.array([dE_next, dN_next], dtype=float) / L_next

            dot = float(np.dot(u_prev, u_next))
            dot = max(-1.0, min(1.0, dot))
            theta = math.acos(dot)

            interior_angle = math.pi - theta
            limit_angle = math.radians(30)

            if interior_angle <= limit_angle:
                Px_post = E[k]
                Py_post = N[k]
                bisector_x = u_prev[0] + u_next[0]
                bisector_y = u_prev[1] + u_next[1]
                if math.hypot(bisector_x, bisector_y) < eps:
                    psi_next = math.atan2(dN_next, dE_next)
                else:
                    psi_next = math.atan2(bisector_y, bisector_x)
                psi_next = self.normalize_angle(psi_next)
            else:
                if theta < 1e-6 or abs(math.pi - theta) < 1e-6:
                    continue
                t = R * math.tan(theta / 2.0)
                t_max_prev = 0.5 * L_prev
                t_max_next = 0.5 * L_next
                t_eff = max(0.0, min(t, t_max_prev, t_max_next))
                Px_post = E[k] + t_eff * u_next[0]
                Py_post = N[k] + t_eff * u_next[1]
                psi_next = self.normalize_angle(math.atan2(dN_next, dE_next))

            Eo.append(Px_post)
            No.append(Py_post)
            Psio.append(psi_next)

        Eo.append(E[-1])
        No.append(N[-1])
        dEl = E[-1] - E[-2]
        dNl = N[-1] - N[-2]
        if abs(dEl) > eps or abs(dNl) > eps:
            psi_last = self.normalize_angle(math.atan2(dNl, dEl))
        else:
            psi_last = 0.0
        Psio.append(psi_last)

        self.E_fly, self.N_fly, self.psi_fly = Eo, No, Psio
        return np.asarray(Eo), np.asarray(No), np.asarray(Psio)

    def _build_flyby_waypoints_from_raw(self):
        if self.raw_waypoints is None or len(self.raw_waypoints) == 0:
            return None
        E = [float(w[0]) for w in self.raw_waypoints]
        N = [float(w[1]) for w in self.raw_waypoints]
        R = getattr(self, "R", None)
        if R is None or R <= 0.0:
            return None
        Eo, No, Psio = self._build_flyby_waypoints(E, N, R)
        flyby_list = []
        for ex, ny, ps in zip(Eo, No, Psio):
            flyby_list.append([float(ex), float(ny), float(ps)])
        return flyby_list

    def _build_waypoint_states(self):
        self.waypoint_states = []
        base_waypoints = self.raw_waypoints if hasattr(self, "raw_waypoints") else self.waypoints
        if base_waypoints is None or len(base_waypoints) == 0:
            return

        if getattr(self, "fly_by", False):
            wp_list = self._build_flyby_waypoints_from_raw()
            if wp_list is None:
                wp_list = base_waypoints
        else:
            wp_list = base_waypoints

        self.waypoints = wp_list
        prev_state = getattr(self, "initial_player_pos", None)
        if prev_state is None:
            prev_E, prev_N, prev_psi = 0.0, 0.0, 0.0
        else:
            prev_E, prev_N, prev_psi = float(prev_state[0]), float(prev_state[1]), float(prev_state[2])

        for wp in wp_list:
            w = np.array(wp, dtype=float)
            if w.size == 2:
                E, N = float(w[0]), float(w[1])
                dE = E - prev_E
                dN = N - prev_N
                if abs(dE) < 1e-6 and abs(dN) < 1e-6:
                    psi = prev_psi
                else:
                    psi = math.atan2(dN, dE)
                state = np.array([E, N, psi], dtype=np.float32)
            elif w.size >= 3:
                state = np.array(w[:3], dtype=np.float32)
                E, N, psi = float(state[0]), float(state[1]), float(state[2])
            else:
                raise ValueError(f"waypoint는 길이 2 또는 3이어야 합니다. got {w}")
            self.waypoint_states.append(state)
            prev_E, prev_N, prev_psi = E, N, psi

    def _build_global_dubins_path(self):
        self._build_waypoint_states()

        if hasattr(self, "waypoint_states") and len(self.waypoint_states) > 0:
            seq = [self.initial_player_pos] + self.waypoint_states
            self.final_goal_pos = self.waypoint_states[-1].copy()
            self.current_wp_index = 0
            self.current_goal_pos = self.waypoint_states[0].copy()
            self.goal_pos = self.final_goal_pos.copy()
        else:
            if not hasattr(self, "goal_pos") or self.goal_pos is None:
                self.global_path = np.zeros((0, 2), dtype=np.float32)
                self.final_goal_pos = None
                self.current_goal_pos = None
                return
            seq = [self.initial_player_pos, self.goal_pos]
            self.final_goal_pos = self.goal_pos.copy()
            self.current_wp_index = 0
            self.current_goal_pos = self.goal_pos.copy()

        path_x_all = []
        path_y_all = []

        try:
            for i in range(len(seq) - 1):
                s = seq[i]
                g = seq[i + 1]
                px, py, pyaw, modes, lengths = self.path_planner.plan(
                    float(s[0]),
                    float(s[1]),
                    float(s[2]),
                    float(g[0]),
                    float(g[1]),
                    float(g[2]),
                )
                if i == 0:
                    path_x_all.extend(px)
                    path_y_all.extend(py)
                else:
                    path_x_all.extend(px[1:])
                    path_y_all.extend(py[1:])

            if len(path_x_all) == 0:
                self.global_path = np.zeros((0, 2), dtype=np.float32)
            else:
                self.global_path = np.stack(
                    [np.asarray(path_x_all), np.asarray(path_y_all)], axis=1
                ).astype(np.float32)
        except Exception as e:
            print("[Dubins] path planning failed:", e)
            self.global_path = np.zeros((0, 2), dtype=np.float32)


    def compute_reference_point_on_global(self, L=None):
        if L is None:
            L = self.reference_L

        if (
                not hasattr(self, "global_path")
                or self.global_path is None
                or self.global_path.shape[0] < 2
        ):
            self.closest_path_idx = None
            self.closest_path_point = None
            return None

        path = np.asarray(self.global_path, dtype=float)
        path_EN = path[:, :2]
        agent = np.array(self.player_pos[:2], dtype=float)

        # 에피소드 시작 시 현재 지점(Index) 초기화
        if getattr(self, "steps", 0) == 0 or not hasattr(self, "last_closest_idx"):
            self.last_closest_idx = 0

        # --- [핵심 수정: 시야(탐색 범위) 대폭 축소] ---
        # 기존 +1500은 맵 스케일에 따라 오른쪽 원 전체를 뛰어넘어버릴 수 있습니다.
        # 뒤로는 조금(-30), 앞으로는 눈앞(+250)만 보도록 시야를 좁힙니다.
        search_start = max(0, self.last_closest_idx - 30)
        search_end = min(len(path_EN), self.last_closest_idx + 250)
        
        diffs = path_EN[search_start:search_end] - agent
        dists = np.hypot(diffs[:, 0], diffs[:, 1])

        # 거리가 가장 짧은 점들 중 첫 번째(인덱스가 가장 작은) 점 선택
        local_min_idx = int(np.argmin(dists))
        cur_idx = search_start + local_min_idx
        
        # 다음 스텝을 위해 현재 지나고 있는 순서를 업데이트
        self.closest_path_idx = cur_idx
        self.last_closest_idx = cur_idx  
        self.closest_path_point = path_EN[cur_idx].copy()
        # ------------------------------------

        if L <= 0.0:
            return self.closest_path_point.copy()

        # 목표점(Reference Point)을 찾기 위해 Look-ahead 거리(L)만큼 앞을 내다봄
        remaining_len = 0.0
        for i in range(cur_idx, len(path_EN) - 1):
            p0 = path_EN[i]
            p1 = path_EN[i + 1]
            remaining_len += float(math.hypot(p1[0] - p0[0], p1[1] - p0[1]))

        if remaining_len <= L:
            return path_EN[-1].copy()

        acc = 0.0
        for i in range(cur_idx, len(path_EN) - 1):
            p0 = path_EN[i]
            p1 = path_EN[i + 1]
            seg_len = float(math.hypot(p1[0] - p0[0], p1[1] - p0[1]))
            acc += seg_len
            if acc >= L:
                return p1.copy()

        return path_EN[-1].copy()

#원본
    # def compute_reference_point_on_global(self, L=None):
    #     if L is None:
    #         L = self.reference_L

    #     if (
    #             not hasattr(self, "global_path")
    #             or self.global_path is None
    #             or self.global_path.shape[0] < 2
    #     ):
    #         self.closest_path_idx = None
    #         self.closest_path_point = None
    #         return None

    #     path = np.asarray(self.global_path, dtype=float)
    #     path_EN = path[:, :2]
    #     agent = np.array(self.player_pos[:2], dtype=float)

    #     # --- [수정된 부분: 이미 지난 지점(역주행) 및 교차점 오류 방지] ---
    #     # 에피소드 시작 시(스텝 0) 이전 인덱스를 0으로 초기화합니다.
    #     if getattr(self, "steps", 0) == 0 or not hasattr(self, "last_closest_idx"):
    #         self.last_closest_idx = 0

    #     # 전체 경로를 탐색하지 않고, '최근에 가장 가까웠던 점'을 기준으로
    #     # 약간 뒤(max -20)부터 앞(max +800)까지만 제한적으로 탐색합니다.
    #     search_start = max(0, self.last_closest_idx - 20)
    #     search_end = min(len(path_EN), self.last_closest_idx + 800)
        
    #     diffs = path_EN[search_start:search_end] - agent
    #     dists = np.hypot(diffs[:, 0], diffs[:, 1])

    #     # 지역 탐색(Local window) 결과 인덱스를 전체 인덱스로 변환
    #     local_min_idx = int(np.argmin(dists))
    #     cur_idx = search_start + local_min_idx
        
    #     self.closest_path_idx = cur_idx
    #     self.last_closest_idx = cur_idx  # 다음 스텝을 위해 현재 위치 기억
    #     self.closest_path_point = path_EN[cur_idx].copy()
    #     # -------------------------------------------------------------------

    #     if L <= 0.0:
    #         return self.closest_path_point.copy()

    #     remaining_len = 0.0
    #     for i in range(cur_idx, len(path_EN) - 1):
    #         p0 = path_EN[i]
    #         p1 = path_EN[i + 1]
    #         remaining_len += float(math.hypot(p1[0] - p0[0], p1[1] - p0[1]))

    #     if remaining_len <= L:
    #         return path_EN[-1].copy()

    #     acc = 0.0
    #     for i in range(cur_idx, len(path_EN) - 1):
    #         p0 = path_EN[i]
    #         p1 = path_EN[i + 1]
    #         seg_len = float(math.hypot(p1[0] - p0[0], p1[1] - p0[1]))
    #         acc += seg_len
    #         if acc >= L:
    #             return p1.copy()

    #     return path_EN[-1].copy()

    def update_reference_point(self):
        self.reference_point = self.compute_reference_point_on_global(
            L=self.reference_L
        )

    def path_tracking(self):
        V = float(self.v)
        L = float(self.reference_L)

        if V <= 1e-6 or L <= 1e-6 or self.reference_point is None:
            self.eta = 0.0
            self.a_cmd = 0.0
            self.w_cmd = 0.0
            return 0.0

        pos = np.array(self.player_pos[:2], dtype=float)
        ref = np.array(self.reference_point[:2], dtype=float)
        dEN = ref - pos
        R = float(np.linalg.norm(dEN))
        if R < 1e-6:
            self.eta = 0.0
            self.a_cmd = 0.0
            self.w_cmd = 0.0
            return 0.0

        chi_ref = math.atan2(dEN[1], dEN[0])
        psi = float(self.player_pos[2])
        eta = self.normalize_angle(chi_ref - psi)
        a_cmd = 2.0 * (V * V / L) * math.sin(eta)
        w_cmd = a_cmd / V

        self.eta = eta
        self.a_cmd = a_cmd
        self.w_cmd = w_cmd

    def _update_current_goal_waypoint(self):
        if not self.waypoint_states:
            return

        idx = getattr(self, "current_wp_index", 0)
        idx = max(0, min(idx, len(self.waypoint_states) - 1))
        cur_wp = self.waypoint_states[idx]

        dist = float(np.linalg.norm(self.player_pos[:2] - cur_wp[:2]))
        heading_ok = math.cos(
            self.player_pos[2] - cur_wp[2]
        ) >= math.cos(math.radians(45))

        if (dist < self.wp_reach_radius) and heading_ok:
            if idx < len(self.waypoint_states) - 1:
                idx += 1

        self.current_wp_index = idx
        self.current_goal_pos = self.waypoint_states[idx].copy()

    def compute_apf_lateral_accel(self):
        dists, rc_indices, centers_body = self.obstacle_cell_distances_at(self.player_pos)
        F_rep_sum = np.zeros(2, dtype=float)
        valid_obs_count = 0

        if dists.size > 0:
            for dist, center in zip(dists, centers_body):
                dist = float(dist)
                if dist < 1e-6 or dist >= self.apf_d0:
                    continue

                xb, yb = float(center[0]), float(center[1])
                r_vec = np.array([xb, yb], dtype=float)
                r_hat = r_vec / dist
                dir_away = -r_hat
                mag = self.apf_K_rep * self.l_rep * dist * math.exp(-self.l_rep * dist ** 2)
                F_vec = mag * dir_away
                F_rep_sum += F_vec
                valid_obs_count += 1

        if valid_obs_count > 0:
            F_rep = F_rep_sum / float(valid_obs_count)
        else:
            F_rep = F_rep_sum

        F_att = np.zeros(2, dtype=float)
        target = None
        if self.current_goal_pos is not None:
            target = self.current_goal_pos
        elif self.final_goal_pos is not None:
            target = self.final_goal_pos

        if target is not None:
            E_g, N_g = float(target[0]), float(target[1])
            xb_g, yb_g = self.world_to_body((E_g, N_g), agent_ENpsi=self.player_pos)
            r_vec_g = np.array([xb_g, yb_g], dtype=float)
            F_att = self.apf_K_att * r_vec_g

        F_total = F_rep + F_att
        a_apf_lat = F_total[1]
        return a_apf_lat

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
            for r in range(L):
                xmin = x_bottoms[r]
                xmax = x_tops[r]
                if xb_o < (xmin - oR) or xb_o > (xmax + oR):
                    continue
                for c in range(L):
                    ymin = y_lefts[c]
                    ymax = y_rights[c]
                    if yb_o < (ymin - oR) or yb_o > (ymax + oR):
                        continue
                    if self._rect_circle_intersects(
                        xmin, xmax, ymin, ymax, xb_o, yb_o, oR
                    ):
                        obs_grid[r, c] = 1.0

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
        dists, rc_list, centers = self.obstacle_cell_distances_at(new_pos)
        if dists.size == 0:
            return {"clear": 0.0, "hard": 0.0}

        S = self.sensor_range
        L = self.local_map_size
        cell_world = (2.0 * S) / max(L, 1)
        eps = 1e-6

        d_sel = dists
        safety_zone = float(self.safety_zone) if hasattr(self, "safety_zone") else 0.5 * cell_world
        d_safe = float(self.agent_radius + safety_zone)
        d_min = float(np.min(d_sel))

        clear = float(np.clip(1.0 - d_min / max(d_safe, eps), 0.0, 1.0))
        d_hard = self.hard_zone
        hard = 1.0 if d_min < d_hard else 0.0
        return {"clear": clear, "hard": hard}

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

    def world_to_screen(self, e, n, target_rect, cell_size=None):
        if cell_size is None:
            cell_size = target_rect.width / self.global_map_size
        sx = target_rect.x + e * cell_size + cell_size / 2.0
        sy = target_rect.y + target_rect.height - (
            n * cell_size + cell_size / 2.0
        )
        return int(sx), int(sy)

    def draw_arrow(
        self, start_xy, end_xy, color, width=4, head_len=10, head_angle=math.pi / 6
    ):
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
        pygame.draw.polygon(
            self.screen, color, [(ex, ey), (left_x, left_y), (right_x, right_y)]
        )

    def draw_x(self, center_xy, size, color, width=3):
        cx, cy = center_xy
        cx = int(cx)
        cy = int(cy)
        half = size / 2.0
        pygame.draw.line(
            self.screen,
            color,
            (cx - half, cy - half),
            (cx + half, cy + half),
            width,
        )
        pygame.draw.line(
            self.screen,
            color,
            (cx - half, cy + half),
            (cx + half, cy - half),
            width,
        )

    def render(self, fps=30):
        if not self.render_option:
            return

        self.screen.fill(self.COLOR_BG)
        pygame.draw.rect(self.screen, self.COLOR_GLOBAL, self.rect_global)
        pygame.draw.rect(self.screen, self.COLOR_INFO, self.rect_info)
        pygame.draw.rect(self.screen, self.COLOR_LOCAL, self.rect_local)

        cell_size = self.rect_global.width / self.global_map_size
        e, n = float(self.player_pos[0]), float(self.player_pos[1])

        self.draw_obstacles_on_global(cell_size)

        if self.fly_by and getattr(self, "raw_waypoints", None) is not None:
            for idx, wp in enumerate(self.raw_waypoints):
                Ew = float(wp[0])
                Nw = float(wp[1])
                wx, wy = self.world_to_screen(
                    Ew, Nw, self.rect_global, cell_size
                )
                pygame.draw.circle(
                    self.screen,
                    self.COLOR_WAYPOINT,
                    (wx, wy),
                    max(3, int(cell_size * 0.6)),
                )
                label = self.small_font.render(f"W{idx+1}", True, (0, 0, 0))
                self.screen.blit(label, (wx + 4, wy + 4))

        if hasattr(self, "waypoint_states") and len(self.waypoint_states) > 0:
            for idx, wp in enumerate(self.waypoint_states):
                wx, wy = self.world_to_screen(
                    float(wp[0]), float(wp[1]), self.rect_global, cell_size
                )
                color = self.COLOR_WAYPOINT_FLYBY if self.fly_by else self.COLOR_WAYPOINT
                pygame.draw.circle(
                    self.screen,
                    color,
                    (wx, wy),
                    max(3, int(cell_size * 0.6)),
                )
                label_text = f"F{idx+1}" if self.fly_by else f"W{idx+1}"
                label = self.small_font.render(label_text, True, (0, 0, 0))
                self.screen.blit(label, (wx + 4, wy + 4))

        if self.final_goal_pos is not None:
            gx, gy = self.world_to_screen(
                float(self.final_goal_pos[0]),
                float(self.final_goal_pos[1]),
                self.rect_global,
                cell_size,
            )
            pygame.draw.circle(self.screen, self.COLOR_GOAL, (gx, gy), int(cell_size / 3))
            dE_g, dN_g = self.dir_from_heading(self.final_goal_pos[2])
            self.draw_arrow(
                (gx, gy),
                (
                    int(gx + 1.5 * cell_size * dE_g),
                    int(gy - 1.5 * cell_size * dN_g),
                ),
                self.COLOR_GOAL_ARROW,
                width=4,
                head_len=10,
            )

        if (
            hasattr(self, "global_path")
            and self.global_path is not None
            and isinstance(self.global_path, np.ndarray)
            and self.global_path.shape[0] >= 2
        ):
            dubins_pts = [
                self.world_to_screen(
                    float(p[0]), float(p[1]), self.rect_global, cell_size
                )
                for p in self.global_path
            ]
            pygame.draw.lines(
                self.screen, self.COLOR_DUBINS, False, dubins_pts, 4
            )

        px, py = self.world_to_screen(e, n, self.rect_global, cell_size)
        pygame.draw.circle(
            self.screen,
            self.COLOR_AGENT,
            (px, py),
            max(2, int(self.agent_radius * cell_size)),
        )
        dE_a, dN_a = self.dir_from_heading(self.player_pos[2])
        self.draw_arrow(
            (px, py),
            (
                int(px + 1.5 * cell_size * dE_a),
                int(py - 1.5 * cell_size * dN_a),
            ),
            self.COLOR_ARROW,
            width=4,
            head_len=10,
        )

        if len(self.visited_path) >= 2:
            pts = [
                self.world_to_screen(
                    p[0], p[1], self.rect_global, cell_size
                )
                for p in self.visited_path
            ]
            pygame.draw.lines(self.screen, self.COLOR_PATH, False, pts, 4)

        if self.reference_point is not None:
            rx, ry = self.world_to_screen(
                self.reference_point[0],
                self.reference_point[1],
                self.rect_global,
                cell_size,
            )
            self.draw_x((rx, ry), size=10, color=self.COLOR_REF_POINT, width=3)

        self.draw_local_range_on_global(cell_size)

        cell_world = (2 * self.sensor_range) / self.local_map_size
        info_x = self.rect_info.x + 10
        self.screen.blit(
            self.font.render(
                f"Return: {float(self.cumulative_reward):.2f}", True, (0, 0, 0)
            ),
            (info_x, 10),
        )
        self.screen.blit(
            self.font.render(f"Steps: {self.steps}", True, (0, 0, 0)),
            (info_x, 50),
        )
        self.screen.blit(
            self.small_font.render(f"v: {self.v:.2f}", True, (0, 0, 0)),
            (info_x, 90),
        )
        self.screen.blit(
            self.small_font.render(f"w: {self.w:.3f}", True, (0, 0, 0)),
            (info_x, 110),
        )
        self.screen.blit(
            self.small_font.render(
                f"a_cmd: {self.a_cmd:.2f}", True, (0, 0, 0)
            ),
            (info_x, 130),
        )
        self.screen.blit(
            self.small_font.render(
                f"Sensor R: {self.sensor_range:.2f}", True, (0, 0, 0)
            ),
            (info_x, 150),
        )
        self.screen.blit(
            self.small_font.render(
                f"Cell = {cell_world:.2f} units", True, (0, 0, 0)
            ),
            (info_x, 175),
        )

        self.render_local_body_grid()

        if self.terminated:
            finished_text = self.font.render("FINISHED", True, (0, 0, 0))
            self.screen.blit(
                finished_text,
                (
                    self.rect_global.x + self.rect_global.width // 2 - 70,
                    self.rect_global.y + self.rect_global.height // 2 - 20,
                ),
            )

        pygame.display.flip()
        self.clock.tick(fps)

    def render_local_body_grid(self):
        r = self.rect_local
        pygame.draw.rect(self.screen, (0, 0, 0), r, width=3)
        self.screen.fill(self.COLOR_LOCAL, r)

        L = self.local_map_size
        cell_px = r.width / L
        obs_grid = self.obs_grid

        for rr in range(L):
            for cc in range(L):
                if obs_grid[rr, cc] > 0.5:
                    x0 = r.x + cc * cell_px
                    y0 = r.y + rr * cell_px
                    rect = pygame.Rect(x0, y0, cell_px, cell_px)
                    pygame.draw.rect(self.screen, self.COLOR_OBS_CELL, rect)

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

        cx = r.x + r.width / 2.0
        cy = r.y + r.height / 2.0
        pygame.draw.circle(self.screen, self.COLOR_AGENT, (int(cx), int(cy)), 8)
        axis_len_px = 0.35 * L * cell_px
        self.draw_arrow(
            (cx, cy),
            (cx, cy - axis_len_px),
            self.COLOR_ARROW,
            width=3,
            head_len=10,
        )
        self.draw_arrow(
            (cx, cy),
            (cx - axis_len_px, cy),
            self.COLOR_ARROW,
            width=3,
            head_len=10,
        )

        S = float(self.sensor_range)

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
                    self.screen, self.COLOR_DUBINS, False, pts_body_px, 4
                )

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
                self.draw_x((sx, sy), size=10, color=self.COLOR_REF_POINT, width=3)

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
                hard_px = max(1, int(hard_r * cell_size))
                pygame.draw.circle(
                    self.screen, self.COLOR_HARD_ZONE, (sx, sy), hard_px, 3
                )

    def close(self):
        if self.render_option:
            pygame.quit()

    def simulation(self, model, sac_model=None, max_steps=None, render=True, fps=30):
        if max_steps is None:
            max_steps = self.max_steps

        obs, info = self.reset()
        traj = [self.player_pos.copy()]
        rewards = []

        done = False

        while not done and self.steps < max_steps:
            if model == "sac":
                if sac_model is None:
                    raise ValueError("model='sac'일 때는 sac_model 인자로 SAC 모델을 넘겨줘야 합니다.")
                action, _states = sac_model.predict(obs, deterministic=True)

            elif model == "apf":
                self.update_reference_point()
                self.path_tracking()
                self._update_current_goal_waypoint()

                a_apf = 0.0
                V = max(self.v, 1e-6)
                if self.is_reference_point_in_obstacle_zone():
                    a_apf = self.compute_apf_lateral_accel()

                a_total = self.a_cmd - a_apf
                w_cmd = a_total / V

                raw = np.clip(w_cmd / self.max_w, -1.0, 1.0)
                action = np.array([raw], dtype=np.float32)

            else:
                raise ValueError("model 인자는 'sac' 또는 'apf'만 가능합니다.")

            obs, r, terminated, truncated, info = self.step(action)

            traj.append(self.player_pos.copy())
            rewards.append(r)

            if render and self.render_option:
                self.render(fps=fps)

            done = terminated or truncated

        return np.array(traj), np.array(rewards)

    def set_player_pos(self, player_pos):
        self.player_pos = np.array(player_pos[:3], dtype=np.float32)

    def get_player_pos(self):
        return self.player_pos.copy()

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



# =========================
# 실행부 (__name__ == "__main__")
# =========================
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import numpy as np
    from math import atan2, radians
    
    # ---------------------------------------------------------
    # [시나리오 정의] 0, 1, 2, 3 모든 시나리오 포함
    # ---------------------------------------------------------

    # ---------------------------------------------------------
    # [시나리오 정의] 3개 맵
    # spec1: 기존 8자
    # spec2: 운동장 타원
    # spec3: rolling circle
    # ---------------------------------------------------------

    def rotate_points(points, center, deg):
        th = np.deg2rad(deg)
        c, s = np.cos(th), np.sin(th)
        rot = np.array([[c, -s], [s, c]])
        pts = np.asarray(points, dtype=float)
        ctr = np.asarray(center, dtype=float)
        pts_rot = (pts - ctr) @ rot.T + ctr
        return pts_rot.tolist()


    def make_stadium_track(center=(750.0, 750.0),
                           straight_half=250.0,
                           radius=170.0,
                           n_straight=18,
                           n_arc=24,
                           rotate_deg=8.0):
        """
        운동장(타원형 트랙) 형태
        """
        cx, cy = center
        pts = []

        x_left = cx - straight_half
        x_right = cx + straight_half
        y_bottom = cy - radius
        y_top = cy + radius

        # bottom straight
        for x in np.linspace(x_left, x_right, n_straight, endpoint=False):
            pts.append([x, y_bottom])

        # right arc
        c_r = np.array([cx + straight_half, cy], dtype=float)
        for th in np.linspace(-np.pi / 2, np.pi / 2, n_arc, endpoint=False):
            pts.append([
                c_r[0] + radius * np.cos(th),
                c_r[1] + radius * np.sin(th)
            ])

        # top straight
        for x in np.linspace(x_right, x_left, n_straight, endpoint=False):
            pts.append([x, y_top])

        # left arc
        c_l = np.array([cx - straight_half, cy], dtype=float)
        for th in np.linspace(np.pi / 2, 3 * np.pi / 2, n_arc + 1):
            pts.append([
                c_l[0] + radius * np.cos(th),
                c_l[1] + radius * np.sin(th)
            ])

        if rotate_deg != 0.0:
            pts = rotate_points(pts, center, rotate_deg)

        return [[round(p[0], 1), round(p[1], 1)] for p in pts]

    def make_rolling_circle_track(center=(900.0, 700.0),
                                      radius=220.0,
                                      cross_angle_deg=145.0,
                                      line_heading_deg=43.0,
                                      line_len=260.0,
                                      n_line=18,
                                      n_circle=160):
            """
            대칭형 rolling circle:
            - 좌하단에서 직선으로 진입
            - 원 위 한 점에서 진입
            - 정확한 원을 한 바퀴 돌고
            - 같은 점으로 돌아오며 직선과 한 점에서 교차
            """

        cx, cy = center

        # 원 위의 교차점(진입/복귀 지점)
        th_cross = np.deg2rad(cross_angle_deg)
        p_cross = np.array([
            cx + radius * np.cos(th_cross),
            cy + radius * np.sin(th_cross)
        ], dtype=float)

        # 좌하단 -> 우상향 직선 방향
        th_line = np.deg2rad(line_heading_deg)
        line_dir = np.array([
                np.cos(th_line),
                np.sin(th_line)
        ], dtype=float)

        # 직선 시작점
        p_start = p_cross - line_len * line_dir

        pts = []

        # 1) 진입 직선
        for t in np.linspace(0.0, 1.0, n_line, endpoint=False):
            p = (1.0 - t) * p_start + t * p_cross
            pts.append([p[0], p[1]])

        # 2) 원 한 바퀴 (clockwise)
        thetas = np.linspace(th_cross, th_cross - 2.0 * np.pi, n_circle + 1)
        for th in thetas:
            pts.append([
                cx + radius * np.cos(th),
                cy + radius * np.sin(th)
            ])

        return [[round(p[0], 1), round(p[1], 1)] for p in pts]

        def quad_bezier(p0, p1, p2, n=12):
            ts = np.linspace(0.0, 1.0, n, endpoint=False)
            pts = []
            for t in ts:
                pt = (1 - t) ** 2 * p0 + 2 * (1 - t) * t * p1 + t ** 2 * p2
                pts.append([pt[0], pt[1]])
            return pts

        cx, cy = center
        pts = []

        # -------------------------------------------------
        # 1) 교차가 보이도록 기준 각도/방향 재설계
        # -------------------------------------------------
        # 실제 cross가 보이도록 circle 진입점은 조금 더 위쪽,
        # approach line은 원의 접선이 아니라 '통과'하는 방향으로 둔다.
        cross_angle_deg = entry_angle_deg - 20.0  # 기본 125 deg 부근
        circle_start_deg = entry_angle_deg + 8.0  # 기본 153 deg 부근
        approach_heading_deg = 43.0  # 좌하단 -> 우상향 직선
        loop_sweep_deg = max(sweep_deg + 22.0, 325.0)  # 거의 한 바퀴

        cross_angle = np.deg2rad(cross_angle_deg)
        circle_start = np.deg2rad(circle_start_deg)
        approach_heading = np.deg2rad(approach_heading_deg)

        approach_dir = np.array([
            np.cos(approach_heading),
            np.sin(approach_heading)
        ], dtype=float)

        # 원 상단 좌측의 교차 기준점
        cross_pt = np.array([
            cx + radius * np.cos(cross_angle),
            cy + radius * np.sin(cross_angle)
        ], dtype=float)

        # 직선 시작점: 좀 더 멀리 뒤에서 시작
        line_start = cross_pt - (line_len + 70.0) * approach_dir

        # 직선이 교차점 지나서 조금 더 나가게 해서 '교차 느낌' 강화
        hook_pt = cross_pt + 0.42 * radius * approach_dir

        # 실제 원호 시작점
        circle_start_pt = np.array([
            cx + radius * np.cos(circle_start),
            cy + radius * np.sin(circle_start)
        ], dtype=float)

        # -------------------------------------------------
        # 2) 진입 직선
        # -------------------------------------------------
        for t in np.linspace(0.0, 1.0, max(n_line, 8), endpoint=False):
            p = line_start + t * (hook_pt - line_start)
            pts.append([p[0], p[1]])

        # -------------------------------------------------
        # 3) 직선 -> 원호 연결용 곡선
        #    (위에서 한번 지나간 뒤 원에 다시 붙는 느낌)
        # -------------------------------------------------
        control_angle_deg = circle_start_deg + 20.0
        control_pt = np.array([
            cx + (radius * 1.55) * np.cos(np.deg2rad(control_angle_deg)),
            cy + (radius * 1.55) * np.sin(np.deg2rad(control_angle_deg))
        ], dtype=float)

        connector_pts = quad_bezier(
            hook_pt,
            control_pt,
            circle_start_pt,
            n=max(10, n_line + 4)
        )
        pts.extend(connector_pts)

        # -------------------------------------------------
        # 4) 메인 원호 (clockwise)
        # -------------------------------------------------
        th_end = circle_start - np.deg2rad(loop_sweep_deg)
        for th in np.linspace(circle_start, th_end, max(n_circle, 90)):
            pts.append([
                cx + radius * np.cos(th),
                cy + radius * np.sin(th)
            ])

        return [[round(p[0], 1), round(p[1], 1)] for p in pts]


    # ---------------------------------------------------------
    # Scenario 1: 기존 spec4/waypoints4 → 8자 맵
    # ---------------------------------------------------------
    spec1 = (
        [150, 750, radians(90)],
        [
            [450, 1050, 40],
            [1050, 450, 40]
        ],
    )

    waypoints1 = [
        [150, 750],
        [153, 789],
        [160, 828],
        [173, 865],
        [190, 900],
        [212, 933],
        [238, 962],
        [267, 988],
        [300, 1010],
        [335, 1027],
        [372, 1040],
        [411, 1047],
        [450, 1050],
        [489, 1047],
        [528, 1040],
        [565, 1027],
        [600, 1010],
        [633, 988],
        [662, 962],
        [688, 933],
        [710, 900],
        [727, 865],
        [740, 828],
        [747, 789],
        [750, 750],
        [753, 711],
        [760, 672],
        [773, 635],
        [790, 600],
        [812, 567],
        [838, 538],
        [867, 512],
        [900, 490],
        [935, 473],
        [972, 460],
        [1011, 453],
        [1050, 450],
        [1089, 453],
        [1128, 460],
        [1165, 473],
        [1200, 490],
        [1233, 512],
        [1262, 538],
        [1288, 567],
        [1310, 600],
        [1327, 635],
        [1340, 672],
        [1347, 711],
        [1350, 750],
        [1347, 789],
        [1340, 828],
        [1327, 865],
        [1310, 900],
        [1288, 933],
        [1262, 962],
        [1233, 988],
        [1200, 1010],
        [1165, 1027],
        [1128, 1040],
        [1089, 1047],
        [1050, 1050],
        [1011, 1047],
        [972, 1040],
        [935, 1027],
        [900, 1010],
        [867, 988],
        [838, 962],
        [812, 933],
        [790, 900],
        [773, 865],
        [760, 828],
        [753, 789],
        [750, 750],
        [747, 711],
        [740, 672],
        [727, 635],
        [710, 600],
        [688, 567],
        [662, 538],
        [633, 512],
        [600, 490],
        [565, 473],
        [528, 460],
        [489, 453],
        [450, 450],
        [411, 453],
        [372, 460],
        [335, 473],
        [300, 490],
        [267, 512],
        [238, 538],
        [212, 567],
        [190, 600],
        [173, 635],
        [160, 672],
        [153, 711],
        [150, 750],
    ]

    # ---------------------------------------------------------
    # Scenario 2: 운동장 타원
    # ---------------------------------------------------------
    stadium_pts = make_stadium_track(
        center=(750.0, 750.0),
        straight_half=250.0,
        radius=170.0,
        n_straight=18,
        n_arc=24,
        rotate_deg=8.0
    )

    stadium_heading = math.atan2(
        stadium_pts[1][1] - stadium_pts[0][1],
        stadium_pts[1][0] - stadium_pts[0][0]
    )

    spec2 = (
        [stadium_pts[0][0], stadium_pts[0][1], stadium_heading],
        [
            [1010, 980, 40],  # 상단 우측 근처
        ],
    )
    waypoints2 = stadium_pts[1:]

    # ---------------------------------------------------------
    # Scenario 3: Rolling Circle
    # ---------------------------------------------------------
    rolling_pts = make_rolling_circle_track(
        center=(900.0, 700.0),
        radius=220.0,
        cross_angle_deg=145.0,
        line_heading_deg=43.0,
        line_len=260.0,
        n_line=18,
        n_circle=160
    )

    rolling_heading = math.atan2(
        rolling_pts[1][1] - rolling_pts[0][1],
        rolling_pts[1][0] - rolling_pts[0][0]
    )

    spec3 = (
        [rolling_pts[0][0], rolling_pts[0][1], rolling_heading],
        [
            [1120, 620, 40],  # 우측 하단 회전부 근처
        ],
    )
    waypoints3 = rolling_pts[1:]






    # ---------------------------------------------------------
    # 1. 환경 설정 (사용할 시나리오 선택)
    # ---------------------------------------------------------
    # 현재는 spec3, waypoints3 (논문용)을 사용하도록 설정됨.
    env = SimpleMazeGrid(
        global_map_size=1500,        
        local_map_size=100,
        v=15.0,                      
        w=[-0.46, 0.46],            
        dt=0.1,
        render_option=False,  
        spec=spec3,            # <--- Scenario 1, 2, 3 중 선택 (spec, spec1...)
        waypoints=waypoints3,  # <--- Scenario 1, 2, 3 중 선택 (waypoints, waypoints1...)
        sensor_range=100.0,          
        reference_L=50.0,  # 전방주시거리, 앞을 얼마나 멀리 내다보는지       
        fly_by=False,
    )

    # ---------------------------------------------------------
    # 2. 비교 시각화 함수 정의
    # ---------------------------------------------------------
    def visualize_result(env, traj_record):
        FS_TITLE = 24
        FS_LABEL = 24
        FS_TICK = 20
        FS_LEGEND = 20

        offset = env.global_map_size / 2.0
        traj_x = traj_record[:, 0] - offset
        traj_y = traj_record[:, 1] - offset

        hard_zone_margin = getattr(env, "hard_zone", 2.0)

        wps = env.raw_waypoints if env.raw_waypoints is not None else env.waypoints
        if wps is not None:
            wps_arr = np.array(wps)
            wps_x = wps_arr[:, 0] - offset
            wps_y = wps_arr[:, 1] - offset
        else:
            wps_arr = None
            wps_x = None
            wps_y = None

        # -------------------------
        # Figure 1: Trajectory Map
        # -------------------------
        plt.figure(figsize=(10, 10))
        ax = plt.gca()

        for i, (oE, oN, oR) in enumerate(env.obstacles):
            lbl_obs = 'Obstacle' if i == 0 else None
            circle = patches.Circle(
                (oE - offset, oN - offset), oR,
                edgecolor='black', facecolor='gray', alpha=0.5, label=lbl_obs
            )
            ax.add_patch(circle)

            lbl_hard = 'Hard Limit' if i == 0 else None
            hard_circle = patches.Circle(
                (oE - offset, oN - offset), oR + hard_zone_margin,
                edgecolor='red', facecolor='none', linewidth=2, linestyle='-', label=lbl_hard
            )
            ax.add_patch(hard_circle)

        if env.global_path is not None and len(env.global_path) > 0:
            plt.plot(
                env.global_path[:, 0] - offset,
                env.global_path[:, 1] - offset,
                color='cyan',
                linewidth=3,
                linestyle='--',
                label="Global Path"
            )

        if wps is not None:
            plt.plot(wps_x, wps_y, 'ro', markersize=10, label="Waypoints")
            for i in range(len(wps_arr)):
                if i == len(wps_arr) - 1:
                    continue
                plt.text(
                    wps_x[i] + 1.5,
                    wps_y[i] + 1.5,
                    f"W{i}",
                    fontsize=FS_TICK,
                    color='red'
                )

        plt.plot(traj_x, traj_y, color='blue', linewidth=3, label="Agent Trajectory")

        plt.xlabel("X Position [m]", fontsize=FS_LABEL)
        plt.ylabel("Y Position [m]", fontsize=FS_LABEL)
        plt.xticks(fontsize=FS_TICK)
        plt.yticks(fontsize=FS_TICK)
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.axis('equal')
        plt.legend(fontsize=FS_LEGEND, loc='upper right')
        plt.tight_layout()
        plt.show()

        if not env.time_table:
            print("[WARN] No time data found.")
            return

        if not hasattr(env, 'dist_error_table') or not env.dist_error_table:
            print("[WARN] No dist_error_table found.")
            err_data = np.zeros(len(env.time_table))
        else:
            err_data = np.array(env.dist_error_table)

        t_data = np.array(env.time_table)
        w_data = np.array(env.w_table)

        min_len = min(len(t_data), len(w_data), len(err_data))
        t_data = t_data[:min_len]
        w_data = w_data[:min_len]
        err_data = err_data[:min_len]

        # -------------------------
        # Figure 2: Yaw Rate
        # -------------------------
        plt.figure(figsize=(10, 6))
        plt.plot(t_data, w_data, linewidth=2.5, color='tab:blue', label="Yaw Rate (w)")
        plt.ylim(env.min_w, env.max_w)
        plt.xlabel("Time [s]", fontsize=FS_LABEL)
        plt.ylabel("w [rad/s]", fontsize=FS_LABEL)
        plt.xticks(fontsize=FS_TICK)
        plt.yticks(fontsize=FS_TICK)
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.legend(fontsize=FS_LEGEND, loc='upper right')
        plt.tight_layout()
        plt.show()

        # -------------------------
        # Figure 3: Cross Track Error
        # -------------------------
        plt.figure(figsize=(10, 6))
        plt.plot(t_data, err_data, linewidth=2.5, color='tab:blue', label="Cross Track Error")
        plt.ylim(0, 220)
        plt.xlabel("Time [s]", fontsize=FS_LABEL)
        plt.ylabel("Error [m]", fontsize=FS_LABEL)
        plt.xticks(fontsize=FS_TICK)
        plt.yticks(fontsize=FS_TICK)
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.legend(fontsize=FS_LEGEND, loc='upper right')
        plt.tight_layout()
        plt.show()

        dt = env.dt
        mean_err = float(np.mean(err_data))
        rms_err = float(np.sqrt(np.mean(err_data ** 2)))
        max_err = float(np.max(err_data))
        p95_err = float(np.percentile(err_data, 95))
        bound_1m = float(np.mean(err_data < 5.0) * 100.0)
        bound_3m = float(np.mean(err_data < 10.0) * 100.0)

        IAE = float(np.sum(np.abs(err_data)) * dt)
        ISE = float(np.sum(err_data ** 2) * dt)

        print("===== Path Tracking Error Metrics =====")
        print(f"Mean Error      : {mean_err:.3f} m")
        print(f"RMSE            : {rms_err:.3f} m")
        print(f"Max Error       : {max_err:.3f} m")
        print(f"95% Error (P95) : {p95_err:.3f} m")
        print(f"Time within 5 m : {bound_1m:.1f} %")
        print(f"Time within 10 m: {bound_3m:.1f} %")
        print(f"IAE (∫|e|dt)    : {IAE:.3f} m·s")
        print(f"ISE (∫e^2 dt)   : {ISE:.3f} m²·s")


    def visualize_comparison(env, results_dict):
        plt.rcParams.update({
            'font.size': 16,
            'axes.titlesize': 20,
            'axes.labelsize': 18,
            'xtick.labelsize': 16,
            'ytick.labelsize': 16,
            'legend.fontsize': 22,
            'figure.titlesize': 22
        })

        import pandas as pd

        offset = env.global_map_size / 2.0
        hard_zone_margin = getattr(env, "hard_zone", 2.0)

        # -------------------------
        # Graph 1: Trajectory
        # -------------------------
        plt.figure(figsize=(10, 10))
        ax = plt.gca()

        for i, (oE, oN, oR) in enumerate(env.obstacles):
            lbl_obs = 'Obstacle' if i == 0 else None
            circle = patches.Circle(
                (oE - offset, oN - offset), oR,
                edgecolor='black', facecolor='gray', alpha=0.5, zorder=1, label=lbl_obs
            )
            ax.add_patch(circle)

            lbl_hard = 'Hard Limit' if i == 0 else None
            hard_circle = patches.Circle(
                (oE - offset, oN - offset), oR + hard_zone_margin,
                edgecolor='red', facecolor='none', linewidth=1.5, linestyle=':', zorder=1, label=lbl_hard
            )
            ax.add_patch(hard_circle)

        if env.global_path is not None and len(env.global_path) > 0:
            plt.plot(
                env.global_path[:, 0] - offset,
                env.global_path[:, 1] - offset,
                color='cyan',
                linewidth=2,
                linestyle='--',
                label="Global Path",
                zorder=1
            )

        for name, data in results_dict.items():
            traj = data['traj']
            plt.plot(
                traj[:, 0] - offset,
                traj[:, 1] - offset,
                color=data['color'],
                linestyle=data['style'],
                linewidth=3,
                label=name,
                alpha=0.9,
                zorder=3
            )

        plt.xlabel("X Position [m]")
        plt.ylabel("Y Position [m]")
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.axis('equal')
        plt.legend(
            loc='lower left',
            framealpha=0.8,
            facecolor='white',
            edgecolor='#cccccc',
            shadow=False,
            fontsize=18
        )
        plt.tight_layout()
        plt.show()

        # -------------------------
        # Graph 2: Cross Track Error
        # -------------------------
        plt.figure(figsize=(10, 6))
        for name, data in results_dict.items():
            min_len = min(len(data['time']), len(data['err']))
            t_data = data['time'][:min_len]
            err_data = data['err'][:min_len]

            plt.plot(
                t_data,
                err_data,
                color=data['color'],
                linestyle=data['style'],
                linewidth=2.5,
                label=name
            )

        plt.xlabel("Time [s]")
        plt.ylabel("Cross Track Error [m]")
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.legend(
            loc='upper right',
            framealpha=0.5,
            facecolor='white',
            edgecolor='#cccccc',
            shadow=False,
            fontsize=18
        )
        plt.tight_layout()
        plt.show()

        # -------------------------
        # Graph 3: Yaw Rate
        # -------------------------
        plt.figure(figsize=(10, 6))
        window_size = 10

        for name, data in results_dict.items():
            if 'w' in data and len(data['w']) > 0:
                min_len = min(len(data['time']), len(data['w']))
                t_data = data['time'][:min_len]
                w_data = data['w'][:min_len]

                plt.plot(
                    t_data,
                    w_data,
                    color=data['color'],
                    linestyle=data['style'],
                    linewidth=1.0,
                    alpha=0.3,
                    label=f"{name}_Raw"
                )

                w_smooth = pd.Series(w_data).rolling(
                    window=window_size,
                    center=True,
                    min_periods=1
                ).mean()

                plt.plot(
                    t_data,
                    w_smooth,
                    color=data['color'],
                    linestyle=data['style'],
                    linewidth=2.5,
                    label=name,
                    alpha=1.0
                )

        plt.xlabel("Time [s]")
        plt.ylabel("Yaw Rate [rad/s]")
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.legend(
            loc='lower left',
            framealpha=0.5,
            facecolor='white',
            edgecolor='#cccccc',
            shadow=False,
            fontsize=18
        )
        plt.tight_layout()
        plt.show()

    # ---------------------------------------------------------
    # 3. 실험 목록 (색상: orange, Blue, Red / 스타일: 실선)
    # ---------------------------------------------------------
    experiments = [
        # {
        #     "name": "Non-linear + APF",
        #     "type": "apf",
        #     "path": None,
        #     "color": "red",  
        #     "style": "-"      
        # },
        {
            "name": "SAC (Eta+Track+Obs)",
            "type": "sac",
            "path": "sac_maze_checkpoint_three", 
            "color": "orange",
            "style": "-"      
        },
        {
            "name": "SAC (Eta+Track+Obs+Time)",
            "type": "sac",
            "path": "sac_maze_checkpoint_four",
            "color": "blue",    
            "style": "-"       
        }
    ]

    results = {}

    print(">>> 논문용 비교 데이터 수집 시작...")
    
    for exp in experiments:
        print(f"Running: {exp['name']} ...")
        
        env.retry() 
        env.time_table = []
        env.dist_error_table = []
        env.w_table = [] 
        
        if exp['type'] == 'apf':
            traj, _ = env.simulation(model="apf", render=False)
            
        elif exp['type'] == 'sac':
            try:
                sac_model = SAC.load(exp['path'])
                traj, _ = env.simulation(model="sac", sac_model=sac_model, render=False)
            except Exception as e:
                print(f"[Error] '{exp['path']}' 로드 실패: {e}")
                # 에러 방지용 더미 데이터
                traj = np.array([[500, 500, 0]])
                env.time_table = [0]
                env.dist_error_table = [0]
                env.w_table = [0]

        # 결과 저장
        results[exp['name']] = {
            'traj': traj,
            'time': np.array(env.time_table),
            'err': np.array(env.dist_error_table) if env.dist_error_table else np.zeros(len(env.time_table)),
            'w': np.array(env.w_table) if env.w_table else np.zeros(len(env.time_table)), 
            'color': exp['color'],
            'style': exp['style']
        }

    # ---------------------------------------------------------
    # 4. 결과 시각화 함수 호출
    # ---------------------------------------------------------
    print(">>> 그래프 생성 중...")
    visualize_comparison(env, results)

    # =========================================================
    # 5. [추가된 부분] 정량적 지표 (비행 시간 & Max Error) 분석 및 출력
    # =========================================================
    print("\n===== 정량적 지표 분석 결과 =====")
    
    for name, data in results.items():
        if 'err' in data and len(data['err']) > 0:
            err_data = np.array(data['err'])
            
            # 1) 총 비행 시간 계산
            total_time = data['time'][-1] if len(data['time']) > 0 else 0.0
            
            # 2) 최대 경로 추종 오차 (Max Error) 계산
            max_err = np.max(err_data)
            
            # 3) 평균 경로 추종 오차 (Mean Error) 계산
            mean_err = np.mean(err_data)
            
            print(f"[{name}]")
            print(f"  - 비행 시간 (Total Time): {total_time:.1f} s")
            print(f"  - 최대 경로 추종 오차 (Max Error): {max_err:.2f} m")
            print(f"  - 평균 경로 추종 오차 (Mean Error): {mean_err:.2f} m\n")