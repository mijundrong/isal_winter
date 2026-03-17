from math import sin, cos, atan2, sqrt, acos, pi, hypot
import numpy as np
from utils import angle_mod, rot_mat_2d
import matplotlib.pyplot as plt
import heapq, time
import math

class dubins_path:
    def __init__(self, curvature=1.0, step_size=0.1, selected_types=None):
        """
        Dubins path planner 클래스

        Parameters
        ----------
        curvature : float
            곡률(1 / 최소 회전반경) [1/m]
        step_size : float
            경로 샘플링 간격 [m]
        selected_types : list[str] or None
            사용할 Dubins 타입 리스트 (예: ["RSL", "RSR"])
            None이면 모든 타입 사용
        """
        self.default_curvature = curvature
        self.default_step_size = step_size
        self.default_selected_types = selected_types

        # path type → 내부 함수 매핑은 생성자에서 초기화
        self._PATH_TYPE_MAP = {
            "LSL": self._LSL,
            "RSR": self._RSR,
            "LSR": self._LSR,
            "RSL": self._RSL,
            "RLR": self._RLR,
            "LRL": self._LRL,
        }

    # ---------------------------------------------------------
    # Public API
    # ---------------------------------------------------------
    def plan(self,
             s_x, s_y, s_yaw,
             g_x, g_y, g_yaw,
             curvature=None,
             step_size=None,
             selected_types=None):
        """
        Dubins path 계획 함수 (클래스 메인 메서드)

        Parameters
        ----------
        s_x, s_y, s_yaw : float
            시작점 위치 및 yaw [m, m, rad]
        g_x, g_y, g_yaw : float
            목표점 위치 및 yaw [m, m, rad]
        curvature : float or None
            곡률(1/회전반경). None이면 생성자에서 설정한 기본값 사용
        step_size : float or None
            샘플링 간격. None이면 기본값 사용
        selected_types : list[str] or None
            사용할 Dubins 타입 (예: ["RSL", "RSR"])
            None이면 모든 타입 사용

        Returns
        -------
        x_list : np.ndarray
        y_list : np.ndarray
        yaw_list : np.ndarray
        modes : list[str]
        lengths : list[float]
            각 세그먼트 길이 (실제 길이 단위 [m])
        """
        if curvature is None:
            curvature = self.default_curvature
        if step_size is None:
            step_size = self.default_step_size
        if selected_types is None:
            selected_types = self.default_selected_types

        if selected_types is None:
            planning_funcs = self._PATH_TYPE_MAP.values()
        else:
            planning_funcs = [self._PATH_TYPE_MAP[ptype] for ptype in selected_types]

        # 시작자세 기준 local frame으로 목표 좌표 변환
        l_rot = rot_mat_2d(s_yaw)
        le_xy = np.stack([g_x - s_x, g_y - s_y]).T @ l_rot
        local_goal_x = le_xy[0]
        local_goal_y = le_xy[1]
        local_goal_yaw = g_yaw - s_yaw

        lp_x, lp_y, lp_yaw, modes, lengths = self._dubins_path_planning_from_origin(
            local_goal_x, local_goal_y, local_goal_yaw, curvature,
            step_size, planning_funcs
        )

        # local → global 변환
        rot = rot_mat_2d(-s_yaw)
        converted_xy = np.stack([lp_x, lp_y]).T @ rot
        x_list = converted_xy[:, 0] + s_x
        y_list = converted_xy[:, 1] + s_y
        yaw_list = angle_mod(np.array(lp_yaw) + s_yaw)

        return x_list, y_list, yaw_list, modes, lengths

    # ---------------------------------------------------------
    # 아래부터는 원래 코드의 내부 함수들 (method로 옮김)
    # ---------------------------------------------------------

    @staticmethod
    def _mod2pi(theta):
        return angle_mod(theta, zero_2_2pi=True)

    @staticmethod
    def _calc_trig_funcs(alpha, beta):
        sin_a = sin(alpha)
        sin_b = sin(beta)
        cos_a = cos(alpha)
        cos_b = cos(beta)
        cos_ab = cos(alpha - beta)
        return sin_a, sin_b, cos_a, cos_b, cos_ab

    # ---- 경로 타입별 세그먼트 길이 계산 ----
    def _LSL(self, alpha, beta, d):
        sin_a, sin_b, cos_a, cos_b, cos_ab = self._calc_trig_funcs(alpha, beta)
        mode = ["L", "S", "L"]
        p_squared = 2 + d ** 2 - (2 * cos_ab) + (2 * d * (sin_a - sin_b))
        if p_squared < 0:
            return None, None, None, mode
        tmp = atan2((cos_b - cos_a), d + sin_a - sin_b)
        d1 = self._mod2pi(-alpha + tmp)
        d2 = sqrt(p_squared)
        d3 = self._mod2pi(beta - tmp)
        return d1, d2, d3, mode

    def _RSR(self, alpha, beta, d):
        sin_a, sin_b, cos_a, cos_b, cos_ab = self._calc_trig_funcs(alpha, beta)
        mode = ["R", "S", "R"]
        p_squared = 2 + d ** 2 - (2 * cos_ab) + (2 * d * (sin_b - sin_a))
        if p_squared < 0:
            return None, None, None, mode
        tmp = atan2((cos_a - cos_b), d - sin_a + sin_b)
        d1 = self._mod2pi(alpha - tmp)
        d2 = sqrt(p_squared)
        d3 = self._mod2pi(-beta + tmp)
        return d1, d2, d3, mode

    def _LSR(self, alpha, beta, d):
        sin_a, sin_b, cos_a, cos_b, cos_ab = self._calc_trig_funcs(alpha, beta)
        p_squared = -2 + d ** 2 + (2 * cos_ab) + (2 * d * (sin_a + sin_b))
        mode = ["L", "S", "R"]
        if p_squared < 0:
            return None, None, None, mode
        d1 = sqrt(p_squared)
        tmp = atan2((-cos_a - cos_b), (d + sin_a + sin_b)) - atan2(-2.0, d1)
        d2 = self._mod2pi(-alpha + tmp)
        d3 = self._mod2pi(-self._mod2pi(beta) + tmp)
        return d2, d1, d3, mode

    def _RSL(self, alpha, beta, d):
        sin_a, sin_b, cos_a, cos_b, cos_ab = self._calc_trig_funcs(alpha, beta)
        p_squared = d ** 2 - 2 + (2 * cos_ab) - (2 * d * (sin_a + sin_b))
        mode = ["R", "S", "L"]
        if p_squared < 0:
            return None, None, None, mode
        d1 = sqrt(p_squared)
        tmp = atan2((cos_a + cos_b), (d - sin_a - sin_b)) - atan2(2.0, d1)
        d2 = self._mod2pi(alpha - tmp)
        d3 = self._mod2pi(beta - tmp)
        return d2, d1, d3, mode

    def _RLR(self, alpha, beta, d):
        sin_a, sin_b, cos_a, cos_b, cos_ab = self._calc_trig_funcs(alpha, beta)
        mode = ["R", "L", "R"]
        tmp = (6.0 - d ** 2 + 2.0 * cos_ab + 2.0 * d * (sin_a - sin_b)) / 8.0
        if abs(tmp) > 1.0:
            return None, None, None, mode
        d2 = self._mod2pi(2 * pi - acos(tmp))
        d1 = self._mod2pi(alpha - atan2(cos_a - cos_b, d - sin_a + sin_b) + d2 / 2.0)
        d3 = self._mod2pi(alpha - beta - d1 + d2)
        return d1, d2, d3, mode

    def _LRL(self, alpha, beta, d):
        sin_a, sin_b, cos_a, cos_b, cos_ab = self._calc_trig_funcs(alpha, beta)
        mode = ["L", "R", "L"]
        tmp = (6.0 - d ** 2 + 2.0 * cos_ab + 2.0 * d * (- sin_a + sin_b)) / 8.0
        if abs(tmp) > 1.0:
            return None, None, None, mode
        d2 = self._mod2pi(2 * pi - acos(tmp))
        d1 = self._mod2pi(-alpha - atan2(cos_a - cos_b, d + sin_a - sin_b) + d2 / 2.0)
        d3 = self._mod2pi(self._mod2pi(beta) - alpha - d1 + self._mod2pi(d2))
        return d1, d2, d3, mode

    # ---- 원점 기준 Dubins path 계산 ----
    def _dubins_path_planning_from_origin(self, end_x, end_y, end_yaw,
                                          curvature, step_size, planning_funcs):

        dx = end_x
        dy = end_y
        d = hypot(dx, dy) * curvature

        theta = self._mod2pi(atan2(dy, dx))
        alpha = self._mod2pi(-theta)
        beta = self._mod2pi(end_yaw - theta)

        best_cost = float("inf")
        b_d1, b_d2, b_d3, b_mode = None, None, None, None

        for planner in planning_funcs:
            d1, d2, d3, mode = planner(alpha, beta, d)
            if d1 is None:
                continue

            cost = (abs(d1) + abs(d2) + abs(d3))
            if best_cost > cost:  # 최소 길이 선택
                b_d1, b_d2, b_d3, b_mode, best_cost = d1, d2, d3, mode, cost

        lengths = [b_d1, b_d2, b_d3]
        x_list, y_list, yaw_list = self._generate_local_course(
            lengths, b_mode, curvature, step_size
        )

        # 실제 길이 단위 [m]로 변환
        lengths = [length / curvature for length in lengths]

        return x_list, y_list, yaw_list, b_mode, lengths

    # ---- 보간 함수들 ----
    def _interpolate(self, length, mode, max_curvature,
                     origin_x, origin_y, origin_yaw,
                     path_x, path_y, path_yaw):

        if mode == "S":
            path_x.append(origin_x + length / max_curvature * cos(origin_yaw))
            path_y.append(origin_y + length / max_curvature * sin(origin_yaw))
            path_yaw.append(origin_yaw)
        else:  # curve
            ldx = sin(length) / max_curvature
            ldy = 0.0
            if mode == "L":
                ldy = (1.0 - cos(length)) / max_curvature
            elif mode == "R":
                ldy = (1.0 - cos(length)) / -max_curvature

            gdx = cos(-origin_yaw) * ldx + sin(-origin_yaw) * ldy
            gdy = -sin(-origin_yaw) * ldx + cos(-origin_yaw) * ldy
            path_x.append(origin_x + gdx)
            path_y.append(origin_y + gdy)

            if mode == "L":
                path_yaw.append(origin_yaw + length)
            elif mode == "R":
                path_yaw.append(origin_yaw - length)

        return path_x, path_y, path_yaw

    def _generate_local_course(self, lengths, modes, max_curvature, step_size):
        p_x, p_y, p_yaw = [0.0], [0.0], [0.0]

        for (mode, length) in zip(modes, lengths):
            if length == 0.0:
                continue

            origin_x, origin_y, origin_yaw = p_x[-1], p_y[-1], p_yaw[-1]
            current_length = step_size

            while abs(current_length + step_size) <= abs(length):
                p_x, p_y, p_yaw = self._interpolate(
                    current_length, mode, max_curvature,
                    origin_x, origin_y, origin_yaw,
                    p_x, p_y, p_yaw
                )
                current_length += step_size

            p_x, p_y, p_yaw = self._interpolate(
                length, mode, max_curvature,
                origin_x, origin_y, origin_yaw,
                p_x, p_y, p_yaw
            )

        return p_x, p_y, p_yaw

class AGPF:
    def __init__(self, K_att, K_rep, step_size):
        self.K_att = K_att
        self.K_rep = K_rep
        self.step_size = step_size

    def blocked(self, cX, cY, dX, dY, matrix):
        if cX + dX < 0 or cX + dX >= matrix.shape[0]:
            return True
        if cY + dY < 0 or cY + dY >= matrix.shape[1]:
            return True
        if dX != 0 and dY != 0:
            if matrix[cX + dX][cY] == 1 and matrix[cX][cY + dY] == 1:
                return True
            if matrix[cX + dX][cY + dY] == 1:
                return True
        else:
            if dX != 0:
                if matrix[cX + dX][cY] == 1:
                    return True
            else:
                if matrix[cX][cY + dY] == 1:
                    return True
        return False

    def heuristic(self, a, b, hchoice):
        if hchoice == 1:
            xdist = math.fabs(b[0] - a[0])
            ydist = math.fabs(b[1] - a[1])
            if xdist > ydist:
                return 14 * ydist + 10 * (xdist - ydist)
            else:
                return 14 * xdist + 10 * (ydist - xdist)
        if hchoice == 2:
            return math.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2)

    def aStar(self, matrix, start, goal, hchoice):
        close_set = set()
        came_from = {}
        gscore = {start: 0}
        fscore = {start: self.heuristic(start, goal, hchoice)}

        pqueue = []

        heapq.heappush(pqueue, (fscore[start], start))

        starttime = time.time()

        while pqueue:

            current = heapq.heappop(pqueue)[1]
            if current == goal:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                path = path[::]
                endtime = time.time()
                return (path, round(endtime - starttime, 6))

            close_set.add(current)
            for dX, dY in [
                (0, 1),
                (0, -1),
                (1, 0),
                (-1, 0),
                (1, 1),
                (1, -1),
                (-1, 1),
                (-1, -1),
            ]:

                if self.blocked(current[0], current[1], dX, dY, matrix):
                    continue

                neighbour = current[0] + dX, current[1] + dY

                if hchoice == 1:
                    if dX != 0 and dY != 0:
                        tentative_g_score = gscore[current] + 14
                    else:
                        tentative_g_score = gscore[current] + 10
                elif hchoice == 2:
                    if dX != 0 and dY != 0:
                        tentative_g_score = gscore[current] + math.sqrt(2)
                    else:
                        tentative_g_score = gscore[current] + 1

                if (
                        neighbour in close_set
                ):  # and tentative_g_score >= gscore.get(neighbour,0):
                    continue

                if tentative_g_score < gscore.get(
                        neighbour, 0
                ) or neighbour not in [i[1] for i in pqueue]:
                    came_from[neighbour] = current
                    gscore[neighbour] = tentative_g_score
                    fscore[neighbour] = tentative_g_score + self.heuristic(
                        neighbour, goal, hchoice
                    )
                    heapq.heappush(pqueue, (fscore[neighbour], neighbour))
            endtime = time.time()
        return (0, round(endtime - starttime, 6))

if __name__ == "__main__":


    # ===== 1. 시작/목표 상태 정의 =====
    # EN 좌표계 기준 (E, N, psi)
    start_x = 60.0
    start_y = 60.0
    start_yaw = -2 * pi / 4  # -90 deg

    goal_x = 20.0
    goal_y = 35.0
    goal_yaw = -2 * pi / 4   # -90 deg

    # 최소 회전반경 R, 곡률 = 1 / R
    v = 1.0          # 선속도 (그냥 R 계산용 개념 변수)
    max_w = 0.25     # 최대 각속도 [rad/s]
    R = v / max_w    # 최소 회전 반경
    curvature = 1.0 / R

    # ===== 2. 플래너 생성 및 경로 계산 =====
    planner = dubins_path(curvature=curvature, step_size=0.1)

    x_list, y_list, yaw_list, modes, seg_lengths = planner.plan(
        start_x, start_y, start_yaw,
        goal_x, goal_y, goal_yaw,
    )
    print(x_list)
    print(y_list)
    print("=== Dubins Path Debug ===")
    print(f"mode: {''.join(modes)}")
    print(f"segment lengths [m]: {seg_lengths}")
    print(f"#points: {len(x_list)}")
    print("first 5 points (E, N, yaw[deg]):")
    for i in range(min(5, len(x_list))):
        print(f"{i}: ({x_list[i]:.2f}, {y_list[i]:.2f}, {yaw_list[i] * 180.0 / pi:.1f})")

    # ===== 3. matplotlib으로 경로 시각화 =====
    plt.figure(figsize=(6, 6))
    plt.plot(x_list, y_list, "-b", label="Dubins path")

    # 시작/목표 포인트 표시
    plt.plot(start_x, start_y, "go", label="Start")
    plt.plot(goal_x, goal_y, "ro", label="Goal")

    # 시작/목표 헤딩 방향 화살표
    arrow_len = 5.0
    sx2 = start_x + arrow_len * cos(start_yaw)
    sy2 = start_y + arrow_len * sin(start_yaw)
    gx2 = goal_x + arrow_len * cos(goal_yaw)
    gy2 = goal_y + arrow_len * sin(goal_yaw)
    plt.arrow(start_x, start_y, sx2 - start_x, sy2 - start_y,
              head_width=1.0, length_includes_head=True, color="g")
    plt.arrow(goal_x, goal_y, gx2 - goal_x, gy2 - goal_y,
              head_width=1.0, length_includes_head=True, color="r")

    plt.xlabel("E [m]")
    plt.ylabel("N [m]")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.title(f"Dubins path: {''.join(modes)}")
    plt.show()






'''

import numpy as np
import random
import pygame
import gymnasium as gym
from gymnasium import spaces
import math
from path_planner import dubins_path


class SimpleMazeGrid(gym.Env):
    """
    EN 좌표계 (E: 동(+x), N: 북(+y))
    Gym / Stable-Baselines3 호환 환경
    Dubins 전역 경로 + 경로 추종(path tracking) + 장애물 회피(기존 보상 기반)
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
        dt=0.05,
        render_option=False,
        random_seed=None,
        spec=None,
        # obstacles
        obstacle_count=0,
        obstacle_min_radius=1.0,
        obstacle_max_radius=3.0,
        # sensor range
        sensor_range=None,  # 미지정 시 local_map_size/2
        # ---- LIDAR 옵션 ----
        use_lidar_edges=True,
        lidar_num_rays=360,
        lidar_fov=2 * math.pi,  # 360 deg
        reference_L=None,
    ):
        super(SimpleMazeGrid, self).__init__()

        # 기본 파라미터
        self.global_map_size = int(global_map_size)
        self.local_map_size = int(local_map_size)
        self.dt = float(dt)
        self.render_option = render_option
        self.spec = spec

        self.terminated = False
        self.terminated_radius = 1.0
        self.goal = False
        self.visited_path = []

        # 이동 파라미터
        self.v = float(v)
        self.w = 0.0
        self.min_w = float(w[0])
        self.max_w = float(w[1])
        self.R = self.v / self.max_w
        self.path_planner = dubins_path(1 / self.R, 0.1)
        self.max_steps = 1500

        # w_cmd + Δw 구조에서 Δw 최대 비율 (|Δw| ≤ max_delta_w_ratio * max_w)
        self.max_delta_w_ratio = 2.0

        # 히스토리 길이
        self.hist_len = 10

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

        self.reference_point = None  # [E, N]
        self.closest_path_point = None
        self.a_cmd = 0.0             # 경로 추종 기반 lateral acceleration command
        self.w_cmd = 0.0             # yaw rate command (이론값)
        self.eta = 0.0               # heading error (ref 방향 - 현재 heading)
        self.path_end_state = None

        # LIDAR 설정
        self.use_lidar_edges = bool(use_lidar_edges)
        self.lidar_num_rays = int(lidar_num_rays)
        self.lidar_fov = float(lidar_fov)
        self.lidar_max_range = float(self.sensor_range)

        # 장애물/안전 파라미터
        self.obstacle_count = int(obstacle_count)
        self.obstacle_min_r = float(obstacle_min_radius)
        self.obstacle_max_r = float(obstacle_max_radius)
        self.agent_radius = 1.0
        self.safety_zone = 10.0
        self.hard_zone = 3.5

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
        self.COLOR_REF_POINT = (200, 0, 200)  # reference X 표시용

        # Action space (각속도 명령, -1~1 → [-max_w, max_w])
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )

        # Reward config
        self.reward_cfg = {
            "w_smooth": 0.1,            # yaw 속도 변화 부드럽게
            "w_obs": 0.5,               # 장애물 회피 (기존 방식)
            "w_time": 0.1,              # 시간 페널티
            "w_track": 0.2,
            "w_eta": 0.1,
            "bonus_goal": 100.0,
            "penalty_collision": -100.0,
            "penalty_timeout": -100.0,
            "clip_per_step": 1.0,
        }

        # 상태 초기화
        if spec is not None:
            self._reset_core_spec(random_seed)
        else:
            self._reset_core(random_seed)

        # Observation space 정의
        self._build_observation_space()

        # Rendering
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
        """
        state dict:
          - w_hist: shape (hist_len,), 각속도 히스토리
          - err_hist: shape (2*hist_len,), eta(경로 추종 heading error)의 cos/sin 히스토리
          - obstacle_pos: shape (1, L, L), 로컬 장애물 그리드 (0~255 uint8)
        """
        L = self.local_map_size

        self.observation_space = spaces.Dict(
            {
                "w_hist": spaces.Box(
                    low=np.full((self.hist_len,), self.min_w, dtype=np.float32),
                    high=np.full((self.hist_len,), self.max_w, dtype=np.float32),
                    dtype=np.float32,
                ),
                "err_hist": spaces.Box(
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
    # Core reset
    # =========================
    def _reset_core(self, random_seed=None):
        self.terminated = False
        self.goal = False
        self.visited_path = []

        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)

        # ===== 1. 에이전트 시작 상태: 맵 중앙 + 랜덤 heading =====
        center_E = self.global_map_size // 2
        center_N = self.global_map_size // 2

        psi0 = random.uniform(-math.pi, math.pi)  # 에이전트 초기 자세 (완전 랜덤)

        self.initial_player_pos = np.array(
            [center_E, center_N, psi0],
            dtype=np.float32,
        )
        self.player_pos = self.initial_player_pos.copy()

        # ===== 2. 목표 상태: 중심에서 거리 40, 랜덤 방향 + 랜덤 heading =====
        radius = 60.0
        phi = random.uniform(-math.pi, math.pi)  # 중심에서 골까지의 방향

        ge = float(center_E + radius * math.cos(phi))
        gn = float(center_N + radius * math.sin(phi))

        # 맵 바깥으로 튀는 것 방지
        ge = float(np.clip(ge, 0.0, self.global_map_size - 1))
        gn = float(np.clip(gn, 0.0, self.global_map_size - 1))

        gpsi = random.uniform(-math.pi, math.pi)  # 목표 heading도 랜덤

        self.goal_pos = np.array([ge, gn, gpsi], dtype=np.float32)

        # ===== 3. Dubins 전역 경로 생성 & reference point 초기화 =====
        self._build_global_dubins_path()
        self.update_reference_point()

        # ===== 4. 나머지 상태 초기화 =====
        self.cumulative_reward = 0.0
        self.steps = 0
        self.visited_path.append(self.player_pos.copy())

        # 장애물 생성
        self.obstacles = []
        self.generate_obstacles(random_seed, midline_offset_ratio=0.05)

        # 히스토리 버퍼 초기화
        self.w_hist = [0.0] * self.hist_len
        self.err_hist = [0.0] * self.hist_len  # eta 기록

        # 로컬 격자 및 기록 테이블
        self.obs_grid = self.compute_local_grids()
        self.time_table, self.v_table, self.w_table = [], [], []

        return self.get_state()

    def _reset_core_spec(self, random_seed=None):
        self.terminated = False
        self.goal = False
        self.visited_path = []

        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)

        initial_player_pos, goal_pos, obs_spec = self.spec

        self.initial_player_pos = np.array(initial_player_pos[0:3], dtype=np.float32)
        self.player_pos = self.initial_player_pos.copy()
        self.goal_pos = np.array(goal_pos[0:3], dtype=np.float32)

        # Dubins 전역 경로 생성
        self._build_global_dubins_path()
        self.update_reference_point()

        self.cumulative_reward = 0.0
        self.steps = 0
        self.visited_path.append(self.player_pos.copy())

        # 히스토리 버퍼
        self.w_hist = [0.0] * self.hist_len
        self.err_hist = [0.0] * self.hist_len

        # 장애물 설정
        self.obstacles = []
        if obs_spec is not None:
            obs_arr = np.array(obs_spec, dtype=float)
            if obs_arr.ndim == 1:
                if obs_arr.size != 3:
                    raise ValueError(
                        f"obs_spec 1D인데 길이가 3이 아님: got {obs_arr.size}"
                    )
                obs_arr = obs_arr.reshape(1, 3)
            for i in range(obs_arr.shape[0]):
                E, N, r = (
                    float(obs_arr[i, 0]),
                    float(obs_arr[i, 1]),
                    float(obs_arr[i, 2]),
                )
                self.obstacles.append((E, N, r))
        else:
            self.generate_obstacles(random_seed)

        # 로컬 격자 / 기록
        self.obs_grid = self.compute_local_grids()
        self.time_table, self.v_table, self.w_table = [], [], []

        return self.get_state()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if self.spec is not None:
            obs = self._reset_core_spec(random_seed=seed)
        else:
            obs = self._reset_core(random_seed=seed)
        info = {}
        return obs, info

    def retry(self):
        self.player_pos = self.initial_player_pos.copy()
        self.terminated = False
        self.cumulative_reward = 0.0
        self.steps = 0
        self.w = 0.0
        self.w_hist = [0.0] * self.hist_len
        self.err_hist = [0.0] * self.hist_len
        self.update_reference_point()
        return self.get_state(), {}

    # =========================
    # Util: angle / heading
    # =========================
    @staticmethod
    def normalize_angle(angle):
        return (angle + np.pi) % (2 * np.pi) - np.pi

    @staticmethod
    def dir_from_heading(psi):
        return math.cos(psi), math.sin(psi)

    # =========================
    # State build & history
    # =========================
    def _build_state(self):
        obs_grid = self.obs_grid  # (L, L)

        w_hist_vec = np.array(self.w_hist, dtype=np.float32)

        err_pairs = []
        for e in self.err_hist:
            err_pairs.extend([math.cos(e), math.sin(e)])
        err_hist_vec = np.array(err_pairs, dtype=np.float32)

        obs_img = (obs_grid * 255).astype(np.uint8)[np.newaxis, :, :]

        state = {
            "w_hist": w_hist_vec,
            "err_hist": err_hist_vec,
            "obstacle_pos": obs_img,
        }
        return state

    def get_state(self):
        return self._build_state()

    def _push_w(self, w):
        self.w_hist = (self.w_hist + [float(w)])[-self.hist_len:]

    def _push_err(self, e):
        self.err_hist = (self.err_hist + [float(e)])[-self.hist_len:]

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

        # ---------- 0) 이전 상태 저장 (거리 차이 계산용) ----------
        old_pos = self.player_pos.copy()  # [E, N, psi]
        old_eta = self.eta
        # ---------- 1) 경로 추종 관련 값 업데이트 ----------
        # 전역 Dubins 경로 기준 reference point 갱신 + closest_path_point 갱신
        self.update_reference_point()
        # 로컬 장애물 그리드 갱신
        self.obs_grid = self.compute_local_grids()
        # 경로 추종 가속도/각속도 계산 (Dubins 기반, APF 없음)
        self.path_tracking()  # self.a_cmd, self.w_cmd, self.eta 갱신됨

        # ✅ 이전 step에서의 "경로 최소거리" 저장
        old_path_dist = None
        if hasattr(self, "closest_path_point") and self.closest_path_point is not None:
            old_path_dist = float(
                np.linalg.norm(old_pos[:2] - self.closest_path_point)
            )

        # ---------- 2) yaw rate = action directly ----------
        raw = float(action[0])  # 기대 범위: [-1, 1]
        raw = np.clip(raw, -1.0, 1.0)

        # action이 곧바로 yaw rate 명령이 되도록 스케일링
        w_target = raw * self.max_w

        # 최종 yaw rate
        self.w = float(np.clip(w_target, self.min_w, self.max_w))

        # 히스토리 업데이트 (yaw, eta)
        self._push_w(self.w)
        self._push_err(self.eta)

        # ---------- 3) 상태 적분 ----------
        new_pos = self.player_pos.copy()
        new_pos[2] = self.normalize_angle(new_pos[2] + self.w * self.dt)

        dE = self.v * math.cos(new_pos[2]) * self.dt
        dN = self.v * math.sin(new_pos[2]) * self.dt

        new_pos[0] = np.clip(new_pos[0] + dE, 0.0, self.global_map_size - 1)
        new_pos[1] = np.clip(new_pos[1] + dN, 0.0, self.global_map_size - 1)

        # 상태 적용
        self.player_pos = new_pos

        # ✅ 현재 위치 기준 "전역 경로 최소 거리점" 다시 계산
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

        # ---------- 4) 보상 계산 ----------

        # (A) yaw 명령 매끄러움
        w_arr = np.array(self.w_hist, dtype=np.float32)
        dw = np.diff(w_arr)
        w_range = max(2.0 * self.max_w, 1e-6)
        dw_n = dw / w_range
        smooth = float(np.mean(dw_n ** 2)) if dw_n.size > 0 else 0.0
        R_smooth = -float(np.clip(smooth, 0.0, 1.0))

        # (B) 장애물 회피 보상: "페널티 감소량" 기반 (이전 - 현재)
        pen_old = self.compute_obstacle_penalties(old_pos)
        pen_new = self.compute_obstacle_penalties(new_pos)

        p_old = np.clip((pen_old["clear"] + pen_old["hard"]) / 3.0, 0.0, 1.0)
        p_new = np.clip((pen_new["clear"] + pen_new["hard"]) / 3.0, 0.0, 1.0)

        # ▶ 장애물에서 멀어지면(p_new < p_old) 양수 보상
        R_obs = -p_new
        # print(self.safety_zone * (p_old - p_new)/(self.v * self.dt))

        # (C) 시간 페널티
        R_time = -5.0 * float(self.dt)

        # (D) 경로 추종 보상: 전역 경로까지 최소거리 "감소량" (이전 - 현재)
        if (old_path_dist is not None) and (new_path_dist is not None):
            delta_d = (old_path_dist - new_path_dist) / (self.v * self.dt)
            # ▶ 경로에 가까워지면(new_path_dist < old_path_dist) 양수 보상
            R_track = float(np.clip(delta_d, -1.0, 1.0))
        else:
            R_track = 0.0

        if abs(new_path_dist) <= 0.15:
            R_track += 1
            R_track = np.clip(R_track, -1.0, 1.0)


        # (E) 경로 추종 보상: 전역 경로까지 최소거리 "감소량" (이전 - 현재)
        new_eta = self.eta
        R_eta = 50 * (old_eta ** 2 - new_eta ** 2)
        R_eta = np.clip(R_eta, -1.0, 1.0)

        if abs(math.degrees(self.eta)) <= 7.5:
            R_eta += 1
            R_eta = np.clip(R_eta, -1.0, 1.0)

        # 최종 reward 합산
        reward = (
            cfg["w_smooth"] * R_smooth
            + cfg["w_obs"] * R_obs       # 장애물에서 멀어질수록 +
            + cfg["w_time"] * R_time     # 시간 지날수록 -
            + cfg["w_track"] * R_track   # 전역 경로에 가까워질수록 +
            + cfg["w_eta"] * R_eta  # 전역 경로에 가까워질수록 +

        )
        reward = float(np.clip(reward, -cfg["clip_per_step"], cfg["clip_per_step"]))

        # ---------- 5) 종료 조건 ----------
        # 1) 충돌
        if self.check_collision(new_pos[:2]):
            reward += cfg["penalty_collision"]
            terminated = True

        # 2) 목표 도달 (위치 + heading)
        #    ✅ 이제는 "목표점"이 아니라 Dubins 전역 경로의 끝(path_end_state) 기준으로 판정
        if getattr(self, "path_end_state", None) is not None:
            target_xy = self.path_end_state[:2]
            target_psi = self.path_end_state[2]
        else:
            # Dubins 경로가 제대로 안 만들어졌을 때는 기존 goal_pos 기준으로 fallback
            target_xy = self.goal_pos[:2]
            target_psi = self.goal_pos[2]

        cur_goal_dist = float(np.linalg.norm(new_pos[:2] - target_xy))
        heading_ok = math.cos(new_pos[2] - target_psi) >= math.cos(math.radians(30))

        if (cur_goal_dist < self.terminated_radius) and heading_ok:
            reward += cfg["bonus_goal"]
            self.goal = True
            terminated = True

        # 3) 스텝 초과
        if self.steps > self.max_steps:
            reward += cfg["penalty_timeout"]
            truncated = True

        # ---------- 6) 상태 / 로그 업데이트 ----------
        next_state = self._build_state()

        self.cumulative_reward += reward
        self.steps += 1
        self.visited_path.append(self.player_pos.copy())
        self.time_table.append(self.steps * self.dt)
        self.w_table.append(self.w)

        self.terminated = terminated or truncated

        info = {}
        return next_state, reward, terminated, truncated, info

    # =========================
    # 장애물 생성 / 충돌 체크
    # =========================
    def generate_obstacles(
        self, random_seed=None, max_attempts=5000,
        ensure_midline=True, midline_offset_ratio=0.2
    ):
        x = np.random.uniform(0, 1)
        if x <= 0.0:
            self.obstacles = []
            return

        self.obstacles = []
        if self.obstacle_count <= 0:
            return

        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)

        E0, N0 = float(self.initial_player_pos[0]), float(self.initial_player_pos[1])
        Eg, Ng = float(self.goal_pos[0]), float(self.goal_pos[1])

        # 시작-목표 중간쯤에 하나 박아두는 옵션
        if ensure_midline:
            d = math.hypot(Eg - E0, Ng - N0)
            if d > 1e-6:
                r0 = random.uniform(self.obstacle_min_r, self.obstacle_max_r)
                mE = 0.5 * (E0 + Eg)
                mN = 0.5 * (N0 + Ng)
                dE = Eg - E0
                dN = Ng - N0
                nE, nN = -dN, dE
                norm = math.hypot(nE, nN)
                if norm > 1e-9:
                    nE /= norm
                    nN /= norm
                else:
                    nE, nN = 0.0, 1.0
                max_offset = midline_offset_ratio * d
                max_offset = float(
                    np.clip(
                        max_offset,
                        0.0,
                        0.4 * min(self.global_map_size, self.sensor_range * 2),
                    )
                )
                offset = random.uniform(-max_offset, max_offset)
                mE_off = float(
                    np.clip(
                        mE + offset * nE,
                        r0 + 1.0,
                        self.global_map_size - (r0 + 1.0),
                    )
                )
                mN_off = float(
                    np.clip(
                        mN + offset * nN,
                        r0 + 1.0,
                        self.global_map_size - (r0 + 1.0),
                    )
                )
                dist_start = math.hypot(mE_off - E0, mN_off - N0)
                dist_goal = math.hypot(mE_off - Eg, mN_off - Ng)
                r_cap = max(
                    0.2, min(dist_start, dist_goal) - (self.agent_radius + 0.5)
                )
                r0 = float(
                    np.clip(r0, self.obstacle_min_r, min(self.obstacle_max_r, r_cap))
                )
                if r0 > 0.0:
                    self.obstacles.append((mE_off, mN_off, r0))

        remaining = max(0, self.obstacle_count - len(self.obstacles))
        margin_rand = self.obstacle_max_r + 1.0

        for _ in range(remaining):
            placed = False
            for _ in range(max_attempts):
                r = random.uniform(self.obstacle_min_r, self.obstacle_max_r)
                E = random.uniform(margin_rand, self.global_map_size - margin_rand)
                N = random.uniform(margin_rand, self.global_map_size - margin_rand)

                if math.hypot(E - E0, N - N0) < (r + self.R * 2.0):
                    continue
                if math.hypot(E - Eg, N - Ng) < (r + self.R * 2.0):
                    continue
                if any(
                    math.hypot(E - oE, N - oN) < (r + oR + 1.0)
                    for (oE, oN, oR) in self.obstacles
                ):
                    continue

                self.obstacles.append((E, N, r))
                placed = True
                break
            if not placed:
                break

    def check_collision(self, pos_EN):
        E, N = float(pos_EN[0]), float(pos_EN[1])
        for (oE, oN, oR) in self.obstacles:
            if math.hypot(E - oE, N - oN) <= (oR + self.agent_radius):
                return True
        return False

    def is_reference_point_in_obstacle_zone(self, margin=None):
        """
        reference_point가 장애물 원 안(또는 margin 포함)으로 들어가 있으면 True.
        """
        if self.reference_point is None:
            return False

        margin = self.hard_zone

        E_r = float(self.reference_point[0])
        N_r = float(self.reference_point[1])

        for (oE, oN, oR) in self.obstacles:
            d = math.hypot(E_r - oE, N_r - oN)
            if d <= (oR + margin):
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

    # =========================
    # Local grid (for CNN)
    # =========================
    def _rect_circle_intersects(self, xmin, xmax, ymin, ymax, cx, cy, r):
        closest_x = min(max(cx, xmin), xmax)
        closest_y = min(max(cy, ymin), ymax)
        dx = cx - closest_x
        dy = cy - closest_y
        return (dx * dx + dy * dy) <= (r * r + 1e-12)

    def compute_local_grids(self):
        """
        로컬 격자 중 '장애물 그리드'만 계산하여 반환.
        반환: obs_grid (L, L), float32, {0.0, 1.0}
        """
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
            xb_o, yb_o = self.world_to_body((oE, oN))
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

    # =========================
    # Coord Transforms
    # =========================
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

    # =========================
    # Dubins Path & Reference Point
    # =========================
    def _build_global_dubins_path(self):
        """
        initial_player_pos → goal_pos Dubins 경로 계산.
        + 마지막 지점에서 10m 직선 구간 추가.
        self.global_path: (N, 2), [E, N]
        """
        try:
            # 1. Dubins 경로 생성
            path_x, path_y, path_yaw, modes, lengths = self.path_planner.plan(
                self.initial_player_pos[0],
                self.initial_player_pos[1],
                self.initial_player_pos[2],
                self.goal_pos[0],
                self.goal_pos[1],
                self.goal_pos[2],
            )

            # [수정] 반환값이 numpy array일 수 있으므로 list로 변환
            path_x = list(path_x)
            path_y = list(path_y)
            path_yaw = list(path_yaw)

            # 2. 직선 구간 10m 추가
            if len(path_x) > 0:
                last_x = path_x[-1]
                last_y = path_y[-1]
                last_yaw = path_yaw[-1]

                extension_len = 25.0  # 10m 연장
                step_size = 0.1       # 포인트 간격
                num_points = int(extension_len / step_size)

                ext_x = []
                ext_y = []

                # 마지막 heading 방향으로 점들 생성
                for i in range(1, num_points + 1):
                    dist = i * step_size
                    new_x = last_x + dist * math.cos(last_yaw)
                    new_y = last_y + dist * math.sin(last_yaw)
                    ext_x.append(new_x)
                    ext_y.append(new_y)

                # 리스트 확장
                path_x.extend(ext_x)
                path_y.extend(ext_y)

            # ✅ 2.5. 경로 끝 상태(E, N, psi_end) 계산
            if len(path_x) >= 2:
                last_x, last_y = path_x[-1], path_y[-1]
                prev_x, prev_y = path_x[-2], path_y[-2]
                psi_end = math.atan2(last_y - prev_y, last_x - prev_x)
                self.path_end_state = np.array([last_x, last_y, psi_end], dtype=np.float32)
            else:
                self.path_end_state = None

            # 3. Numpy 변환
            self.global_path = np.stack([path_x, path_y], axis=1).astype(np.float32)

        except Exception as e:
            print("[Dubins] path planning failed:", e)
            self.global_path = np.zeros((0, 2), dtype=np.float32)
            self.path_end_state = None   # ✅ 실패 시 초기화

    def compute_reference_point_on_global(self, L=None):
        """
        전역 Dubins 경로(self.global_path) 위에서
        에이전트로부터 L만큼 떨어진 지점을 찾는다 (남은 길이 부족하면 끝점).

        추가:
          - self.closest_path_idx   : 에이전트와 가장 가까운 path index
          - self.closest_path_point : 그 지점의 [E, N]
        """
        if L is None:
            L = self.reference_L

        if (
                not hasattr(self, "global_path")
                or self.global_path is None
                or self.global_path.shape[0] < 2
        ):
            # 가장 가까운 점 정보도 초기화
            self.closest_path_idx = None
            self.closest_path_point = None
            return None

        path = np.asarray(self.global_path, dtype=float)
        path_EN = path[:, :2]

        agent = np.array(self.player_pos[:2], dtype=float)
        diffs = path_EN - agent
        dists = np.hypot(diffs[:, 0], diffs[:, 1])

        # ✅ 에이전트와 가장 가까운 path index/점 저장
        cur_idx = int(np.argmin(dists))
        self.closest_path_idx = cur_idx
        self.closest_path_point = path_EN[cur_idx].copy()

        if L <= 0.0:
            # L=0이면 그냥 가장 가까운 점 리턴
            return self.closest_path_point.copy()

        # 남은 경로 길이 계산
        remaining_len = 0.0
        for i in range(cur_idx, len(path_EN) - 1):
            p0 = path_EN[i]
            p1 = path_EN[i + 1]
            remaining_len += float(math.hypot(p1[0] - p0[0], p1[1] - p0[1]))

        # 남은 길이가 L보다 짧으면 종점
        if remaining_len <= L:
            return path_EN[-1].copy()

        # cur_idx부터 앞으로 누적 길이가 L이 되는 세그먼트의 끝점 선택
        acc = 0.0
        for i in range(cur_idx, len(path_EN) - 1):
            p0 = path_EN[i]
            p1 = path_EN[i + 1]
            seg_len = float(math.hypot(p1[0] - p0[0], p1[1] - p0[1]))
            acc += seg_len
            if acc >= L:
                return p1.copy()

        return path_EN[-1].copy()

    def update_reference_point(self):
        self.reference_point = self.compute_reference_point_on_global(
            L=self.reference_L
        )

    # =========================
    # Path Tracking (Dubins 기반, APF 없음)
    # =========================
    def path_tracking(self):
        """
        Dubins 전역 경로 위 reference_point를 추종하는 경로 추종.

        a_cmd = 2 * V^2 / L * sin(eta)

        여기서
          - eta: ref 방향(LOS) - 현재 heading
        """

        V = float(self.v)
        L = float(self.reference_L)

        if V <= 1e-6 or L <= 1e-6 or self.reference_point is None:
            self.eta = 0.0
            self.a_cmd = 0.0
            self.w_cmd = 0.0
            return 0.0

        # 현재 위치 (E,N)
        pos = np.array(self.player_pos[:2], dtype=float)
        # reference point (E,N)
        ref = np.array(self.reference_point[:2], dtype=float)

        # 기체 -> reference 방향 (LOS)
        dEN = ref - pos
        R = float(np.linalg.norm(dEN))
        if R < 1e-6:
            self.eta = 0.0
            self.a_cmd = 0.0
            self.w_cmd = 0.0
            return 0.0

        chi_ref = math.atan2(dEN[1], dEN[0])  # ref 방향 (EN 기준)
        psi = float(self.player_pos[2])       # 현재 heading

        # eta = ref 방향 - 현재 heading
        eta = self.normalize_angle(chi_ref - psi)

        # 경로 추종 가속도
        a_cmd = 2.0 * (V * V / L) * math.sin(eta)

        # yaw rate (이론값)
        w_cmd = a_cmd / V

        # 저장
        self.eta = eta
        self.a_cmd = a_cmd
        self.w_cmd = w_cmd

    # =========================
    # 장애물 셀 거리 (penalty용)
    # =========================
    def obstacle_cell_distances_at(self, agent_ENpsi):
        """
        agent_ENpsi = [E,N,psi] 기준 로컬 그리드에서
        obs_grid==1로 채워지는 셀 중심까지 거리(바디 프레임)를 계산.
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
        """
        장애물 페널티용 스칼라:
          - clear: 최소 안전거리 위반 정도 [0,1]
          - hard : 하드 임계 [0,1]
        """
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

        # d_min >= d_safe → 0, d_min → 0 → 1
        clear = float(np.clip(1.0 - d_min / max(d_safe, eps), 0.0, 1.0))

        d_hard = self.hard_zone
        hard = 2.0 if d_min < d_hard else 0.0

        return {"clear": clear, "hard": hard}

    # =========================
    # LIDAR helper (grid 채움)
    # =========================
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
    # Rendering utils
    # =========================
    def world_to_screen(self, e, n, target_rect, cell_size=None):
        if cell_size is None:
            cell_size = target_rect.width / self.global_map_size
        sx = target_rect.x + e * cell_size + cell_size / 2.0
        sy = target_rect.y + target_rect.height - (
            n * cell_size + cell_size / 2.0
        )
        return int(sx), int(sy)

    def draw_arrow(
        self, start_xy, end_xy, color, width=3, head_len=10, head_angle=math.pi / 6
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

    def draw_x(self, center_xy, size, color, width=2):
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

    # =========================
    # Rendering
    # =========================
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

        # 장애물 (global)
        self.draw_obstacles_on_global(cell_size)

        # 목표
        gx, gy = self.world_to_screen(ge, gn, self.rect_global, cell_size)
        pygame.draw.circle(self.screen, self.COLOR_GOAL, (gx, gy), int(cell_size / 3))
        dE_g, dN_g = self.dir_from_heading(self.goal_pos[2])
        self.draw_arrow(
            (gx, gy),
            (
                int(gx + 1.5 * cell_size * dE_g),
                int(gy - 1.5 * cell_size * dN_g),
            ),
            self.COLOR_GOAL_ARROW,
            width=3,
            head_len=10,
        )

        # Dubins 전역 경로 라인
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
                self.screen, self.COLOR_DUBINS, False, dubins_pts, 2
            )

        # 에이전트
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
            width=3,
            head_len=10,
        )

        # 이동 궤적
        if len(self.visited_path) >= 2:
            pts = [
                self.world_to_screen(
                    p[0], p[1], self.rect_global, cell_size
                )
                for p in self.visited_path
            ]
            pygame.draw.lines(self.screen, self.COLOR_PATH, False, pts, 2)

        # reference point (global X표시)
        if self.reference_point is not None:
            rx, ry = self.world_to_screen(
                self.reference_point[0],
                self.reference_point[1],
                self.rect_global,
                cell_size,
            )
            self.draw_x((rx, ry), size=10, color=self.COLOR_REF_POINT, width=2)

        # 센서 범위 원
        self.draw_local_range_on_global(cell_size)

        # Info 패널
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

        # 로컬(바디) 패널
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


# =========================
# 메인 테스트
# =========================
if __name__ == "__main__":
    spec = (
        [90, 90, -2 * math.pi / 4],
        [20, 35, -2 * math.pi / 4],
        [79, 82, 5],
    )

    env = SimpleMazeGrid(
        global_map_size=180,
        local_map_size=100,
        v=1.0,
        w=[-0.25, 0.25],
        dt=0.1,
        render_option=True,
        random_seed=None,
        spec=spec,  # 고정 시나리오
        obstacle_count=1,
        obstacle_min_radius=1.0,
        obstacle_max_radius=4.0,
        sensor_range=10.0,
        use_lidar_edges=True,
        lidar_num_rays=360,
        lidar_fov=math.pi * 2,
        reference_L=5.0,
    )

    env.render()
    env.handle_keyboard_input()
    env.close()


'''