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
