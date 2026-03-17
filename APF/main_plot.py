import numpy as np
import matplotlib
matplotlib.use("TkAgg")  # 또는 "Qt5Agg"
import matplotlib.pyplot as plt
from matplotlib import patches
from pathlib import Path


# ================================
# 스타일(폰트 크기) 설정
# ================================
FS_LABEL = 24
FS_TICK = 20
FS_LEGEND = 20

LW_TRAJ = 3.0
LW_LINE = 2.5
MS_WAYPOINT = 10


def load_episode_rewards_auto(base_dir, candidates=None):
    """
    base_dir에서 에피소드 보상 로그를 자동 탐색해서 rewards (1D) 반환.
    지원:
      - monitor.csv (Stable-Baselines3 Monitor)
      - episode_rewards.txt / rewards.txt 등 txt (1열 또는 [episode, reward] 2열)
      - rewards.npy / episode_rewards.npy
    """
    base_dir = Path(base_dir)

    if candidates is None:
        candidates = [
            "monitor.csv",
            "episode_rewards.txt",
            "episode_reward.txt",
            "rewards.txt",
            "reward.txt",
            "episode_rewards.npy",
            "rewards.npy",
        ]

    # 1) SB3 monitor.csv
    mon = base_dir / "monitor.csv"
    if mon.exists():
        # monitor.csv는 첫 줄에 '#{json...}' 주석이 들어갈 수 있음
        lines = mon.read_text(encoding="utf-8", errors="ignore").splitlines()
        # 주석(#) 라인 제거 후, "r,l,t" 헤더부터 시작하는 지점 찾기
        start_idx = None
        for i, line in enumerate(lines):
            if len(line) > 0 and not line.lstrip().startswith("#"):
                start_idx = i
                break
        if start_idx is None:
            raise RuntimeError("monitor.csv는 있는데 데이터 라인을 찾지 못했습니다.")

        # 남은 라인을 numpy로 읽기
        from io import StringIO
        data = np.genfromtxt(StringIO("\n".join(lines[start_idx:])),
                             delimiter=",", names=True)
        # SB3 Monitor 컬럼: r,l,t  (reward, length, time)
        rewards = np.asarray(data["r"], dtype=float)
        return rewards

    # 2) 후보 파일들 순회 (txt / npy)
    for name in candidates:
        p = base_dir / name
        if not p.exists():
            continue

        if p.suffix == ".npy":
            arr = np.load(p)
            arr = np.asarray(arr).squeeze()
            if arr.ndim == 2 and arr.shape[1] >= 2:
                rewards = arr[:, 1]
            else:
                rewards = arr.reshape(-1)
            return rewards.astype(float)

        # txt/csv류
        try:
            arr = np.loadtxt(p)
        except Exception:
            continue

        arr = np.asarray(arr)
        if arr.ndim == 0:
            continue
        if arr.ndim == 1:
            rewards = arr
        else:
            # [episode, reward] 형태면 reward 컬럼 사용
            rewards = arr[:, 1] if arr.shape[1] >= 2 else arr[:, 0]
        return rewards.astype(float)

    raise FileNotFoundError(
        f"에피소드 보상 파일을 찾지 못했습니다. base_dir={base_dir}\n"
        f"찾는 후보: {candidates} (또는 monitor.csv)"
    )


def plot_episode_reward_curve(
    base_dir,
    algo="SAC",
    window=50,
    save_path=None,
    show=True,
    ylim=None,
):
    """
    논문용 에피소드 보상 학습곡선:
      - Raw reward (얇게)
      - Moving average (굵게)
      - Moving average ±1σ (rolling std) 음영
    """
    rewards = load_episode_rewards_auto(base_dir)
    rewards = np.asarray(rewards, dtype=float)

    n = len(rewards)
    if n < window:
        print(f"[WARN] rewards 길이({n}) < window({window}) 입니다. window를 줄이세요.")
        window = max(5, n // 5)

    # ---- rolling mean/std (trailing window) : O(N) 누적합 방식 ----
    w = int(window)
    c1 = np.cumsum(np.insert(rewards, 0, 0.0))
    c2 = np.cumsum(np.insert(rewards**2, 0, 0.0))

    ma = np.full(n, np.nan, dtype=float)
    ms = np.full(n, np.nan, dtype=float)

    # i = w-1 부터 계산 (마지막 w개로 trailing)
    for i in range(w - 1, n):
        s1 = c1[i + 1] - c1[i + 1 - w]
        s2 = c2[i + 1] - c2[i + 1 - w]
        mean = s1 / w
        var = max(0.0, (s2 / w) - mean**2)
        ma[i] = mean
        ms[i] = np.sqrt(var)

    episodes = np.arange(1, n + 1)

    # ---- Figure ----
    fig, ax = plt.subplots(figsize=(14, 6))

    # Raw rewards: 너무 튀지 않게 연하게
    ax.plot(episodes, rewards, linewidth=1.5, alpha=0.25, label="Raw")

    # Moving avg line
    ax.plot(episodes, ma, linewidth=3.0, label=f"Moving avg (win={w})")

    # Moving avg ±1σ band
    ax.fill_between(
        episodes,
        ma - ms,
        ma + ms,
        alpha=0.18,
        label="Moving avg ±1σ",
    )

    # ---- 꾸미기 (논문용 크게) ----
    ax.set_xlabel("Episode", fontsize=FS_LABEL)
    ax.set_ylabel("Score / Reward", fontsize=FS_LABEL)

    ax.tick_params(labelsize=FS_TICK)
    ax.grid(True, linestyle="--", alpha=0.4)

    if ylim is not None:
        ax.set_ylim(ylim)

    ax.legend(fontsize=FS_LEGEND, loc="best")
    fig.tight_layout()

    # (논문용) PDF/PNG 저장 추천: 텍스트 깨짐 방지 옵션
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=600, bbox_inches="tight")  # 논문용: 600dpi
        print(f"[SAVE] {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, ax


# ================================
# 데이터 로더
# ================================
def load_uav_pose_time_vel(base_dir, uav_idx):
    """
    uav{idx}_pose.txt, uav{idx}_time.txt, uav{idx}_vel.txt 를 읽어서
    길이를 맞춰서 반환.
    pose: (N, 6) [x, y, z, roll, pitch, yaw]
    vel:  (N, 6) 가정 (vx, vy, vz, wx, wy, wz) - 실제 컬럼 인덱스는 아래에서 선택
    """
    base_dir = Path(base_dir)

    pose = np.loadtxt(base_dir / f"uav{uav_idx}_pose.txt")
    t    = np.loadtxt(base_dir / f"uav{uav_idx}_time.txt")
    vel  = np.loadtxt(base_dir / f"uav{uav_idx}_vel.txt")

    n = min(len(pose), len(t), len(vel))
    pose = pose[:n]
    t    = t[:n]
    vel  = vel[:n]

    return t, pose, vel


def load_uav_error(base_dir, uav_idx):
    """
    follower의 tracking error 파일 (uav{i}_error.txt) 로딩
    없으면 None 반환
    """
    base_dir = Path(base_dir)
    err_path = base_dir / f"uav{uav_idx}_error.txt"
    if not err_path.exists():
        return None, None

    err = np.loadtxt(err_path)
    t   = np.loadtxt(base_dir / f"uav{uav_idx}_time.txt")

    n = min(len(err), len(t))
    return t[:n], err[:n]


def load_leader_dubins_path(base_dir):
    """
    uav0_dubins_path.txt (E N, header 1줄 포함) → x_path, y_path
    없으면 (None, None)
    """
    base_dir = Path(base_dir)
    path_file = base_dir / "uav0_dubins_path.txt"
    if not path_file.exists():
        return None, None

    data = np.loadtxt(path_file, skiprows=1)  # 첫 줄 'E N' 헤더 스킵
    return data[:, 0], data[:, 1]

def compute_and_print_error_metrics(t, err, name="Agent", bound1=1.0, bound2=2.0):
    """
    t: 시간 벡터 (1D)
    err: 에러 벡터 (1D)
    name: 출력할 에이전트 이름 문자열

    네가 준 코드의 정량 지표 계산을 그대로 사용하되,
    env.dt 대신 시간 벡터 t에서 dt를 추정해서 사용.
    """
    t = np.asarray(t)
    err = np.asarray(err)

    if len(err) == 0:
        print(f"[WARN] No error data for {name}")
        return

    if len(t) > 1:
        dt = float(np.mean(np.diff(t)))
    else:
        dt = 0.0  # 샘플이 하나뿐이면 적분 지표는 0에 가깝다고 보고 처리

    # 기본 통계량
    mean_err = float(np.mean(err))
    rms_err  = float(np.sqrt(np.mean(err**2)))
    max_err  = float(np.max(err))
    p95_err  = float(np.percentile(err, 95))

    # 허용 오차 내 비율 (예시: 2 m, 3 m)
    bound_1 = float(np.mean(err < bound1) * 100.0)  # [%]
    bound_2 = float(np.mean(err < bound2) * 100.0)  # [%]

    # 누적 오차 (IAE, ISE)
    IAE = float(np.sum(np.abs(err)) * dt)   # Integral of |e| dt
    ISE = float(np.sum(err**2) * dt)        # Integral of e^2 dt

    print(f"===== Path Tracking Error Metrics ({name}) =====")
    print(f"Mean Error      : {mean_err:.3f} m")
    print(f"RMSE            : {rms_err:.3f} m")
    print(f"Max Error       : {max_err:.3f} m")
    print(f"95% Error (P95) : {p95_err:.3f} m")
    print(f"Time within {bound1:.1f} m : {bound_1:.1f} %")
    print(f"Time within {bound2:.1f} m : {bound_2:.1f} %")
    print(f"IAE (∫|e|dt)    : {IAE:.3f} m·s")
    print(f"ISE (∫e² dt)    : {ISE:.3f} m²·s")
    print()  # 한 줄 띄우기
# ================================
# 1) 리더 궤적 + global path + waypoint + 장애물
# ================================
def plot_leader_trajectory_with_path(
    base_dir,
    waypoints,
    obstacles,
    agent_colors,
    hard_zone_margin=None,
):
    """
    - 리더의 실제 궤적
    - uav0_dubins_path.txt (global path)
    - waypoint, 장애물 원 표시
    """
    # 데이터 로딩
    t0, pose0, _ = load_uav_pose_time_vel(base_dir, uav_idx=0)
    x0 = pose0[:, 0]
    y0 = pose0[:, 1]

    path_x, path_y = load_leader_dubins_path(base_dir)

    fig, ax = plt.subplots(figsize=(10, 10))

    # 장애물: (x, y, r) 리스트 가정
    for i, (ox, oy, orad) in enumerate(obstacles):
        lbl_obs = "Obstacle" if i == 0 else None
        circle = patches.Circle(
            (ox, oy), orad,
            edgecolor="black",
            facecolor="gray",
            alpha=0.5,
            label=lbl_obs,
        )
        ax.add_patch(circle)

        # hard zone margin 있을 경우 외곽선 한 번 더
        if hard_zone_margin is not None:
            lbl_hard = "Hard Limit" if i == 0 else None
            hard_circle = patches.Circle(
                (ox, oy),
                orad + hard_zone_margin,
                edgecolor="red",
                facecolor="none",
                linewidth=2,
                linestyle="-",
                label=lbl_hard,
            )
            ax.add_patch(hard_circle)

    # global path (Dubins)
    if path_x is not None:
        ax.plot(
            path_x,
            path_y,
            linestyle="--",
            linewidth=LW_TRAJ,
            color="cyan",
            label="Global Path",
        )

    # waypoint
    if waypoints is not None and len(waypoints) > 0:
        wps = np.asarray(waypoints)
        wx, wy = wps[:, 0], wps[:, 1]
        ax.plot(wx, wy, "ro", markersize=MS_WAYPOINT, label="Waypoints")
        for i in range(len(wps) - 1):
            ax.text(
                wx[i] + 0.5,
                wy[i] + 0.5,
                f"W{i}",
                fontsize=FS_TICK,
                color="red",
            )



    # 리더 궤적
    leader_color = agent_colors[0]
    ax.plot(
        x0,
        y0,
        linewidth=LW_TRAJ,
        color='blue',
        label="Leader",
    )

    ax.set_xlabel("X Position [m]", fontsize=FS_LABEL)
    ax.set_ylabel("Y Position [m]", fontsize=FS_LABEL)
    ax.tick_params(labelsize=FS_TICK)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(fontsize=FS_LEGEND, loc="best")
    fig.tight_layout()
    plt.show()


# ================================
# 2) 팔로워 tracking error (y축 0~10)
# ================================
def plot_follower_tracking_errors(base_dir, num_uav, agent_colors, groups=None):
    """
    follower(uav1~)의 tracking error(t) 그래프를 group 기준으로 나눠서 그림.

    groups 예시:
      - None 이면: 기존처럼 uav1~num_uav-1 전부 한 plot
      - [[1,2,3,4],[5,6]] 이면:
          Figure1: Follower 1~4
          Figure2: Follower 5~6
    y축: [0, 10]
    + 각 팔로워별 정량 지표 출력
    """

    # groups가 None이면 기존 동작(전체 follower 하나로)
    if groups is None:
        groups = [list(range(1, num_uav))]

    # 혹시 tuple, np.array 등 들어오면 리스트화
    groups = [list(g) for g in groups]

    for gi, g in enumerate(groups, start=1):
        fig, ax = plt.subplots(figsize=(10, 6))

        plotted_any = False

        for uav_idx in g:
            if uav_idx <= 0 or uav_idx >= num_uav:
                print(f"[WARN] group[{gi}]에 잘못된 uav index 포함: {uav_idx} (무시)")
                continue

            t, err = load_uav_error(base_dir, uav_idx)
            if t is None or err is None:
                print(f"[WARN] No error data for Follower {uav_idx}")
                continue

            color = agent_colors[uav_idx]
            label = f"Follower {uav_idx}"

            # ==== 정량 지표 출력 ====
            compute_and_print_error_metrics(t, err, name=label)

            # ==== 그래프 ====
            ax.plot(
                t,
                err,
                linewidth=LW_LINE,
                color=color,
                label=label,
            )
            plotted_any = True

        ax.set_xlabel("Time [s]", fontsize=FS_LABEL)
        ax.set_ylabel("Error [m]", fontsize=FS_LABEL)
        ax.set_ylim(0, 10)
        ax.tick_params(labelsize=FS_TICK)
        ax.grid(True, linestyle="--", alpha=0.4)

        # 그룹별 제목(원하면)

        if plotted_any:
            ax.legend(fontsize=FS_LEGEND, loc="best")

        fig.tight_layout()
        plt.show()


# ================================
# 3) 리더 + 팔로워 전체 궤적
#    (색은 2번에서 쓴 follower 색과 동일)
# ================================
def plot_all_trajectories(
    base_dir,
    num_uav,
    waypoints,
    obstacles,
    agent_colors,
    hard_zone_margin=None,
    graph=None,
    snapshot_dt=None,
):
    """
    - 리더 + 모든 팔로워 궤적
    - waypoint, obstacle 같이 표시
    - 각 에이전트는 고정된 색상(agent_colors)을 사용
    - graph(adj matrix)가 주어지면,
      1) 시뮬레이션 시작 위치에서 한 번 무조건 formation edge 그림
      2) snapshot_dt 간격으로 formation edge 추가
    """
    base_dir = Path(base_dir)

    fig, ax = plt.subplots(figsize=(10, 10))

    # -------------------------
    # 장애물
    # -------------------------
    for i, (ox, oy, orad) in enumerate(obstacles):
        lbl_obs = "Obstacle" if i == 0 else None
        circle = patches.Circle(
            (ox, oy), orad,
            edgecolor="black",
            facecolor="gray",
            alpha=0.5,
            label=lbl_obs,
        )
        ax.add_patch(circle)

        if hard_zone_margin is not None:
            lbl_hard = "Hard Limit" if i == 0 else None
            hard_circle = patches.Circle(
                (ox, oy),
                orad + hard_zone_margin,
                edgecolor="red",
                facecolor="none",
                linewidth=2,
                linestyle="-",
                label=lbl_hard,
            )
            ax.add_patch(hard_circle)

    # -------------------------
    # 웨이포인트
    # -------------------------
    if waypoints is not None and len(waypoints) > 0:
        wps = np.asarray(waypoints)
        wx, wy = wps[:, 0], wps[:, 1]
        ax.plot(wx, wy, "ro", markersize=MS_WAYPOINT, label="Waypoints")
        for i in range(len(wps) - 1):
            ax.text(
                wx[i] + 0.5,
                wy[i] + 0.5,
                f"W{i}",
                fontsize=FS_TICK,
                color="red",
            )

    # -------------------------
    # 각 UAV 궤적 로딩 + 플로팅
    # -------------------------
    all_t = []
    all_xy = []

    for uav_idx in range(num_uav):
        t, pose, _ = load_uav_pose_time_vel(base_dir, uav_idx)
        x = pose[:, 0]
        y = pose[:, 1]

        all_t.append(t)
        all_xy.append(np.stack([x, y], axis=1))  # (N, 2)

        color = agent_colors[uav_idx]
        if uav_idx == 0:
            label = "Leader"
        else:
            label = f"Follower {uav_idx}"

        ax.plot(
            x,
            y,
            linewidth=LW_TRAJ,
            color=color,
            label=label,
        )

    # -------------------------
    # (추가) 시작 위치에서 편대 선 무조건 한 번 그림
    # -------------------------
    if graph is not None:
        graph = np.asarray(graph)
        # 각 UAV의 첫 샘플 위치 사용 (시간과 상관없이 index 0)
        positions_0 = []
        for uav_idx in range(num_uav):
            if len(all_xy[uav_idx]) == 0:
                positions_0.append(None)
                continue
            positions_0.append(all_xy[uav_idx][0])  # (x0, y0)

        # legend용 플래그
        first_edge = True

        for i in range(num_uav):
            for j in range(i + 1, num_uav):
                if graph[i, j] == 0:
                    continue
                if positions_0[i] is None or positions_0[j] is None:
                    continue

                x_i, y_i = positions_0[i]
                x_j, y_j = positions_0[j]

                if first_edge:
                    ax.plot(
                        [x_i, x_j],
                        [y_i, y_j],
                        color="k",
                        linewidth=1.8,
                        alpha=0.5,
                    )
                    first_edge = False
                else:
                    ax.plot(
                        [x_i, x_j],
                        [y_i, y_j],
                        color="k",
                        linewidth=1.8,
                        alpha=0.5,
                    )
    else:
        first_edge = False  # 아래 snapshot에서 사용

    # -------------------------
    # snapshot_dt 간격 formation snapshot
    # -------------------------
    if graph is not None and snapshot_dt is not None and snapshot_dt > 0:
        graph = np.asarray(graph)
        # 모든 UAV가 공통으로 커버하는 시간 구간 (min of max t)
        t_max = min(t_arr[-1] for t_arr in all_t if len(t_arr) > 0)
        snapshot_times = np.arange(0.0, t_max + 1e-6, snapshot_dt)

        # 위에서 first_edge를 이미 썼을 수 있으니 그대로 이어서 사용
        for ts in snapshot_times:
            positions_ts = []
            for uav_idx in range(num_uav):
                t_arr = all_t[uav_idx]
                xy_arr = all_xy[uav_idx]
                if len(t_arr) == 0:
                    positions_ts.append(None)
                    continue

                if ts < t_arr[0] or ts > t_arr[-1]:
                    positions_ts.append(None)
                    continue

                k = int(np.argmin(np.abs(t_arr - ts)))
                positions_ts.append(xy_arr[k])

            for i in range(num_uav):
                for j in range(i + 1, num_uav):
                    if graph[i, j] == 0:
                        continue
                    if positions_ts[i] is None or positions_ts[j] is None:
                        continue

                    x_i, y_i = positions_ts[i]
                    x_j, y_j = positions_ts[j]

                    if first_edge:
                        ax.plot(
                            [x_i, x_j],
                            [y_i, y_j],
                            color="k",
                            linewidth=1.5,
                            alpha=0.5,
                        )
                        first_edge = False
                    else:
                        ax.plot(
                            [x_i, x_j],
                            [y_i, y_j],
                            color="k",
                            linewidth=1.5,
                            alpha=0.5,
                        )

    # -------------------------
    # axis / label / legend
    # -------------------------
    ax.set_xlabel("X Position [m]", fontsize=FS_LABEL)
    ax.set_ylabel("Y Position [m]", fontsize=FS_LABEL)
    ax.tick_params(labelsize=FS_TICK)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(fontsize=FS_LEGEND, loc="best")
    fig.tight_layout()
    plt.show()



# ================================
# 4) 각 에이전트 vx, vy, w 속도 그래프 (subplot)
# ================================
def plot_velocity_profiles(base_dir, num_uav, agent_colors,
                           vx_idx=0, vy_idx=1, w_idx=-1):
    """
    - 하나의 Figure에 3개의 subplot (vx, vy, w)
    - 각 subplot에 모든 에이전트의 곡선 표시
    - agent_colors에 따라 색상 고정
    - vx_idx, vy_idx, w_idx 는 uav*_vel.txt의 컬럼 인덱스
      (지금은 예시로 [0,1,5] 가정 → 실제 구조에 맞게 수정 가능)
    """
    fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    ax_vx, ax_vy, ax_w = axs

    for uav_idx in range(num_uav):
        t, _, vel = load_uav_pose_time_vel(base_dir, uav_idx)

        vx = vel[:, vx_idx]
        vy = vel[:, vy_idx]
        w  = vel[:, w_idx]

        color = agent_colors[uav_idx]
        if uav_idx == 0:
            label = "Leader "
        else:
            label = f"Follower {uav_idx}"

        # vx
        ax_vx.plot(
            t,
            vx,
            linewidth=LW_LINE,
            color=color,
            label=label,
        )

        # vy
        ax_vy.plot(
            t,
            vy,
            linewidth=LW_LINE,
            color=color,
        )

        # w
        ax_w.plot(
            t,
            w,
            linewidth=LW_LINE,
            color=color,
        )

    # 라벨/그리드/폰트
    ax_vx.set_ylabel("vx [m/s]", fontsize=FS_LABEL)
    ax_vy.set_ylabel("vy [m/s]", fontsize=FS_LABEL)
    ax_w.set_ylabel("w [rad/s]", fontsize=FS_LABEL)
    ax_w.set_xlabel("Time [s]", fontsize=FS_LABEL)

    for ax in axs:
        ax.tick_params(labelsize=FS_TICK)
        ax.grid(True, linestyle="--", alpha=0.4)

    # legend는 맨 위 subplot에만
    ax_vx.legend(fontsize=FS_LEGEND, loc="best")
    ax_w.set_ylim(-0.25, 0.25)

    fig.tight_layout()
    plt.show()

def plot_leader_tracking_error(base_dir, agent_colors):
    """
    리더(uav0)의 tracking error 그래프.
    y축: [0, 10]
    + 정량 지표 출력
    """
    t, err = load_uav_error(base_dir, uav_idx=0)
    if t is None or err is None:
        print("[WARN] Leader error file (uav0_error.txt) not found.")
        return

    # ==== 정량 지표 출력 ====
    compute_and_print_error_metrics(t, err, name="Leader (uav0)")

    # ==== 그림 ====
    fig, ax = plt.subplots(figsize=(10, 6))

    color = agent_colors[0]  # 리더 색상 고정
    ax.plot(
        t,
        err,
        linewidth=LW_LINE,
        color=color,
        label="Leader",
    )

    ax.set_xlabel("Time [s]", fontsize=FS_LABEL)
    ax.set_ylabel("Error [m]", fontsize=FS_LABEL)
    ax.set_ylim(0, 10)  # 요구사항: 0~10
    ax.tick_params(labelsize=FS_TICK)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(fontsize=FS_LEGEND, loc="best")
    fig.tight_layout()
    plt.show()

def plot_leader_angular_velocity_only(base_dir, agent_colors, w_idx=5, ylim=(-0.25, 0.25)):
    """
    리더(uav0) 각속도(w)만 1개의 Figure에 표시
    - w_idx: uav0_vel.txt에서 각속도 컬럼 인덱스 (기본 5)
    """
    t, _, vel = load_uav_pose_time_vel(base_dir, uav_idx=0)
    w = vel[:, w_idx]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(
        t,
        w,
        linewidth=LW_LINE,
        color=agent_colors[0],
        label="Leader (uav0)",
    )

    ax.set_xlabel("Time [s]", fontsize=FS_LABEL)
    ax.set_ylabel("w [rad/s]", fontsize=FS_LABEL)
    if ylim is not None:
        ax.set_ylim(ylim)

    ax.tick_params(labelsize=FS_TICK)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(fontsize=FS_LEGEND, loc="best")

    fig.tight_layout()
    plt.show()



# ================================
# 메인 실행부 예시
# ================================
if __name__ == "__main__":

    case = 2

    if case == 1:
        # txt 파일들이 있는 폴더
        graph = [[0, 1, 1, 0, 0],[1, 0, 0, 1, 0],[1, 0, 0, 0, 1],[0, 1, 0, 0, 0],[0, 0, 1, 0, 0]]
        group = [[1, 2], [3, 4]]
        graph = np.array([
            [0, 1, 1, 0, 0],
            [1, 0, 0, 1, 0],
            [1, 0, 0, 0, 1],
            [0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0],
        ], dtype=int)

        SNAPSHOT_DT = 20.0  # 20초 간격

        DATA_DIR = r"Z:\Jaewan\개인 연구\강화학습\개인 연구\stable baselines3 항적\v자"  # <- 실제 경로로 바꿔줘

        # uav0 = 리더, uav1~uav6 = 팔로워 → 총 7대
        NUM_UAV = 5

        # waypoint, obstacles 는 네가 직접 넣으면 됨 (예시)
        waypoints = [
        [0, 0],
        [30, 0],
        [40, 20],
        [0, 40],
        [0, 0]
    ]
        # (x, y, radius)
        obstacles = [
            (20, 29.75, 2),
        ]

    else:
        graph = np.array([
            [0, 1, 1, 1, 1, 0, 0],
            [1, 0, 0, 0, 0, 1, 0],
            [1, 0, 0, 0, 0, 1, 0],
            [1, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 1],
            [0, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 1, 0, 0],
        ], dtype=int)

        group = [[1, 2, 3, 4], [5, 6]]


        SNAPSHOT_DT = 20.0  # 20초 간격
        # txt 파일들이 있는 폴더
        DATA_DIR = r"Z:\Jaewan\개인 연구\강화학습\개인 연구\stable baselines3 항적\육각형1"  # <- 실제 경로로 바꿔줘

        # uav0 = 리더, uav1~uav6 = 팔로워 → 총 7대
        NUM_UAV = 7

        # waypoint, obstacles 는 네가 직접 넣으면 됨 (예시)
        waypoints = [
            [0.0, 0.0],
            [20.0, -18.0],
            [20.0, -35.0],
            [40.0, -35.0],
            [40.0, 0.0],
            [0.0, -0.0],
        ]
        # (x, y, radius)
        obstacles = [
            (41.0, -18.0, 1.0),
        ]

    HARD_ZONE_MARGIN = 2.0  # 필요하면 2.0 같은 값 넣으면 됨

    # 에이전트별 색상 고정 (matplotlib 기본 color cycle 활용)
    base_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    agent_colors = {i: base_colors[i % len(base_colors)] for i in range(NUM_UAV)}

    # 1) 리더 궤적 + Dubins path + waypoint + obstacle
    plot_leader_trajectory_with_path(
        DATA_DIR,
        waypoints,
        obstacles,
        agent_colors,
        hard_zone_margin=HARD_ZONE_MARGIN,
    )

    # 2-1) 리더 tracking error
    plot_leader_tracking_error(
        DATA_DIR,
        agent_colors,
    )

    plot_follower_tracking_errors(DATA_DIR, NUM_UAV, agent_colors, groups=group)

    # 3) 전체 궤적 (리더 + 팔로워, 색 고정)
    plot_all_trajectories(
        DATA_DIR,
        NUM_UAV,
        waypoints,
        obstacles,
        agent_colors,
        hard_zone_margin=HARD_ZONE_MARGIN,
        graph=graph,
        snapshot_dt=SNAPSHOT_DT,
    )

    # 4) vx, vy, w 속도 프로파일 (subplot)
    #   ※ uav*_vel.txt 의 컬럼 구조에 맞게 인덱스 조정해도 됨
    plot_velocity_profiles(
        DATA_DIR,
        NUM_UAV,
        agent_colors,
        vx_idx=0,
        vy_idx=1,
        w_idx=5,   # 예시: 마지막 컬럼을 w로 사용 (필요하면 변경)
    )

    DATA_DIR1 = r"Z:\Jaewan\개인 연구\강화학습\개인 연구\stable baselines3 항적"  # <- 실제 경로로 바꿔줘

    plot_episode_reward_curve(
        DATA_DIR1,
        algo="SAC",
        window=50,
        save_path=Path(DATA_DIR1) / "learning_curve_episode_reward.png",
        show=True,          # 저장만 하고 싶으면 False
        ylim=None,          # 예: (-300, 250) 처럼 고정하고 싶으면 넣기
    )

    plot_leader_angular_velocity_only(
        DATA_DIR,
        agent_colors,
        w_idx=5,  # 필요하면 수정
        ylim=(-0.25, 0.25)  # 필요 없으면 None
    )
