import math
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from simple_maze_grid_goal50 import SimpleMazeGrid


# =========================
# 1. 에피소드 리워드 저장 콜백
# =========================
class EpisodeRewardCallback(BaseCallback):
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.episode_num = 0
        self.episode_rewards = []  # 에피소드별 리워드 저장

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                self.episode_num += 1
                ep_reward = info["episode"]["r"]
                ep_length = info["episode"]["l"]
                self.episode_rewards.append(ep_reward)

                # 로그가 너무 많으면 주석 처리 하셔도 됩니다.
                print(
                    f"[Episode {self.episode_num}] "
                    f"Reward = {ep_reward:.2f}, Length = {ep_length}"
                )
        return True


# =========================
# 2. Env factory
# =========================
def make_env():
    def _init():
        env = SimpleMazeGrid(
            global_map_size=400,
            local_map_size=125,
            v=5.0,
            w=[-0.46, 0.46],
            dt=0.1,
            render_option=False,
            random_seed=None,
            spec=None,

            obstacle_count=1,
            obstacle_min_radius=2.0,
            obstacle_max_radius=10.0,

            hard_zone=2.0,
            safety_zone=4.0,

            enable_two_obstacle_corridor=True,
            corridor_episode_interval=5,
            corridor_path_clearance=(1.0, 2.0),
            corridor_sample_ratio=(0.35, 0.65),
            corridor_anchor_jitter=8.0,
            path_obstacle_clearance=(0.5, 2.0),
            path_obstacle_sample_ratio=(0.20, 0.80),
            path_obstacle_anchor_jitter=10.0,

            sensor_range=25.0,
            use_lidar_edges=True,
            lidar_num_rays=360,
            lidar_fov=math.pi * 2,
            reference_L=15.0,
            goal_obstacle_min_dist=50.0,
        )
        env = Monitor(env)
        return env

    return _init


# =========================
# 3. 데이터 처리 및 플롯 유틸 함수
# =========================
def rolling_mean_std(x: np.ndarray, window: int):
    n = x.shape[0]
    w = max(1, min(window, n))
    cumsum = np.cumsum(np.insert(x, 0, 0.0))
    mean = (cumsum[w:] - cumsum[:-w]) / w
    std = np.empty_like(mean)
    for i in range(mean.shape[0]):
        seg = x[i: i + w]
        std[i] = seg.std(ddof=0)
    pad = np.full(w - 1, np.nan)
    return np.concatenate([pad, mean]), np.concatenate([pad, std])


def prepare_data(arr: np.ndarray, window: int):
    a = np.asarray(arr)
    if a.ndim == 1:
        a = a[None, :]
    elif a.ndim > 2:
        a = a.reshape((-1, a.shape[-1]))
    lengths = [len(r) for r in a]
    if len(set(lengths)) > 1:
        T = min(lengths)
        a = np.array([r[:T] for r in a])
    mean = a.mean(axis=0)
    std = a.std(axis=0, ddof=0)
    smean, sstd = rolling_mean_std(mean, window)
    return mean, std, smean, sstd


# [수정] 그래프 그리는 로직을 함수로 분리
def save_plot(episode_rewards, save_path, window=50, xtick_step=1000):
    """현재까지 수집된 리워드로 그래프를 그리고 저장하는 함수"""
    if len(episode_rewards) == 0:
        return

    plt.figure(figsize=(10, 5.5))
    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["axes.unicode_minus"] = False

    mean, std, smean, sstd = prepare_data(episode_rewards, window)
    x = np.arange(mean.shape[0])

    plt.fill_between(x, mean - std, mean + std, alpha=0.15, label="Raw ±1σ")
    plt.plot(x, smean, linewidth=2.0, label=f"Moving avg (win={window})")
    plt.fill_between(x, smean - sstd, smean + sstd, alpha=0.25, label="Moving avg ±1σ")

    plt.xlabel("Episode")
    plt.ylabel("Score / Reward")
    plt.title("Learning Curve (Episode Reward, SAC)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="best")

    if xtick_step > 0 and len(x) > 0:
        # X축 눈금 설정 (데이터 길이에 맞춰 조정)
        step = xtick_step if len(x) < 10000 else xtick_step * 5
        plt.xticks(np.arange(0, len(x) + 1, step))

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()  # 메모리 누수 방지를 위해 닫기 (show 대신 close 사용)
    print(f"Plot saved: {save_path}")


# =========================
# 4. 학습 + 저장 + 플롯 (SAC)
# =========================
if __name__ == "__main__":
    # ---------- 학습 설정 ----------
    num_envs = 4
    vec_env = DummyVecEnv([make_env() for _ in range(num_envs)])
    eval_env = DummyVecEnv([make_env()])

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./logs/best_model",
        log_path="./logs/results",
        eval_freq=2000,
        deterministic=True,
        render=False,
        verbose=1
    )

    model = SAC(
        "MultiInputPolicy",
        vec_env,
        learning_rate=3e-4,
        buffer_size=100_000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        learning_starts=10_000,
        ent_coef="auto",
        target_update_interval=1,
        verbose=1,
        tensorboard_log="sac_maze_tensorboard",
    )

    reward_callback = EpisodeRewardCallback()

    total_timesteps = 1_000_000
    chunk = 10_000
    timesteps = 0

    print("Training Start...")

    # [수정] 반복문 내에서 플롯 저장 로직 추가
    while timesteps < total_timesteps:
        model.learn(
            total_timesteps=chunk,
            reset_num_timesteps=False,
            callback=[reward_callback, eval_callback],
        )
        timesteps += chunk

        # 1. 모델 체크포인트 저장
        model.save(f"sac_maze_checkpoint_{timesteps}")

        # 2. 리워드 데이터 저장 (Numpy)
        current_rewards = np.array(reward_callback.episode_rewards, dtype=float)
        np.save("episode_rewards_current.npy", current_rewards)

        # 3. [핵심] 현재까지의 결과로 그래프 저장 (중간 확인용)
        # 매번 덮어쓰기 하려면 파일명을 고정(e.g., "current_plot.png")
        # 히스토리를 남기려면 파일명에 timesteps 포함
        plot_filename = f"episode_rewards_plot_{timesteps}.png"
        save_plot(current_rewards, plot_filename, window=50)

        print(f"Current steps: {timesteps} completed. Checkpoint and plot saved.")

    # 최종 모델 저장
    model.save("sac_final")
    print("Training Finished.")
    print("Best model is saved at './logs/best_model/best_model.zip'")

    vec_env.close()
    eval_env.close()