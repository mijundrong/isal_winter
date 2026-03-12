import math
import os
import time

import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import SAC

from simple_maze_grid import SimpleMazeGrid


def make_eval_env():
    env = SimpleMazeGrid(
        global_map_size=400,
        local_map_size=125,
        v=5.0,
        w=[-0.46, 0.46],
        dt=0.1,
        render_option=True,
        random_seed=None,
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
    return env


def save_yaw_rate_plot(env, save_path="yaw_rate_history.png"):
    if len(env.time_table) == 0 or len(env.w_table) == 0:
        return None

    times = np.asarray(env.time_table, dtype=float)
    yaw_rates = np.asarray(env.w_table, dtype=float)

    plt.figure(figsize=(8, 4.5))
    plt.plot(times, yaw_rates, linewidth=2.0, label="Yaw rate (w)")
    plt.xlabel("Time [s]")
    plt.ylabel("Yaw rate [rad/s]")
    plt.title("Time vs Angular Velocity")
    plt.grid(True, linestyle=":", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    return save_path


def run_once(model_path):
    if not os.path.exists(model_path) and not os.path.exists(model_path + ".zip"):
        print(f"[ERROR] 모델 파일을 찾을 수 없습니다: {model_path}")
        return

    print(f"Loading model from: {model_path}")
    model = SAC.load(model_path)

    env = make_eval_env()
    obs, info = env.reset()
    total_reward = 0.0

    for i in range(3000):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        env.render(fps=30)
        total_reward += reward

        if terminated or truncated:
            print(f"Episode Finished. Result: {'Success' if env.goal else 'Fail/Collision'}")
            break

    print(f"도달 시간: {env.steps * env.dt:.2f} sec")
    print(f"Total Reward: {total_reward:.2f}")
    print(f"Obstacle layout: {env.latest_layout_name}")

    paper_fig_path = env.save_publication_figure("paper_figure_run.png", dpi=300, show_local=True)
    yaw_fig_path = save_yaw_rate_plot(env, "yaw_rate_history.png")

    if paper_fig_path is not None:
        print(f"Saved paper figure: {paper_fig_path}")
    if yaw_fig_path is not None:
        print(f"Saved yaw-rate figure: {yaw_fig_path}")

    time.sleep(1.0)
    env.close()


if __name__ == "__main__":
    model_path = "./sac_maze_checkpoint_130000.zip"
    run_once(model_path)
