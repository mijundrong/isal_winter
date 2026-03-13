import math
import os
import time

from stable_baselines3 import SAC

from simple_maze_grid_goal50 import SimpleMazeGrid


MAX_STEPS = 3000
FPS = 30
NUM_RANDOM_EPISODES = 5
BASE_SEED = 0


def make_train_like_env(render_option=True):
    """train_sac_goal50.py와 동일한 맵 생성 매커니즘을 쓰는 평가용 환경."""
    return SimpleMazeGrid(
        global_map_size=400,
        local_map_size=125,
        v=5.0,
        w=[-0.46, 0.46],
        dt=0.1,
        render_option=render_option,
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

def make_env_with_obstacle_count(obstacle_count, render_option=True):
    return SimpleMazeGrid(
        global_map_size=400,
        local_map_size=125,
        v=5.0,
        w=[-0.46, 0.46],
        dt=0.1,
        render_option=render_option,
        random_seed=None,
        spec=None,
        obstacle_count=obstacle_count,
        obstacle_min_radius=2.0,
        obstacle_max_radius=10.0,
        hard_zone=2.0,
        safety_zone=4.0,
        enable_two_obstacle_corridor=False,
        sensor_range=25.0,
        use_lidar_edges=True,
        lidar_num_rays=360,
        lidar_fov=math.pi * 2,
        reference_L=15.0,
        goal_obstacle_min_dist=50.0,
    )

def run_episode(model, env, scenario_name, reset_seed=None, max_steps=MAX_STEPS, fps=FPS):
    obs, info = env.reset(seed=reset_seed)
    total_reward = 0.0

    print("\n" + "=" * 70)
    print(f"[Scenario] {scenario_name}")
    print("=" * 70)
    print(f"seed={reset_seed}, mode={env.current_episode_mode}, obstacles={env.obstacles}")

    for step_idx in range(max_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        env.render(fps=fps)
        total_reward += float(reward)

        if terminated or truncated:
            print(f"Episode Finished. Result: {'Success' if env.goal else 'Fail/Collision'}")
            break

    elapsed_time = env.steps * env.dt
    success = bool(env.goal)

    print(f"도달 시간: {elapsed_time:.2f} sec")
    print(f"Total Reward: {total_reward:.2f}")
    print(f"총 step 수: {env.steps}")

    time.sleep(0.5)
    env.close()

    env.plot_w_history(
        show=True,
        save_path=None,
        title=f"{scenario_name} - Time vs Angular Velocity (Yaw Rate)",
    )

    return {
        "scenario": scenario_name,
        "success": success,
        "steps": int(env.steps),
        "time_sec": float(elapsed_time),
        "total_reward": float(total_reward),
        "seed": reset_seed,
        "mode": env.current_episode_mode,
    }


def run_random_scenarios(model_path, num_episodes=NUM_RANDOM_EPISODES, base_seed=BASE_SEED):
    if not os.path.exists(model_path) and not os.path.exists(model_path + ".zip"):
        print(f"[ERROR] 모델 파일을 찾을 수 없습니다: {model_path}")
        return

    print(f"Loading model from: {model_path}")
    model = SAC.load(model_path)

    results = []
    for ep_idx in range(num_episodes):
        env = make_train_like_env(render_option=True)
        seed = base_seed + ep_idx
        results.append(
            run_episode(
                model=model,
                env=env,
                scenario_name=f"Random map episode {ep_idx + 1}",
                reset_seed=seed,
            )
        )

    print("\n" + "#" * 70)
    print("최종 요약")
    print("#" * 70)
    for item in results:
        print(
            f"{item['scenario']}: "
            f"seed={item['seed']} | mode={item['mode']} | "
            f"{'Success' if item['success'] else 'Fail'} | "
            f"time={item['time_sec']:.2f}s | "
            f"steps={item['steps']} | "
            f"reward={item['total_reward']:.2f}"
        )


if __name__ == "__main__":
    model_path = "./sac_maze_checkpoint_130000.zip"

    obstacle_count = int(input("obstacle_count 입력 (1 또는 2): ").strip())
    num_cases = int(input("몇 개 seed를 돌릴지 입력: ").strip())
    base_seed = int(time.time()) % 100000
    print(f"자동 생성된 base_seed: {base_seed}")

    model = SAC.load(model_path)
    results = []

    for i in range(num_cases):
        env = make_env_with_obstacle_count(
            obstacle_count=obstacle_count,
            render_option=True,
        )
        seed = base_seed + i
        results.append(
            run_episode(
                model=model,
                env=env,
                scenario_name=f"obs={obstacle_count}, seed={seed}",
                reset_seed=seed,
            )
        )
