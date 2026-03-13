import math
import os
import time

import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import SAC

from simple_maze_grid import SimpleMazeGrid


MAX_STEPS = 3000
FPS = 30


# -----------------------------
# Scenario builders
# -----------------------------
def make_general_env(render_option=True):
    """일반 평가용: 기존과 같은 random obstacle 1개 시나리오."""
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
        sensor_range=25.0,
        use_lidar_edges=True,
        lidar_num_rays=360,
        lidar_fov=math.pi * 2,
        reference_L=15.0,
    )


def make_two_obstacle_corridor_env(render_option=True):
    """
    장애물 2개 사이를 지나가도록 만든 고정 평가 시나리오.
    - start -> goal 이 거의 직선으로 이어지도록 두고
    - 그 직선 근처 위/아래에 장애물 2개를 배치
    - hard_zone(2m) 안으로 path가 들어가지 않도록 충분한 간격 확보
    """
    corridor_spec = (
        [120.0, 200.0, 0.0],
        [280.0, 200.0, 0.0],
        [
            [200.0, 212.0, 8.0],
            [200.0, 188.0, 8.0],
        ],
    )

    return SimpleMazeGrid(
        global_map_size=400,
        local_map_size=125,
        v=5.0,
        w=[-0.46, 0.46],
        dt=0.1,
        render_option=render_option,
        random_seed=None,
        spec=corridor_spec,
        obstacle_count=0,  # spec로 직접 넣으므로 추가 랜덤 장애물 없음
        obstacle_min_radius=2.0,
        obstacle_max_radius=10.0,
        sensor_range=25.0,
        use_lidar_edges=True,
        lidar_num_rays=360,
        lidar_fov=math.pi * 2,
        reference_L=15.0,
    )


# -----------------------------
# Episode runner
# -----------------------------
def run_episode(model, env, scenario_name, reset_seed=None, max_steps=MAX_STEPS, fps=FPS):
    obs, info = env.reset(seed=reset_seed)
    total_reward = 0.0

    print("\n" + "=" * 70)
    print(f"[Scenario] {scenario_name}")
    print("=" * 70)

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

    # pygame 창 닫고, 그 다음 각속도 그래프 창 표시
    time.sleep(0.8)
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
    }


# -----------------------------
# Main entry
# -----------------------------
def run_both_scenarios(model_path):
    if not os.path.exists(model_path) and not os.path.exists(model_path + ".zip"):
        print(f"[ERROR] 모델 파일을 찾을 수 없습니다: {model_path}")
        return

    print(f"Loading model from: {model_path}")
    model = SAC.load(model_path)

    results = []

    # 1) 일반 시나리오
    general_env = make_general_env(render_option=True)
    results.append(
        run_episode(
            model=model,
            env=general_env,
            scenario_name="1. 일반 모델 시나리오",
            reset_seed=7,
        )
    )

    # 2) 장애물 2개 사이 통과 시나리오
    corridor_env = make_two_obstacle_corridor_env(render_option=True)
    results.append(
        run_episode(
            model=model,
            env=corridor_env,
            scenario_name="2. 장애물 2개 사이 통과 시나리오",
            reset_seed=0,
        )
    )

    print("\n" + "#" * 70)
    print("최종 요약")
    print("#" * 70)
    for item in results:
        print(
            f"{item['scenario']}: "
            f"{'Success' if item['success'] else 'Fail'} | "
            f"time={item['time_sec']:.2f}s | "
            f"steps={item['steps']} | "
            f"reward={item['total_reward']:.2f}"
        )

if __name__ == "__main__":
    model_path = "./sac_maze_checkpoint_130000.zip"
    run_both_scenarios(model_path)
