import time
import math
import os
import numpy as np                  # 데이터 처리를 위해 추가
import matplotlib.pyplot as plt     # 그래프 출력을 위해 추가
from stable_baselines3 import SAC
from simple_maze_grid import SimpleMazeGrid


def make_eval_env():
    # 평가용 시나리오 및 환경 설정
    env = SimpleMazeGrid(
        global_map_size= 400,
        local_map_size=125,
        v=5.0,
        w=[-0.46, 0.46],
        dt=0.1,
        render_option=True,  # 시각화 ON
        random_seed=None,
        spec=None,
        obstacle_count=1,
        obstacle_min_radius=2.0,
        obstacle_max_radius=10.0,
        sensor_range=25.0,  
        use_lidar_edges=True,
        lidar_num_rays=360,
        lidar_fov=math.pi * 2,
        reference_L=15.0
    )
    return env


def run_once(model_path):
    # 경로 확인
    if not os.path.exists(model_path) and not os.path.exists(model_path + ".zip"):
        print(f"[ERROR] 모델 파일을 찾을 수 없습니다: {model_path}")
        return

    print(f"Loading model from: {model_path}")

    # 모델 로드
    model = SAC.load(model_path)

    env = make_eval_env()
    obs, info = env.reset()

    total_reward = 0.0

    # 시뮬레이션 루프
    for i in range(3000):  # max_steps
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        env.render(fps=30)

        total_reward += reward

        if terminated or truncated:
            print(f"Episode Finished. Result: {'Success' if env.goal else 'Fail/Collision'}")
            break

    print(f"도달 시간: {env.steps * env.dt:.2f} sec")
    print(f"Total Reward: {total_reward:.2f}")

    # ========================================================
    # [수정됨] x축: 시간, y축: 각속도(Yaw Rate) 그래프 그리기
    # ========================================================
    if len(env.time_table) > 0 and len(env.w_table) > 0:
        # 시간 데이터
        times = np.array(env.time_table)
        
        # 각속도 데이터 (w_table)
        yaw_rates = np.array(env.w_table)

        plt.figure(figsize=(10, 5))
        plt.plot(times, yaw_rates, label='Yaw Rate (w)', color='red')
        plt.xlabel("Time [sec]")
        plt.ylabel("Yaw Rate [rad/s]")
        plt.title("Time vs Angular Velocity (Yaw Rate)")
        plt.grid(True)
        plt.legend()
        
        # 그래프 창을 띄웁니다 (닫아야 프로그램이 종료됨)
        plt.show()
    # ========================================================

    # 종료 후 잠시 대기 (결과 화면 확인용)
    time.sleep(1.0)
    env.close()


if __name__ == "__main__":
    # EvalCallback으로 저장된 Best Model 경로 지정
    # model_path = "./logs/best_model/best_model"
    model_path = "./sac_maze_checkpoint_130000.zip"
    run_once(model_path)