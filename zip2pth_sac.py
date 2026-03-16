import torch
import torch.nn as nn
from stable_baselines3 import SAC
from pathlib import Path


class MazeSACActor(nn.Module):
    """
    SB3 SAC MultiInputPolicy의 actor 부분만 그대로 복제한 네트워크
    - obs dict:
      - "err_hist": (B, 20)
      - "w_hist"  : (B, 10)
      - "obstacle_pos": (B, 1, 100, 100)  # 0~255
    """
    def __init__(self, err_dim=20, w_dim=10, img_h=100, img_w=100, action_dim=1):
        super().__init__()

        # NatureCNN과 동일
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        # Conv 출력: 64 x 9 x 9 = 5184
        self.cnn_linear = nn.Sequential(
            nn.Linear(5184, 256),
            nn.ReLU(),
        )

        vec_dim = err_dim + w_dim  # 20 + 10 = 30
        feat_dim = 256 + vec_dim   # 256 + 30 = 286

        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )

        self.mu = nn.Linear(256, action_dim)
        self.log_std = nn.Linear(256, action_dim)

    def forward(self, obs_dict):
        """
        obs_dict:
          - "obstacle_pos": torch.uint8 or float, (B,1,H,W)
          - "err_hist"    : (B,20)
          - "w_hist"      : (B,10)
        """
        img = obs_dict["obstacle_pos"].float() / 255.0   # (B,1,100,100)
        err = obs_dict["err_hist"].float()               # (B,20)
        w_hist = obs_dict["w_hist"].float()              # (B,10)

        img_feat = self.cnn(img)          # (B,5184)
        img_feat = self.cnn_linear(img_feat)  # (B,256)

        # ★ SB3 CombinedExtractor와 동일한 순서: [err, obs_feat, w_hist]
        feat = torch.cat([err, img_feat, w_hist], dim=-1)  # (B,286)

        h = self.mlp(feat)           # (B,256)
        mu = self.mu(h)              # (B,1)
        log_std = self.log_std(h)    # (B,1)
        return mu, log_std


def convert_sb3_sac_to_ros_pth(zip_path: str, out_pth: str):
    zip_path = Path(zip_path)
    out_pth = Path(out_pth)

    # 1) SAC 모델 로드
    model = SAC.load(str(zip_path))
    sd_sb3 = model.policy.state_dict()

    print("=== SB3 policy ===")
    print(model.policy)

    # 2) ROS용 actor 네트워크 생성
    ros_actor = MazeSACActor(err_dim=20, w_dim=10, img_h=100, img_w=100, action_dim=1)
    sd_ros = ros_actor.state_dict()

    # 3) 키 매핑 (actor 기준)
    mapping = {}

    # 3-1) CNN conv layers (0,2,4만 Conv2d)
    conv_indices = [0, 2, 4]
    for i in conv_indices:
        mapping[f"cnn.{i}.weight"] = f"actor.features_extractor.extractors.obstacle_pos.cnn.{i}.weight"
        mapping[f"cnn.{i}.bias"]   = f"actor.features_extractor.extractors.obstacle_pos.cnn.{i}.bias"

    # 3-2) CNN Linear
    mapping["cnn_linear.0.weight"] = "actor.features_extractor.extractors.obstacle_pos.linear.0.weight"
    mapping["cnn_linear.0.bias"]   = "actor.features_extractor.extractors.obstacle_pos.linear.0.bias"

    # 3-3) MLP (latent_pi)
    mapping["mlp.0.weight"] = "actor.latent_pi.0.weight"
    mapping["mlp.0.bias"]   = "actor.latent_pi.0.bias"
    mapping["mlp.2.weight"] = "actor.latent_pi.2.weight"
    mapping["mlp.2.bias"]   = "actor.latent_pi.2.bias"

    # 3-4) mu, log_std
    mapping["mu.weight"]      = "actor.mu.weight"
    mapping["mu.bias"]        = "actor.mu.bias"
    mapping["log_std.weight"] = "actor.log_std.weight"
    mapping["log_std.bias"]   = "actor.log_std.bias"

    # 4) SB3 → ROS state_dict 복사
    for k_ros, k_sb3 in mapping.items():
        if k_sb3 not in sd_sb3:
            raise KeyError(f"SB3 state_dict에 {k_sb3} 없음 (매핑 확인 필요)")
        if sd_ros[k_ros].shape != sd_sb3[k_sb3].shape:
            raise ValueError(
                f"shape mismatch for {k_ros}: ros={sd_ros[k_ros].shape}, sb3={sd_sb3[k_sb3].shape}"
            )
        sd_ros[k_ros] = sd_sb3[k_sb3].clone()

    # 5) 로드 테스트
    ros_actor.load_state_dict(sd_ros)

    # 6) 저장
    torch.save(sd_ros, out_pth)
    print(f"[OK] Converted SB3 SAC actor → ROS MazeSACActor pth: {out_pth}")


if __name__ == "__main__":
    ZIP_PATH = "best_model.zip"   # SAC 학습 결과 .zip
    PTH_PATH = "sac_model.pth"
    convert_sb3_sac_to_ros_pth(ZIP_PATH, PTH_PATH)
