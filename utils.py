# utils/angle.py
import numpy as np

def angle_mod(angle, zero_2_2pi=False):
    """
    각도 wrap 함수
    - zero_2_2pi = False: [-pi, pi) 범위로 변환
    - zero_2_2pi = True : [0, 2*pi) 범위로 변환
    """
    if zero_2_2pi:
        return np.mod(angle, 2.0 * np.pi)
    else:
        a = np.mod(angle + np.pi, 2.0 * np.pi) - np.pi
        # np.mod 때문에 pi 근처에서 약간의 수치오차가 생길 수 있어 보정
        if isinstance(a, float) or np.isscalar(a):
            if a == -np.pi:
                a = np.pi
        else:
            a[a == -np.pi] = np.pi
        return a


def rot_mat_2d(theta):
    """
    2D 회전행렬 R(theta) 생성
    [ cosθ  -sinθ ]
    [ sinθ   cosθ ]
    """
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([[c, -s],
                     [s,  c]])
