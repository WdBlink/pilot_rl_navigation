import numpy as np
import pygame
import gymnasium as gym
from gymnasium import spaces


class DroneNavigationEnv(gym.Env):
    """无人机导航环境"""

    def __init__(self):
        super(DroneNavigationEnv, self).__init__()

        # 动作空间：推力 [0,1]，俯仰角 [-30,30]，偏航角 [-180,180]
        self.action_space = spaces.Box(
            low=np.array([0, -30, -180]),
            high=np.array([1, 30, 180]),
            dtype=np.float32
        )

        # 观测空间：当前坐标 (x,y) + 目标方向向量 (dx, dy) + 当前速度向量 (vx, vy)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(6,),
            dtype=np.float32
        )

        # 物理参数
        self.max_speed = 0.02  # 最大速度（度/秒）
        self.dt = 1.0          # 时间步长
        self.wind_std = 0.005  # 风速标准差
        self.target_threshold = 0.001  # 到达判定阈值
        self.max_steps = 900  # 让推理能运行30秒 (fps≈30)

        # 可视化参数
        self.screen_size = 800
        self.scale = 5000  # 1度=5000像素
        self.screen = None
        self.history = []  # 记录历史轨迹

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # 初始化起点和目标点（地理坐标）
        self.start = np.random.uniform(-1.0, 1.0, 2)  # 扩大训练范围
        self.target = np.random.uniform(-1.0, 1.0, 2)

        self.state = np.concatenate([self.start, self.target - self.start, np.array([0.0, 0.0])]).astype(np.float32)
        self.prev_state = self.state.copy()
        self.history = [self.start.copy()]
        self.steps = 0

        return self.state.copy(), {}

    def step(self, action):
        thrust, pitch, yaw = action

        # 计算基础速度
        base_speed = thrust * self.max_speed * np.cos(np.deg2rad(pitch))

        # 计算运动分量
        dx = base_speed * np.cos(np.deg2rad(yaw)) * self.dt
        dy = base_speed * np.sin(np.deg2rad(yaw)) * self.dt

        # 添加风扰动
        dx += np.random.normal(0, self.wind_std)
        dy += np.random.normal(0, self.wind_std)

        # 更新位置
        new_pos = self.state[:2] + np.array([dx, dy])
        self.history.append(new_pos)  # 记录轨迹
        self.state[:2] = new_pos

        # 计算奖励
        prev_distance = np.linalg.norm(self.prev_state[:2] - self.prev_state[2:4])
        distance = np.linalg.norm(self.state[:2] - self.state[2:4])
        progress = prev_distance - distance  # 计算朝目标的进展

        reward = progress * 50  # 鼓励前进
        reward -= distance * 10  # 距离惩罚
        self.prev_state = self.state.copy()

        done = False
        if distance < self.target_threshold:
            reward += 100
            done = True

        self.steps += 1
        if self.steps >= self.max_steps:
            done = True

        # ✅ **修正目标方向计算**
        self.state[2:4] = self.state[2:4] - self.state[:2]  # 计算目标方向向量
        self.state[4:6] = np.array([dx, dy])  # 速度向量

        return self.state.copy(), reward, done, False, {}

    def render(self):
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((self.screen_size, self.screen_size))

        def transform(pos):
            return int(pos[0] * self.scale + self.screen_size / 2), int(-pos[1] * self.scale + self.screen_size / 2)

        self.screen.fill((255, 255, 255))

        # 绘制目标点（五角星）
        target_pos = transform(self.state[2:])
        pygame.draw.polygon(self.screen, (255, 0, 0), [
            (target_pos[0], target_pos[1] - 10),
            (target_pos[0] + 6, target_pos[1] - 3),
            (target_pos[0] + 10, target_pos[1] + 4),
            (target_pos[0] - 10, target_pos[1] + 4),
            (target_pos[0] - 6, target_pos[1] - 3)
        ])

        # 绘制历史轨迹
        if len(self.history) > 1:
            pygame.draw.lines(self.screen, (0, 255, 0), False, [transform(p) for p in self.history], 2)

        # 绘制无人机（小飞机）
        drone_pos = transform(self.state[:2])
        pygame.draw.polygon(self.screen, (0, 0, 255), [
            (drone_pos[0], drone_pos[1] - 7),
            (drone_pos[0] - 5, drone_pos[1] + 7),
            (drone_pos[0] + 5, drone_pos[1] + 7)
        ])

        pygame.display.flip()

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None