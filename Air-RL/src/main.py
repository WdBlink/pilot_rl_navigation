import numpy as np
import pygame
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env


class DroneNavigationEnv(gym.Env):
    """基于真实地理坐标的无人机导航环境"""

    def __init__(self, start_pos=None, target_pos=None):
        super(DroneNavigationEnv, self).__init__()

        # 假设任务范围在省级（例如 30°N~31°N, 120°E~121°E）
        self.geo_bounds = {"lat_min": 30.0, "lat_max": 31.0,
                           "lon_min": 120.0, "lon_max": 121.0}

        # 手动设置初始位置和目标位置
        self.start_pos = np.array(start_pos) if start_pos else np.random.uniform(
            [self.geo_bounds["lat_min"], self.geo_bounds["lon_min"]],
            [self.geo_bounds["lat_max"], self.geo_bounds["lon_max"]]
        )
        self.target_pos = np.array(target_pos) if target_pos else np.random.uniform(
            [self.geo_bounds["lat_min"], self.geo_bounds["lon_min"]],
            [self.geo_bounds["lat_max"], self.geo_bounds["lon_max"]]
        )

        # 动作空间
        self.action_space = spaces.Box(low=np.array([-1, -1, -1]), high=np.array([1, 1, 1]), dtype=np.float32)

        # 状态空间 (6维)：当前位置(x, y) + 目标位置(x, y) + 速度(vx, vy)
        # 状态空间 (7维)：当前位置(x, y) + 目标位置(x, y) + 速度(vx, vy) + 朝向(yaw)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32)

        # 物理参数
        self.max_speed = 0.0005
        self.dt = 0.1
        self.max_steps = 30000
        self.wind_std = 0.0001
        self.target_threshold = 0.00005

        # 轨迹记录
        self.history = []

        # 可视化窗口
        self.screen_size = 1000
        pygame.init()
        self.screen = pygame.display.set_mode((self.screen_size, self.screen_size))

        # 计算缩放比例
        lat_range = self.geo_bounds["lat_max"] - self.geo_bounds["lat_min"]
        lon_range = self.geo_bounds["lon_max"] - self.geo_bounds["lon_min"]
        self.scale = self.screen_size / max(lat_range, lon_range)  # 取较大范围作为基准

    def _calculate_new_position(self, action):
        """计算新位置（确保速度方向与yaw一致）"""
        thrust = (action[0] + 1) / 2
        yaw_delta = action[2] * 30  # 增大转向幅度

        # 更新yaw
        self.yaw += yaw_delta
        self.yaw %= 360  # 确保角度在0~360之间

        # 加速度方向基于当前yaw
        acceleration = thrust * self.max_speed * np.array([
            np.cos(np.deg2rad(self.yaw)),
            np.sin(np.deg2rad(self.yaw))
        ])

        # 更新速度
        self.velocity += acceleration * self.dt
        # 速度钳制
        if np.linalg.norm(self.velocity) > self.max_speed:
            self.velocity = self.velocity / np.linalg.norm(self.velocity) * self.max_speed

        # 计算位移
        dx = self.velocity[0] * self.dt + np.random.normal(0, self.wind_std)
        dy = self.velocity[1] * self.dt + np.random.normal(0, self.wind_std)

        return self.state[:2] + np.array([dx, dy])

    def reset(self, seed=None, options=None):
        """重置环境，初始方向朝向目标"""
        super().reset(seed=seed)

        self.start_pos = np.random.uniform(
            [self.geo_bounds["lat_min"], self.geo_bounds["lon_min"]],
            [self.geo_bounds["lat_max"], self.geo_bounds["lon_max"]]
        )
        self.target_pos = np.random.uniform(
            [self.geo_bounds["lat_min"], self.geo_bounds["lon_min"]],
            [self.geo_bounds["lat_max"], self.geo_bounds["lon_max"]]
        )

        # 计算初始方向
        delta_lon = self.target_pos[1] - self.start_pos[1]
        delta_lat = self.target_pos[0] - self.start_pos[0]
        self.yaw = np.degrees(np.arctan2(delta_lat, delta_lon))
        # 状态：当前位置(x, y) + 目标位置(x, y) + 速度(vx, vy) + 朝向(yaw)
        self.state = np.concatenate([self.start_pos, self.target_pos, np.array([0.0, 0.0]), np.array([self.yaw])])
        self.velocity = (self.target_pos - self.start_pos) / np.linalg.norm(self.target_pos - self.start_pos) * 0.0001
        self.steps = 0
        self.history = [self.start_pos.copy()]

        return np.array(self.state, dtype=np.float32), {}

    def step(self, action):
        """环境更新"""
        # 计算新位置
        new_pos = self._calculate_new_position(action)
        self.state[:2] = new_pos
        self.state[4:6] = self.velocity
        self.state[6] = self.yaw  # ✅ 存储朝向

        self.history.append(new_pos.copy())

        # 计算奖励：接近目标时给予正奖励
        distance = np.linalg.norm(self.state[:2] - self.state[2:4])
        # 主要目标：靠近目标
        reward = -np.log(distance + 1e-6) * 50

        # 速度方向与目标方向一致时给予更大奖励
        direction_to_target = (self.target_pos - self.state[:2]) / np.linalg.norm(self.target_pos - self.state[:2])
        velocity_direction = self.velocity / (np.linalg.norm(self.velocity) + 1e-6)
        alignment_reward = np.dot(direction_to_target, velocity_direction) * np.linalg.norm(self.velocity) * 1000

        reward += alignment_reward * 100  # ✅ 额外奖励，鼓励速度方向与目标一致

        # 额外的朝向奖励：yaw 角度朝向目标时奖励
        delta_lon = self.target_pos[1] - self.state[1]
        delta_lat = self.target_pos[0] - self.state[0]
        yaw_to_target = np.degrees(np.arctan2(delta_lat, delta_lon))
        yaw_error = abs(yaw_to_target - self.yaw) % 360
        yaw_error = min(yaw_error, 360 - yaw_error)  # 取最小角度差
        yaw_reward = np.cos(np.radians(yaw_error)) * 50  # 角度误差越小，奖励越高
        reward += yaw_reward

        # 终止条件：到达目标或超过最大步数
        if distance < self.target_threshold or self.steps >= self.max_steps:
            done = True
        else:
            done = False

        # 到达目标奖励
        reward += 100 if distance < self.target_threshold else 0
        self.steps += 1
        return np.array(self.state, dtype=np.float32), reward, self.steps >= self.max_steps, False, {}

    def render(self):
        """渲染环境"""

        def transform(pos):
            """地理坐标 → 画布坐标"""
            x = (pos[1] - self.geo_bounds["lon_min"]) * self.scale
            y = (self.geo_bounds["lat_max"] - pos[0]) * self.scale
            return int(x), int(y)

        self.screen.fill((255, 255, 255))

        # 绘制目标点（五角星）
        def draw_star(surface, color, center, size):
            points = []
            for i in range(5):
                angle = np.deg2rad(i * 144)
                x = center[0] + size * np.cos(angle)
                y = center[1] + size * np.sin(angle)
                points.append((x, y))
            pygame.draw.polygon(surface, color, points)

        draw_star(self.screen, (255, 0, 0), transform(self.target_pos), 15)

        # 绘制历史轨迹
        if len(self.history) > 1:
            pygame.draw.lines(self.screen, (0, 255, 0), False, [transform(p) for p in self.history], 2)

        # 直接使用无人机的yaw值，并转换为屏幕旋转角度
        screen_angle = (-self.yaw) % 360  # 地理角度 → 屏幕顺时针角度

        def draw_airplane(surface, color, center, angle, size=12):
            points = [
                (size, 0),
                (-size, -size / 2),
                (-size / 2, 0),
                (-size, size / 2),
            ]
            rotated_points = []
            for px, py in points:
                x = px * np.cos(np.radians(angle)) - py * np.sin(np.radians(angle))
                y = px * np.sin(np.radians(angle)) + py * np.cos(np.radians(angle))
                rotated_points.append((center[0] + x, center[1] + y))
            pygame.draw.polygon(surface, color, rotated_points)

        draw_airplane(self.screen, (0, 0, 255), transform(self.state[:2]), screen_angle)

        pygame.display.flip()

    def close(self):
        pygame.quit()

# 先固定目标位置训练：
env = DroneNavigationEnv(start_pos=[30.3, 120.5], target_pos=[30.6, 120.6])
check_env(env)
# 训练 PPO：
model = PPO("MlpPolicy", env,
            verbose=1,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=128,
            n_epochs=15,
            gamma=0.995,
            ent_coef=0.01)
model.learn(total_timesteps=100_000)

# 再训练随机目标点：
env = DroneNavigationEnv()
model.set_env(env)
model.learn(total_timesteps=500_000)

# 保存模型
model.save("output/ppo_drone_navigation")

# 可视化测试
obs, _ = env.reset()
#加载模型
model = PPO.load("output/ppo_drone_navigation")
for _ in range(3000000):  # 最多渲染 3000000 步
    action, _ = model.predict(obs)
    obs, _, done, _, _ = env.step(action)
    env.render()
    if done:
        break

env.close()