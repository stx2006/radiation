import numpy as np
from .data import SourceConfig, StationData


# 辐射源
class Source:
    def __init__(self, source_config: SourceConfig):
        self.a = source_config.a  # 初始辐射波幅度
        self.f = source_config.f  # 初始辐射波频率
        self.v = source_config.v  # m/s (5 - 60 km/h 转换为 m/s)
        self.x = np.inf  # x坐标
        self.y = np.inf  # y坐标
        self.time = 0  # 记录时间

    # 更新设置
    def update_config(self, *args, **kwargs) -> bool:
        pass

    # 产生辐射
    def generate_radiation(self):
        # 更新幅度和频率
        # 这里我们假设幅度和频率随时间呈正弦变化
        self.a = self.a + 0.2 * np.sin(0.1 * self.time)
        # 让频率在 25MHz 左右的 40kHz 内变化
        frequency_range = 40e3
        self.f = self.f + 0.5 * frequency_range * np.sin(0.2 * self.time)  # 你需要实现的部分

        return self.a, self.f

    # 接收角度
    def receive_data(self):
        pass

    # 解算位置
    def calculate_position(self, station_data: tuple[StationData] | list[StationData]):
        # 计算从基站 1 到辐射源的射线的角度
        ray_angle_1 = station_data[0].angle - station_data[0].theta
        # 计算从基站 2 到辐射源的射线的角度
        ray_angle_2 = station_data[1].angle - station_data[1].theta

        # 两条射线的斜率
        m1 = np.tan(ray_angle_1)
        m2 = np.tan(ray_angle_2)

        # 检查两条射线是否平行
        if np.isclose(m1, m2):
            print("两条射线平行，无法计算交点。")
            return None, None

        # 两条射线的截距
        c1 = station_data[0].y - m1 * station_data[0].x
        c2 = station_data[1].y - m2 * station_data[1].x

        # 计算交点的 x 坐标
        x = (c2 - c1) / (m1 - m2)
        # 计算交点的 y 坐标
        y = m1 * x + c1

        self.x = x
        self.y = y

        return self.x, self.y


# 辐射源仿真器
class SourceSimulator:
    def __init__(self, source_configs: tuple[SourceConfig] | list[SourceConfig], dt: float = 0.5):
        self.sources = [Source(source_config) for source_config in source_configs]  # 辐射源列表
        self.x = []  # 辐射源x坐标
        self.y = []  # 辐射源y坐标
        self.motion_types = []  # 每个辐射源的运动类型
        self.speeds = []  # 每个辐射源的速度
        self.radii = []  # 每个辐射源的半径（如果是圆周运动）
        self.time = 0  # 当前时间
        self.dt = dt

    # 更新设置
    def update_config(self, *args, **kwargs):
        pass

    # 更新辐射源位置
    def update_position(self):
        """更新dt秒后的位置、幅度和频率"""
        # 更新时间
        self.time += self.dt

        new_x = []
        new_y = []
        for i in range(len(self.sources)):
            motion_type = self.motion_types[i]
            speed = self.speeds[i]
            if motion_type == 'linear':
                # 假设x方向运动
                new_x_i = self.x[i] + speed * dt
                new_y_i = self.y[i]
            elif motion_type == 'circular':
                radius = self.radii[i]
                # 圆弧运动实现
                omega = speed / radius
                # 计算新的x和y坐标，不依赖self.thetas
                delta_theta = omega * dt
                cos_delta_theta = np.cos(delta_theta)
                sin_delta_theta = np.sin(delta_theta)
                new_x_i = self.x[i] * cos_delta_theta - self.y[i] * sin_delta_theta
                new_y_i = self.x[i] * sin_delta_theta + self.y[i] * cos_delta_theta
            else:
                raise ValueError(f'Unknown motion type: {motion_type}')
            new_x.append(new_x_i)
            new_y.append(new_y_i)

        self.x = new_x
        self.y = new_y

    # 从测向站模拟器获取数据
    def receive_data(self):
        pass

    # 向测向站模拟器发送数据
    def send_data(self):
        pass

    def calculate_position(self, station_data):
        for i, source in enumerate(self.sources):
            source.calculate_position(station_data[i])

    def simulate(self):
        self.update_position(0.5)  # 更新位置
        self.send_data()
        station_data = self.receive_data()
        self.calculate_position(station_data)
