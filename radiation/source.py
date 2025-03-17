import numpy as np
from scipy.constants import c
from .data import SourceConfig, SourceMotionConfig, StationData


# 辐射源
class Source:
    def __init__(self, source_config: SourceConfig):
        self.a = source_config.a  # 初始辐射波幅度
        self.f = source_config.f  # 初始辐射波频率
        self.x = np.inf  # x坐标
        self.y = np.inf  # y坐标
        self.time = 0  # 记录时间

    # 更新设置
    def update_config(self, *args, **kwargs):
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
        # 判定是否有足够数据计算辐射源位置
        if len(station_data) < 2:
            return self.x, self.y

        # 计算从基站 1 到辐射源的射线的角度
        ray_angle_1 = station_data[0].angle - station_data[0].theta[self.f]
        # 计算从基站 2 到辐射源的射线的角度
        ray_angle_2 = station_data[1].angle - station_data[1].theta[self.f]

        # 两条射线的斜率
        m1 = np.tan(ray_angle_1 / 180 * np.pi)
        m2 = np.tan(ray_angle_2 / 180 * np.pi)

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

        print(f"calculated position : x = {x}, y = {y}")

        return self.x, self.y


# 辐射源仿真器
class SourceSimulator:
    def __init__(self, source_configs: tuple[SourceConfig] | list[SourceConfig],
                 source_motion_configs: tuple[SourceMotionConfig] | list[SourceMotionConfig],
                 dt: float = 0.5):
        self.sources = [Source(source_config) for source_config in source_configs]  # 辐射源列表
        self.x = [source_motion_config.x for source_motion_config in source_motion_configs]  # 辐射源x坐标
        self.y = [source_motion_config.y for source_motion_config in source_motion_configs]  # 辐射源y坐标
        self.motion_types = [source_motion_config.motion_type for source_motion_config in source_motion_configs]  # 运动类型
        self.v = [source_motion_config.v for source_motion_config in source_motion_configs]  # 每个辐射源的速度
        self.r = [source_motion_config.r for source_motion_config in source_motion_configs]  # 每个辐射源的半径（如果是圆周运动）
        self.t = 0  # 当前时间
        self.dt = dt  # 模拟间隔时间
        self.station_datas = None  # 测向站数据

    # 更新设置
    def update_config(self, *args, **kwargs):
        pass

    # 更新辐射源位置
    def update_position(self):
        """更新dt秒后的位置、幅度和频率"""
        # 更新时间
        print(f"now time = {self.t}")
        self.t += self.dt

        new_x = []
        new_y = []
        for i in range(len(self.sources)):
            motion_type = self.motion_types[i]
            speed = self.v[i] / 3.6
            if motion_type == 'linear':
                # 假设x方向运动
                new_x_i = self.x[i] + speed * self.dt
                new_y_i = self.y[i]
            elif motion_type == 'circular':
                radius = self.r[i]
                # 圆弧运动实现
                omega = speed / radius
                # 计算新的x和y坐标，不依赖self.thetas
                delta_theta = omega * self.dt
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

        print(f"real position : x = {self.x}, y = {self.y}")

    # 从测向站模拟器获取数据
    def receive_data(self) -> StationData:
        pass

    # 向测向站模拟器发送数据
    def send_data(self):
        pass

    # 计算位置
    def calculate_position(self):
        # 对每个辐射源
        for source in self.sources:
            # 计算位置
            source.calculate_position([station_data for station_data in self.station_datas])

    def simulate(self):
        # 更新位置
        self.update_position()
        # 发送数据
        self.send_data()
        # 接收测向站数据
        self.receive_data()
        # 计算位置
        self.calculate_position()
