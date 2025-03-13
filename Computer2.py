import math
import time
from dataclasses import dataclass
from typing import Callable
import numpy as np


# 测向站设置
@dataclass
class StationConfig(object):
    x: int | float = 0  # x 坐标
    y: int | float = 0  # y 坐标
    angle: int | float = 0  # 角度
    n: int = 8  # 阵列单元数(8-16)
    d: int | float = 6  # 阵列间距(半波长6m)
    sample_rate: int | float = 100_000  # 采样频率(Hz)
    time: int | float = 0.05  # 采样时间(s)


# 辐射源数据
@dataclass
class SourceData(object):
    x: int | float = 0
    y: int | float = 0
    a: int | float = 0
    f: int | float = 0


# 辐射源
class Source:
    def __init__(self, x, y):
        self.x = x
        self.y = y


# 辐射源仿真器
class SourceSimulator(object):
    def __init__(self):
        self.sources = []


# 阵元
class Element(object):
    def __init__(self, number: int):
        self.number = number  # 序号
        self.data = []

    # 采集数据
    def get_data(self):
        pass


# 测向站
class Station(object):
    def __init__(self, x=0, y=0, angle=0, n=8, d=6):
        self.x = x  # x坐标
        self.y = y  # y坐标
        self.angle = angle  # 角度
        self.n = n  # 阵元数量
        self.elements = [Element(i) for i in range(n)]  # 阵元列表
        self.d = d  # 阵列间距，最好取半波长间距
        self.sample_rate = 100_000  # 采样率(Hz)
        self.time = 0.05  # 采样时间(s)

    # 采集数据
    def get_data(self):
        pass

    # 计算角度
    def calculate_angle(self):
        pass

    # 发送数据
    def send_data(self):
        pass


# 测向站模拟器
class StationSimulator(object):
    def __init__(self, station_configs: tuple[StationConfig] | list[StationConfig],
                 wave_length: tuple[int | float] | list[int | float] = 3,
                 noise_power: int | float | tuple[int | float] | list[int | float] = 0.5, sleep_time=0.5):
        # 添加测向站
        self.stations = [Station(x=i.x, y=i.y, angle=i.angle, n=i.n, d=i.d) for i in station_configs]
        self.source_number = len(wave_length)  # 辐射源数量
        self.wave_length = wave_length  # 波长
        self.noise_power = noise_power  # 噪声功率
        self.sleep_time = sleep_time  # 仿真时间间隔

    @staticmethod
    def atan2(y, x):
        """
        :param y:
        :param x:
        :return:
        """
        theta = math.atan2(y, x)
        return theta if theta > 0 else math.pi + theta

    # 接受数据
    def get_data(self) -> tuple[SourceData] | list[SourceData]:
        pass

    # 发送数据
    def send_data(self):
        pass

    def _simulate(self):
        # 获取数据
        sources = self.get_data()
        # 对每个测向站
        for i, station in enumerate(self.stations):
            # 阵列信号
            element_signal = np.zeros(station.n, dtype=complex)
            # 采样数量
            n = int(station.time * station.sample_rate)
            # 对每个信号源
            for j, source in enumerate(sources):
                # 计算相对位置
                x = source.x - station.x
                y = source.y - station.y
                # 计算相对距离
                d = math.hypot(x, y)
                # 距离判定
                if d < 20 or d > 60:
                    continue
                # 计算角度
                theta = station.angle - self.atan2(y, x)
                if theta > math.pi:
                    theta = 2 * math.pi - theta
                elif theta < -math.pi:
                    theta = 2 * math.pi + theta
                # 角度判定
                if abs(theta) > math.pi / 3:
                    continue
                # 生成阵元 0 采样数据
                t = np.linspace(0, station.time, n)  # 生成时间切片
                fi0 = np.random.random()
                s0 = source.a * np.cos(2 * np.pi * source.f * t + fi0)  # 生成采样
                # 生成方向矢量
                a = np.exp(1j * 2 * np.pi * station.d * np.arange(station.n) * np.sin(theta * np.pi / 180.0) / self.wave_length[i])
                # 信号叠加
                element_signal += np.dot(a.reshape(2, 1), s0.reshape(1, 2))

            # 生成高斯噪声
            if isinstance(self.wave_length, tuple | list):
                noise_power = self.noise_power[i]
            elif isinstance(self.wave_length, int | float):
                noise_power = self.noise_power
            else:
                raise ValueError(f'Invalid noise_power: {self.noise_power} must be int, float, tuple or list')
            noise = np.sqrt(noise_power / 2) * (np.random.randn(station.n, n) + 1j * np.random.randn(station.n, n))
            # 生成观测信号
            element_signal += noise

            # 对每个阵元
            for j, element in enumerate(station.elements):
                element.data = list(element_signal[i])

            # 计算角度
            station.calculate_angle()

            # 发送数据
            self.send_data()

    def simulate(self):
        self._simulate()
        time.sleep(self.sleep_time)
