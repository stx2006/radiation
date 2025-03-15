import math
import time
import warnings
from dataclasses import dataclass
import numpy as np
import scipy.signal as sgl
from scipy.optimize import fsolve
import matplotlib.pyplot as plt


# 测向站设置
@dataclass
class StationConfig:
    x: int | float = 0  # x 坐标
    y: int | float = 0  # y 坐标
    angle: int | float = 0  # 角度
    n: int = 8  # 阵列单元数(8-16)
    d: int | float = 6  # 阵列间距(半波长6m)
    sample_rate: int | float = 100_000  # 采样频率(Hz)
    t: int | float = 0.05  # 采样时间(s)


# 辐射源数据
@dataclass
class SourceData(object):
    x: int | float = 0
    y: int | float = 0
    a: int | float = 0
    f: int | float = 0


# 辐射源
class Source:
    def __init__(self, x, y, speed, initial_amplitude=0.5, initial_frequency=25e6, x1=0, y1=0, a1=0, x2=0, y2=0, a2=0):
        self.speed = speed  # m/s (5 - 60 km/h 转换为 m/s)
        self.x = x  # x坐标
        self.y = y  # y坐标
        self.amplitude = initial_amplitude  # 初始辐射波幅度
        self.frequency = initial_frequency  # 初始辐射波频率
        self.time = 0  # 记录时间
        self.x1 = x1
        self.y1 = y1
        self.a1 = a1
        self.x2 = x2
        self.y2 = y2
        self.a2 = a2

    # 产生辐射
    def generate_radiation(self):
        # 更新幅度和频率
        # 这里我们假设幅度和频率随时间呈正弦变化
        self.amplitude = self.amplitude + 0.2 * np.sin(0.1 * self.time)
        # 让频率在 25MHz 左右的 40kHz 内变化
        frequency_range = 40e3
        self.frequency = self.frequency + 0.5 * frequency_range * np.sin(0.2 * self.time)  # 你需要实现的部分

        return self.amplitude, self.frequency

    # 解算角度
    def calculate_position(self, theta1, theta2):
        # 计算从基站 1 到辐射源的射线的角度
        ray_angle_1 = self.a1 - theta1
        # 计算从基站 2 到辐射源的射线的角度
        ray_angle_2 = self.a2 - theta2

        # 两条射线的斜率
        m1 = math.tan(ray_angle_1)
        m2 = math.tan(ray_angle_2)

        # 两条射线的截距
        c1 = self.y1 - m1 * self.x1
        c2 = self.y2 - m2 * self.x2

        # 检查两条射线是否平行
        if math.isclose(m1, m2):
            print("两条射线平行，无法计算交点。")
            return None, None

        # 计算交点的 x 坐标
        x = (c2 - c1) / (m1 - m2)
        # 计算交点的 y 坐标
        y = m1 * x + c1

        self.x = x
        self.y = y

        return self.x, self.y


# 辐射源仿真器
class SourceSimulator(object):
    def __init__(self):
        self.sources = []  # 辐射源列表
        self.x = []  # 辐射源x坐标
        self.y = []  # 辐射源y坐标
        self.motion_types = []  # 每个辐射源的运动类型
        self.speeds = []  # 每个辐射源的速度
        self.radii = []  # 每个辐射源的半径（如果是圆周运动）
        self.time = 0  # 当前时间

    # 更新辐射源位置
    def update_position(self, dt):
        """更新dt秒后的位置、幅度和频率"""
        # 更新时间
        self.time += dt

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

    def send_data(self, data):
        pass

    def get_data(self):
        pass


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
        # 辐射源数据
        self._lambda = [12, ]
        self._theta = [100, ]

    # 采集数据
    def get_data(self):
        pass

    # 计算角度
    def calculate_angle(self, element_signal):
        """
        计算信号源的到达角度（DOA）

        :param element_signal: 阵列接收到的信号数据，形状为 (n,)，其中 n 是阵元数量
        :return: 估计的信号源角度
        """
        for i in range(len(self._lambda)):
            # 定义角度搜索范围
            angles = np.linspace(-90, 90, 181)  # 从 -90 度到 90 度，步长为 1 度
            # 响应字典
            response = dict()
            # 遍历所有角度
            for angle in angles:
                # 计算当前角度下的波束形成权重
                w = self.beam_w(az=angle, M=self.n, dspace=self.d / self._lambda[i])

                # 计算阵列在该角度下的响应
                response[angle] = np.abs(np.dot(w.T.conj(), element_signal)).sum()

            angle1 = max(response, key=response.get)
            if angle1 == -90:
                angle2 = angles[2]
            elif angle1 == 90:
                angle2 = angles[-2]
            else:
                angle2 = angle1 - 1 if response[angle1 - 1] > response[angle1 + 1] else angle1 + 1

            return sorted([angle1, angle2])

    # 计算a(θ)
    @staticmethod
    def a_theta(az: float | np.ndarray = 0, M: int = 8, dspace: float = 0.5):
        return np.exp(-1j * np.arange(M).reshape((M, 1)) * 2 * np.pi * dspace * np.sin(az * np.pi / 180.0))

    # 生成波束方向图
    def dirfun(self, w, plotcur=False):
        if w.ndim == 1:
            m = len(w)
        else:
            m, n = w.shape
        t = np.linspace(-90, 90, 1000)
        y = w.T.conj().dot(self.a_theta(t, m))

        ya = np.abs(y)

        if plotcur:
            plt.figure()
            plt.plot(t, 20 * np.log10(ya.T))
            plt.grid()

        return ya

    # 生成滤波系数w
    def beam_w(self, az: float | np.ndarray[int | float] = 0, M: int = 8, dspace: float = 0.5):
        # 生成滤波系数
        w = self.a_theta(az, M, dspace)

        # 切比雪夫滤波
        warnings.filterwarnings("ignore", category=UserWarning)
        win = sgl.windows.chebwin(M, at=20)  # 生成切比雪夫窗，旁瓣衰减率为20
        warnings.filterwarnings("default", category=UserWarning)
        win = win / np.sum(win)
        if w.ndim == 2:
            for i in range(w.shape[1]):
                w[:, i] *= win
        elif w.ndim == 1:
            w *= win
        return w

    def dir_fun_p(self, w, b):

        wH = w.T.conj()
        m = len(w)

        def calc_s(x):
            y = wH.dot(self.a_theta(x, m))
            return np.abs(y[0]) / m - b

        return calc_s

    def sum_minus_cur(self, w1, w2, b):

        def wrapper(x):
            f1 = self.dir_fun_p(w1, 0)
            y1 = f1(x)
            f2 = self.dir_fun_p(w2, 0)
            y2 = f2(x)
            return (y1 - y2) / (y1 + y2) - b

        return wrapper

    def amp_comp(self, b, w1, w2, t1, t2):

        f3 = self.sum_minus_cur(w1, w2, b)
        sol = fsolve(f3, (t1 + t2) / 2)  # 求解 f3=0 的解
        return sol  # 返回解

    @staticmethod
    def wave_gen(f, a, N=5000):
        fs = 100e3
        dt = 1 / fs
        t = np.arange(0, dt * N, dt)
        f = np.array(f)
        x = np.cos(2 * np.pi * f.reshape(len(f), 1) * t.reshape(1, N))
        a = np.array(a)
        return t, a.reshape(len(a), 1) * x

    @staticmethod
    def find_targets_beam(yf2):
        yfabs = (yf2 * yf2.conj()).real
        yfsum = np.sum(yfabs, axis=0)
        yfdb = np.log10(yfsum) * 10
        thr = np.median(yfdb) + 20
        peaks, _ = sgl.find_peaks(yfdb, threshold=thr)  # 寻找峰值
        beams = []
        for peak in peaks:
            beams.append(np.argmax(yfabs[:, peak]))
        return beams, peaks

    # 发送数据
    def send_data(self):
        pass


# 测向站模拟器
class StationSimulator(object):
    def __init__(self, station_configs: StationConfig | tuple[StationConfig] | list[StationConfig],
                 wave_length: tuple[int | float] | list[int | float] = (3,),
                 noise_power: int | float | tuple[int | float] | list[int | float] = 0, sleep_time=0.5):
        # 添加测向站
        if isinstance(station_configs, StationConfig):
            self.stations = [Station(station_configs.x, station_configs.y, station_configs.angle, station_configs.n,
                                     station_configs.d)] * 2
        elif isinstance(station_configs, tuple | list):
            self.stations = [Station(x=i.x, y=i.y, angle=i.angle, n=i.n, d=i.d) for i in station_configs]
        else:
            raise ValueError(f'Unknown station configs: {station_configs}')
        self.source_number = len(wave_length)  # 辐射源数量
        self.wave_length = wave_length  # 波长
        self.noise_power = noise_power  # 噪声功率
        self.sleep_time = sleep_time  # 仿真时间间隔
        self.theta = []  # 测向站角度

    @staticmethod
    def atan2(y, x):
        """
        :param y:
        :param x:
        :return:
        """
        theta = math.atan2(y, x)
        return theta if theta > 0 else math.pi + theta

    # 计算导向矢量a(θ)
    @staticmethod
    def a_theta(_theta: float = 0, _n: int = 8, _d: float = 6, _lambda: float = 12) -> np.ndarray:
        return np.exp(-2j * np.pi * np.arange(_n) * _d * np.sin(_theta) / _lambda)

    # 计算阵列流形矩阵
    @staticmethod
    def A_theta(_theta: np.array, _n: int = 8, _d: float = 6, _lambda: float = 12) -> np.ndarray:
        return np.exp(2j * np.pi * np.arange(_n).reshape((-1, 1)) * _d * np.sin(_theta.reshape((1, -1))) / _lambda)

    # 计算复包络向量s(t)
    @staticmethod
    def s_t() -> np.ndarray:
        pass

    # 计算噪声向量n(t)
    @staticmethod
    def n_t() -> np.ndarray:
        pass

    # 计算阵元接收到的信号x(t)
    def x_t(self, _theta) -> np.ndarray:
        return np.dot(self.A_theta(_theta=_theta), self.s_t()) + self.n_t()

    # 接受数据
    @staticmethod
    def get_data() -> tuple[SourceData] | list[SourceData]:
        return (SourceData(x=20, y=20 * math.sqrt(3), a=10, f=20_000),)

    # 发送数据
    def send_data(self):
        print(f"theta = {self.theta}")

    # 模拟一次
    def _simulate(self):
        # 获取数据
        source_data = self.get_data()

        # 重置测得角度
        self.theta = []

        # 对每个测向站
        for i, station in enumerate(self.stations):

            # 采样数量
            n = int(station.time * station.sample_rate)

            # 阵列信号
            element_signal = np.zeros((station.n, n), dtype=complex)

            # 对每个信号源
            for j, source in enumerate(source_data):

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
                a = self.a_theta(_theta=theta, _n=station.n, _d=station.d, _lambda=station._lambda[j])
                # 信号叠加
                element_signal += np.dot(a.reshape(station.n, 1), s0.reshape(1, n))

            # 生成高斯噪声
            if isinstance(self.noise_power, tuple | list):
                noise_power = self.noise_power[i]
            elif isinstance(self.noise_power, int | float):
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
            self.theta.append(station.calculate_angle(element_signal))

            # 发送数据
            self.send_data()

    def simulate(self):
        # 模拟一次
        self._simulate()
        # 延时
        time.sleep(self.sleep_time)


if __name__ == '__main__':
    station_config = StationConfig(x=0, y=0, angle=90 / 180 * math.pi, n=8, d=6, sample_rate=100_000, t=0.05)
    station_simulator = StationSimulator(station_config)
    while True:
        station_simulator.simulate()
