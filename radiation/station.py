import warnings
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
import scipy.signal as sgl
from scipy.constants import c
from .data import SourceConfig, SourceData, SimulatedSourceData, StationConfig


# 阵元
class Element:
    def __init__(self, number: int):
        self.number = number  # 序号
        self.data = []

    # 采集数据
    def get_data(self):
        pass


# 测向站
class Station:
    def __init__(self, station_config: StationConfig, source_configs: tuple[SourceConfig] | list[SourceConfig]):
        self.x = station_config.x  # x坐标
        self.y = station_config.y  # y坐标
        self.angle = station_config.angle  # 角度(弧度)
        self.n = station_config.n  # 阵元数量
        self.elements = [Element(i) for i in range(self.n)]  # 阵元列表
        self.d = station_config.d  # 阵列间距，最好取半波长间距
        self.sample_rate = station_config.sample_rate  # 采样率(Hz)
        self.time = station_config.t  # 采样时间(s)
        # 辐射源数据
        self._lambda = [c / source_config.f for source_config in source_configs]
        self._a = [source_config.a for source_config in source_configs]
        self.theta = dict()  # 辐射源测向站角度

    # 更新辐射源设置
    def update_config(self, *args, **kwargs) -> bool:
        pass

    # 采集数据
    def get_data(self) -> tuple[SourceData] | list[SourceData]:
        pass

    # 计算角度
    def calculate_angle(self, element_signal):
        """
        计算信号源的到达角度（DOA）

        :param element_signal: 阵列接收到的信号数据，形状为 (n,)，其中 n 是阵元数量
        :return: 估计的信号源角度
        """
        # 定义角度搜索范围
        angles = np.linspace(-90, stop=90, num=181)  # 从 -90 度到 90 度，步长为 1 度
        # 响应字典
        response = dict()
        # 遍历所有角度
        for angle in angles:
            # 计算当前角度下的波束形成权重
            w = self.beam_w(az=angle, M=self.n, dspace=self.d / 12)

            # 计算阵列在该角度下的响应
            response[angle] = np.abs(np.dot(w.T.conj(), element_signal)).sum()

        angle1 = max(response, key=response.get)
        if angle1 == -90:
            angle2 = angles[2]
        elif angle1 == 90:
            angle2 = angles[-2]
        else:
            angle2 = angle1 - 1 if response[angle1 - 1] > response[angle1 + 1] else angle1 + 1

        # 记录角度
        self.theta[c / self._lambda[0]] = angle1
        print(f"calculated angle = {angle1}")

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
    def beam_w(self, az: float | np.ndarray[float] = 0, M: int = 8, dspace: float = 0.5):
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
class StationSimulator:
    def __init__(self, station_configs: tuple[StationConfig] | list[StationConfig],
                 source_configs: tuple[SourceConfig] | list[SourceConfig],
                 noise_power: float | tuple[float] | list[float] = 0.5, dt=0.5):
        # 添加测向站
        self.stations = [Station(station_config, source_configs) for station_config in station_configs]
        self.source_number = len(source_configs)  # 辐射源数量
        self.noise_power = noise_power  # 噪声功率
        self.dt = dt  # 仿真时间间隔
        self.source_data = None  # 辐射源数据
        self.signal = None  # 阵列信号

    # 更新参数
    def update_config(self, *args, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    @staticmethod
    def atan2(y, x):
        """
        :param y:
        :param x:
        :return:
        """
        theta = np.arctan2(y, x)
        return theta if theta > 0 else np.pi + theta

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
    def s_t(n: int = 5000) -> np.ndarray:
        pass

    # 计算噪声向量n(t)
    @staticmethod
    def n_t() -> np.ndarray:
        pass

    # 计算阵元接收到的信号x(t)
    def x_t(self, _theta) -> np.ndarray:
        return np.dot(self.A_theta(_theta=_theta), self.s_t()) + self.n_t()

    # 计算阵列接收到的信号X(t)
    def X_t(self, _theta) -> np.ndarray:
        pass

    # 从辐射源模拟器接收数据
    def receive_data(self) -> tuple[SimulatedSourceData] | list[SimulatedSourceData]:
        pass

    # 向辐射源模拟器发送数据
    def send_data(self):
        pass

    def calculate_signal(self):
        # 初始化阵列信号
        self.signal = []

        # 对每个测向站
        for i, station in enumerate(self.stations):

            # 采样数量
            n = int(station.time * station.sample_rate)

            # 阵列信号
            station_signal = np.zeros((station.n, n), dtype=complex)

            # 对每个信号源
            for j, source in enumerate(self.source_data):

                # 计算相对位置
                x = source.x - station.x
                y = source.y - station.y

                # 计算相对距离
                d = np.hypot(x, y)

                # 距离判定
                if d < 20 or d > 60:
                    continue

                # 计算角度
                theta = station.angle / 180 * np.pi - self.atan2(y, x)
                if theta > np.pi:
                    theta = 2 * np.pi - theta
                elif theta < -np.pi:
                    theta = 2 * np.pi + theta

                # 角度判定
                if abs(theta) > np.pi / 3:
                    continue

                # 生成阵元 0 采样数据
                t = np.linspace(0, station.time, n)  # 生成时间切片
                fi0 = np.random.random()
                s0 = source.a * np.cos(2 * np.pi * source.f * t + fi0)  # 生成采样
                # 生成方向矢量
                a = self.a_theta(_theta=theta, _n=station.n, _d=station.d, _lambda=12)
                # 信号叠加
                station_signal += np.dot(a.reshape(station.n, 1), s0.reshape(1, n))

            # 生成高斯噪声
            if isinstance(self.noise_power, tuple | list):
                noise_power = self.noise_power[i]
            elif isinstance(self.noise_power, int | float):
                noise_power = self.noise_power
            else:
                raise ValueError(f'Invalid noise_power: {self.noise_power} must be int, float, tuple or list')
            noise = np.sqrt(noise_power / 2) * (np.random.randn(station.n, n) + 1j * np.random.randn(station.n, n))

            # 生成观测信号
            station_signal += noise

            # 对每个阵元
            for j, element in enumerate(station.elements):
                element.data = list(station_signal[i])

            # 记录测向站信号
            self.signal.append(station_signal)

    def calculate_theta(self):
        # 对每个测向站
        for i, station in enumerate(self.stations):
            station.calculate_angle(self.signal[i])

    # 模拟一次
    def simulate(self):
        # 接受辐射源数据
        self.receive_data()
        # 计算信号
        self.calculate_signal()
        # 计算角度
        self.calculate_theta()
        # 发送角度数据
        self.send_data()

