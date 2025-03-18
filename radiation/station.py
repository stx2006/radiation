import warnings
import socket
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
import scipy.signal as sgl
from scipy.fft import fft
from scipy.constants import c
from .data import SourceConfig, SourceData, SimulatedSourceData, StationConfig, StationData


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
    carrier_f = 10_000_000
    carrier_wavelength = 12  # 载波波长 12 米

    def __init__(self, station_config: StationConfig, source_configs: tuple[SourceConfig] | list[SourceConfig]):
        self.x = station_config.x  # x坐标
        self.y = station_config.y  # y坐标
        self.angle = station_config.angle  # 角度(弧度)
        self.n = station_config.n  # 阵元数量
        self.elements = [Element(i) for i in range(self.n)]  # 阵元列表
        self.d = station_config.d  # 阵列间距，最好取半波长间距
        self.sample_rate = station_config.sample_rate  # 采样率(Hz)
        self.t = station_config.t  # 采样时间(s)
        # 辐射源数据
        self._lambda = [c / source_config.f for source_config in source_configs]
        self._a = [source_config.a for source_config in source_configs]
        self.theta = dict()  # 辐射源测向站角度

    # 更新辐射源设置
    def update_config(self, *args, **kwargs) -> bool:
        pass

    # 采集数据
    def receive_data(self) -> tuple[SourceData] | list[SourceData]:
        pass

    # 计算角度
    def calculate_thetas(self, x):
        """
        计算信号源的到达角度（DOA）

        :param x: 阵列接收到的信号数据，形状为 (n,)，其中 n 是阵元数量
        :return: 估计的信号源角度
        """
        n = int(self.sample_rate * self.t)  # 采样数
        beam_a = np.linspace(-90, stop=90, num=37)  # 生成角度搜索范围，从 -90 度到 90 度，步长为 1 度
        beam_w = self.beam_w(beam_a)  # 生成空间滤波权重
        y = np.dot(beam_w.T.conj(), x)  # 空间滤波
        yf = fft(y, axis=1)  # 快速傅里叶变换
        yf2 = yf[:, 0:n // 2]  # 取前一半
        beams, peaks = self.find_targets_beam(yf2)  # 寻找峰值
        for beam, peak in zip(beams, peaks):
            w1 = beam_w[:, beam[0]]  # 空间滤波权重1
            w2 = beam_w[:, beam[1]]  # 空间滤波权重2
            y1 = np.abs(np.dot(w1.T.conj(), x).sum())  # beam 1 output amplitude
            y2 = np.abs(np.dot(w2.T.conj(), x).sum())  # beam 2 output amplitude
            b = (y1 - y2) / (y1 + y2)  # 计算 b
            theta = self.calculate_theta(b, w1, w2, beam_a[beam[0]], beam_a[beam[1]])[0]  # 解方程算角度
            self.theta[peak * self.sample_rate / n] = theta  # 记录角度(f = k * sample_rate / n)
            print(f"calculated angle = {theta}")

    # 计算a(θ)
    def a_theta(self, az: float | np.ndarray = 0):
        """
        计算一个均匀线性阵列的转向矢量。

        参数:
        - az：方位角，单位为度。
        —M：表示数组中元素的个数。
        - dspace：以波长为单位的元素间距。

        返回:
        -给定参数的转向矢量。
        """
        return np.exp(
            2j * np.arange(self.n).reshape((-1, 1)) * np.pi * self.d / self.carrier_wavelength * np.sin(np.radians(az)))

    # 生成波束方向图
    def dirfun(self, w, plotcur=False):
        """
        计算和绘制方向模式。

        参数:
        - w：数组的权重向量或矩阵。
        - plotcur：布尔值，表示是否绘制当前模式。

        返回:
        -包含计算模式的绝对值的数组。
        """
        t = np.linspace(-90, 90, 1000)
        y = w.T.conj().dot(self.a_theta(t))

        ya = np.abs(y)

        if plotcur:
            plt.figure()
            plt.plot(t, 20 * np.log10(ya.T))
            plt.grid()

        return ya

    # 生成滤波系数w
    def beam_w(self, beam_a):
        """
        生成波束形成权重向量或矩阵。

        参数:
        - az：方位角，单位为度。
        —M：表示数组中元素的个数。
        - dspace：以波长为单位的元素间距。

        返回:
        -权重向量或矩阵与应用切比雪夫窗口。
        """
        # 生成滤波系数
        w = self.a_theta(beam_a)

        # 切比雪夫滤波
        warnings.filterwarnings("ignore", category=UserWarning)
        win = sgl.windows.chebwin(self.n, at=20)  # 生成切比雪夫窗，旁瓣衰减率为20
        warnings.filterwarnings("default", category=UserWarning)
        win = win / np.sum(win)
        if w.ndim == 2:
            for i in range(w.shape[1]):
                w[:, i] *= win
        elif w.ndim == 1:
            w *= win
        return w

    # 返回方向图 f(θ) 函数
    def f_theta(self, w):
        def f_theta(theta):
            return np.abs(np.dot(w.T.conj(), self.a_theta(theta)))

        return f_theta

    # 返回和差比幅曲线 k(θ) 曲线
    def k_theta(self, w1, w2):
        f_theta1 = self.f_theta(w1)
        f_theta2 = self.f_theta(w2)

        def k_theta(theta):
            f1 = f_theta1(theta)
            f2 = f_theta2(theta)
            return (f1 - f2) / (f1 + f2)

        return k_theta

    # 等式 k(θ) = b
    def k_theta_equal_b(self, w1, w2, b):
        """
        封装用于计算两个模式截止值之差的函数。

        参数:
        - w1：第一个波束束的权重向量。
        - w2：第二个波束的权重向量。
        - b：期望的模式截止值。

        返回:
        -用于计算模式之间差异的包装函数。
        """
        k_theta = self.k_theta(w1=w1, w2=w2)

        def k_theta_equal_b(theta):
            return k_theta(theta) - b

        return k_theta_equal_b

    # 计算角度
    def calculate_theta(self, b, w1, w2, t1, t2):
        """
        计算两个波束之间的振幅比较。

        参数:
        - b：两束之间期望的幅度比。
        - w1：第一个波束的权重向量。
        - w2：第二个波束的权重向量。
        - t1：第一个波束的方位角。
        - t2：第二个波束的方位角。

        返回:
        -方位角的解。
        """
        equation = self.k_theta_equal_b(w1, w2, b)  # k(θ) = b
        theta = fsolve(func=equation, x0=(t1 + t2) / 2)  # 求解 k(θ) = b 的解
        return theta  # 返回解

    @staticmethod
    def find_targets_beam(yf2):
        yfabs = (yf2 * yf2.conj()).real  # 计算频谱的幅度平方（即功率谱密度）
        yfsum = np.sum(yfabs, axis=0)  # 绝对值求和
        yfdb = np.log10(yfsum) * 10  # 转换为分贝（dB）
        threshold = np.median(yfdb) + 20  # 设置阈值
        peaks, _ = sgl.find_peaks(yfdb, threshold=threshold)  # 寻找峰值，返回峰值索引
        beams = []  # 空间滤波波束
        for peak in peaks:
            beam1 = np.argmax(yfabs[:, peak])
            if beam1 == 0:
                beam2 = 1
            elif beam1 == yfdb.shape[0] - 1:
                beam2 = beam1 - 1
            else:
                beam2 = beam1 - 1 if yfdb[beam1 - 1] > yfdb[beam1 + 1] else beam1 + 1
            beams.append((beam1, beam2))
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
        self.sock = None  # 客户端套接字
        self.source_datas = []  # 辐射源数据
        self.signal = None  # 阵列信号

    def communication_init(self, client_ip='127.0.0.1', timeout=1, port=8080):
        assert 1000 <= port < 65536
        self.sock = socket.socket()  # 建立套接字对象
        self.sock.settimeout(timeout)  # 设置超时时间
        self.sock.connect((client_ip, port))  # 连接到服务器
        print(self.sock)

    # 从辐射源模拟器接收数据
    def receive_data(self):
        serialized_data = self.sock.recv(1024)
        self.source_datas = pickle.loads(serialized_data)

    # 向辐射源模拟器发送数据
    def send_data(self):
        station_data = [StationData(x=station.x, y=station.y, angle=station.angle, theta=station.theta) for station in
                        self.stations]
        serialized_data = pickle.dumps(station_data)
        self.sock.sendall(serialized_data)

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

    # 计算方向矢量 a(θ)
    @staticmethod
    def a_theta(_theta: float = 0, _n: int = 8, _d: float = 6, _lambda: float = 12) -> np.ndarray:
        return np.exp(2j * np.pi * np.arange(_n) * _d * np.sin(_theta) / _lambda)

    # 计算阵列流形矩阵
    @staticmethod
    def A_theta(_theta: np.array, _n: int = 8, _d: float = 6, _lambda: float = 12) -> np.ndarray:
        return np.exp(-2j * np.pi * np.arange(_n).reshape((-1, 1)) * _d * np.sin(_theta.reshape((1, -1))) / _lambda)

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

    def calculate_signal(self):
        # 初始化阵列信号
        self.signal = []

        # 对每个测向站
        for i, station in enumerate(self.stations):

            # 采样数量
            n = int(station.t * station.sample_rate)

            # 阵列信号
            signal = np.zeros((station.n, n), dtype=complex)

            # 对每个信号源
            for j, source in enumerate(self.source_datas):
                # 计算相对位置
                x = source.x - station.x
                y = source.y - station.y

                # 计算相对距离
                d = np.hypot(x, y)

                # 距离判定
                if d < 20 or d > 60:
                    continue

                # 计算角度
                theta = np.radians(station.angle) - self.atan2(y, x)
                if theta > np.pi:
                    theta = 2 * np.pi - theta
                elif theta < -np.pi:
                    theta = 2 * np.pi + theta

                # 角度判定
                if abs(theta) > np.pi / 3:
                    continue

                # 生成阵元 0 采样数据
                t = np.linspace(0, station.t, n)  # 生成时间切片
                fi0 = np.random.random()
                s0 = source.a * np.cos(2 * np.pi * source.f * t + fi0)  # 生成采样
                # 生成方向矢量
                a = self.a_theta(_theta=theta, _n=station.n, _d=station.d, _lambda=12)
                # 信号叠加
                signal += np.dot(a.reshape(station.n, 1), s0.reshape(1, n))

            # 生成高斯噪声
            if isinstance(self.noise_power, tuple | list):
                noise_power = self.noise_power[i]
            elif isinstance(self.noise_power, int | float):
                noise_power = self.noise_power
            else:
                raise ValueError(f'Invalid noise_power: {self.noise_power} must be int, float, tuple or list')
            noise = np.sqrt(noise_power / 2) * (np.random.randn(station.n, n) + 1j * np.random.randn(station.n, n))

            # 生成观测信号
            signal += noise

            # 对每个阵元
            for j, element in enumerate(station.elements):
                element.data = list(signal[i])

            # 记录测向站信号
            self.signal.append(signal)

    def calculate_theta(self):
        # 对每个测向站
        for i, station in enumerate(self.stations):
            station.calculate_thetas(self.signal[i])

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
