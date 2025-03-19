import warnings
import socket
import pickle
import numpy as np
from scipy.ndimage import uniform_filter
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.signal import chirp, windows, savgol_filter, find_peaks
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
    carrier_f = 25_000_000  # 载波频率 25 MHz
    carrier_wavelength = 12  # 载波波长 12 m

    def __init__(self, station_config: StationConfig, source_configs: tuple[SourceConfig] | list[SourceConfig]):
        self.x = station_config.x  # x坐标
        self.y = station_config.y  # y坐标
        self.angle = station_config.angle  # 角度(弧度)
        self.n = station_config.n  # 阵元数量
        self.elements = [Element(i) for i in range(self.n)]  # 阵元列表
        self.d = station_config.d  # 阵列间距，最好取半波长间距
        self.sample_rate = station_config.sample_rate  # 采样率(Hz)
        self.t = station_config.t  # 采样时间(s)
        self.m = int(self.sample_rate * self.t)  # 采样数
        # 辐射源数据
        self.f = [source_config.f for source_config in source_configs]
        self.a = [source_config.a for source_config in source_configs]
        self.mode = [source_config.mode for source_config in source_configs]
        self.theta = dict()  # 辐射源测向站角度

    # 更新辐射源设置
    def update_config(self, *args, **kwargs) -> bool:
        pass

    # 采集数据
    def receive_data(self) -> tuple[SourceData] | list[SourceData]:
        pass

    # 计算角度
    def calculate_thetas(self, x: np.ndarray) -> None:
        """
        计算信号源的到达角度（DOA）

        :param x: 阵列接收到的信号数据，形状为 (n,)，其中 n 是阵元数量
        :return: 估计的信号源角度
        """
        beam_a = np.linspace(-60, stop=60, num=25)  # 生成角度搜索范围，从 -60 度到 60 度，步长为 1 度
        beam_w = self.beam_w(beam_a)  # 生成空间滤波权重
        y = np.dot(beam_w.T.conj(), x)  # 空间滤波
        yf = fft(y, axis=1)  # 快速傅里叶变换
        yf2 = yf[:, 0:self.m // 2]  # 取前一半
        beams, peaks = self.find_targets_beam(yf2)  # 寻找峰值
        for beam, peak in zip(beams, peaks):
            w1 = beam_w[:, beam[0]]  # 空间滤波权重1
            w2 = beam_w[:, beam[1]]  # 空间滤波权重2
            y1 = np.abs(np.dot(w1.T.conj(), x).sum())  # beam 1 output amplitude
            y2 = np.abs(np.dot(w2.T.conj(), x).sum())  # beam 2 output amplitude
            b = (y1 - y2) / (y1 + y2)  # 计算 b
            theta = self.calculate_theta(b, w1, w2, beam_a[beam[0]], beam_a[beam[1]])[0]  # 解方程算角度
            self.theta[peak * self.sample_rate / self.m] = theta  # 记录角度(f = k * sample_rate / n)
            print(f"calculated angle = {theta}")

    # 计算a(θ)
    def a_theta(self, az: float | np.ndarray = 0):
        """
        计算一个均匀线性阵列的转向矢量。

        参数:
        - az：方位角，单位为度。
        - M：表示数组中元素的个数。
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
        win = windows.chebwin(self.n, at=20)  # 生成切比雪夫窗，旁瓣衰减率为20
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
        yffiltered = uniform_filter(yfsum, size=5)  # 平滑处理
        yfdb = np.log10(yffiltered) * 10  # 转换为分贝（dB）

        threshold = np.median(yfdb) + 20  # 设置阈值
        peaks, _ = find_peaks(yfdb, threshold=threshold)  # 寻找峰值，返回峰值索引
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
    carrier_f = 25_000_000  # 载波频率 25 MHz
    carrier_wavelength = 12  # 载波波长 12 m

    def __init__(self, station_configs: tuple[StationConfig] | list[StationConfig],
                 source_configs: tuple[SourceConfig] | list[SourceConfig],
                 noise_power: int | float = 0.5, dt: int | float = 0.5):
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
        theta = np.arctan2(y, x)
        return theta if theta > 0 else np.pi + theta

    # 计算角度
    def calculate_theta(self, source: SimulatedSourceData, station: Station):
        # 计算相对位置
        x = source.x - station.x
        y = source.y - station.y

        # 计算相对距离
        d = np.hypot(x, y)

        # 距离判定
        if d < 20 or d > 60:
            return None

        # 计算角度
        theta = np.radians(station.angle) - self.atan2(y, x)
        if theta > np.pi:
            theta = 2 * np.pi - theta
        elif theta < -np.pi:
            theta = 2 * np.pi + theta

        # 角度判定
        if abs(theta) > np.pi / 3:
            return None

        return theta

    # 计算方向矢量 a(θ)
    def a_theta(self, theta: float = 0, n: int = 8, d: float = 6) -> np.ndarray:
        return np.exp(2j * np.pi * np.arange(n) * d * np.sin(theta) / self.carrier_wavelength).reshape(-1, 1)

    # 计算方向矢量 A(θ)
    def A_theta(self, theta: np.array, n: int = 8, d: float = 6) -> np.ndarray:
        return np.exp(2j * np.pi * np.arange(n).reshape((-1, 1)) * d * np.sin(theta.reshape((1, -1))) / self.carrier_wavelength)

    # 计算信号 s(t)
    @classmethod
    def s_t(cls, station: Station, source: SimulatedSourceData) -> np.ndarray:
        n = int(station.t * station.sample_rate)  # 采样数量
        t = np.linspace(0, station.t, n)  # 时间轴
        if source.mode.upper() == 'FM':  # 调频
            f = 50  # 调制信号频率
            modulated_signal = chirp(t, source.f - f, float(t[-1]), source.f + f, method='linear')
            return modulated_signal.reshape(1, -1)
        elif source.mode.upper() == 'AM':  # 调幅
            f = 50  # 调制信号频率
            carrier_signal = np.sin(2 * np.pi * source.f * t)  # 载波信号
            modulating_signal = np.sin(2 * np.pi * f * t)  # 调制信号
            modulated_signal = (1 + modulating_signal) * carrier_signal  # 幅度调制（AM）
            return modulated_signal.reshape(1, -1)
        else:  # 不调制
            carrier_signal = np.sin(2 * np.pi * source.f * t)  # 载波信号
            return carrier_signal.reshape(1, -1)

    # 计算噪声向量n(t)
    @staticmethod
    def n_t(station: Station, noise_power: int | float = 0) -> np.ndarray:
        n, m = station.n, station.m  # 阵元数，采样数
        noise = np.sqrt(noise_power / 2) * (np.random.randn(n, m) + 1j * np.random.randn(n, m))
        return noise

    # 计算阵元接收到的信号x(t)
    def x_t(self, station: Station, source: SimulatedSourceData) -> np.ndarray | None:
        theta = self.calculate_theta(station=station, source=source)  # 计算角度
        if theta is None:
            return None
        return np.dot(self.a_theta(theta=theta, n=station.n), self.s_t(station=station, source=source))

    # 计算阵列接收到的信号X(t)
    def X_t(self, station: Station, sources: tuple[SimulatedSourceData] | list[SimulatedSourceData],
            noise_power: int | float = 0) -> np.ndarray:
        signal = np.zeros((station.n, station.m), dtype=complex)  # 阵列信号
        for source in sources:
            _signal = self.x_t(station=station, source=source)
            if _signal is None:
                continue
            signal += _signal
        signal += self.n_t(station=station, noise_power=noise_power)
        return signal

    def calculate_signal(self):
        # 初始化阵列信号
        self.signal = []

        # 对每个测向站
        for station in self.stations:
            # 生成观测信号
            signal = self.X_t(station=station, sources=self.source_datas, noise_power=self.noise_power)

            # 阵元获取数据
            for j, element in enumerate(station.elements):
                element.data = list(signal[j])

            # 记录测向站信号
            self.signal.append(signal)

    def calculate_thetas(self) -> None:
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
        self.calculate_thetas()
        # 发送角度数据
        self.send_data()
