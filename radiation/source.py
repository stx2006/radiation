import socket
import pickle
import struct
import numpy as np
from scipy.constants import c
from .data import SourceConfig, SourceMotionConfig, SourceData, SimulatedSourceData, StationData
import matplotlib.pyplot as plt


# 辐射源
class Source:
    def __init__(self, source_config: SourceConfig):
        self.a = source_config.a  # 初始辐射波幅度
        self.f = source_config.f  # 初始辐射波频率
        self.mode = source_config.mode  # 调制方式
        self.x = np.inf  # x坐标
        self.y = np.inf  # y坐标
        self.time = 0  # 记录时间
        self.x_history = []  # 记录 x 位置
        self.y_history = []  # 记录 y 位置
        self.calculated_x_history = []  # 记录计算出的 x 位置
        self.calculated_y_history = []  # 记录计算出的 y 位置

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
    def calculate_position(self, station_datas: tuple[StationData] | list[StationData]):
        # 判定是否有足够数据计算辐射源位置
        if station_datas is None:
            print("Out of detection zone.")
            return self.x, self.y
        else:
            number = 0
            for station in station_datas:
                if self.f in station.theta.keys():
                    number += 1
            if number < 2:
                print("Out of detection zone.")
                return self.x, self.y

        # 构建最小二乘法所需的矩阵 A 和向量 b
        A = []
        B = []
        for station_data in station_datas:
            # 计算从基站到辐射源的射线的角度
            ray_angle = station_data.angle - station_data.theta[self.f]
            # 射线的斜率
            k = np.tan(np.radians(ray_angle))
            # 射线的截距
            b = station_data.y - k * station_data.x
            # 构建矩阵 A 和向量 b
            A.append([k, -1])
            B.append(-b)

        A = np.array(A)
        B = np.array(B)

        # 使用最小二乘法求解
        try:
            x, y = np.linalg.lstsq(A, B, rcond=None)[0]
        except np.linalg.LinAlgError:
            print("最小二乘法求解失败。")
            return self.x, self.y

        self.x = x
        self.y = y
        self.calculated_x_history.append(x)
        self.calculated_y_history.append(y)

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
        self.center_x = [config.x for config in source_motion_configs]
        self.center_y = [config.y for config in source_motion_configs]
        self.t = 0  # 当前时间
        self.dt = dt  # 模拟间隔时间
        self.sock = None  # 套接字
        self.client = None  # 客户端
        self.station_datas = None  # 测向站数据

    # 连接网络
    def connect(self, client_ip='127.0.0.1', timeout=9999, port=8080):
        assert 1000 <= port < 65536
        self.sock = socket.socket(family=socket.AF_INET, type=socket.SOCK_STREAM)  # 创建服务器端套接字对象
        self.sock.settimeout(timeout)  # 设置超时时间
        self.sock.bind((client_ip, port))  # 绑定地址
        self.sock.listen(1)  # 监听连接
        self.client, _ = self.sock.accept()  # 接受连接

    # 断开连接
    def disconnect(self):
        self.sock.close()
        self.client.close()

    # 更新设置
    def update_config(self, *args, **kwargs):
        if args:
            if len(args) >= 1 and args[0]:
                self.sources = [Source(config) for config in args[0]]
            if len(args) >= 2 and args[1]:
                source_motion_configs = args[1]
                self.x = [config.x for config in source_motion_configs]
                self.y = [config.y for config in source_motion_configs]
                self.motion_types = [config.motion_type for config in source_motion_configs]
                self.v = [config.v for config in source_motion_configs]
                self.r = [config.r for config in source_motion_configs]
                self.center_x = [config.x for config in source_motion_configs]
                self.center_y = [config.y for config in source_motion_configs]
            if len(args) >= 3 and args[2]:
                self.dt = args[2]

        if 'source_configs' in kwargs:
            self.sources = [Source(config) for config in kwargs['source_configs']]
        if 'source_motion_configs' in kwargs:
            source_motion_configs = kwargs['source_motion_configs']
            self.x = [config.x for config in source_motion_configs]
            self.y = [config.y for config in source_motion_configs]
            self.motion_types = [config.motion_type for config in source_motion_configs]
            self.v = [config.v for config in source_motion_configs]
            self.r = [config.r for config in source_motion_configs]
        if 'dt' in kwargs:
            self.dt = kwargs['dt']

    # 更新辐射源位置
    def update_position(self):
        """更新dt秒后的位置、幅度和频率"""
        # 更新时间
        self.t += self.dt

        new_x = []
        new_y = []
        for i, source in enumerate(self.sources):
            motion_type = self.motion_types[i]
            speed = self.v[i] / 1e3  # 单位换算 km/h -> km/s
            if motion_type == 'linear':
                # 假设x方向运动
                new_x_i = self.x[i] + speed * self.dt
                new_y_i = self.y[i]
            elif motion_type == 'circular':
                radius = self.r[i]
                # 圆弧运动实现，以 (x, y) 处为圆心
                omega = speed / radius
                delta_theta = omega * self.t
                new_x_i = self.center_x[i] + radius * np.cos(delta_theta)
                new_y_i = self.center_y[i] - radius * np.sin(delta_theta)
            else:
                raise ValueError(f'Unknown motion type: {motion_type}')
            new_x.append(new_x_i)
            new_y.append(new_y_i)
            source.x_history.append(new_x_i)
            source.y_history.append(new_y_i)

        self.x = new_x
        self.y = new_y

    # 向测向站模拟器发送数据
    def send_data(self):
        simulated_source_data = [SimulatedSourceData(a=source.a, f=source.f, mode=source.mode, x=self.x[i], y=self.y[i])
                                 for i, source in enumerate(self.sources)]  # 生成辐射源模拟器数据
        serialized_data = pickle.dumps(simulated_source_data)  # 序列化辐射源模拟器数据
        self.client.sendall(serialized_data)  # 发送数据

    # 从测向站模拟器获取数据
    def receive_data(self):
        serialized_data = self.client.recv(1024)
        self.station_datas = pickle.loads(serialized_data)

    # 计算位置
    def calculate_position(self):
        # 对每个辐射源
        for source in self.sources:
            # 计算位置
            source.calculate_position([station_data for station_data in self.station_datas])

    # 输出调试信息
    def log(self):
        # 输出时间
        print(f"now time = {self.t}")
        # 输出真实坐标
        print("real position:")
        real_position = zip(self.x, self.y)
        for i, (x, y) in enumerate(real_position):
            print(f"{self.sources[i].f} : x = {x}, y = {y}")
        # 输出计算坐标
        print("calculated position:")
        for source in self.sources:
            print(f"{source.f} : x = {source.x}, y = {source.y}")

        # 计算误差
        print("position error:")
        for i, source in enumerate(self.sources):
            error_x = abs(self.x[i] - source.x)
            error_y = abs(self.y[i] - source.y)
            print(f"{source.f} : error_x = {error_x}, error_y = {error_y}")

        # # 动态显示辐射源的真实位置和计算位置，并保留路径
        # plt.close('all')
        # for i, source in enumerate(self.sources):
        #     plt.plot(np.array(source.x_history), np.array(source.y_history), 'b-')
        #     plt.scatter(np.array(source.x_history), np.array(source.y_history), color='blue')
        #     plt.plot(np.array(source.calculated_x_history), np.array(source.calculated_y_history), 'r--')
        #     plt.scatter(np.array(source.calculated_x_history), np.array(source.calculated_y_history), color='red')
        # plt.plot([], [], 'b-', label='Real Path')
        # plt.plot([], [], 'r--', label='Calculated Path')
        # plt.scatter([], [], color='blue', label='Real Position')
        # plt.scatter([], [], color='red', label='Calculated Position')
        # plt.title('Real vs Calculated Positions')
        # plt.xlim(-100, 100)  # 设置x轴范围
        # plt.ylim(-100, 100)  # 设置y轴范围
        # plt.grid(True)  # 显示网格
        # plt.legend()
        # plt.xlabel('X Position')
        # plt.ylabel('Y Position')
        # plt.show()

        # if len(self.sources[0].x_history) % 10 == 0:
        #     plt.plot(self.sources[0].x_history, self.sources[0].y_history, 'b-')
        #     plt.plot(self.sources[0].calculated_x_history, self.sources[0].calculated_y_history, 'r--')
        #     plt.scatter(self.sources[0].x_history, self.sources[0].y_history, color='blue')
        #     plt.scatter(self.sources[0].calculated_x_history, self.sources[0].calculated_y_history, color='red')
        #     plt.title('Real vs Calculated Positions')
        #     plt.xlim(-100, 100)
        #     plt.ylim(-100, 100)
        #     plt.grid(True)
        #     plt.xlabel('X Position')
        #     plt.ylabel('Y Position')
        #     plt.show()

    def simulate(self):
        # 更新位置
        self.update_position()
        # 发送数据
        self.send_data()
        # 接收测向站数据
        self.receive_data()
        # 计算位置
        self.calculate_position()
        # 输出调试信息
        self.log()
