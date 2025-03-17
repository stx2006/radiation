from typing import Any
from multiprocessing import Process, Pipe, Manager
from .data import SourceConfig, SourceMotionConfig, SimulatedSourceData, StationConfig, StationData
from .source import SourceSimulator
from .station import StationSimulator


# 计算机 1 模拟器
class Computer1Simulator(SourceSimulator):
    def __init__(self, source_configs: tuple[SourceConfig] | list[SourceConfig],
                 source_motion_configs: tuple[SourceMotionConfig] | list[SourceMotionConfig],
                 conn: Pipe, dt: int | float = 0.5):
        super().__init__(source_configs=source_configs, source_motion_configs=source_motion_configs, dt=dt)
        self.conn = conn  # 建立管道

    def simulate(self, *args: Any, **kwargs: Any):
        # 更新位置
        self.update_position()
        # 发送数据
        data = [SimulatedSourceData(x=self.x[i], y=self.y[i], a=source.a, f=source.f) for i, source in
                enumerate(self.sources)]
        self.conn.send(data)
        # 接收数据
        self.station_datas = self.conn.recv()
        # 计算位置
        self.calculate_position()
        # 输出实际位置
        print(f"x : {self.x}, y : {self.y}")
        # 输出计算位置
        print(f"x : {self.sources[0].x}, y : {self.sources[0].y}")


# 计算机 2 模拟器
class Computer2Simulator(StationSimulator):
    def __init__(self, station_configs: tuple[StationConfig] | list[StationConfig],
                 source_configs: tuple[SourceConfig] | list[SourceConfig],
                 conn: Pipe, noise_power: float | tuple[float] | list[float] = 0, dt=0.5):
        super().__init__(station_configs=station_configs, source_configs=source_configs,
                         noise_power=noise_power, dt=dt)
        self.conn = conn  # 建立管道

    def simulate(self, *args: Any, **kwargs: Any):
        # 接收数据
        self.source_data = self.conn.recv()
        # 计算信号
        self.calculate_signal()
        # 计算角度
        self.calculate_theta()
        # 发送数据
        self.conn.send(
            [StationData(x=station.x, y=station.y, angle=station.angle, theta=station.theta) for station in
             self.stations])


class ComputerSimulator:
    def __init__(self, station_configs: tuple[StationConfig] | list[StationConfig],
                 source_configs: tuple[SourceConfig] | list[SourceConfig],
                 source_motion_configs: tuple[SourceMotionConfig] | list[SourceMotionConfig],
                 noise_power=0, dt=0.5):
        # 创建管道
        computer1_conn, computer2_conn = Pipe()
        # 创建 computer 1 和 computer 2
        self.computer1 = Computer1Simulator(source_configs=source_configs, source_motion_configs=source_motion_configs,
                                            conn=computer1_conn)
        self.computer2 = Computer2Simulator(station_configs=station_configs, source_configs=source_configs,
                                            conn=computer2_conn, noise_power=noise_power, dt=dt)

    # 计算机 1 模拟
    def computer1_simulate(self):
        while True:
            self.computer1.simulate()

    # 计算机 2 模拟
    def computer2_simulate(self):
        while True:
            self.computer2.simulate()

    def simulate(self) -> None:
        process1 = Process(target=self.computer1_simulate, args=())
        process2 = Process(target=self.computer2_simulate, args=())
        process1.start()
        process2.start()
        process1.join()
        process2.join()
