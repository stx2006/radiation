import pickle
from multiprocessing import Process, Pipe
from multiprocessing.managers import BaseManager
from .data import SourceConfig, SourceMotionConfig, SimulatedSourceData, StationConfig, StationData
from .source import SourceSimulator
from .station import StationSimulator


# 计算机 1 模拟器
class Computer1Simulator(SourceSimulator):
    def __init__(self, source_configs: tuple[SourceConfig] | list[SourceConfig],
                 source_motion_configs: tuple[SourceMotionConfig] | list[SourceMotionConfig],
                 dt: int | float = 0.5):
        super().__init__(source_configs=source_configs, source_motion_configs=source_motion_configs, dt=dt)


# 计算机 2 模拟器
class Computer2Simulator(StationSimulator):
    def __init__(self, station_configs: tuple[StationConfig] | list[StationConfig],
                 source_configs: tuple[SourceConfig] | list[SourceConfig],
                 noise_power: float | tuple[float] | list[float] = 0, dt=0.5):
        super().__init__(station_configs=station_configs, source_configs=source_configs,
                         noise_power=noise_power, dt=dt)


class ComputerManager(BaseManager):
    pass


# 注册自定义管理器
ComputerManager().register(typeid='Computer1Simulator', callable=Computer1Simulator)
ComputerManager().register(typeid='Computer2Simulator', callable=Computer2Simulator)


# 计算机模拟器
class ComputerSimulator:
    def __init__(self, station_configs: tuple[StationConfig] | list[StationConfig],
                 source_configs: tuple[SourceConfig] | list[SourceConfig],
                 source_motion_configs: tuple[SourceMotionConfig] | list[SourceMotionConfig],
                 noise_power: int | float = 0, dt: int | float = 0.5):
        # 创建计算机管理器
        computer_manager = ComputerManager()
        computer_manager.start()
        # 创建 computer 1
        self.computer1 = computer_manager.Computer1Simulator(source_configs=source_configs,
                                                             source_motion_configs=source_motion_configs)
        # 创建 computer 2
        self.computer2 = computer_manager.Computer2Simulator(station_configs=station_configs,
                                                             source_configs=source_configs,
                                                             noise_power=noise_power, dt=dt)

    # 连接
    def connect(self, computer1_args: tuple = (), computer2_args: tuple = ()):
        process1 = Process(target=self.computer1.connect, args=computer1_args)
        process2 = Process(target=self.computer2.connect, args=computer2_args)
        process1.start()
        process2.start()
        process1.join()
        process2.join()

    def simulate(self) -> None:
        process1 = Process(target=self.computer1.simulate, args=())
        process2 = Process(target=self.computer2.simulate, args=())
        process1.start()
        process2.start()
        process1.join()
        process2.join()
