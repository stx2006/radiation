from multiprocessing import Process
from multiprocessing.managers import BaseManager
from .data import SourceConfig, SourceMotionConfig, StationConfig
from .source import SourceSimulator
from .station import StationSimulator


class SimulatorManager(BaseManager):
    pass


# 注册自定义管理器
SimulatorManager().register(typeid='SourceSimulator', callable=SourceSimulator)
SimulatorManager().register(typeid='StationSimulator', callable=StationSimulator)


# 计算机模拟器
class Simulator:
    def __init__(self, station_configs: tuple[StationConfig] | list[StationConfig],
                 source_configs: tuple[SourceConfig] | list[SourceConfig],
                 source_motion_configs: tuple[SourceMotionConfig] | list[SourceMotionConfig],
                 noise_power: int | float = 0, dt: int | float = 0.5):
        # 创建计算机管理器
        simulator_manager = SimulatorManager()
        simulator_manager.start()
        # 创建辐射源模拟器
        self.source_simulator = simulator_manager.SourceSimulator(source_configs=source_configs,
                                                             source_motion_configs=source_motion_configs)
        # 创建测向站模拟器
        self.station_simulator = simulator_manager.StationSimulator(station_configs=station_configs,
                                                             source_configs=source_configs,
                                                             noise_power=noise_power, dt=dt)

    # 连接
    def connect(self, computer1_args: tuple = (), computer2_args: tuple = ()):
        process1 = Process(target=self.source_simulator.connect, args=computer1_args)
        process2 = Process(target=self.station_simulator.connect, args=computer2_args)
        process1.start()
        process2.start()
        process1.join()
        process2.join()

    def simulate(self) -> None:
        process1 = Process(target=self.source_simulator.simulate, args=())
        process2 = Process(target=self.station_simulator.simulate, args=())
        process1.start()
        process2.start()
        process1.join()
        process2.join()
