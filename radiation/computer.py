from .data import SourceConfig, SimulatedSourceData, StationConfig
from source import SourceSimulator
from station import StationSimulator


class Computer1Simulator(SourceSimulator):
    def __init__(self, source_configs: tuple[SourceConfig] | list[SourceConfig]):
        super().__init__(source_configs)

    def receive_data(self):
        pass

    def send_data(self):
        pass


class Computer2Simulator(StationSimulator):
    def __init__(self, station_configs: tuple[StationConfig] | list[StationConfig],
                 source_configs: tuple[SourceConfig] | list[SourceConfig],
                 noise_power: float | tuple[float] | list[float] = 0, dt=0.5):
        super().__init__(station_configs=station_configs, source_configs=source_configs, noise_power=noise_power,
                         dt=dt)

    # 获取数据
    def receive_data(self) -> tuple[SimulatedSourceData] | list[SimulatedSourceData]:
        pass

    # 发送数据
    def send_data(self):
        pass


class ComputerSimulator:
    def __init__(self, station_configs: tuple[StationConfig] | list[StationConfig],
                 source_configs: tuple[SourceConfig] | list[SourceConfig], noise_power=0, dt=0.5):
        self.computer1 = Computer1Simulator(source_configs=source_configs)
        self.computer2 = Computer2Simulator(station_configs=station_configs, source_configs=source_configs, noise_power=noise_power, dt=dt)

    def simulate(self):
        self.computer1.update_position()
        self.computer1.send_data()
        self.computer2.receive_data()
        self.computer2.send_data()
        self.computer1.receive_data()
        self.computer1.send_data()
