from radiation import SourceConfig, StationConfig, StationSimulator

if __name__ == '__main__':
    station1_config = StationConfig(x=0, y=0, angle=90, n=16, d=6, sample_rate=100_000, t=0.05)
    station2_config = StationConfig(x=20, y=0, angle=135, n=16, d=6, sample_rate=100_000, t=0.05)
    source1_config = SourceConfig(a=10, f=2_000)
    source2_config = SourceConfig(a=8, f=3_200)
    source3_config = SourceConfig(a=6, f=2_800)
    station_simulator = StationSimulator(station_configs=[station1_config, station2_config],
                                        source_configs=[source1_config, source2_config, source3_config],
                                        noise_power=0, dt=0.5)
    station_simulator.connect()
    # 计算机模拟
    while True:
        station_simulator.simulate()
