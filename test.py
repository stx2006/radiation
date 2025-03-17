from radiation import SourceConfig, SourceMotionConfig, StationConfig, ComputerSimulator

if __name__ == '__main__':
    station1_config = StationConfig(x=0, y=0, angle=90, n=16, d=6, sample_rate=100_000, t=0.05)
    station2_config = StationConfig(x=20, y=0, angle=135, n=16, d=6, sample_rate=100_000, t=0.05)
    source1_config = SourceConfig(a=10, f=20_000)
    source1_motion_config = SourceMotionConfig(x=20, y=20, v=5, motion_type='circular', r=5)
    computer = ComputerSimulator(station_configs=[station1_config, station2_config],
                                 source_configs=[source1_config], source_motion_configs=[source1_motion_config])

    # 计算机模拟
    computer.simulate()
