from radiation import *

if __name__ == '__main__':
    station1_config = StationConfig(x=0, y=0, angle=90, n=16, d=6, sample_rate=100_000, t=0.05)
    station2_config = StationConfig(x=20, y=0, angle=135, n=16, d=6, sample_rate=100_000, t=0.05)
    source1_config = SourceConfig(a=10, f=2_000, mode='FM')
    source1_motion_config = SourceMotionConfig(x=0, y=30, v=5, motion_type='circular', r=5)
    source2_config = SourceConfig(a=8, f=3_200, mode='FM')
    source2_motion_config = SourceMotionConfig(x=0, y=40, v=5, motion_type='circular', r=5)
    source3_config = SourceConfig(a=6, f=2_800, mode='FM')
    source3_motion_config = SourceMotionConfig(x=10, y=30, v=5, motion_type='circular', r=5)
    simulator = Simulator(station_configs=[station1_config, station2_config],
                          source_configs=[source1_config, source2_config, source3_config],
                          source_motion_configs=[source1_motion_config, source2_motion_config, source3_motion_config])

    # 连接
    simulator.connect()
    # 计算机模拟
    while True:
        simulator.simulate()
