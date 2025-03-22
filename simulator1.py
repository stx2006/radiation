from radiation import SourceConfig, SourceMotionConfig, SourceSimulator

if __name__ == '__main__':
    source1_config = SourceConfig(a=10, f=2_000)
    source1_motion_config = SourceMotionConfig(x=0, y=30, v=0, motion_type='circular', r=5)
    source2_config = SourceConfig(a=8, f=3_200)
    source2_motion_config = SourceMotionConfig(x=0, y=40, v=0, motion_type='circular', r=5)
    source3_config = SourceConfig(a=6, f=2_800)
    source3_motion_config = SourceMotionConfig(x=10, y=30, v=0, motion_type='circular', r=5)
    source_simulator = SourceSimulator(source_configs=[source1_config, source2_config, source3_config],
                                       source_motion_configs=[source1_motion_config, source2_motion_config,
                                                              source3_motion_config], dt=0.5)
    source_simulator.connect()
    # 计算机模拟
    while True:
        source_simulator.simulate()
