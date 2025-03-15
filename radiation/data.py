from dataclasses import dataclass


# 辐射源设置
@dataclass
class SourceConfig:
    a: int | float = 0  # 振幅
    f: int | float = 0  # 频率
    v: int | float = 0  # 速度


# 辐射源数据
@dataclass
class SourceData:
    a: int | float = 0  # 振幅
    f: int | float = 0  # 频率


# 模拟辐射源数据
@dataclass
class SimulatedSourceData(SourceData):
    x: int | float = 0  # x坐标
    y: int | float = 0  # y坐标


# 测向站设置
@dataclass
class StationConfig:
    x: int | float = 0  # x 坐标
    y: int | float = 0  # y 坐标
    angle: int | float = 0  # 角度
    n: int = 8  # 阵列单元数(8-16)
    d: int | float = 6  # 阵列间距(半波长6m)
    sample_rate: int | float = 100_000  # 采样频率(Hz)
    t: int | float = 0.05  # 采样时间(s)


# 测向站数据
@dataclass
class StationData:
    x: int | float = 0  # x坐标
    y: int | float = 0  # y坐标
    angle: int | float = 0  # 朝向
    theta: int | float = 0  # 角度
