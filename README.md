# External Radiation Source Location System Simulation

## 项目简介
本项目模拟了一个外辐射源定位系统，包括辐射源、测向站和计算机模拟器。通过该系统，可以实现对辐射源位置的动态计算和可视化。

## 目录结构
```
radiation
├── radiation
│   ├── __init__.py
│   ├── data.py
│   ├── simulator.py
│   ├── source.py
│   └── station.py
├── src
│   ├── icon.png
│   ├── source_1.json
│   ├── source_3.json
│   ├── station_1.json
│   └── station_3.json
├── .gitignore
├── CONTRIBUTING.md
├── LICENSE
├── README.md
├── requirements.txt
├── simulator.py
├── SourceUI.py
└── StationUI.py
```

## 文件说明

### radiation/__init__.py
初始化模块，导入项目中的主要类和函数，便于外部调用。

### radiation/data.py
定义了项目中使用的数据类，包括：
- `SourceConfig`：辐射源设置类，包含辐射源的振幅、频率和调制方式。
- `SourceMotionConfig`：辐射源运动设置类，包含辐射源的初始坐标、速度、半径和运动类型。
- `SourceData`：辐射源数据类，包含辐射源的振幅和频率。
- `SimulatedSourceData`：模拟辐射源数据类，继承自 `SourceData`，增加了坐标和调制方式。
- `StationConfig`：测向站设置类，包含测向站的坐标、角度、阵列单元数、阵列间距、采样频率和采样时间。
- `StationData`：测向站数据类，包含测向站的坐标、朝向和角度数据。

### radiation/source.py
定义了辐射源类和辐射源仿真器类，负责辐射源的属性、行为和仿真。
- `Source`：辐射源类，包含辐射源的属性和行为。
  - `__init__`：初始化辐射源。
  - `update_config`：更新辐射源设置。
  - `generate_radiation`：产生辐射。
  - `receive_data`：接收数据。
  - `calculate_position`：解算位置。
- `SourceSimulator`：辐射源仿真器类，负责辐射源的仿真。
  - `__init__`：初始化辐射源仿真器。
  - `connect`：连接网络。
  - `disconnect`：断开网络。
  - `update_config`：更新设置。
  - `update_position`：更新辐射源位置。
  - `send_data`：向测向站模拟器发送数据。
  - `receive_data`：从测向站模拟器获取数据。
  - `calculate_position`：计算位置。
  - `log`：输出调试信息。
  - `simulate`：运行仿真。

### radiation/station.py
定义了测向站类和测向站仿真器类，负责测向站的属性、行为和仿真。
- `Element`：阵元类，包含阵元的属性和行为。
  - `__init__`：初始化阵元。
  - `get_data`：采集数据。
- `Station`：测向站类，包含测向站的属性和行为。
  - `__init__`：初始化测向站。
  - `update_config`：更新辐射源设置。
  - `receive_data`：采集数据。
  - `calculate_thetas`：计算角度。
  - `a_theta`：计算转向矢量。
  - `dirfun`：生成波束方向图。
  - `beam_w`：生成滤波系数。
  - `f_theta`：返回方向图函数。
  - `k_theta`：返回和差比幅曲线。
  - `k_theta_equal_b`：等式 k(θ) = b。
  - `calculate_theta`：计算角度。
  - `find_targets_beam`：寻找目标波束。
  - `send_data`：发送数据。
- `StationSimulator`：测向站仿真器类，负责测向站的仿真。
  - `__init__`：初始化测向站仿真器。
  - `connect`：连接网络。
  - `disconnect`：断开网络。
  - `update_config`：更新参数。
  - `send_data`：向辐射源模拟器发送数据。
  - `calculate_signal`：计算信号。
  - `calculate_thetas`：计算角度。
  - `receive_data`：从辐射源模拟器接收数据。
  - `log`：输出调试信息。
  - `simulate`：运行仿真。
  - `atan2`：计算角度。
  - `calculate_theta`：计算角度。
  - `a_theta`：计算方向矢量。
  - `A_theta`：计算方向矢量。
  - `s_t`：计算信号。
  - `n_t`：计算噪声向量。
  - `x_t`：计算阵元接收到的信号。
  - `X_t`：计算阵列接收到的信号。

### radiation/simulator.py
定义了模拟器类，用一个模拟器同时实现辐射源模拟器和测向站模拟器的运行，便于开发人员调试和管理。
- `SimulatorManager`：自定义管理器类，用于管理辐射源模拟器和测向站模拟器。
- `Simulator`：模拟器类，管理辐射源模拟器和测向站模拟器的运行。
  - `__init__`：初始化模拟器。
  - `connect`：连接网络。
  - `simulate`：运行仿真。

### src/icon.png
图标文件，用于用户界面。

### src/source_1.json, src/source_3.json, src/station_1.json, src/station_3.json
JSON 配置文件，包含辐射源和测向站的配置数据。

### .gitignore
Git 忽略文件，指定应忽略的文件和目录。

### CONTRIBUTING.md
贡献指南，说明如何参与项目贡献。

### LICENSE
许可证文件，说明项目的许可证信息。

### README.md
项目的自述文件，包含项目简介、目录结构、文件说明、安装和使用说明、许可证、贡献指南和联系方式。

### requirements.txt
依赖项文件，列出了项目所需的 Python 包。

### simulator.py
定义了计算机模拟器类，用于模拟测向站的运行和数据处理。
- `SimulatorManager`：自定义管理器类，用于管理辐射源模拟器和测向站模拟器。
- `Simulator`：计算机模拟器类，管理辐射源模拟器和测向站模拟器的运行。
  - `__init__`：初始化计算机模拟器。
  - `connect`：连接网络。
  - `simulate`：运行仿真。

### SourceUI.py
辐射源端的用户界面，提供了辐射源的配置和仿真控制功能。
- `ParamDialog`：参数对话框类，用于设置数据更新率。
  - `__init__`：初始化参数对话框。
  - `initUI`：初始化用户界面。
  - `get_parameters`：获取用户输入的参数。
- `MainWindow`：主窗口类，提供辐射源的配置和仿真控制功能。
  - `__init__`：初始化主窗口。
  - `initUI`：初始化用户界面。
  - `initWindow`：初始化窗口。
  - `initToolBar`：初始化工具栏。
  - `loadSettings`：加载设置。
  - `saveSettings`：保存设置。
  - `showSettings`：显示设置对话框。
  - `showHelp`：显示帮助信息。
  - `initLayout`：初始化布局。
  - `initLeftDockWidget`：初始化左侧停靠部件。
  - `initBottomDockWidget`：初始化底部停靠部件。
  - `applySettings`：应用设置。
  - `startSimulation`：开始仿真。
  - `stopSimulation`：停止仿真。
  - `runSimulation`：运行仿真。
  - `connect`：连接网络。
  - `_disconnect`：断开网络。
  - `initConfigTab`：初始化配置选项卡。
  - `initPlotTab`：初始化图像选项卡。
  - `updatePlot`：更新图像。
  - `addSource`：增加辐射源。
  - `removeSource`：减少辐射源。

### StationUI.py
测向站端的用户界面，提供了测向站的配置和仿真控制功能。
- `ParamDialog`：参数对话框类，用于设置噪声功率和数据更新率。
  - `__init__`：初始化参数对话框。
  - `initUI`：初始化用户界面。
  - `get_parameters`：获取用户输入的参数。
- `MainWindow`：主窗口类，提供测向站的配置和仿真控制功能。
  - `__init__`：初始化主窗口。
  - `initUI`：初始化用户界面。
  - `initWindow`：初始化窗口。
  - `initToolBar`：初始化工具栏。
  - `loadSettings`：加载设置。
  - `saveSettings`：保存设置。
  - `showSettings`：显示设置对话框。
  - `showHelp`：显示帮助信息。
  - `initLayout`：初始化布局。
  - `initLeftDockWidget`：初始化左侧停靠部件。
  - `initBottomDockWidget`：初始化底部停靠部件。
  - `applySettings`：应用设置。
  - `startSimulation`：开始仿真。
  - `runSimulation`：运行仿真。
  - `stopSimulation`：停止仿真。
  - `connect`：连接网络。
  - `_disconnect`：断开网络。
  - `initConfigTab`：初始化配置选项卡。
  - `updateStationData`：更新测向站数据。
  - `addStation`：增加测向站。
  - `removeStation`：减少测向站。
  - `addSource`：增加辐射源。
  - `removeSource`：减少辐射源。

## 项目特色
- **动态仿真**：支持对辐射源和测向站的动态仿真，允许用户根据实际情况设置多个辐射源和测向站，实时计算和显示辐射源位置。用户可以通过界面实时观察辐射源和测向站的状态变化，提供了高度的互动性和可操作性。
- **多种调制方式**：支持多种辐射源调制方式，包括调频（FM）和调幅（AM）。用户可以根据实际需求选择不同的调制方式，模拟不同的辐射源信号，增加了系统的灵活性和适应性。
- **可视化**：提供详细的可视化功能，动态显示辐射源的真实位置和计算位置，并保留路径。通过图形界面，用户可以直观地看到辐射源和测向站的运动轨迹和相互关系，便于分析和调试。
- **用户界面**：提供友好的用户界面，便于用户配置和控制仿真。界面设计简洁明了，功能布局合理，用户可以轻松上手，快速完成各种配置和操作。
- **多电脑支持**：支持多电脑协同仿真，分别模拟辐射源和测向站。通过网络连接，不同电脑上的模拟器可以实时通信和数据交换，实现复杂的分布式仿真场景。
- **高扩展性**：面向对象开发使得该项目具有高度的可读性和科拓展性，用户系统设计具有高度的模块化和扩展性，用户可以根据需要添加新的辐射源类型、测向站类型和仿真算法，满足不同的研究和应用需求。
- **数据记录和回放**：系统支持仿真数据的记录和回放功能，用户可以保存仿真过程中的数据，并在需要时进行回放和分析，便于结果验证和优化。
- **多种信号处理算法**：内置多种信号处理算法，包括滤波、傅里叶变换、波束形成等，用户可以根据实际情况选择合适的算法，提高仿真精度和效率。
- **实时通信**：支持实时通信功能，辐射源和测向站之间可以实时交换数据，模拟真实环境中的信号传输和处理过程，增强仿真的真实性和可靠性。
- **跨平台支持**：系统支持多种操作系统，包括 Windows、Linux 和 macOS，用户可以在不同平台上运行仿真程序，增加了系统的适用范围。
- **详细文档**：提供详细的用户手册和开发文档，帮助用户快速了解系统功能和使用方法，同时为开发者提供参考，便于二次开发和功能扩展。

## 安装和使用

### 安装依赖
在项目根目录下运行以下命令以安装所需的依赖项：
```bash
pip install -r requirements.txt
```

### 运行示例
#### 单电脑仿真模拟
```bash
python simulator.py
```

#### 多电脑仿真模拟
##### 电脑 1 （辐射源模拟）
```bash
python simulator1.py
```
##### 电脑 2 （测向站模拟）
```bash
python simulator2.py
```

## 用户界面
### 辐射源模拟器用户界面

### 模拟器用户界面

## 许可证
本项目采用 GNU General Public License v3.0 许可证。详情见 LICENSE 文件。

## 贡献
欢迎贡献！请阅读 CONTRIBUTING.md 了解如何参与。

## 联系
如有任何问题或建议，可以直接提交 Issue。
