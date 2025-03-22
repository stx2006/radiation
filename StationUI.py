import sys
import json
import webbrowser
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from radiation import *


class ParamDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.initUI()

    def initUI(self):
        # 创建标签和输入框
        self.label1 = QLabel("噪声功率:")
        self.input1 = QLineEdit()

        self.label2 = QLabel("数据更新率(次/秒):")
        self.input2 = QLineEdit()

        # 创建按钮
        self.ok_button = QPushButton("OK")
        self.ok_button.clicked.connect(self.accept)  # 点击 OK 按钮时关闭对话框并返回 QDialog.Accepted

        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)  # 点击 Cancel 按钮时关闭对话框并返回 QDialog.Rejected

        # 设置布局
        form_layout = QVBoxLayout()
        form_layout.addWidget(self.label1)
        form_layout.addWidget(self.input1)
        form_layout.addWidget(self.label2)
        form_layout.addWidget(self.input2)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.ok_button)
        button_layout.addWidget(self.cancel_button)

        main_layout = QVBoxLayout()
        main_layout.addLayout(form_layout)
        main_layout.addLayout(button_layout)

        self.setLayout(main_layout)

        self.setWindowTitle('Set Parameters')
        self.setFixedSize(300, 150)

    def get_parameters(self):
        # 返回用户输入的参数
        text1 = self.input1.text()
        text2 = self.input2.text()
        flag = True
        if text1.isdecimal():
            text1 = float(text1)
        else:
            text1 = 0
            flag = False
        if text2.isdecimal():
            text2 = float(text2)
        else:
            text2 = 0.5
            flag = False
        return text1, text2, flag


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.station_simulator = None  # 辐射源模拟器
        self.noise_power = 0
        self.dt = 0.5
        self.station_count = 0  # 初始化测向站计数
        self.source_count = 0  # 初始化辐射源计数
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.runSimulation)
        self.initUI()  # 初始化界面

    def initUI(self):
        self.initWindow()  # 初始化窗口
        self.initToolBar()  # 初始化工具栏
        self.initLayout()  # 初始化布局
        self.initLeftDockWidget()  # 初始化左侧停靠部件
        self.initBottomDockWidget()  # 初始化底部停靠部件
        self.initConfigTab()  # 初始化主选项卡
        self.show()  # 显示窗口

    # 初始化窗口
    def initWindow(self):
        self.setWindowTitle("外辐射源定位系统模拟(测向站端)")
        self.setWindowIcon(QIcon('src/icon.png'))
        self.setGeometry(100, 100, 2400, 1200)

    # 初始化工具栏
    def initToolBar(self):
        # 创建工具栏
        toolbar = QToolBar("Tool Bar")
        self.addToolBar(toolbar)

        # 创建“文件”动作并添加子菜单
        file_action = QAction("File", self)
        file_menu = QMenu()

        save_settings_action = QAction("Save Configuration", self)
        save_settings_action.triggered.connect(self.saveSettings)
        file_menu.addAction(save_settings_action)

        load_settings_action = QAction("Load Configuration", self)
        load_settings_action.triggered.connect(self.loadSettings)
        file_menu.addAction(load_settings_action)

        file_action.setMenu(file_menu)
        toolbar.addAction(file_action)

        # 创建“设置”动作（示例，无子菜单）
        settings_action = QAction("Settings", self)
        settings_action.triggered.connect(self.showSettings)
        toolbar.addAction(settings_action)

        # 创建“帮助”动作（示例，无子菜单）
        help_action = QAction("Help", self)
        help_action.triggered.connect(self.showHelp)
        toolbar.addAction(help_action)

    # 工具栏函数
    # 加载设置
    def loadSettings(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "加载设置", "", "JSON Files (*.json);;All Files (*)",
                                                   options=options)
        if file_name:
            with open(file_name, 'r') as file:
                settings = json.load(file)
                for source in settings['sources']:
                    self.addSource()
                    group_box = self.source_layout.itemAt(self.source_count - 1).widget()
                    source_layout = group_box.layout()
                    for i in range(source_layout.count()):
                        item = source_layout.itemAt(i)
                        if isinstance(item, QHBoxLayout):
                            for j in range(item.count()):
                                widget = item.itemAt(j).widget()
                                if isinstance(widget, QLabel):
                                    label_text = widget.text()
                                elif isinstance(widget, QDoubleSpinBox):
                                    widget.setValue(source[label_text])
                                elif isinstance(widget, QSpinBox):
                                    widget.setValue(source[label_text])
                for station in settings['stations']:
                    self.addStation()
                    group_box = self.station_layout.itemAt(self.station_count - 1).widget()
                    station_layout = group_box.layout()
                    for i in range(station_layout.count()):
                        item = station_layout.itemAt(i)
                        if isinstance(item, QHBoxLayout):
                            for j in range(item.count()):
                                widget = item.itemAt(j).widget()
                                if isinstance(widget, QLabel):
                                    label_text = widget.text()
                                elif isinstance(widget, QDoubleSpinBox):
                                    widget.setValue(station[label_text])
                                elif isinstance(widget, QSpinBox):
                                    widget.setValue(station[label_text])
            QMessageBox.information(self, "加载设置", "设置加载成功！")

    # 保存设置
    def saveSettings(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getSaveFileName(self, "保存设置", "", "JSON Files (*.json);;All Files (*)",
                                                   options=options)
        if file_name:
            settings = {'sources': [], 'stations': []}
            for i in range(self.source_layout.count()):
                group_box = self.source_layout.itemAt(i).widget()
                if isinstance(group_box, QGroupBox):
                    source_layout = group_box.layout()
                    source_params = {}
                    for j in range(source_layout.count()):
                        item = source_layout.itemAt(j)
                        if isinstance(item, QHBoxLayout):
                            for k in range(item.count()):
                                widget = item.itemAt(k).widget()
                                if isinstance(widget, QLabel):
                                    label_text = widget.text()
                                elif isinstance(widget, QDoubleSpinBox):
                                    source_params[label_text] = widget.value()
                                elif isinstance(widget, QSpinBox):
                                    source_params[label_text] = widget.value()
                    settings['sources'].append(source_params)
            for i in range(self.station_layout.count()):
                group_box = self.station_layout.itemAt(i).widget()
                if isinstance(group_box, QGroupBox):
                    station_layout = group_box.layout()
                    station_params = {}
                    for j in range(station_layout.count()):
                        item = station_layout.itemAt(j)
                        if isinstance(item, QHBoxLayout):
                            for k in range(item.count()):
                                widget = item.itemAt(k).widget()
                                if isinstance(widget, QLabel):
                                    label_text = widget.text()
                                elif isinstance(widget, QDoubleSpinBox):
                                    station_params[label_text] = widget.value()
                                elif isinstance(widget, QSpinBox):
                                    station_params[label_text] = widget.value()
                    settings['stations'].append(station_params)
            with open(file_name, 'w') as file:
                json.dump(settings, file, indent=4)
            QMessageBox.information(self, "保存设置", "设置保存成功！")

    # 设置
    def showSettings(self):
        dialog = ParamDialog(self)
        if dialog.exec_() == QDialog.Accepted:  # 如果用户点击了 OK 按钮
            self.noise_power, self.dt, flag = dialog.get_parameters()
            if flag:
                QMessageBox.information(self, "保存设置", "设置保存成功！")
            else:
                QMessageBox.warning(self, "保存设置", "设置保存失败：输入不合法！")

    # 帮助
    def showHelp(self):
        url = "https://github.com/stx2006/radiation"
        webbrowser.open(url)

    # 初始化布局
    def initLayout(self):
        main_widget = QWidget()  # 创建主部件
        self.setCentralWidget(main_widget)  # 设置中心主部件
        main_layout = QHBoxLayout(main_widget)  # 创建主布局
        self.tabs = QTabWidget()  # 创建选项卡部件
        main_layout.addWidget(self.tabs)  # 向主布局添加选项卡部件

    # 初始化左侧停靠部件(对象栏)
    def initLeftDockWidget(self):
        left_dock_widget = QDockWidget("Objects", self)
        self.addDockWidget(Qt.LeftDockWidgetArea, left_dock_widget)

        left_dock_content = QWidget()
        left_dock_layout = QVBoxLayout(left_dock_content)

        # 测向站对象栏
        self.station_tree = QTreeWidget(left_dock_content)
        self.station_tree.setHeaderLabels(["Stations"])
        left_dock_layout.addWidget(self.station_tree)
        # 增加测向站按钮
        add_station_button = QPushButton("Add Station")
        add_station_button.clicked.connect(self.addStation)
        left_dock_layout.addWidget(add_station_button)
        # 减少测向站按钮
        remove_station_button = QPushButton("Remove Station")
        remove_station_button.clicked.connect(self.removeStation)
        left_dock_layout.addWidget(remove_station_button)

        # 辐射源对象栏
        self.source_tree = QTreeWidget(left_dock_content)
        self.source_tree.setHeaderLabels(["Sources"])
        left_dock_layout.addWidget(self.source_tree)
        # 增加辐射源按钮
        add_source_button = QPushButton("Add Source")
        add_source_button.clicked.connect(self.addSource)
        left_dock_layout.addWidget(add_source_button)
        # 减少辐射源按钮
        remove_source_button = QPushButton("Remove Source")
        remove_source_button.clicked.connect(self.removeSource)
        left_dock_layout.addWidget(remove_source_button)

        left_dock_widget.setWidget(left_dock_content)

    # 初始化底部停靠部件(控制栏)
    def initBottomDockWidget(self):
        bottom_dock_widget = QDockWidget("Controls", self)
        self.addDockWidget(Qt.BottomDockWidgetArea, bottom_dock_widget)

        bottom_dock_content = QWidget()
        bottom_dock_layout = QHBoxLayout(bottom_dock_content)

        # 左侧按钮区域
        button_layout = QVBoxLayout()
        # 应用设置
        apply_settings_button = QPushButton("Apply Settings")
        apply_settings_button.setStyleSheet("background-color: yellow; color: black")
        apply_settings_button.clicked.connect(self.applySettings)
        button_layout.addWidget(apply_settings_button)
        # 开始仿真
        start_simulation_button = QPushButton("Start Simulation")
        start_simulation_button.setStyleSheet("background-color: green; color: black")
        start_simulation_button.clicked.connect(self.startSimulation)
        button_layout.addWidget(start_simulation_button)
        # 停止仿真
        stop_simulation_button = QPushButton("Stop Simulation")
        stop_simulation_button.setStyleSheet("background-color: red; color: black")
        stop_simulation_button.clicked.connect(self.stopSimulation)
        button_layout.addWidget(stop_simulation_button)
        bottom_dock_layout.addLayout(button_layout)

        # 右侧网络连接区域F
        network_layout = QFormLayout()
        # ip地址输入
        self.ip_address_input = QLineEdit("127.0.0.1")
        self.ip_address_input.setPlaceholderText("127.0.0.1")  # 设置占位符
        # 端口输入
        self.port_input = QLineEdit("8080")
        self.port_input.setPlaceholderText("8080")  # 设置占位符
        # 连接
        connect_button = QPushButton("Connect")
        connect_button.setStyleSheet("background-color: green; color: white")
        connect_button.clicked.connect(self.connect)
        # 断开
        break_button = QPushButton("Break")
        break_button.setStyleSheet("background-color: red; color: white")
        break_button.clicked.connect(self._disconnect)
        # 添加组件
        network_layout.addRow(QLabel("IP Address:"), self.ip_address_input)
        network_layout.addRow(QLabel("Port:"), self.port_input)
        network_layout.addWidget(connect_button)
        network_layout.addWidget(break_button)
        bottom_dock_layout.addLayout(network_layout)

        bottom_dock_widget.setWidget(bottom_dock_content)

    # 应用设置
    def applySettings(self):
        station_configs = []  # 测向站参数
        # 处理参数
        for i in range(self.station_layout.count()):
            group_box = self.station_layout.itemAt(i).widget()
            if isinstance(group_box, QGroupBox):
                station_layout = group_box.layout()
                parameters = {}
                for j in range(station_layout.count()):
                    item = station_layout.itemAt(j)
                    if isinstance(item, QHBoxLayout):
                        for k in range(item.count()):
                            widget = item.itemAt(k).widget()
                            if isinstance(widget, QLabel):
                                label_text = widget.text()
                            elif isinstance(widget, QDoubleSpinBox):
                                parameters[label_text] = widget.value()
                            elif isinstance(widget, QSpinBox):
                                parameters[label_text] = widget.value()
                station_configs.append(StationConfig(*parameters.values()))
        source_configs = []  # 辐射源参数
        # 处理参数
        for i in range(self.source_layout.count()):
            group_box = self.source_layout.itemAt(i).widget()
            if isinstance(group_box, QGroupBox):
                source_layout = group_box.layout()
                parameters = {}
                for j in range(source_layout.count()):
                    item = source_layout.itemAt(j)
                    if isinstance(item, QHBoxLayout):
                        for k in range(item.count()):
                            widget = item.itemAt(k).widget()
                            if isinstance(widget, QLabel):
                                label_text = widget.text()
                            elif isinstance(widget, QDoubleSpinBox):
                                parameters[label_text] = widget.value()
                            elif isinstance(widget, QSpinBox):
                                parameters[label_text] = widget.value()
                source_configs.append(SourceConfig(*parameters.values()))
        self.station_simulator = StationSimulator(station_configs=station_configs,
                                                                            source_configs=source_configs,
                                                                            noise_power=self.noise_power,
                                                                            dt=self.dt)  # 创建辐射源模拟器
        QMessageBox.information(self, "应用设置", "设置成功！")

    # 开始仿真
    def startSimulation(self):
        self.timer.start(1000)  # 每秒调用一次 simulate 方法

    # 运行仿真
    def runSimulation(self):
        self.station_simulator.simulate()

    # 停止仿真
    def stopSimulation(self):
        self.timer.stop()

    # 连接网络
    def connect(self):
        if self.station_simulator is not None:
            self.station_simulator.connect()
            QMessageBox.information(self, "网络连接", "连接成功！")
        else:
            QMessageBox.information(self, "网络连接", "连接失败：未应用设置！")

    # 断开网络
    def _disconnect(self):
        self.station_simulator.disconnect()
        QMessageBox.information(self, "网络连接", "退出连接！")

    # 主选项卡初始化
    def initConfigTab(self):
        # 创建一个示例选项卡
        tab_config = QWidget()
        self.tabs.addTab(tab_config, "Configuration")

        # 创建标签布局
        tab_layout = QVBoxLayout(tab_config)

        # 测向站区域
        station_scroll_area = QScrollArea()
        station_scroll_area.setWidgetResizable(True)
        station_content = QWidget()
        self.station_layout = QVBoxLayout(station_content)
        station_scroll_area.setWidget(station_content)
        tab_layout.addWidget(QLabel("Stations"))
        tab_layout.addWidget(station_scroll_area)

        # 辐射源区域
        source_scroll_area = QScrollArea()
        source_scroll_area.setWidgetResizable(True)
        source_content = QWidget()
        self.source_layout = QVBoxLayout(source_content)
        source_scroll_area.setWidget(source_content)
        tab_layout.addWidget(QLabel("Sources"))
        tab_layout.addWidget(source_scroll_area)

    # 增加测向站
    def addStation(self):
        self.station_count += 1  # 测向站数量 + 1
        object_name = f"Station {self.station_count}"

        group_box = QGroupBox(object_name)  # 创建新的分组框
        station_layout = QFormLayout()  # 创建新布局

        # 第一行
        row1_layout = QHBoxLayout()
        x = QDoubleSpinBox()  # 创建数值输入框
        x.setRange(-100, 100)  # 设置输入框的数值范围
        row1_layout.addWidget(QLabel("x(km)"))  # 创建标签
        row1_layout.addWidget(x)  # 将控件添加到网格布局的指定行和列

        y = QDoubleSpinBox()  # 创建数值输入框
        y.setRange(-100, 100)  # 设置输入框的数值范围
        row1_layout.addWidget(QLabel("y(km)"))  # 创建标签
        row1_layout.addWidget(y)  # 将控件添加到网格布局的指定行和列
        station_layout.addRow(row1_layout)

        # 第二行
        row2_layout = QHBoxLayout()
        angle = QDoubleSpinBox()  # 创建数值输入框
        angle.setRange(0, 360)  # 设置输入框的数值范围
        row2_layout.addWidget(QLabel("角度(°)"))  # 创建标签
        row2_layout.addWidget(angle)  # 将控件添加到网格布局的指定行和列

        n = QSpinBox()
        n.setRange(8, 16)
        row2_layout.addWidget(QLabel("阵列单元数"))
        row2_layout.addWidget(n)
        station_layout.addRow(row2_layout)

        # 第三行
        row3_layout = QHBoxLayout()
        d = QDoubleSpinBox()
        d.setRange(0, 100)
        d.setValue(6)  # 设置默认值
        row3_layout.addWidget(QLabel("阵元间距(m)"))
        row3_layout.addWidget(d)

        f = QDoubleSpinBox()
        f.setRange(0, 1_000_000)
        f.setValue(100_000)  # 设置默认值
        row3_layout.addWidget(QLabel("采样频率(Hz)"))
        row3_layout.addWidget(f)
        station_layout.addRow(row3_layout)

        # 第四行
        row4_layout = QHBoxLayout()
        t = QDoubleSpinBox()
        t.setValue(0.05)  # 设置默认值
        row4_layout.addWidget(QLabel("采样时间(s)"))
        row4_layout.addWidget(t)
        station_layout.addRow(row4_layout)

        group_box.setLayout(station_layout)
        self.station_layout.addWidget(group_box)

    # 减少测向站
    def removeStation(self):
        if self.station_count > 0:
            # 移除最后一个对象
            item = self.station_tree.topLevelItem(self.station_count - 1)
            self.station_tree.takeTopLevelItem(self.station_count - 1)
            del item  # 删除 QTreeWidgetItem 对象

            # 移除最后一个分组框
            if self.station_layout.count() > 0:
                widget_to_remove = self.station_layout.itemAt(self.station_layout.count() - 1).widget()
                self.station_layout.removeWidget(widget_to_remove)
                widget_to_remove.deleteLater()

            self.station_count = max(0, self.station_count - 1)

    # 增加辐射源
    def addSource(self):
        self.source_count += 1  # 辐射源数量 + 1
        object_name = f"Source {self.source_count}"
        group_box = QGroupBox(object_name)  # 创建新的分组框
        source_layout = QFormLayout()  # 创建新布局

        # 第一行
        row1_layout = QHBoxLayout()
        source_a1 = QDoubleSpinBox()
        source_a1.setRange(0, 10000)
        source_a1.setValue(0)
        row1_layout.addWidget(QLabel("辐射源振幅"))
        row1_layout.addWidget(source_a1)

        source_fl1 = QDoubleSpinBox()
        source_fl1.setRange(0, 10000)
        source_fl1.setValue(0)
        row1_layout.addWidget(QLabel("辐射源频率(Hz)"))
        row1_layout.addWidget(source_fl1)
        source_layout.addRow(row1_layout)

        group_box.setLayout(source_layout)  # 设置布局
        self.source_layout.addWidget(group_box)  # 加入分组框

    # 减少辐射源
    def removeSource(self):
        if self.source_count > 0:
            # 移除最后一个对象
            item = self.source_tree.topLevelItem(self.source_count - 1)
            self.source_tree.takeTopLevelItem(self.source_count - 1)
            del item  # 删除 QTreeWidgetItem 对象

            # 移除最后一个分组框
            if self.source_layout.count() > 0:
                widget_to_remove = self.source_layout.itemAt(self.source_layout.count() - 1).widget()
                self.source_layout.removeWidget(widget_to_remove)
                widget_to_remove.deleteLater()

            self.source_count = max(0, self.source_count - 1)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_win = MainWindow()
    sys.exit(app.exec_())
