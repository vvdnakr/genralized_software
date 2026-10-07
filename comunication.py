import sys
import os
import yaml
import random
from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtCore import QCoreApplication
import pexpect
from PyQt6.QtWidgets import QInputDialog
import shutil
import subprocess
import re
import platform
import subprocess
from subprocess import call

# Check for Wayland or X11
# if sys.platform == 'windows' :
#     print("Running under windows. Will not use requestActivate()")
#     os.environ["QT_QPA_PLATFORM"] = "windows"
# else:
#     print("Running under X11. Using requestActivate()")
#     os.environ["QT_QPA_PLATFORM"] = "xcb"  
# os.environ["QT_QPA_PLATFORM"] = "windows"
if platform.system() == "Windows":
    os.environ["QT_QPA_PLATFORM"] = "windows"
else:
    os.environ["QT_QPA_PLATFORM"] = "xcb"


class Ui_MainWindow(object):
    def __init__(self):
        # Initialize configuration dictionaries
        self.area = {}      
        self.side_com = {}
        self.area_com = {}
        self.check_area = {}        
        self.cam_key_map = {}  # maps cam-UI-name to internal key

        # Flag to indicate if communication data is already saved
        self.comm_data_exists = False

    def onSideChanged(self, index):
        # ——— 1) save old side ———
        

        # ——— 2) clear ALL inputs ———

        # ── Clear Results controls ───────────────────────────────
        self.chkAreaWise.setChecked(False)
        self.chkCombine .setChecked(False)
        self.chkDetectAll.setChecked(False)
        self.finalResult1.clear()
        self.finalResult2.clear()

        # Side panel
        self.inFromPlcSideInput.clear()
        self.outFromVisionSideInput.clear()

        # Cam panel
        self.plcInfoTextEdit.clear()

        # Area panel
        self.areaComboBox.clear()
        self.areaComboBox.setVisible(False)
        self.areaLabel.setVisible(False)
        self.inputBox.setVisible(False)
        self.inFromPlcInput.clear()
        self.outFromVisionOkInput.clear()
        self.outFromVisionNgInput.clear()

        # ——— 3) rebuild combo-boxes for new side ———
        self.updateSideLabel()       # repopulates camComboBoxbox
        self.updateCamLabel()        # repopulates PLC info (blank for now)
        # leave areaComboBox hidden until cam is picked

        # ——— 4) if we already have comm-data for new side, fill it back in ———
        new_side = self.sideComboBox.currentText()
        side_conf = self.side_com.get(new_side)
        if side_conf:
            self.inFromPlcSideInput.setText(side_conf.get('PLC', ''))
            self.outFromVisionSideInput.setText(side_conf.get('vision', ''))

        # also immediately load camera-level PLC if it exists
        selected_cam = self.camComboBox.currentText()
        cam_conf = self.area_com.get(new_side, {}).get(selected_cam, {})
        if cam_conf:
            self.plcInfoTextEdit.setPlainText(cam_conf.get('PLC', ''))



    def loadCommunicationData(self):
            """
            Check if the active communication YAML file is empty.
            An empty file should contain {}.
            """
            folder_path = os.path.join("vision_data", self.active_part, "data", "yaml", "communication_data")
            self.file_path = os.path.join(folder_path, "comm_data.yaml")
            if os.path.exists(self.file_path):
                with open(self.file_path, "r") as file:
                    data = yaml.safe_load(file)
                    # Treat empty dictionary as no communication data
                    if data and data != {}:
                        self.comm_data_exists = True
                    else:
                        self.comm_data_exists = False
            else:
                self.comm_data_exists = False


    def save_data_to_yaml(self, file_path):
        # Load existing data to preserve unsaved cams
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                try:
                    existing_data = yaml.safe_load(f) or {}
                except yaml.YAMLError:
                    existing_data = {}
        else:
            existing_data = {}

        for side_key, side_config in self.side_com.items():
            entry = {
                'PLC':   side_config.get('PLC', ''),
                'Vision': side_config.get('vision', ''),
                'AreaWiseResult':  side_config.get('area_wise', False),
                'CombineResult':   side_config.get('combine',   False),
                'DetectAll':       side_config.get('detect_all', False),
                'FinalResult':     side_config.get('final_result', ['', '']),
            }

            # preserve cams if not overwritten below
            preserved = existing_data.get(side_key, {})
            preserved.update(entry)
            existing_data[side_key] = preserved

            if side_key in self.area_com:
                for ui_cam_label, cam_config in self.area_com[side_key].items():
                    # Convert cam-1 (UI) to cam-0 (YAML)
                    if ui_cam_label.startswith("cam-") and ui_cam_label.split("-")[1].isdigit():
                        shifted_cam_index = int(ui_cam_label.split("-")[1]) - 1
                        if shifted_cam_index < 0:
                            continue
                        shifted_cam_key = f"cam-{shifted_cam_index}"
                    else:
                        shifted_cam_key = ui_cam_label

                    cam_data = {}
                    if cam_config.get('PLC'):
                        cam_data['PLC'] = cam_config['PLC']

                    for area_key, area_values in cam_config.items():
                        if area_key.startswith('area_') and any(area_values.values()):
                            cam_data[area_key] = {
                                'PLC': area_values.get('PLC', ''),
                                'OK':  area_values.get('OK',  ''),
                                'NG':  area_values.get('NG',  '')
                            }

                    if cam_data:
                        existing_data[side_key][shifted_cam_key] = cam_data

        with open(file_path, 'w') as file:
            yaml.dump(existing_data, file, default_flow_style=False, sort_keys=False)
    def setupUi(self, MainWindow, active_part="default_active_part"):
        self.active_part = active_part
        # Make sure the folder exists
        folder_path = os.path.join("vision_data", self.active_part, "data", "yaml", "communication_data")
        os.makedirs(folder_path, exist_ok=True)
        self.file_path = os.path.join(folder_path, "comm_data.yaml")
        # Only create a new YAML file if it does not exist or is empty
        if not os.path.exists(self.file_path) or os.stat(self.file_path).st_size == 0:
            self.save_data_to_yaml(file_path=self.file_path)
        # Check if communication data exists (i.e. not None)
        self.loadCommunicationData()
        self.loadYamlData()
        # remove or comment out:
# self.sideComboBox.currentIndexChanged.connect(self.updateSideLabel)

# add instead:


        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1200,800)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.centralwidget.setStyleSheet("background-color: white")
        
        self.stackedWidget = QtWidgets.QStackedWidget(self.centralwidget)
        self.stackedWidget.setGeometry(QtCore.QRect(0, 0, 1200, 800))
        self.stackedWidget.setObjectName("stackedWidget")

        # ------------------ Communication Page ------------------
        self.communicationPage = QtWidgets.QWidget()
        self.communicationPage.setObjectName("communicationPage")

        self.communicationLabel = QtWidgets.QLabel(self.communicationPage)
        self.communicationLabel.setGeometry(QtCore.QRect(450, 200, 300, 50))
        font = QtGui.QFont()
        font.setPointSize(20)
        font.setBold(True)
        self.communicationLabel.setFont(font)
        self.communicationLabel.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.communicationLabel.setText("Communication")

        self.hardwareButton = QtWidgets.QPushButton(self.communicationPage)
        self.hardwareButton.setGeometry(QtCore.QRect(350, 300, 150, 40))
        self.hardwareButton.setText("Hardware")
        self.hardwareButton.setStyleSheet("background-color: #243447")
        self.hardwareButton.clicked.connect(self.hardwarePlaceholder)

        self.softwareButton = QtWidgets.QPushButton(self.communicationPage)
        self.softwareButton.setGeometry(QtCore.QRect(700, 300, 150, 40))
        self.softwareButton.setText("Software")
        self.softwareButton.setStyleSheet("background-color: #243447")
        self.softwareButton.clicked.connect(self.softwareButtonClicked)

        self.stackedWidget.addWidget(self.communicationPage)

        # ------------------ IP and TCP Configuration Page ------------------
        self.ipConfigPage = QtWidgets.QWidget()
        self.ipConfigPage.setObjectName("ipConfigPage")

        self.ipLabel = QtWidgets.QLabel(self.ipConfigPage)
        self.ipLabel.setGeometry(QtCore.QRect(450, 150, 100, 30))
        font.setPointSize(12)
        self.ipLabel.setFont(font)
        self.ipLabel.setText("IP Address:")

        self.ipInput = QtWidgets.QLineEdit(self.ipConfigPage)
        self.ipInput.setGeometry(QtCore.QRect(600, 150, 200, 30))
        ip_regex = QtCore.QRegularExpression(r'^(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$')
        self.ipInput.setValidator(QtGui.QRegularExpressionValidator(ip_regex))
        self.ipInput.setPlaceholderText("Enter IP Address")

        self.tcpLabel = QtWidgets.QLabel(self.ipConfigPage)
        self.tcpLabel.setGeometry(QtCore.QRect(450, 200, 100, 30))
        self.tcpLabel.setFont(font)
        self.tcpLabel.setText("TCP Port:")

        self.tcpInput = QtWidgets.QLineEdit(self.ipConfigPage)
        self.tcpInput.setGeometry(QtCore.QRect(600, 200, 200, 30))
        tcp_regex = QtCore.QRegularExpression(r'^[0-9]+$')
        self.tcpInput.setValidator(QtGui.QRegularExpressionValidator(tcp_regex))
        self.tcpInput.setPlaceholderText("Enter TCP Port")

        self.toggalLable = QtWidgets.QLabel(self.ipConfigPage)
        self.toggalLable.setGeometry(450,250,200,30)
        self.toggalLable.setFont(font)
        self.toggalLable.setText("Share Image Data : No")


        self.toggal = QtWidgets.QSlider(self.ipConfigPage)
        self.toggal.setOrientation(QtCore.Qt.Orientation.Horizontal)
        self.toggal.setRange(0, 1)
        self.toggal.setValue(0)
        self.toggal.setGeometry(650,250,30,30)
        self.toggal.valueChanged.connect(self.updatelabel)

        self.saveIpButton = QtWidgets.QPushButton(self.ipConfigPage)
        self.saveIpButton.setGeometry(QtCore.QRect(555, 300, 100, 30))
        self.saveIpButton.setText("Save")
        self.saveIpButton.clicked.connect(self.saveIpConfiguration)

        # Back button remains
        self.backButtonIp = QtWidgets.QPushButton(self.ipConfigPage)
        self.backButtonIp.setGeometry(QtCore.QRect(30, 400, 100, 30))
        self.backButtonIp.setText("Back")
        self.backButtonIp.clicked.connect(self.show_option_HARD_SOFT)

        # New: Show and Edit buttons (only visible when comm data exists)
        self.showButton = QtWidgets.QPushButton(self.ipConfigPage)
        self.showButton.setGeometry(QtCore.QRect(500, 400, 100, 30))
        self.showButton.setText("Show")
        self.showButton.setStyleSheet("background-color: #2E8B57")
        self.showButton.clicked.connect(self.showCommDataPage)
        self.showButton.setVisible(False)

        self.editButton = QtWidgets.QPushButton(self.ipConfigPage)
        self.editButton.setGeometry(QtCore.QRect(620, 400, 100, 30))
        self.editButton.setText("Edit")
        self.editButton.setStyleSheet("background-color: #4682B4")
        self.editButton.clicked.connect(self.editCommData)
        self.editButton.setVisible(False)

        self.stackedWidget.addWidget(self.ipConfigPage)


        


        # ------------------ Combo Box Page (for side/area configuration) ------------------
        self.comboBoxPage = QtWidgets.QWidget()
        self.comboBoxPage.setObjectName("comboBoxPage")
        # [Existing code for comboBoxPage unchanged …]
        # Side ComboBox and Label
        self.sideComboBox = QtWidgets.QComboBox(self.comboBoxPage)
        self.sideComboBox.setGeometry(QtCore.QRect(50, 50, 250, 35))
        self.sideComboBox.setObjectName("sideComboBox")
        # Populate the combo box with sides generated from YAML data
        self.sideComboBox.addItem("SIDE")
        self.sideComboBox.addItems(self.yaml_sides)
        self.sideComboBox.currentIndexChanged.connect(self.onSideChanged)


        self.camComboBox = QtWidgets.QComboBox(self.comboBoxPage)
        self.camComboBox.setGeometry(QtCore.QRect(50, 400, 250, 35))  # 100px below side combo
        self.camComboBox.setObjectName("camComboBox")
        # Populate the combo box with sides generated from YAML data
        self.camComboBox.addItem("CAM")
        # self.camComboBox.addItems(self.yaml_cams)
        self.camComboBox.currentIndexChanged.connect(self.updateCamLabel)
        # In setupUi() under comboBoxPage setup
        self.camComboBox.currentIndexChanged.connect(self.updateAreaComboBox)

        

        

        self.sideLabel = QtWidgets.QLabel(self.comboBoxPage)
        self.sideLabel.setGeometry(QtCore.QRect(50, 90, 250, 20))  # Below side combo
        font.setPointSize(12)
        self.sideLabel.setFont(font)
        self.sideLabel.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.sideLabel.setText("Selected Side: side 1")
        self.sideComboBox.currentIndexChanged.connect(self.updateSideLabel)

        self.camLabel = QtWidgets.QLabel(self.comboBoxPage)
        self.camLabel.setGeometry(QtCore.QRect(50, 350, 250, 20))  # Below cam combo
        font.setPointSize(12)
        self.camLabel.setFont(font)
        self.camLabel.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.camLabel.setText("Selected CAM: CAM ")
        # self.camComboBox.currentIndexChanged.connect(self.updateCLabel)

        # Side Configuration Group Box
        self.sideConfigBox = QtWidgets.QGroupBox(self.comboBoxPage)
        self.sideConfigBox.setGeometry(QtCore.QRect(700, 10, 400, 330))
        self.sideConfigBox.setFont(font)
        self.sideConfigBox.setTitle("Side Configuration")

        self.camConfigBox = QtWidgets.QGroupBox(self.comboBoxPage)
        self.camConfigBox.setGeometry(QtCore.QRect(700, 350, 400, 200))  # Reduced height
        self.camConfigBox.setFont(font)
        self.camConfigBox.setTitle("Cam Configuration")
############################################################
                # ── Results Configuration ───────────────────────────────
        self.resultsGroup = QtWidgets.QGroupBox(self.sideConfigBox)
        self.resultsGroup.setTitle("Results")
        self.resultsGroup.setGeometry(QtCore.QRect(20, 110, 360, 150))
        twenty = 20
        forty = 40
        sixty = 60

        # Checkboxes
        self.chkAreaWise = QtWidgets.QCheckBox("Area-wise result", self.resultsGroup)
        self.chkAreaWise.setGeometry(20,  30, 150, 20)

        self.chkCombine  = QtWidgets.QCheckBox("Combine result",   self.resultsGroup)
        self.chkCombine.setGeometry(20,  65, 150, 20)

        self.chkDetectAll = QtWidgets.QCheckBox("Detect all",      self.resultsGroup)
        self.chkDetectAll.setGeometry(20,  95, 150, 20)

        # Final result inputs
        lblFinal = QtWidgets.QLabel("Final result:", self.resultsGroup)
        lblFinal.setGeometry(200, 20, 100, 20)
        self.finalResult1 = QtWidgets.QLineEdit(self.resultsGroup)
        self.finalResult1.setGeometry(200,  forty, 70,  twenty)
        self.finalResult2 = QtWidgets.QLineEdit(self.resultsGroup)
        self.finalResult2.setGeometry(280,  forty, 70,  twenty)
#######################################################
        self.plcInfoLabel = QtWidgets.QLabel(self.camConfigBox)
        self.plcInfoLabel.setGeometry(QtCore.QRect(20, 30, 150, 20))  # Smaller label
        self.plcInfoLabel.setText("In from PLC:")
        
        self.plcInfoTextEdit = QtWidgets.QTextEdit(self.camConfigBox)
        self.plcInfoTextEdit.setGeometry(QtCore.QRect(20, 55, 150, 30))  # Compact size
        # Save Cam Button
        self.saveCamButton = QtWidgets.QPushButton(self.camConfigBox)
        self.saveCamButton.setGeometry(QtCore.QRect(150, 150, 200, 35))  # Moved down
        self.saveCamButton.setText("Save Camera Configuration")
        self.saveCamButton.clicked.connect(self.saveCamConfiguration)


        self.inFromPlcSideLabel = QtWidgets.QLabel(self.sideConfigBox)
        self.inFromPlcSideLabel.setGeometry(QtCore.QRect(20, 30, 150, 30))
        self.inFromPlcSideLabel.setText("In from PLC:")
        self.inFromPlcSideInput = QtWidgets.QLineEdit(self.sideConfigBox)
        self.inFromPlcSideInput.setGeometry(QtCore.QRect(180, 30, 200, 30))

        self.outFromVisionSideLabel = QtWidgets.QLabel(self.sideConfigBox)
        self.outFromVisionSideLabel.setGeometry(QtCore.QRect(20, 80, 150, 30))
        self.outFromVisionSideLabel.setText("Out from Vision:")
        self.outFromVisionSideInput = QtWidgets.QLineEdit(self.sideConfigBox)
        self.outFromVisionSideInput.setGeometry(QtCore.QRect(180, 80, 200, 30))

        self.saveSideButton = QtWidgets.QPushButton(self.sideConfigBox)
        self.saveSideButton.setGeometry(QtCore.QRect(150, 280, 100, 30))
        self.saveSideButton.setText("Save Side")
        self.saveSideButton.clicked.connect(self.saveSideConfiguration)

        # self.inFromPlcCamLabel = QtWidgets.QLabel(self.sideConfigBox)
        # self.inFromPlcCamLabel.setGeometry(QtCore.QRect(20, 30, 150, 30))
        # self.inFromPlcCamLabel.setText("In from PLC:")
        # self.inFromPlcCamInput = QtWidgets.QLineEdit(self.sideConfigBox)
        # self.inFromPlcCamInput.setGeometry(QtCore.QRect(180, 30, 200, 30))
        

        
        # self.saveSideButton = QtWidgets.QPushButton(self.sideConfigBox)
        # self.saveSideButton.setGeometry(QtCore.QRect(150, 130, 100, 30))
        # self.saveSideButton.setText("Save Side")
        # self.saveSideButton.clicked.connect(self.saveSideConfiguration)


        # Area ComboBox and Label
        self.areaComboBox = QtWidgets.QComboBox(self.comboBoxPage)
        self.areaComboBox.setGeometry(QtCore.QRect(50, 650, 200, 30))
        self.areaComboBox.setObjectName("areaComboBox")
        self.areaComboBox.setVisible(False)

        self.areaLabel = QtWidgets.QLabel(self.comboBoxPage)
        self.areaLabel.setGeometry(QtCore.QRect(50, 600, 200, 30))
        self.areaLabel.setFont(font)
        self.areaLabel.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.areaLabel.setText("Select Area:")
        self.areaLabel.setVisible(False)


        # Input Box for Area Configuration (Initially Hidden)
        self.inputBox = QtWidgets.QGroupBox(self.comboBoxPage)
        self.inputBox.setGeometry(QtCore.QRect(700, 560, 400, 220))
        self.inputBox.setFont(font)
        self.inputBox.setTitle("Area Configuration")
        self.inputBox.setVisible(True)

        self.inFromPlcLabel = QtWidgets.QLabel(self.inputBox)
        self.inFromPlcLabel.setGeometry(QtCore.QRect(20, 30, 150, 30))
        self.inFromPlcLabel.setText("In from PLC:")
        self.inFromPlcInput = QtWidgets.QLineEdit(self.inputBox)
        self.inFromPlcInput.setGeometry(QtCore.QRect(180, 30, 200, 30))

        self.outFromVisionOkLabel = QtWidgets.QLabel(self.inputBox)
        self.outFromVisionOkLabel.setGeometry(QtCore.QRect(20, 80, 150, 30))
        self.outFromVisionOkLabel.setText("Out from Vision/OK:")
        self.outFromVisionOkInput = QtWidgets.QLineEdit(self.inputBox)
        self.outFromVisionOkInput.setGeometry(QtCore.QRect(180, 80, 200, 30))

        self.outFromVisionNgLabel = QtWidgets.QLabel(self.inputBox)
        self.outFromVisionNgLabel.setGeometry(QtCore.QRect(20, 130, 150, 30))
        self.outFromVisionNgLabel.setText("Out from Vision/NG:")
        self.outFromVisionNgInput = QtWidgets.QLineEdit(self.inputBox)
        self.outFromVisionNgInput.setGeometry(QtCore.QRect(180, 130, 200, 30))

        self.saveAreaButton = QtWidgets.QPushButton(self.inputBox)
        self.saveAreaButton.setGeometry(QtCore.QRect(150, 180, 100, 30))
        self.saveAreaButton.setText("Save Area")
        self.saveAreaButton.clicked.connect(self.saveAreaConfiguration)
        import config
        self.setTheme(isDark=config.IS_DARK)

        self.backButtonCombo = QtWidgets.QPushButton(self.comboBoxPage)
        self.backButtonCombo.setGeometry(QtCore.QRect(50, 700, 100, 35))  # Moved down
        self.backButtonCombo.setText("Back")
        self.backButtonCombo.clicked.connect(self.showIpConfigPage)

        self.stackedWidget.addWidget(self.comboBoxPage)
        self.camComboBox.currentIndexChanged.connect(self.updateAreaComboBox)
        self.areaComboBox.currentIndexChanged.connect(self.showInputBox)

        # (rest of your comboBoxPage and dynamic resizing code remains as is)
        # …
        MainWindow.setCentralWidget(self.centralwidget)
        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        # Internal variables
        self.ip_data = {}
        self.side_data = {}
        self.area_data = {}

        self.base_width = 1130
        self.base_height = 609
        self.original_geometries = {}
        widgets_to_scale = [
            self.plcInfoTextEdit,
            self.plcInfoLabel,
            self.saveCamButton,
            self.camConfigBox,
            self.stackedWidget,
            self.communicationLabel,
            self.hardwareButton,
            self.softwareButton,
            self.ipLabel,
            self.ipInput,
            self.tcpLabel,
            self.tcpInput,
            self.saveIpButton,
            self.backButtonIp,
            self.sideComboBox,
            self.sideLabel,
            self.sideConfigBox,
            self.inFromPlcSideLabel,
            self.inFromPlcSideInput,
            self.outFromVisionSideLabel,
            self.outFromVisionSideInput,
            self.saveSideButton,
            self.areaComboBox,
            self.areaLabel,
            self.inputBox,
            self.inFromPlcLabel,
            self.inFromPlcInput,
            self.outFromVisionOkLabel,
            self.outFromVisionOkInput,
            self.outFromVisionNgLabel,
            self.outFromVisionNgInput,
            self.saveAreaButton,
            self.backButtonCombo
            # … include other widgets as needed
        ]
        for widget in widgets_to_scale:
            self.original_geometries[widget] = widget.geometry()

    def updatelabel(self,value):
        if value ==1:
            self.toggalLable.setText("Share Image Data : Yes")
        else:
            self.toggalLable.setText("Share Image Data : No")


    def showCenteredMessage(self, parent, title, text, icon=QtWidgets.QMessageBox.Icon.Information, timeout=None):
        """
        Shows a QMessageBox centered over `parent`.
        If timeout (ms) is given, auto‑closes after that many milliseconds.
        """
        msg = QtWidgets.QMessageBox(parent)
        msg.setWindowTitle(title)
        msg.setText(text)
        msg.setIcon(icon)
        msg.setStandardButtons(QtWidgets.QMessageBox.StandardButton.Ok)

        # Center over parent
        parent_rect = parent.frameGeometry()
        msg_rect    = msg.frameGeometry()
        x = parent_rect.x() + (parent_rect.width()  - msg_rect.width())  // 2
        y = parent_rect.y() + (parent_rect.height() - msg_rect.height()) // 2
        msg.move(x, y)

        if timeout:
            QtCore.QTimer.singleShot(timeout, msg.accept)
        msg.exec()



    def saveSideConfiguration(self):
        selected_side = self.sideComboBox.currentText()
        if selected_side == "SIDE":
            return self.showCenteredMessage(self.centralwidget, "Input Required",
           "Please choose a side first!", QtWidgets.QMessageBox.Icon.Warning, 1000)
        in_from_plc = self.inFromPlcSideInput.text()
        out_from_vision = self.outFromVisionSideInput.text()

        if in_from_plc and out_from_vision:
            self.side_com[selected_side]['PLC'] = in_from_plc
            self.side_com[selected_side]['vision'] = out_from_vision
            # ── Capture Results settings ─────────────────────────────
            self.side_com[selected_side]['area_wise']    = self.chkAreaWise.isChecked()
            self.side_com[selected_side]['combine']      = self.chkCombine.isChecked()
            self.side_com[selected_side]['detect_all']   = self.chkDetectAll.isChecked()
            self.side_com[selected_side]['final_result'] = [
                self.finalResult1.text(),
                self.finalResult2.text()
            ]
            self.save_data_to_yaml(file_path=self.file_path)
            # QtWidgets.QMessageBox.information(None, "Success", f"Configuration for {selected_side} saved!")
            self.showCenteredMessage(
            parent=self.centralwidget,
            title="Success",
            text=f"Configuration for {selected_side} saved!",
            icon=QtWidgets.QMessageBox.Icon.Information,
            timeout=1000  # auto‑close after 1s
        )
        else:
            # QtWidgets.QMessageBox.warning(None, "Input Required", "Please fill all fields for Side Configuration!")
            msg_box = QtWidgets.QMessageBox()
            msg_box.setWindowTitle("Input Required")
            msg_box.setText( "Please fill all fields for Side Configuration!")
            msg_box.setIcon(QtWidgets.QMessageBox.Icon.Information)
            msg_box.setStandardButtons(QtWidgets.QMessageBox.StandardButton.Ok)
            QtCore.QTimer.singleShot(1000, lambda: msg_box.done(0))
            msg_box.exec()

    def showInputBox(self):
        side = self.sideComboBox.currentText()
        cam = self.camComboBox.currentText()
        area_display = self.areaComboBox.currentText()
        
        if area_display:
            # Convert "Area 1" to "area_0"
            area_num = int(area_display.split()[-1]) - 1
            area_key = f"area_{area_num}"
            
            if (side in self.area_com and 
                cam in self.area_com[side] and 
                area_key in self.area_com[side][cam]):
                config = self.area_com[side][cam][area_key]
                self.inFromPlcInput.setText(config.get('PLC', ''))
                self.outFromVisionOkInput.setText(config.get('OK', ''))
                self.outFromVisionNgInput.setText(config.get('NG', ''))
        
        self.inputBox.setVisible(True)

    def saveAreaConfiguration(self):
        side = self.sideComboBox.currentText()
        cam = self.camComboBox.currentText()
        area_display = self.areaComboBox.currentText()
        
        if not area_display:
            return
            
        # Convert "Area 1" to "area_0"
        area_num = int(area_display.split()[-1]) - 1
        area_key = f"area_{area_num}"
        
        if side not in self.area_com:
            self.area_com[side] = {}
        if cam not in self.area_com[side]:
            self.area_com[side][cam] = {}
        
        self.area_com[side][cam][area_key] = {
            "PLC": self.inFromPlcInput.text(),
            "OK": self.outFromVisionOkInput.text(),
            "NG": self.outFromVisionNgInput.text()
        }
        self.save_data_to_yaml(self.file_path)
        self.showCenteredMessage(
            parent=self.centralwidget,
            title="Success",
            text="Area details saved!",
            icon=QtWidgets.QMessageBox.Icon.Information,
            timeout=1000
        )
    


    def loadYamlData(self):
        """
        Load sides, cameras, and areas from the active part's YAML file.
        Supports both 'area0' and 'area_0' key formats.
        """
        yaml_path = os.path.join(
            "vision_data",
            self.active_part,
            "data",
            "yaml",
            f"{self.active_part}.yaml"
        )
        if not os.path.exists(yaml_path):
            # No YAML file: initialize empty structures
            self.yaml_sides = []
            self.total_rois = {}
            return

        with open(yaml_path, "r") as file:
            yaml_data = yaml.safe_load(file) or {}

        # Sides in the YAML
        self.yaml_sides = list(yaml_data.keys())
        self.total_rois = {}

        # Build total_rois: side -> cam -> [area indices]
        for side, cams in yaml_data.items():
            self.total_rois[side] = {}
            for cam_key, cam_content in cams.items():
                if not str(cam_key).startswith("cam-"):
                    continue
                self.total_rois[side][cam_key] = []
                # Find all area keys matching 'area0', 'area_0', etc.
                for area_key in cam_content:
                    match = re.match(r"^area_?(\d+)$", area_key)
                    if match:
                        idx = int(match.group(1))
                        self.total_rois[side][cam_key].append(idx)

        # Optionally sort area indices for consistency
        for side in self.total_rois:
            for cam in self.total_rois[side]:
                self.total_rois[side][cam].sort()
        self.buildConfigurationsFromYaml()
     

    def buildConfigurationsFromYaml(self):
        self.side_com = {}
        self.area_com = {}
        
        for side in self.yaml_sides:
            # Initialize side config
            self.side_com[side] = {'PLC': '', 'vision': ''}
            
            # Initialize camera structure
            self.area_com[side] = {}
            for cam in self.total_rois.get(side, {}):
                self.area_com[side][cam] = {'PLC': ''}
                # Initialize areas using stored indices
                for area_idx in self.total_rois[side][cam]:
                    area_key = f"area_{area_idx}"
                    self.area_com[side][cam][area_key] = {
                        'PLC': '',
                        'OK': '',
                        'NG': ''
                    }

    def updateSideLabel(self):
        selected_side = self.sideComboBox.currentText()
        self.sideLabel.setText(f"Selected Side: {selected_side}")
        
        self.camComboBox.clear()
        
        if selected_side in self.total_rois:
            cams = list(self.total_rois[selected_side].keys())
            
            # Display cam-1, cam-2, ... instead of cam-0, cam-1, ...
            display_names = [f"cam-{int(cam.split('-')[1]) + 1}" for cam in cams]
            self.camComboBox.addItems(display_names)
            
            # Map displayed cam names back to actual keys
            self.cam_key_map = {display: original for display, original in zip(display_names, cams)}
        
        self.updateCamLabel()

    def updateCamLabel(self):
        selected_side = self.sideComboBox.currentText()
        selected_cam_ui = self.camComboBox.currentText()

        # Convert from UI cam label (e.g., cam-1) to internal key (e.g., cam-0)
        selected_cam = self.cam_key_map.get(selected_cam_ui, selected_cam_ui)

        try:
            cam_number = int(selected_cam.split('-')[1]) + 1
        except:
            cam_number = selected_cam_ui  # fallback if unexpected format

        self.camLabel.setText(f"Selected Camera: {cam_number}")

        # Load existing PLC info if available
        if selected_side in self.side_com and selected_cam in self.side_com[selected_side]:
            self.plcInfoTextEdit.setPlainText(
                self.side_com[selected_side][selected_cam].get("PLC_Info", "")
            )

    def saveCamConfiguration(self):
        selected_side = self.sideComboBox.currentText()
        selected_cam = self.camComboBox.currentText()
        plc_info = self.plcInfoTextEdit.toPlainText()

        # Update data structure
        if selected_side not in self.area_com:
            self.area_com[selected_side] = {}
        
        if selected_cam not in self.area_com[selected_side]:
            self.area_com[selected_side][selected_cam] = {}
        
        self.area_com[selected_side][selected_cam]['PLC'] = plc_info
        self.save_data_to_yaml(self.file_path)
        

        # Show a 1-second “Saved!” notification
        self.showCenteredMessage(
            parent=self.centralwidget,
            title="Success",
            text="Camera details saved!",
            icon=QtWidgets.QMessageBox.Icon.Information,
            timeout=1000
        )
        # self.showCenteredMessage("Success", "Camera configuration saved!", timeout=1000)



    def updateAreaComboBox(self):
        selected_side = self.sideComboBox.currentText()
        selected_cam_ui = self.camComboBox.currentText()

        # If it doesn't look like "cam-<number>", bail out early
        parts = selected_cam_ui.split('-')
        if len(parts) != 2 or not parts[1].isdigit():
            self.areaComboBox.clear()
            self.areaComboBox.setVisible(False)
            self.areaLabel.setVisible(False)
            return

        # Convert back to zero-based key
        cam_index = int(parts[1]) - 1
        internal_cam_key = f"cam-{cam_index}"

        # Now show the area list only if that camera really exists
        if (selected_side in self.total_rois and 
            internal_cam_key in self.total_rois[selected_side]):
            
            self.areaComboBox.clear()
            self.areaLabel.setVisible(True)
            self.areaComboBox.setVisible(True)

            areas = [
                f"Area {idx + 1}"
                for idx in self.total_rois[selected_side][internal_cam_key]
            ]
            self.areaComboBox.addItems(areas)
        else:
            # No areas for this side/cam combo
            self.areaComboBox.clear()
            self.areaComboBox.setVisible(False)
            self.areaLabel.setVisible(False)



    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Communication"))

    # ------------------ Signal Handlers ------------------
    def hardwarePlaceholder(self):
        QtWidgets.QMessageBox.information(None, "Hardware", "Hardware functionality is not implemented yet!")

    def softwareButtonClicked(self):
        # When Software is clicked, show the IP config page.
        self.stackedWidget.setCurrentWidget(self.ipConfigPage)
        # If communication data exists, reveal the Show/Edit buttons.
        if self.comm_data_exists:
            self.showButton.setVisible(True)
            self.editButton.setVisible(True)
        else:
            self.showButton.setVisible(False)
            self.editButton.setVisible(False)

    def show_option_HARD_SOFT(self):
        self.stackedWidget.setCurrentWidget(self.communicationPage) 

    def saveIpConfiguration(self):
        # Get IP and Port from inputs
        ip_address = self.ipInput.text().strip()
        tcp_port   = self.tcpInput.text().strip()

        if not ip_address or not tcp_port:
            return self.showCenteredMessage(
                parent=self.centralwidget,
                title="Input Error",
                text="Both IP Address and TCP Port are required!",
                icon=QtWidgets.QMessageBox.Icon.Warning,
                timeout=2000
            )

        # Ask for sudo password
        path = os.path.join("user_pass.yaml")

        if not os.path.exists(path):
            return self.showCenteredMessage(
                parent=self.centralwidget,
                title="Error",
                text="User password file not found!",
                icon=QtWidgets.QMessageBox.Critical,
                timeout=2000
            )

        with open(path, 'r') as file:
            user_data = yaml.safe_load(file) or {}

        if 'sudo_password' not in user_data:
            sudo_password, ok = QInputDialog.getText(
                self.centralwidget,
                "Authentication Required",
                "Enter sudo password:",
                QtWidgets.QLineEdit.Password
            )
            if not ok or not sudo_password:
                return self.showCenteredMessage(
                    parent=self.centralwidget,
                    title="Cancelled",
                    text="Operation cancelled by user",
                    icon=QtWidgets.QMessageBox.Information,
                    timeout=1000
                )

            # Save the new sudo password
            user_data['sudo_password'] = sudo_password
            with open(path, 'w') as file:
                yaml.dump(user_data, file)

            print("Sudo password saved successfully.")
            return self.showCenteredMessage(
                parent=self.centralwidget,
                title="Success",
                text="Sudo password saved successfully.",
                icon=QtWidgets.QMessageBox.Information,
                timeout=1000
            )
        else:
            sudo_password = user_data['sudo_password']
            print(sudo_password, "==================")

            
        
            # self.stackedWidget.setCurrentWidget(self.comboBoxPage)
        


        


        def change_system_ip(ip, password):
            if platform.system() != "windows":
                try:
                    command = "ip link show | grep -E '^[0-9]+: (en|eth)' | awk '{print $2}' | sed 's/:$//'"
                    # command="ls /etc/netplan/"

                    result = subprocess.run(command, shell=True, capture_output=True, text=True)
                    eter_name=result.stdout.replace("\n",'')
                    print(eter_name,"++++++++++++++++++++++++++++++===")
                    name=f'/etc/netplan/01-network-manager-all.yaml'

                    # Print the output
                    pwd=password

                    cmd=f'touch {name}'

                    call('echo {} | sudo -S {}'.format(pwd, cmd), shell=True)


                    cmd=f'chmod o+w {name}'

                    call('echo {} | sudo -S {}'.format(pwd, cmd), shell=True)
                    print(eter_name,"------------")
                    dicte={'network':
                                {'version': 2,
                                'renderer': 'NetworkManager',
                                'ethernets':
                                {eter_name:
                                    {'dhcp4': "no",
                                    'addresses': [ f"{ip}/24"]}}}}


                    with open(name, "w") as f:
                            yaml.dump(dicte,f,default_flow_style=False)

                    cmd=f'chmod o-w {name}'

                    call('echo {} | sudo -S {}'.format(pwd, cmd), shell=True)


                    cmd=f'netplan apply'
                    call('echo {} | sudo -S {}'.format(pwd, cmd), shell=True)


                    command=f"cat {name}"

                    result = subprocess.run(command, shell=True, capture_output=True, text=True)
                    out=result.stdout
                    err = result.stderr or ""
                    # Print the output
                    print(out,"------------")
                
                    if result.returncode == 0:
                        # vision_data/111/data/yaml/communication_data/comm_data.yaml
                        yaml_path = os.path.join(
                            "vision_data", self.active_part,
                            "data", "yaml", "communication_data", "ip_config.yaml"
                        )
                        # existing_data = {"ip":ip "port":tcp_port}
                        existing_data = {}
                        
                        existing_data["ip"] = ip
                        existing_data["port"] = tcp_port
                        existing_data["share_image_data"] = self.toggal.value()
                        

                        with open(yaml_path, 'w') as file:
                             yaml.dump(existing_data, file, default_flow_style=False, sort_keys=False)
                             

                        return True, out.strip() or "IP changed successfully"
                    return False, err.strip() or f"Exit code {result.returncode}"
                except Exception as e:
                    return False, str(e)
            elif platform.system == "windows":
                massage = "no work for windows under process"
                print(massage)
                return True,massage

        success, message = change_system_ip(ip_address, sudo_password)
        if success:
            self.ip_data = {"IP Address": ip_address, "TCP Port": tcp_port}
            self.showCenteredMessage(
                parent=self.centralwidget,
                title="Success",
                text=f"Configuration Saved!\n{message}",
                icon=QtWidgets.QMessageBox.Icon.Information,
                timeout=1500
            )
            if not self.comm_data_exists:
                QtCore.QTimer.singleShot(1700, lambda:
                    self.stackedWidget.setCurrentWidget(self.comboBoxPage)
                )
        else:
            self.showCenteredMessage(
                parent=self.centralwidget,
                title="Configuration Failed",
                text=f"Could not save configuration:\n{message}",
                icon=QtWidgets.QMessageBox.Icon.Critical,
                timeout=3000
            )

        
    # Columns: Side | Camera | Area | Field | Value
    def showCommDataPage(self):
        """
        Display communication data in a hierarchical tree: Side -> Camera -> Area.
        """
        # 1) Create the page and tree widget if needed
        if not hasattr(self, 'commDataShowPage'):
            self.commDataShowPage = QtWidgets.QWidget()
            layout = QtWidgets.QVBoxLayout(self.commDataShowPage)

            # Hierarchical tree widget
            self.commTree = QtWidgets.QTreeWidget()
            self.commTree.setHeaderLabels(["Name", "PLC", "Vision/OK", "NG"])
            self.commTree.header().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)
            layout.addWidget(self.commTree)

            # Back button
            backBtn = QtWidgets.QPushButton("Back")
            # backBtn.clicked.connect(lambda:
            #     self.stackedWidget.setCurrentWidget(self.comboBoxPage)
            # )
            layout.addWidget(backBtn)
            self.stackedWidget.addWidget(self.commDataShowPage)

        # 2) Load the YAML file
        yaml_path = os.path.join(
            "vision_data", self.active_part,
            "data", "yaml", "communication_data", "comm_data.yaml"
        )
        with open(yaml_path, 'r') as f:
            comm = yaml.safe_load(f) or {}

        # 3) Populate the tree hierarchy
        self.commTree.clear()
        for side, sc in comm.items():
            side_item = QtWidgets.QTreeWidgetItem([
                side,
                sc.get('PLC', ''),
                sc.get('Vision', ''),
                ""
            ])
            self.commTree.addTopLevelItem(side_item)

            for cam_key, cam_conf in sc.items():
                if not cam_key.startswith("cam-"):
                    continue

                # Display as cam-1, cam-2, ...
                cam_num = int(cam_key.split('-')[1]) + 1
                cam_display = f"cam-{cam_num}"

                cam_item = QtWidgets.QTreeWidgetItem([
                    cam_display,
                    cam_conf.get('PLC', ''),
                    "",
                    ""
                ])
                side_item.addChild(cam_item)

                # Areas under each camera
                area_keys = sorted(
                    [k for k in cam_conf if k.startswith('area_')],
                    key=lambda x: int(x.split('_')[1])
                )
                for area_key in area_keys:
                    idx = int(area_key.split('_')[1]) + 1  # display starts from 1
                    ac = cam_conf[area_key]
                    area_item = QtWidgets.QTreeWidgetItem([
                        f"Area {idx}",
                        ac.get('PLC', ''),
                        ac.get('OK', ''),
                        ac.get('NG', '')
                    ])
                    cam_item.addChild(area_item)

        # 4) Show the page
        self.stackedWidget.setCurrentWidget(self.commDataShowPage)






    def editCommData(self):
        """
        Load communication YAML and pre-fill Side, Camera, and Area configurations, including Camera PLC info.
        """
        yaml_path = os.path.join(
            "vision_data", self.active_part,
            "data", "yaml", "communication_data", "comm_data.yaml"
        )
        if not os.path.exists(yaml_path):
            QtWidgets.QMessageBox.warning(None, "File Not Found", "Saved communication data file not found.")
            return

        # Load YAML data
        with open(yaml_path, "r") as file:
            data = yaml.safe_load(file) or {}

        # Reset configurations
        self.side_com = {}
        self.area_com = {}

        # Build in-memory structures from YAML
        for side_key, config in data.items():
            # Side-level PLC, Vision, and result settings
            self.side_com[side_key] = {
                "PLC": config.get("PLC", ""), 
                "vision": config.get("Vision", ""),
                "AreaWiseResult": config.get('AreaWiseResult', False),
                "CombineResult": config.get('CombineResult', False),
                "DetectAll": config.get('DetectAll', False),
                "FinalResult": config.get('FinalResult', ['', ''])
            }
            
            # Camera entries under each side
            for cam_key, cam_conf in config.items():
                if not str(cam_key).startswith("cam-"):
                    continue
                # Ensure side exists in area_com
                self.area_com.setdefault(side_key, {})
                # Camera-level PLC
                self.area_com[side_key][cam_key] = { 'PLC': cam_conf.get('PLC', '') }
                # Area entries under each camera
                for area_key, area_vals in cam_conf.items():
                    if not area_key.startswith('area_'):
                        continue
                    # Copy PLC, OK, NG
                    self.area_com[side_key][cam_key][area_key] = {
                        'PLC': area_vals.get('PLC', ''),
                        'OK':  area_vals.get('OK', ''),
                        'NG':  area_vals.get('NG', '')
                    }

        # Populate UI controls
        # 1. Side combobox
        self.sideComboBox.clear()
        self.sideComboBox.addItems(self.side_com.keys())
        # Choose first side or keep current
        first_side = next(iter(self.side_com), None)
        if first_side:
            side_conf = self.side_com[first_side]
            self.sideComboBox.setCurrentText(first_side)
            self.inFromPlcSideInput.setText(side_conf['PLC'])
            self.outFromVisionSideInput.setText(side_conf['vision'])
            
            # Set result checkboxes and fields
            self.chkAreaWise.setChecked(side_conf.get('AreaWiseResult', False))
            self.chkCombine.setChecked(side_conf.get('CombineResult', False))
            self.chkDetectAll.setChecked(side_conf.get('DetectAll', False))
            
            # Set final result fields
            final_result = side_conf.get('FinalResult', ['', ''])
            self.finalResult1.setText(final_result[0])
            self.finalResult2.setText(final_result[1])

        # 2. Camera combobox
        self.camComboBox.clear()
        cams = list(self.area_com.get(first_side, {}).keys())
        print(cams)
        display_cams = [f"cam-{int(cam.split('-')[1]) + 1}" for cam in cams]
        print(display_cams,"==========================")
        self.camComboBox.addItems(display_cams)
        self.cam_key_map = {display: actual for display, actual in zip(display_cams, cams)}

        first_cam_display = display_cams[0] if display_cams else None
        print(first_cam_display)
        first_cam = self.cam_key_map.get(first_cam_display) if first_cam_display else None
        print(first_cam,"----------------------------")

        if first_cam:
            self.camComboBox.setCurrentText(first_cam_display)

            # Auto-fill camera PLC info
            self.plcInfoTextEdit.setPlainText(
                self.area_com[first_side][first_cam].get('PLC', '')
            )

        # 3. Area combobox and inputs
        self.updateAreaComboBox()
        if self.areaComboBox.count() > 0:
            # Select first area
            self.areaComboBox.setCurrentIndex(0)
            self.showInputBox()

        # Switch back to configuration page
        self.stackedWidget.setCurrentWidget(self.comboBoxPage)

    def showIpConfigPage(self):
        self.stackedWidget.setCurrentWidget(self.ipConfigPage)

    # (Rest of your methods like updateSideLabel, updateAreaComboBox, loadYamlData, buildConfigurationsFromYaml, etc., remain unchanged)
    # … 
    def setTheme(self,isDark):
        if isDark :
            self.centralwidget.setStyleSheet("background-color: #171717 ; color : #e2e2d2 ")
            self.hardwareButton.setStyleSheet(""" QPushButton {
                    color: #e2e2d2; 
                    background-color: #171717; 
                    border: 1px solid #e2e2d2;
                    border-radius: 10px;
                    text-align: center;
                    font-size:14px;
                }
                QPushButton:hover {
                    border: 2px solid #e2e2d2;
                }
            """)            
            self.softwareButton.setStyleSheet(""" QPushButton {
                    color: #e2e2d2; 
                    background-color: #171717; 
                    border: 1px solid #e2e2d2;
                    border-radius: 10px;
                    text-align: center;
                    font-size:14px;
                }
                QPushButton:hover {
                    border: 2px solid #e2e2d2;
                }
            """)
            self.saveIpButton.setStyleSheet(""" QPushButton {
                    color: #e2e2d2; 
                    background-color: #171717; 
                    padding: 10px; 
                    border: 1px solid #e2e2d2;
                    border-radius: 10px;
                    text-align: center;
                    font-size:14px;
                }
                QPushButton:hover {
                    border: 2px solid #e2e2d2;
                }
            """)
            self.backButtonIp.setStyleSheet(""" QPushButton {
                    color: #e2e2d2; 
                    background-color: #171717;
                    border: 1px solid #e2e2d2;
                    border-radius: 10px;
                    text-align: center;
                    font-size:14px;
                }
                QPushButton:hover {
                    border: 2px solid #e2e2d2;
                }
            """)
            self.saveSideButton.setStyleSheet(""" QPushButton {
                    color: #e2e2d2; 
                    background-color: #171717;  
                    border: 1px solid #e2e2d2;
                    border-radius: 10px;
                    text-align: center;
                    font-size:14px;
                }
                QPushButton:hover {
                    border: 2px solid #e2e2d2;
                }
            """)
            self.saveAreaButton.setStyleSheet(""" QPushButton {
                    color: #e2e2d2; 
                    background-color: #171717;  
                    border: 1px solid #e2e2d2;
                    border-radius: 10px;
                    text-align: center;
                    font-size:14px;
                }
                QPushButton:hover {
                    border: 2px solid #e2e2d2;
                }
            """)
            self.saveCamButton.setStyleSheet(""" QPushButton {
                    color: #e2e2d2; 
                    background-color: #171717;  
                    border: 1px solid #e2e2d2;
                    border-radius: 10px;
                    text-align: center;
                    font-size:14px;
                }
                QPushButton:hover {
                    border: 2px solid #e2e2d2;
                }
            """)
            # self.backButtonCombo.setStyleSheet(""" QPushButton {
            #         color: #e2e2d2; 
            #         background-color: #171717; 
            #         padding: 10px; 
            #         border: 1px solid #e2e2d2;
            #         border-radius: 10px;
            #         text-align: center;
            #         font-size:14px;
            #     }
            #     QPushButton:hover {
            #         border: 2px solid #e2e2d2;
            #     }
            # """)
        else :
            self.centralwidget.setStyleSheet("background-color: #e2e2d2 ; color : black")
            self.hardwareButton.setStyleSheet(""" QPushButton {
                    color: #000000; 
                    background-color: #d3d3d3; 
                    border: 1px solid #000000;
                    border-radius: 10px;
                    text-align: center;
                    font-size:14px;
                }
                QPushButton:hover {
                    border: 2px solid #000000;
                }
            """)            
            self.softwareButton.setStyleSheet(""" QPushButton {
                    color: #000000; 
                    background-color: #d3d3d3; 
                    border: 1px solid #000000;
                    border-radius: 10px;
                    text-align: center;
                    font-size:14px;
                }
                QPushButton:hover {
                    border: 2px solid #000000;
                }
            """)
            self.saveIpButton.setStyleSheet(""" QPushButton {
                    color: #000000; 
                    background-color: #d3d3d3; 
                    padding: 10px; 
                    border: 1px solid #000000;
                    border-radius: 10px;
                    text-align: center;
                    font-size:14px;
                }
                QPushButton:hover {
                    border: 2px solid #000000;
                }
            """)
            self.saveCamButton.setStyleSheet(""" QPushButton {
                    color: #000000; 
                    background-color: #d3d3d3;
                    border: 1px solid #000000;
                    border-radius: 10px;
                    text-align: center;
                    font-size:14px;
                }
                QPushButton:hover {
                    border: 2px solid #000000;
                }
            """)
            self.backButtonIp.setStyleSheet(""" QPushButton {
                    color: #000000; 
                    background-color: #d3d3d3;
                    border: 1px solid #000000;
                    border-radius: 10px;
                    text-align: center;
                    font-size:14px;
                }
                QPushButton:hover {
                    border: 2px solid #000000;
                }
            """)
            self.saveSideButton.setStyleSheet(""" QPushButton {
                    color: #000000; 
                    background-color: #d3d3d3;  
                    border: 1px solid #000000;
                    border-radius: 10px;
                    text-align: center;
                    font-size:14px;
                }
                QPushButton:hover {
                    border: 2px solid #000000;
                }
            """)
            self.saveAreaButton.setStyleSheet(""" QPushButton {
                    color: #000000; 
                    background-color: #d3d3d3;  
                    border: 1px solid #000000;
                    border-radius: 10px;
                    text-align: center;
                    font-size:14px;
                }
                QPushButton:hover {
                    border: 2px solid #000000;
                }
            """)
            # self.backButtonCombo.setStyleSheet(""" QPushButton {
            #         color: #000000; 
            #         background-color: #d3d3d3; 
            #         padding: 10px; 
            #         border: 1px solid #000000;
            #         border-radius: 10px;
            #         text-align: center;
            #         font-size:14px;
            #     }
            #     QPushButton:hover {
            #         border: 2px solid #000000;
            #     }
            # """)

# ------------------ MainWindow Subclass ------------------
class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        Ui_MainWindow.__init__(self)
        self.setupUi(self, active_part="ASD")  

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    mainWindow = MainWindow()
    mainWindow.show()
    sys.exit(app.exec())
