import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import seaborn as sns
import torch
from PyQt5 import QtWidgets
from PyQt5.QtCore import QAbstractTableModel
from PyQt5.QtCore import QObject
from PyQt5.QtCore import Qt, QPoint, QTimer, QThread, pyqtSignal
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtGui import QTransform
from PyQt5.QtWidgets import QApplication, QMainWindow, QDialog
from PyQt5.QtWidgets import QFileDialog, QMenu
from PyQt5.QtWidgets import QInputDialog
from PyQt5.QtWidgets import QLineEdit, QVBoxLayout, QLabel, QDialogButtonBox
from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtWidgets import QSizePolicy
from ultralytics import YOLO

from main_win.win import Ui_mainWindow
from models.experimental import attempt_load
from utils.CustomMessageBox import MessageBox
from utils.capnums import Camera
from utils.datasets import LoadImages
from utils.general import check_img_size, increment_path
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device

sns.set(style='darkgrid', palette='pastel')
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  #



imgsz=1280
save_crop = False
augment = False
classes = None
project = ROOT / 'result' # save results to project/name
hide_labels=False  # hide labels
hide_conf=False  # hide confidences
line_thickness=3  # bounding box thickness (pixels)



#检测线程
class DetThread(QThread):
    """这段代码定义了6个信号，用于在PyQt程序中进行通信。
send_img信号用于发送图像数组，
send_raw信号用于发送原始数组，
send_statistic信号用于发送统计字典数据，
send_msg信号用于发送分析(检测)的状态消息，
send_percent信号用于发送分析(检测)进度，
最后send_fps用于发送帧率数"""
    send_img = pyqtSignal(np.ndarray)
    send_raw = pyqtSignal(np.ndarray)
    send_statistic = pyqtSignal(dict)
    # emit：detecting/pause/stop/finished/error msg
    send_msg = pyqtSignal(str)
    send_percent = pyqtSignal(int)
    send_fps = pyqtSignal(str)

    def __init__(self):
        super(DetThread, self).__init__()
        self.weights = './yolov5s.pt'# 设置权重
        self.current_weight = './yolov5s.pt'# 当前权重
        self.source = '0'# 视频源
        self.conf_thres = 0.25 # 置信度
        self.iou_thres = 0.45# iou
        self.jump_out = False                   # jump out of the loop # 跳出循环
        self.is_continue = True                 # continue/pause# 进度条
        self.percent_length = 1000              # progress bar# 是否启用延时
        self.rate_check = True                  # Whether to enable delay# 延时HZ
        self.rate = 100
        self.save_fold = './result' # 保存文件夹



    @torch.no_grad()
    def run(self,
            # imgsz=1280,  # inference size (pixels)
            max_det=1000,  # maximum detections per image
            device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
            view_img=True,  # show results
            # save_crop=False,  # save cropped prediction boxes
            # augment=False,  # augmented inference
            # classes=None,  # filter by class: --class 0, or --class 0 2 3
            # project=ROOT / 'result',  # save results to project/name
            name='exp',  # save results to project/name
            exist_ok=False,  # existing project/name ok, do not increment， False是自增，true是不自增覆盖
            # line_thickness=3,  # bounding box thickness (pixels)
            # hide_labels=False,  # hide labels
            # hide_conf=False,  # hide confidences
            save_txt=False,  # save results to *.txt
            save_conf=False,  # save confidences in --save-txt labels
            update=False,  # update all models
            nosave=False,  # do not save images/videos
            prune_model =True
            ):

        global imgsz
        # Initialize
        try:
            device = select_device(device)

            # Load model
            model = attempt_load(self.weights, map_location=device)  # load FP32 model
            #自己的模型加载方式
            my_model = YOLO(self.weights)

            num_params = 0
            for param in model.parameters():
                num_params += param.numel()
            stride = int(model.stride.max())  # model stride
            print("model:", self.weights)
            #print("stride:",stride)
            #print("imgsz:", imgsz)
            imgsz = check_img_size(imgsz, s=stride)  # check image size
            names = model.module.names if hasattr(model, 'module') else model.names  # get class names

            # 检查 names 是否为字典类型，如果是，则转换为列表
            if isinstance(names, dict):
                names = [names[i] for i in range(len(names))]
            print("sorts:", names)

            dataset = LoadImages(self.source, img_size=imgsz, stride=stride)

            # Run inference
            if device.type != 'cpu':
                model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once

            count = 0
            jump_count = 0
            start_time = time.time()
            dataset = iter(dataset)

            global save_crop
            #print("save_crop", save_crop)
            global augment
            #print("augment", augment)
            global classes
            #print("classes:", classes)
            global hide_labels
            #print("hide_labels:", hide_labels)
            global hide_conf
            #print("hide_conf:", hide_conf)
            global line_thickness
            #print("line_thickness:", line_thickness)

            while True:

                # 手动停止
                if self.jump_out:
                    self.vid_cap.release()
                    self.send_percent.emit(0)
                    self.send_msg.emit('Stop')
                    if hasattr(self, 'out'):
                        self.out.release()
                    break

                # 临时更换模型
                if self.current_weight != self.weights:
                    model = attempt_load(self.weights, map_location=device)  # load FP32 model
                    my_model = YOLO(self.weights)

                    num_params = 0
                    for param in model.parameters():
                        num_params += param.numel()
                    stride = int(model.stride.max())  # model stride
                    imgsz = check_img_size(imgsz, s=stride)  # check image size
                    names = model.module.names if hasattr(model, 'module') else model.names  # get class names
                    if isinstance(names, dict):
                        names = [names[i] for i in range(len(names))]
                    if device.type != 'cpu':
                        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
                    self.current_weight = self.weights

                # 暂停开关,这个就是主程序
                if self.is_continue:
                    try:
                        path, img, im0s, self.vid_cap = next(dataset)
#                       path：当前处理的视频文件路径。
#                       img：当前视频帧经过填充和调整大小后的图像，用于模型输入，格式为 (C, H, W) 且已转换为 RGB。
#                       img0：当前视频帧的原始图像，格式为 BGR。
#                       cap：OpenCV 的视频捕获对象 cv2.VideoCapture，用于读取视频帧
                    except StopIteration:
                        self.send_msg.emit('finished')
                        break


                    #帧计数
                    count += 1
                    if count % 30 == 0 and count >= 30:
                        fps = int(30 / (time.time() - start_time))
                        self.send_fps.emit('fps：' + str(fps))
                        start_time = time.time()
                    if self.vid_cap:
                        percent = int(count / self.vid_cap.get(cv2.CAP_PROP_FRAME_COUNT) * self.percent_length)*2
                        self.send_percent.emit(percent)
                    else:
                        percent = self.percent_length

                    # 初始化 statistic_dic
                    statistic_dic = {name: 0 for name in names}

                    global project
                    # 循环迭代时自动调用，从视频流中读取下一帧图像，并返回一个布尔值 ret_val 和读取的图像数据 img0
                    if self.vid_cap is not None:
                        ret_val, img0 = self.vid_cap.read()
                        # 如果是视频，载入img0并在之后的循环中加载之后帧
                        my_results = my_model.predict(img0, stream=True)
                    else:
                        # 如果是图片，则载入im0s
                        my_results = my_model.predict(im0s)

                    for my_result in my_results:
                        print("boxes data:",my_result.boxes.data)
                        if my_result.keypoints is not None:
                            print("points data:",my_result.keypoints.cpu().numpy().data)
                        pred = my_result.boxes.data
                        pred2 = [pred.clone()]
                        #print("Model prediction output:", pred2)
                        for i, det in enumerate(pred2):  # detections per image
                            #print(f"Index: {i}, Detection: {det}")
                            im0 = img0.copy() if self.vid_cap is not None else im0s.copy()
                            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
                            statistic_dic = {name: 0 for name in names}
                            if len(det):
                                for *xyxy, conf, cls in reversed(det):
                                    #如果置信度小于阈值，则不处理当前帧
                                    if conf<self.conf_thres:
                                        continue
                                    c = int(cls)  # integer class
                                    #print("xyxy:", xyxy)
                                    if c >= len(names):
                                        c %= len(names)
                                    statistic_dic[names[c]] += 1
                                    #是否显示label（参数设置页面）
                                    label = None if hide_labels else (
                                        names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                                    #画识别框图+label
                                    annotator.box_label(xyxy, label, color=colors(c, True))
                                    #画姿势点线
                                    if my_result.keypoints is not None:
                                        keypoints = my_result.keypoints  # Keypoints object for pose outputs
                                        keypoints = keypoints.cpu().numpy()  # convert to numpy array
                                        annotator.draw_points(keypoints)
                                    #保存裁剪的图片
                                    im0 = im0.copy() if save_crop else im0  # for save_crop
                                    if save_crop:
                                        file_path = 'result'
                                        save_dir = increment_path(Path(project) / name,
                                                                  exist_ok=True)  # increment run
                                        save_dir.mkdir(parents=True, exist_ok=True)  # make dir
                                        save_one_box(xyxy, im0, file=save_dir / 'crops' / names[c] / f'{file_path}.jpg',
                                                     BGR=True)
                                        print("crop已保存")
                                        # save_one_box(xyxy, im0, file=save_dir / 'crops' / names[c] / 'file_path.jpg', BGR=True)
                        im0 = annotator.result()  # Return annotated image as array   np.asarray(im0)
                        self.send_img.emit(im0)  # 发送np.asarray(im0)
                        print("send_img")
                        # self.send_raw.emit(img0 if isinstance(im0s, np.ndarray) else im0s[0])
                        self.send_raw.emit(img0 if self.vid_cap is not None else im0s)
                        print("send_raw")
                        self.send_statistic.emit(statistic_dic)
                        if self.rate_check:
                            time.sleep(1 / self.rate)
                    # 保存识别结果,目录可选，默认保存到./result下
                    if self.save_fold:
                        os.makedirs(self.save_fold, exist_ok=True)
                        if self.vid_cap is None:
                            save_path = os.path.join(self.save_fold,
                                                     time.strftime('%Y_%m_%d_%H_%M_%S',
                                                                   time.localtime()) + '.jpg')
                            cv2.imwrite(save_path, im0)
                        else:
                            if count == 1: # 第一帧时初始化录制
                                # 以视频原始帧率的一半进行录制
                                ori_fps = int(self.vid_cap.get(cv2.CAP_PROP_FPS))/2
                                if ori_fps == 0:
                                    ori_fps = 25
                                width, height = im0.shape[1], im0.shape[0]
                                save_path = os.path.join(self.save_fold, time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime()) + '.mp4')
                                self.out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"mp4v"), ori_fps,
                                                           (width, height))
                            self.out.write(im0)
                    #进度条
                    if percent == self.percent_length:
                        self.send_percent.emit(0)
                        self.send_msg.emit('finished')
                        if hasattr(self, 'out'):
                            self.out.release()
                        break


        except Exception as e:
            self.send_msg.emit('finished')
            print("error:", e)
            print("errorType:", type(e))  # 查看异常的具体类型




#插入弹窗数据库
class AddRecordDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super(AddRecordDialog, self).__init__(parent)

        self.sign_type_line_edit = QtWidgets.QLineEdit()
        self.sign_count_line_edit = QtWidgets.QLineEdit()
        self.additional_info_line_edit = QtWidgets.QLineEdit()

        button_box = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)

        layout = QtWidgets.QFormLayout()
        layout.addRow("Sign Type:", self.sign_type_line_edit)
        layout.addRow("Sign Count:", self.sign_count_line_edit)
        layout.addRow("Additional Info:", self.additional_info_line_edit)
        layout.addWidget(button_box)

        self.setLayout(layout)

    def get_record_data(self):
        sign_type = self.sign_type_line_edit.text()
        sign_count = self.sign_count_line_edit.text()
        additional_info = self.additional_info_line_edit.text()

        return sign_type, sign_count, additional_info
#总体数据库
class PymysqlTableModel(QAbstractTableModel):
    def __init__(self, data, headers):
        super().__init__()
        self._data = data
        self._headers = headers

    def rowCount(self, parent=None):
        return len(self._data)

    def columnCount(self, parent=None):
        return len(self._data[0])

    def data(self, index, role=Qt.DisplayRole):
        if role == Qt.DisplayRole:
            return self._data[index.row()][index.column()]
        return None

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role == Qt.DisplayRole and orientation == Qt.Horizontal:
            return self._headers[section]
        return None
    def filter_data(self, filter_function):
        self._filtered_data = list(filter(filter_function, self._data))
        self.layoutChanged.emit()

#图像增强线程
import os
import cv2
import random
from typing import List, Tuple
import numpy as np
from typing import Optional

class ImageAugmentor(QObject):
    progress_updated = pyqtSignal(int)
    def __init__(self):
        super().__init__()
        self.operations = [
            self.flip, self.random_crop, self.random_scale, self.random_rotate,
            self.random_brightness_contrast_saturation_hue, self.random_erase,
            self.random_noise, self.cutout, self.mixup, self.mosaic
        ]
        self.input_folder = None  # 添加此行



    def process_images(self, input_folder: str, output_folder: str, probabilities: List[float]) -> Tuple[int, int, int]:
        self.input_folder = input_folder
        success_count = 0
        failure_count = 0
        image_count = sum(1 for file in os.listdir(input_folder) if file.lower().endswith(('.png', '.jpg', '.jpeg')))


        for index, file in enumerate(os.listdir(input_folder)):
            filepath = os.path.join(input_folder, file)
            if os.path.isfile(filepath) and file.lower().endswith(('.png', '.jpg', '.jpeg')):
                # 读取图像
                image = cv2.imdecode(np.fromfile(filepath, dtype=np.uint8), cv2.IMREAD_COLOR)

                # 检查图像是否为空
                if image is None:
                    print(f"Error loading image: {filepath}")
                    failure_count += 1
                    continue

                for i, probability in enumerate(probabilities):
                    if random.random() < probability:
                        if i in {8, 9}:  # Mixup and Mosaic operations
                            extra_image = self.load_random_image(input_folder)
                            image = self.apply_operation(image, i, extra_image)
                        else:
                            image = self.apply_operation(image, i)

                output_path = os.path.join(output_folder, file)

                # 保存图像
                encoded_image = cv2.imencode('.jpg', image)[1]
                if encoded_image is not None:
                    encoded_image.tofile(output_path)
                    success_count += 1
                else:
                    failure_count += 1
            self.progress_updated.emit(success_count + failure_count)

        return success_count, failure_count, image_count

    def load_random_image(self, input_folder: str) -> np.ndarray:
        files = [file for file in os.listdir(input_folder) if file.lower().endswith(('.png', '.jpg', '.jpeg'))]
        random_file = random.choice(files)
        filepath = os.path.join(input_folder, random_file)
        return cv2.imdecode(np.fromfile(filepath, dtype=np.uint8), cv2.IMREAD_COLOR)

    def apply_operation(self, image: np.ndarray, operation_index: int,
                        extra_image: Optional[np.ndarray] = None) -> np.ndarray:
        if operation_index == 8:  # Mixup
            if extra_image is not None:
                return self.mixup(image, extra_image)
            else:
                return image
        elif operation_index == 9:  # Mosaic
            if extra_image is not None:
                return self.mosaic(image, extra_image)
            else:
                return image
        else:
            return self.operations[operation_index](image)

    # 1. 图像翻转
    def flip(self, image):
        mode = random.choice(['horizontal', 'vertical'])
        print("flip启用")
        return cv2.flip(image, 1 if mode == 'horizontal' else 0)

    # 2. 随机裁剪
    def random_crop(self, image):
        height, width, _ = image.shape
        crop_ratio = random.uniform(0.5, 1.0)
        new_height, new_width = int(height * crop_ratio), int(width * crop_ratio)

        y = random.randint(0, height - new_height)
        x = random.randint(0, width - new_width)
        print("random_crop启用")
        return image[y:y + new_height, x:x + new_width]

    # 3. 随机缩放
    def random_scale(self, image):
        scale_factor = random.uniform(0.5, 2.0)
        print("random_scale启用")
        return cv2.resize(image, None, fx=scale_factor, fy=scale_factor)

    # 4. 随机旋转
    def random_rotate(self, image):
        angle = random.randint(-180, 180)
        height, width = image.shape[:2]
        matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)
        print("random_rotate启用")
        return cv2.warpAffine(image, matrix, (width, height))

    # 5. 随机亮度、对比度、饱和度、色调调整
    def random_brightness_contrast_saturation_hue(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        brightness = random.uniform(0.5, 1.5)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * brightness, 0, 255)

        contrast = random.uniform(0.5, 1.5)
        image = np.clip((image - 128) * contrast + 128, 0, 255)

        saturation = random.uniform(0.5, 1.5)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation, 0, 255)

        hue_shift = random.uniform(-30, 30)
        hsv[:, :, 0] = np.clip(hsv[:, :, 0] + hue_shift, 0, 180)
        print("random_brightness_contrast_saturation_hue启用")
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # 6. 随机擦除
    def random_erase(self, image):
        height, width, _ = image.shape
        erase_size = random.randint(10, min(height, width) // 2)

        y = random.randint(0, height - erase_size)
        x = random.randint(0, width - erase_size)

        image[y:y + erase_size, x:x + erase_size, :] = 0
        print("random_erase启用")
        return image

    # 7. 随机加噪
    def random_noise(self, image):
        row, col, ch = image.shape
        mean = 0
        sigma = random.uniform(10, 50)
        gauss = np.random.normal(mean, sigma, (row, col, ch))
        gauss = gauss.reshape(row, col, ch)
        noisy = image + gauss
        print("random_noise启用")
        return np.clip(noisy, 0, 255).astype(np.uint8)

    # 8. Mixup
    def mixup(self, image):
        # 在这个示例中，我们将随机选择另一张图像进行混合。您可以根据需要自定义图像选择方法。
        image_list = [f for f in os.listdir(self.input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        random_image_path = os.path.join(self.input_folder, random.choice(image_list))
        mixup_image = cv2.imread(random_image_path)

        height, width, _ = image.shape
        mixup_image = cv2.resize(mixup_image, (width, height))

        alpha = random.uniform(0, 1)
        print("mixup启用")
        return cv2.addWeighted(image, alpha, mixup_image, 1 - alpha, 0)

    # 9. CutOut
    def cutout(self, image):
        height, width, _ = image.shape
        cutout_size = random.randint(10, min(height, width) // 2)

        y = random.randint(0, height - cutout_size)
        x = random.randint(0, width - cutout_size)

        image[y:y + cutout_size, x:x + cutout_size, :] = 0
        print("cutout启用")
        return image

    # 10. Mosaic
    def mosaic(self, image: np.ndarray, extra_image: np.ndarray) -> np.ndarray:
        image_list = [os.path.join(self.input_folder, f) for f in os.listdir(self.input_folder) if
                      f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        random_images = [cv2.imread(random.choice(image_list)) for _ in range(3)]

        height, width, _ = image.shape
        height = height - height % 2  # Make height an even number
        width = width - width % 2  # Make width an even number

        # Resize the input image
        image = cv2.resize(image, (width, height))

        for i in range(3):
            random_images[i] = cv2.resize(random_images[i], (width // 2, height // 2))

        mosaic_image = np.zeros((height, width, 3), dtype=np.uint8)

        mosaic_image[:height // 2, :width // 2] = image[:height // 2, :width // 2]
        mosaic_image[height // 2:, :width // 2] = random_images[0][:height // 2, :width // 2]
        mosaic_image[:height // 2, width // 2:] = random_images[1][:height // 2, :width // 2]
        mosaic_image[height // 2:, width // 2:] = random_images[2][:height // 2, :width // 2]
        print("mosaic启用")
        return mosaic_image


#主UI线程
class MainWindow(QMainWindow, Ui_mainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)
        self.m_flag = False
        self.open_stackedWidget(2)#默认打开识别界面
        # style 1: window can be stretched
        # self.setWindowFlags(Qt.CustomizeWindowHint | Qt.WindowStaysOnTopHint)
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        # self.setWindowFlags(Qt.CustomizeWindowHint | Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)

        # style 2: window can not be stretched
        # self.setWindowFlags(Qt.Window | Qt.FramelessWindowHint
        #                     | Qt.WindowSystemMenuHint | Qt.WindowMinimizeButtonHint | Qt.WindowMaximizeButtonHint)
        # self.setWindowOpacity(0.85)  # Transparency of window

        # 自定义标题栏按钮
        self.minButton.clicked.connect(self.showMinimized)
        self.maxButton.clicked.connect(self.max_or_restore)
        # show Maximized window
        # self.maxButton.animateClick(10)
        self.closeButton.clicked.connect(self.close)

        # 定时清空自定义状态栏上的文字
        self.qtimer = QTimer(self)
        self.qtimer.setSingleShot(True)
        self.qtimer.timeout.connect(lambda: self.statistic_label.clear())

        # 自动搜索模型
        self.comboBox.clear()
        self.pt_list = os.listdir('./pt')
        self.pt_list = [file for file in self.pt_list if file.endswith('.pt')]
        self.pt_list.sort(key=lambda x: os.path.getsize('./pt/'+x))
        self.comboBox.clear()
        self.comboBox.addItems(self.pt_list)
        self.qtimer_search = QTimer(self)
        self.qtimer_search.timeout.connect(lambda: self.search_pt())
        self.qtimer_search.start(2000)

        #  yolov5线程
        self.det_thread = DetThread()
        self.model_type = self.comboBox.currentText()
        self.det_thread.weights = "./pt/%s" % self.model_type # 权重
        self.det_thread.source = '0'     # 默认打开本机摄像头，无需保存到配置文件
        self.det_thread.percent_length = self.progressBar.maximum()
        self.det_thread.send_raw.connect(lambda x: self.show_image(x, self.raw_video))
        self.det_thread.send_img.connect(lambda x: self.show_image(x, self.out_video))
        self.det_thread.send_statistic.connect(self.show_statistic)
        self.det_thread.send_msg.connect(lambda x: self.show_msg(x))
        self.det_thread.send_percent.connect(lambda x: self.progressBar.setValue(x))
        self.det_thread.send_fps.connect(lambda x: self.fps_label.setText(x))


        self.fileButton.clicked.connect(self.open_file)
        self.cameraButton.clicked.connect(self.chose_cam)
        #self.rtspButton.clicked.connect(self.chose_rtsp)

        self.runButton.clicked.connect(self.run_or_continue)
        self.stopButton.clicked.connect(self.stop)

        self.comboBox.currentTextChanged.connect(self.change_model)
        self.confSpinBox.valueChanged.connect(lambda x: self.change_val(x, 'confSpinBox'))
        self.confSlider.valueChanged.connect(lambda x: self.change_val(x, 'confSlider'))
        self.iouSpinBox.valueChanged.connect(lambda x: self.change_val(x, 'iouSpinBox'))
        self.iouSlider.valueChanged.connect(lambda x: self.change_val(x, 'iouSlider'))
        self.rateSpinBox.valueChanged.connect(lambda x: self.change_val(x, 'rateSpinBox'))
        self.rateSlider.valueChanged.connect(lambda x: self.change_val(x, 'rateSlider'))

        self.checkBox.clicked.connect(self.checkrate)
        self.saveCheckBox.clicked.connect(self.is_save)
        self.load_setting()

# 在您的窗口类的 __init__ 方法中，连接按钮的 clicked 信号到相应的槽函数
        self.pushButton.clicked.connect(lambda: self.open_stackedWidget(2))#标志识别按钮
        #self.pushButton_4.clicked.connect(lambda: self.open_stackedWidget(0))#数据库按钮
        self.refreshButton_2.clicked.connect(lambda: self.open_stackedWidget(1))#图像处理按钮
        self.pushButton_2.clicked.connect(lambda: self.open_stackedWidget(3))
        self.pushButton_5.clicked.connect(lambda: self.open_stackedWidget(4))



#高参数大模块
        #self.visualize_checkbox1.stateChanged.connect(self.on_visualize_checkbox_stateChanged)
        #self.SavedatabaseCheckBox.stateChanged.connect(self.on_databases_checkbox_stateChanged)
        #self.agnostic_nmscheckbox.stateChanged.connect(self.on_agnostic_nmscheckbox_checkbox_stateChanged)
        #self.halfcheckbox_2.stateChanged.connect(self.on_halfcheckbox_2_nmscheckbox_checkbox_stateChanged)
        # self.augmentedcheckbox_3.stateChanged.connect(self.on_augment_nmscheckbox_checkbox_stateChanged)
        self.save_cropcheckbox_2.stateChanged.connect(self.on_save_crop_nmscheckbox_checkbox_stateChanged)
        self.hide_labelscheckbox_4.stateChanged.connect(self.on_hide_labelscheckbox_checkbox_checkbox_stateChanged)
        self.hide_confscheckbox_6.stateChanged.connect(self.on_hide_confcheckbox_checkbox_checkbox_stateChanged)

        self.imgsz_spinbox12.valueChanged.connect(self.on_imgsz_spinbox_valueChanged)
        self.line_thickness_spinbox12_2.valueChanged.connect(self.on_line_thickness_spinbox_valueChanged)
        self.lineEdit.textChanged.connect(self.update_classes)
        self.browse_result_folder_button.clicked.connect(self.on_browse_result_folder_button_clicked)





#批处理图像增强大模块
        # 创建一个 ImageAugmentor 实例
        self.image_augmentor = ImageAugmentor()

        # 将 UI 控件连接到相关方法
        self.inputFolderButton.clicked.connect(self.select_input_folder)
        self.outputFolderButton.clicked.connect(self.select_output_folder)
        self.startProcessingButton.clicked.connect(self.start_processing)

        self.init_progress_bar()
        # 在MainWindow类的__init__方法中为每个复选框添加事件处理程序
        for checkbox in [self.enhanceMethod1Checkbox, self.enhanceMethod2Checkbox, self.enhanceMethod3Checkbox,
                         self.enhanceMethod4Checkbox, self.enhanceMethod5Checkbox, self.enhanceMethod6Checkbox,
                         self.enhanceMethod7Checkbox, self.enhanceMethod8Checkbox, self.enhanceMethod9Checkbox,
                         self.enhanceMethod10Checkbox]:
            checkbox.clicked.connect(self.checkbox_clicked)

        self.helpButton.clicked.connect(self.show_help)
        # 高级参数设置
        #self.advancedSettingsButton.clicked.connect(self.open_advanced_settings)
        # 设置默认值
        #self.enhanceMethod2Parameter = QLineEdit("0.8, 1.0")  # 示例：设置增强方法2的默认参数值



# 单个图像处理大模块
        # 加载缩放
        self.loadImageButton.clicked.connect(self.load_image1)
        self.scaleSlider.valueChanged.connect(self.scale_image)
        # # 翻转
        # self.comboBox_3.currentTextChanged.connect(self.on_comboBox_3_currentIndexChanged)
        # 旋转
        self.rotationSlider.valueChanged.connect(self.rotate_image)

# 正常显示和预览
        self.imageEnhanceButton.clicked.connect(lambda: self.apply_image_enhance(preview=False))
        self.previewEnhanceButton.clicked.connect(lambda: self.apply_image_enhance(preview=True))

        self.imageFilterButton.clicked.connect(lambda: self.apply_image_filter(preview=False))
        self.previewFilterButton.clicked.connect(lambda: self.apply_image_filter(preview=True))

        self.morphologyButton.clicked.connect(lambda: self.apply_morphology(preview=False))
        self.previewmorphologyButton.clicked.connect(lambda: self.apply_morphology(preview=True))

        self.histogramButton.clicked.connect(lambda: self.apply_histogram(preview=False))
        self.previewhistogramButton.clicked.connect(lambda: self.apply_histogram(preview=True))

        #打开一个文件夹选择上一个下一个
        self.image_list = []
        self.current_image_index = -1
        self.save_folder = ""
        #文件按钮
        self.openFolderButton.clicked.connect(self.open_folder)
        self.saveFolderButton.clicked.connect(self.select_save_folder)
        self.nextImageButton.clicked.connect(self.next_image)
        self.previousImageButton.clicked.connect(self.previous_image)
        self.saveButton.clicked.connect(self.save_image)


# 自定义参数
        self.slider_adjust_contrast.valueChanged.connect(self.slider_adjustment)
        self.apply_custom_adjustments_imageEnhance.clicked.connect(self.apply_custom_adjustments_with_value)

#这里只提供图像增强的滑条和设置具体数值的按钮，其他的自行按照这种模式添加
    def slider_adjustment(self, value):
        operation = self.imageEnhanceComboBox.currentText()
        value = value / 10.0
        if operation == '对比度增强':
            self.adjust_contrast(alpha=value, preview=False)
        elif operation == '饱和度调整':
            self.adjust_saturation(value=value, preview=False)
        elif operation == '调整色相':
            self.adjust_hue(value=value, preview=False)
        elif operation == '调整亮度':
            self.adjust_brightness1(value=value, preview=False)
        elif operation == 'Gamma 校正':
            self.adjust_gamma(gamma=value, preview=False)
        elif operation == '二值化':
            self.binarize_image(threshold=value, preview=False)

    def apply_custom_adjustments_with_value(self):
        operation = self.imageEnhanceComboBox.currentText()
        value, ok = QInputDialog.getInt(self, "输入参数", f"设置{operation}的值：", min=0, max=100)

        if ok:
            if operation == '对比度增强':
                self.adjust_contrast(alpha=value, preview=False)
            elif operation == '饱和度调整':
                self.adjust_saturation(value=value, preview=False)
            elif operation == '调整色相':
                self.adjust_hue(value=value, preview=False)
            elif operation == '调整亮度':
                self.adjust_brightness1(value=value, preview=False)
            elif operation == 'Gamma 校正':
                self.adjust_gamma(gamma=value, preview=False)
            elif operation == '二值化':
                self.binarize_image(threshold=value, preview=False)

   # 二值化
    def binarize_image(self, threshold=128, preview=True):

        try:
            if not self.imageLabel.pixmap():
                QMessageBox.warning(self, "提示", "请先加载一张图片")
                return

            original_image = self.pixmap_to_cv2_image(self.imageLabel.pixmap())
            if original_image is None:
                QMessageBox.warning(self, "提示", "无法处理此图片")
                return

            gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
            _, result_image = cv2.threshold(gray_image, threshold, 255, cv2.THRESH_BINARY)

            pixmap = self.cv2_image_to_pixmap(result_image)

            # scaled_pixmap = pixmap.scaled(self.scrollArea.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)

            if preview:
                label_size = self.previewLabel.size()
                scaled_pixmap = pixmap.scaled(label_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.previewLabel.setPixmap(scaled_pixmap)

            else:
                label_size = self.imageLabel_2.size()
                scaled_pixmap = pixmap.scaled(label_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.imageLabel_2.setPixmap(scaled_pixmap)

        except Exception as e:
            print(f"Error in binarize_image: {e}")




    def adjust_gamma(self, gamma=1.0, preview=True):
        try:
            if not self.imageLabel.pixmap():
                QMessageBox.warning(self, "提示", "请先加载一张图片")
                return

            original_image = self.pixmap_to_cv2_image(self.imageLabel.pixmap())
            if original_image is None:
                QMessageBox.warning(self, "提示", "无法处理此图片")
                return

            result_image = np.power(original_image, gamma)

            pixmap = self.cv2_image_to_pixmap(result_image)

            # scaled_pixmap = pixmap.scaled(self.scrollArea.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)

            if preview:
                label_size = self.previewLabel.size()
                scaled_pixmap = pixmap.scaled(label_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.previewLabel.setPixmap(scaled_pixmap)

            else:
                label_size = self.imageLabel_2.size()
                scaled_pixmap = pixmap.scaled(label_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.imageLabel_2.setPixmap(scaled_pixmap)
        except Exception as e:
            print(f"Error in adjust_gamma: {e}")


    #调整亮度
    def adjust_brightness1(self, value=1.5, preview=True):
        try:
            if not self.imageLabel.pixmap():
                QMessageBox.warning(self, "提示", "请先加载一张图片")
                return

            original_image = self.pixmap_to_cv2_image(self.imageLabel.pixmap())
            if original_image is None:
                QMessageBox.warning(self, "提示", "无法处理此图片")
                return

            hsv_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2HSV)

            brightness_multiplier = np.ones(hsv_image.shape[:2], dtype=np.float32) * value
            hsv_image[..., 2] = cv2.add(hsv_image[..., 2], brightness_multiplier, dtype=cv2.CV_32F)

            hsv_image[..., 2] = cv2.convertScaleAbs(hsv_image[..., 2])

            result_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

            pixmap = self.cv2_image_to_pixmap(result_image)

            # scaled_pixmap = pixmap.scaled(self.scrollArea.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)

            if preview:
                label_size = self.previewLabel.size()
                scaled_pixmap = pixmap.scaled(label_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.previewLabel.setPixmap(scaled_pixmap)

            else:
                label_size = self.imageLabel_2.size()
                scaled_pixmap = pixmap.scaled(label_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.imageLabel_2.setPixmap(scaled_pixmap)
        except Exception as e:
            print(f"Error in adjust_brightness1: {e}")


        # ...
#调整色相
    def adjust_hue(self, value=1.5, preview=True):
        try:
            if not self.imageLabel.pixmap():
                QMessageBox.warning(self, "提示", "请先加载一张图片")
                return

            original_image = self.pixmap_to_cv2_image(self.imageLabel.pixmap())
            if original_image is None:
                QMessageBox.warning(self, "提示", "无法处理此图片")
                return

            hsv_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2HSV)

            hue_multiplier = np.ones(hsv_image.shape[:2], dtype=np.float32) * value
            hsv_image[..., 0] = cv2.add(hsv_image[..., 0], hue_multiplier, dtype=cv2.CV_32F)

            hsv_image[..., 0] = cv2.convertScaleAbs(hsv_image[..., 0])

            result_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

            pixmap = self.cv2_image_to_pixmap(result_image)

            # scaled_pixmap = pixmap.scaled(self.scrollArea.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)

            if preview:
                label_size = self.previewLabel.size()
                scaled_pixmap = pixmap.scaled(label_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.previewLabel.setPixmap(scaled_pixmap)

            else:
                label_size = self.imageLabel_2.size()
                scaled_pixmap = pixmap.scaled(label_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.imageLabel_2.setPixmap(scaled_pixmap)
        except Exception as e:
            print(f"Error in adjust_hue: {e}")


    def adjust_saturation(self, value=1.5, preview=True):
        try:
            if not self.imageLabel.pixmap():
                QMessageBox.warning(self, "提示", "请先加载一张图片")
                return

                # 获取原始图像
            original_image = self.pixmap_to_cv2_image(self.imageLabel.pixmap())
            if original_image is None:
                QMessageBox.warning(self, "提示", "无法处理此图片")
                return

            # 转换为 HSV 颜色空间
            hsv_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2HSV)

            # 增加饱和度
            saturation_multiplier = np.ones(hsv_image.shape[:2], dtype=np.float32) * value
            hsv_image[..., 1] = cv2.multiply(hsv_image[..., 1], saturation_multiplier, dtype=cv2.CV_32F)

            hsv_image[..., 1] = cv2.convertScaleAbs(hsv_image[..., 1])

            # 转换回 BGR 颜色空间
            result_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

            # 将结果显示在界面上
            pixmap = self.cv2_image_to_pixmap(result_image)

            # Scale the pixmap to fit the scrollArea
            # scaled_pixmap = pixmap.scaled(self.scrollArea.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)

            if preview:
                label_size = self.previewLabel.size()
                scaled_pixmap = pixmap.scaled(label_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.previewLabel.setPixmap(scaled_pixmap)

            else:
                label_size = self.imageLabel_2.size()
                scaled_pixmap = pixmap.scaled(label_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.imageLabel_2.setPixmap(scaled_pixmap)
        except Exception as e:
            print(f"Error in adjust_saturation: {e}")




    def adjust_contrast(self, alpha=1.5, preview=True):
        try :
            if not self.imageLabel.pixmap():
                QMessageBox.warning(self, "提示", "请先加载一张图片")
                return

            original_image = self.pixmap_to_cv2_image(self.imageLabel.pixmap())
            original_image = original_image.astype(np.float64)
            contrast_multiplier = np.ones(original_image.shape, dtype=np.float64) * alpha
            contrast_image = cv2.multiply(original_image, contrast_multiplier)
            contrast_image = cv2.convertScaleAbs(contrast_image)
            pixmap = self.cv2_image_to_pixmap(contrast_image)

            # Scale the pixmap to fit the scrollArea
            # scaled_pixmap = pixmap.scaled(self.scrollArea.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)

            if preview:
                label_size = self.previewLabel.size()
                scaled_pixmap = pixmap.scaled(label_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.previewLabel.setPixmap(scaled_pixmap)

            else:
                label_size = self.imageLabel_2.size()
                scaled_pixmap = pixmap.scaled(label_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.imageLabel_2.setPixmap(scaled_pixmap)
        except Exception as e:
            print(f"Error in adjust_contrast: {e}")


    def adjust_brightness(self, beta=50, preview=True):
        try:
            if not self.imageLabel.pixmap():
                QMessageBox.warning(self, "提示", "请先加载一张图片")
                return

            original_image = self.pixmap_to_cv2_image(self.imageLabel.pixmap())
            brightness_image = cv2.add(original_image, np.full(original_image.shape, beta, dtype=np.uint8))

            pixmap = self.cv2_image_to_pixmap(brightness_image)
            # scaled_pixmap = pixmap.scaled(self.scrollArea.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            if preview:
                label_size = self.previewLabel.size()
                scaled_pixmap = pixmap.scaled(label_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.previewLabel.setPixmap(scaled_pixmap)

            else:
                label_size = self.imageLabel_2.size()
                scaled_pixmap = pixmap.scaled(label_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.imageLabel_2.setPixmap(scaled_pixmap)
        except Exception as e:
            print(f"Error in adjust_brightness: {e}")

    def sharpen_image(self, preview=True):
        try:
            if not self.imageLabel.pixmap():
                QMessageBox.warning(self, "提示", "请先加载一张图片")
                return

            original_image = self.pixmap_to_cv2_image(self.imageLabel.pixmap())
            kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
            sharpened_image = cv2.filter2D(original_image, -1, kernel)
            pixmap = self.cv2_image_to_pixmap(sharpened_image)
            # scaled_pixmap = pixmap.scaled(self.scrollArea.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            if preview:
                label_size = self.previewLabel.size()
                scaled_pixmap = pixmap.scaled(label_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.previewLabel.setPixmap(scaled_pixmap)

            else:
                label_size = self.imageLabel_2.size()
                scaled_pixmap = pixmap.scaled(label_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.imageLabel_2.setPixmap(scaled_pixmap)
        except Exception as e:
            print(f"Error in sharpen_image: {e}")

    def smooth_image(self, preview=True):
        try:
            if not self.imageLabel.pixmap():
                QMessageBox.warning(self, "提示", "请先加载一张图片")
                return

            original_image = self.pixmap_to_cv2_image(self.imageLabel.pixmap())
            smooth_image = cv2.GaussianBlur(original_image, (7, 7), 0)
            pixmap = self.cv2_image_to_pixmap(smooth_image)
            # scaled_pixmap = pixmap.scaled(self.scrollArea.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            if preview:
                label_size = self.previewLabel.size()
                scaled_pixmap = pixmap.scaled(label_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.previewLabel.setPixmap(scaled_pixmap)

            else:
                label_size = self.imageLabel_2.size()
                scaled_pixmap = pixmap.scaled(label_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.imageLabel_2.setPixmap(scaled_pixmap)
        except Exception as e:
            print(f"Error in smooth_image: {e}")

    def flip_image(self,  orientation='水平翻转',preview=True):
        try:
            if not self.imageLabel.pixmap():
                QMessageBox.warning(self, "提示", "请先加载一张图片")
                return

            original_image = self.pixmap_to_cv2_image(self.imageLabel.pixmap())

            if orientation == '水平翻转':
                flipped_image = cv2.flip(original_image, 1)
            elif orientation == '垂直翻转':
                flipped_image = cv2.flip(original_image, 0)
            elif orientation == '水平垂直翻转':
                flipped_image = cv2.flip(original_image, -1)

            pixmap = self.cv2_image_to_pixmap(flipped_image)
            # scaled_pixmap = pixmap.scaled(self.scrollArea.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            if preview:
                label_size = self.previewLabel.size()
                scaled_pixmap = pixmap.scaled(label_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.previewLabel.setPixmap(scaled_pixmap)

            else:
                label_size = self.imageLabel_2.size()
                scaled_pixmap = pixmap.scaled(label_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.imageLabel_2.setPixmap(scaled_pixmap)
                # self.imageLabel_2.setPixmap(scaled_pixmap)
                # self.imageLabel_2.setScaledContents(False)
                # self.imageLabel_2.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)
        except Exception as e:
            print(f"Error in flip_image: {e}")
    def histogram_equalization(self, preview=True):
        try:
            if not self.imageLabel.pixmap():
                QMessageBox.warning(self, "提示", "请先加载一张图片")
                return

            original_image = self.pixmap_to_cv2_image(self.imageLabel.pixmap())
            gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
            equalized_image = cv2.equalizeHist(gray_image)
            pixmap = self.cv2_image_to_pixmap(equalized_image)
            scaled_pixmap = pixmap.scaled(self.scrollArea.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            if preview:
                label_size = self.previewLabel.size()
                scaled_pixmap = pixmap.scaled(label_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.previewLabel.setPixmap(scaled_pixmap)

            else:
                label_size = self.imageLabel_2.size()
                scaled_pixmap = pixmap.scaled(label_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.imageLabel_2.setPixmap(scaled_pixmap)
        except Exception as e:
            print(f"Error in histogram_equalization: {e}")

    def pixmap_to_cv2_image(self, pixmap):
        try:
            qimage = pixmap.toImage()
            if not qimage.constBits():
                return None
            qimage = qimage.convertToFormat(QImage.Format_ARGB32)
            buffer = qimage.constBits().asarray(qimage.byteCount())
            image = np.frombuffer(buffer, dtype=np.uint8).reshape((qimage.height(), qimage.width(), 4))
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
            return image
        except Exception as e:
            print(f"Error in pixmap_to_cv2_image: {e}")

    def cv2_image_to_pixmap(self, image):
        try:
            qimage = QImage(image.data, image.shape[1], image.shape[0], QImage.Format.Format_RGB888).rgbSwapped()
            pixmap = QPixmap.fromImage(qimage)
            return pixmap
        except Exception as e:
            print(f"Error in cv2_image_to_pixmap: {e}")

    def morphology_operation(self, operation, preview=True):
        try:
            if not self.imageLabel.pixmap():
                QMessageBox.warning(self, "提示", "请先加载一张图片")
                return

            original_image = self.pixmap_to_cv2_image(self.imageLabel.pixmap())
            gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
            _, binary_image = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY)

            kernel = np.ones((5, 5), np.uint8)

            if operation == 'dilate':
                result_image = cv2.dilate(binary_image, kernel, iterations=1)
            elif operation == 'erode':
                result_image = cv2.erode(binary_image, kernel, iterations=1)
            elif operation == 'opening':
                result_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)
            elif operation == 'closing':
                result_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)

            pixmap = self.cv2_image_to_pixmap(result_image)
            scaled_pixmap = pixmap.scaled(self.scrollArea.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            if preview:
                label_size = self.previewLabel.size()
                scaled_pixmap = pixmap.scaled(label_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.previewLabel.setPixmap(scaled_pixmap)

            else:
                label_size = self.imageLabel_2.size()
                scaled_pixmap = pixmap.scaled(label_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.imageLabel_2.setPixmap(scaled_pixmap)
        except Exception as e:
            print(f"Error in morphology_operation: {e}")
    def mean_filter(self, preview=True):
        try:
            if not self.imageLabel.pixmap():
                QMessageBox.warning(self, "提示", "请先加载一张图片")
                return

            original_image = self.pixmap_to_cv2_image(self.imageLabel.pixmap())
            mean_filtered_image = cv2.blur(original_image, (5, 5))
            pixmap = self.cv2_image_to_pixmap(mean_filtered_image)
            scaled_pixmap = pixmap.scaled(self.scrollArea.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            if preview:
                label_size = self.previewLabel.size()
                scaled_pixmap = pixmap.scaled(label_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.previewLabel.setPixmap(scaled_pixmap)

            else:
                label_size = self.imageLabel_2.size()
                scaled_pixmap = pixmap.scaled(label_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.imageLabel_2.setPixmap(scaled_pixmap)
        except Exception as e:
            print(f"Error in mean_filter: {e}")

    def gaussian_filter(self, preview=True):
        try:
            if not self.imageLabel.pixmap():
                QMessageBox.warning(self, "提示", "请先加载一张图片")
                return

            original_image = self.pixmap_to_cv2_image(self.imageLabel.pixmap())
            gaussian_filtered_image = cv2.GaussianBlur(original_image, (5, 5), 0)
            pixmap = self.cv2_image_to_pixmap(gaussian_filtered_image)
            scaled_pixmap = pixmap.scaled(self.scrollArea.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            if preview:
                label_size = self.previewLabel.size()
                scaled_pixmap = pixmap.scaled(label_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.previewLabel.setPixmap(scaled_pixmap)

            else:
                label_size = self.imageLabel_2.size()
                scaled_pixmap = pixmap.scaled(label_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.imageLabel_2.setPixmap(scaled_pixmap)
        except Exception as e:
            print(f"Error in gaussian_filter: {e}")

    def median_filter(self, preview=True):
        try:
            if not self.imageLabel.pixmap():
                QMessageBox.warning(self, "提示", "请先加载一张图片")
                return

            original_image = self.pixmap_to_cv2_image(self.imageLabel.pixmap())
            median_filtered_image = cv2.medianBlur(original_image, 5)
            pixmap = self.cv2_image_to_pixmap(median_filtered_image)
            scaled_pixmap = pixmap.scaled(self.scrollArea.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            if preview:
                label_size = self.previewLabel.size()
                scaled_pixmap = pixmap.scaled(label_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.previewLabel.setPixmap(scaled_pixmap)

            else:
                label_size = self.imageLabel_2.size()
                scaled_pixmap = pixmap.scaled(label_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.imageLabel_2.setPixmap(scaled_pixmap)
        except Exception as e:
            print(f"Error in median_filter: {e}")

    def load_image1(self, file_name):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        file_name, _ = QFileDialog.getOpenFileName(self, "Load Image", "",
                                                   "Images (*.png *.xpm *.jpg *.bmp);;All Files (*)", options=options)

        if file_name:
            # Load and display the image
            pixmap = QPixmap(file_name)
            self.imageLabel.setPixmap(
                pixmap.scaled(self.imageLabel.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
            self.imageLabel.setScaledContents(False)
            self.imageLabel.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)

            # 将图片添加到self.image_list并设置self.current_image_index为0
            self.image_list = [file_name]
            self.current_image_index = 0

    def load_image(self, file_name):
        if file_name:
            pixmap = QPixmap(file_name)
            self.imageLabel.setPixmap(
                pixmap.scaled(self.imageLabel.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
            self.imageLabel.setScaledContents(False)
            self.imageLabel.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)

    def open_folder(self):
        folder_name = QFileDialog.getExistingDirectory(self, "打开文件夹")
        if folder_name:
            self.image_list = [os.path.join(folder_name, f) for f in os.listdir(folder_name)
                               if os.path.splitext(f)[-1].lower() in ['.png', '.jpg', '.bmp', '.xpm']]
            self.current_image_index = 0
            self.load_image(self.image_list[self.current_image_index])

    def select_save_folder(self):
        self.save_folder = QFileDialog.getExistingDirectory(self, "选择保存文件夹")

    def next_image(self):
        if not self.image_list:
            QMessageBox.warning(self, "提示", "请先打开一个文件夹")
            return

        self.current_image_index = (self.current_image_index + 1) % len(self.image_list)
        self.load_image(self.image_list[self.current_image_index])

    def previous_image(self):
        if not self.image_list:
            QMessageBox.warning(self, "提示", "请先打开一个文件夹")
            return

        self.current_image_index = (self.current_image_index - 1) % len(self.image_list)
        self.load_image(self.image_list[self.current_image_index])

    def save_image(self):
        if not self.imageLabel_2.pixmap():
            QMessageBox.warning(self, "提示", "没有处理后的图片可供保存")
            return

        if not self.save_folder:
            QMessageBox.warning(self, "提示", "请先选择保存文件夹")
            return

        if not self.image_list or self.current_image_index >= len(self.image_list):
            QMessageBox.warning(self, "提示", "当前没有图像可供保存")
            return

        file_name = os.path.basename(self.image_list[self.current_image_index])
        save_path = os.path.join(self.save_folder, "processed_" + file_name)
        self.imageLabel_2.pixmap().save(save_path)
        QMessageBox.information(self, "提示", f"图片已成功保存到：{save_path}")




    def apply_image_enhance(self, preview):
        operation = self.imageEnhanceComboBox.currentText()

        if operation == '对比度增强':
            self.adjust_contrast(preview=preview)
        # ... and so on for other operations

        elif operation == '亮度增强':
            self.adjust_brightness(preview=preview)
        elif operation == '锐化':
            self.sharpen_image(preview=preview)
        elif operation == '平滑':
            self.smooth_image(preview=preview)
        else:
            self.flip_image(operation,preview=preview)


    def apply_image_filter(self,preview):
        operation = self.imageFilterComboBox.currentText()

        if operation == '均值滤波':
            self.mean_filter(preview=preview)
            #高斯滤波
        elif operation == '高斯滤波':
            self.gaussian_filter(preview=preview)
            #中值滤波
        elif operation == '中值滤波':
            self.median_filter(preview=preview)

    def apply_morphology(self,preview):
        operation = self.morphologyComboBox.currentText()

        if operation == '膨胀':
            self.morphology_operation('dilate', preview=preview)
        elif operation == '腐蚀':
            self.morphology_operation('erode', preview=preview)
        elif operation == '开运算':
            self.morphology_operation('opening', preview=preview)
        elif operation == '闭运算':
            self.morphology_operation('closing', preview=preview)

    def apply_histogram(self,preview):
        operation = self.histogramComboBox.currentText()

        if operation == '直方图均衡化':
            self.histogram_equalization(preview=preview)


    def scale_image(self):
        try:
            # 获取滑块的值（将滑块的值除以100，使其表示为百分比，然后用200减去当前值实现反转）
            scale_value = (200 - self.scaleSlider.value()) / 100.0
            # 检查原始图像是否已加载
            if not self.imageLabel.pixmap():
                QMessageBox.warning(self, "提示", "请先加载一张图片")
                return
            # 获取原始图像
            original_pixmap = self.imageLabel.pixmap()
            original_size = original_pixmap.size()
            # 根据缩放比例计算新的图像尺寸
            new_size = original_size * scale_value
            # 对原始图像进行缩放
            scaled_pixmap = original_pixmap.scaled(new_size, Qt.KeepAspectRatio)
            # 在 imageLabel 上显示缩放后的图像
            self.imageLabel.setPixmap(scaled_pixmap)
        except Exception as e:
            print(f"Error in scale_image: {e}")

    def rotate_image(self):
        try:
            rotation_angle = self.rotationSlider.value()
            if not self.imageLabel.pixmap():
                QMessageBox.warning(self, "提示", "请先加载一张图片")
                return

            original_pixmap = self.imageLabel.pixmap()
            original_size = original_pixmap.size()

            transform = QTransform().rotate(rotation_angle)
            rotated_pixmap = original_pixmap.transformed(transform, mode=Qt.SmoothTransformation)

            self.imageLabel_2.setPixmap(
                rotated_pixmap.scaled(self.scrollArea.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
            self.imageLabel_2.setScaledContents(False)
            self.imageLabel_2.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)
        except Exception as e:
            print(f"Error in rotate_image: {e}")



    def show_help(self):
        help_text = (
            "<h3>图像增强操作详细信息：</h3>"
            "<ul>"
            "<li><b>翻转</b>：水平或垂直翻转图像。</li>"
            "<li><b>随机裁剪</b>：从图像中随机裁剪一部分。</li>"
            "<li><b>随机缩放</b>：对图像进行随机缩放。</li>"
            "<li><b>随机旋转</b>：在一定范围内随机旋转图像。</li>"
            "<li><b>随机亮度/对比度/饱和度/色调</b>：调整图像的亮度、对比度、饱和度和色调。</li>"
            "<li><b>随机擦除</b>：在图像上随机擦除一个矩形区域。</li>"
            "<li><b>随机噪声</b>：在图像上添加随机噪声。</li>"
            "<li><b>剪切</b>：在图像上随机剪切一个矩形区域。</li>"
            "<li><b>Mixup</b>：将两个图像按一定比例混合在一起。</li>"
            "<li><b>Mosaic</b>：将四个图像组合成一个马赛克图像。</li>"
            "</ul>"
            "<p><b>注意：</b>Mixup 和 Mosaic 操作不能与其他操作同时使用。</p>"
            "<h3>建议的图像增强组合：</h3>"
            "<ol>"
            "<li>翻转 + 随机裁剪 + 随机缩放 + 随机旋转</li>"
            "<li>随机亮度/对比度/饱和度/色调 + 随机擦除 + 随机噪声 + 剪切</li>"
            "<li>使用 Mixup 或 Mosaic 作为单独操作。</li>"
            "</ol>"
        )

        style_sheet = """
            QMessageBox {
                background-color: rgba(0, 0, 0, 50%);
                color: white;
                font-size: 14px;
                font-weight: bold;
                border-radius: 10px;
            }

            QMessageBox QLabel {
                color: white;
                font-size: 14px;
            }

            QPushButton {
                background-color: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 rgb(3, 85, 160), stop:1 rgb(13, 206, 197));
                color: white;
                font-size: 14px;
                font-weight: bold;
                border-radius: 5px;
                padding: 5px;
                min-width: 100px;
            }

            QPushButton:hover {
                background-color: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 rgb(13, 206, 197), stop:1 rgb(3, 85, 160));
            }

            QPushButton:pressed {
                background-color: rgb(20, 120, 100);
            }
        """

        # 创建自定义 QMessageBox
        message_box = QMessageBox(self)
        message_box.setWindowTitle("帮助")
        message_box.setText(help_text)
        message_box.setIcon(QMessageBox.Information)


        message_box.setStyleSheet(style_sheet)

        message_box.exec_()

    # 添加新的 checkbox_clicked 方法
    def checkbox_clicked(self):
            incompatible_operations = {8, 9}  # Mixup 和 Mosaic 的索引

            # 获取当前已选中的复选框索引
            checked_indexes = [i for i, checkbox in enumerate(
                [self.enhanceMethod1Checkbox, self.enhanceMethod2Checkbox, self.enhanceMethod3Checkbox,
                 self.enhanceMethod4Checkbox, self.enhanceMethod5Checkbox, self.enhanceMethod6Checkbox,
                 self.enhanceMethod7Checkbox, self.enhanceMethod8Checkbox, self.enhanceMethod9Checkbox,
                 self.enhanceMethod10Checkbox]) if checkbox.isChecked()]

            # 检查是否选中了不兼容的操作
            if any(index in checked_indexes for index in incompatible_operations) and len(checked_indexes) > 1:
                QMessageBox.warning(self, "警告", "Mixup 或 Mosaic 操作不能与其他操作同时使用，请取消选择不兼容的操作。")


    def init_progress_bar(self):
        self.image_augmentor.progress_updated.connect(self.update_progress)
        self.progressBar_2.setValue(0)  # 假设 progressBar 是您在 Qt Designer 中添加的 QProgressBar 对象的名称

    @pyqtSlot(int)
    def update_progress(self, value):
        self.progressBar_2.setValue(self.progressBar_2.value() + value)






    # 在这里添加 select_input_folder、select_output_folder 和start_processing 槽函数的实现。
    def select_input_folder(self):
        folder_path = QFileDialog.getExistingDirectory()
        self.inputFolderLabel.setText(folder_path)

    def select_output_folder(self):
        folder_path = QFileDialog.getExistingDirectory()
        self.outputFolderLabel.setText(folder_path)

    def start_processing(self):
        input_folder = self.inputFolderLabel.text()
        output_folder = self.outputFolderLabel.text()
        probabilities = [float(self.enhanceProbabilityInput.text()) if checkbox.isChecked() else 0 for checkbox in
                         [self.enhanceMethod1Checkbox, self.enhanceMethod2Checkbox, self.enhanceMethod3Checkbox,
                          self.enhanceMethod4Checkbox, self.enhanceMethod5Checkbox, self.enhanceMethod6Checkbox,
                          self.enhanceMethod7Checkbox, self.enhanceMethod8Checkbox, self.enhanceMethod9Checkbox,
                          self.enhanceMethod10Checkbox]]

        success_count, failure_count, image_count = self.image_augmentor.process_images(input_folder, output_folder,
                                                                                        probabilities)

        # 设置进度条的最大值
        self.progressBar_2.setMaximum(image_count)

        # 显示弹窗
        QMessageBox.information(self, "处理完成", f"成功处理 {success_count} 个图像，失败 {failure_count} 个")

        self.progressBar_2.setValue(0)

    def open_advanced_settings(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("高级参数设置")

        layout = QVBoxLayout()

        # 为每个增强方法添加一个 QLineEdit，用于输入自定义参数
        # 添加默认值和输入规范的 QLabel
        enhance_method2_parameter = QLineEdit(self.enhanceMethod2Parameter.text())  # 使用默认值作为 QLineEdit 的初始值
        enhance_method2_label = QLabel("scale_range (tuple): (min, max), e.g., (0.8, 1.0)")
        layout.addWidget(enhance_method2_label)
        layout.addWidget(enhance_method2_parameter)

        # ...在此处添加更多的增强方法参数和 QLabel...

        # 添加确认和取消按钮
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)

        dialog.setLayout(layout)

        # 如果用户点击“确认”，请使用新的参数值更新增强方法
        if dialog.exec() == QDialog.Accepted:
            self.enhanceMethod2Parameter.setText(enhance_method2_parameter.text())
            # ...在此处更新更多增强方法参数值...










#检测线程的参数


    def on_save_crop_nmscheckbox_checkbox_stateChanged(self, state):
        global save_crop
        save_crop = state == 2
        print(f"on_save_crop_nmscheckbox is now: {save_crop}")
    def on_augment_nmscheckbox_checkbox_stateChanged(self, state):
        global augment
        augment = state == 2
        print(f"augmentedcheckbox_3 is now: {augment}")
    def on_imgsz_spinbox_valueChanged(self, value):
        global imgsz
        imgsz = value
        print(f"imgsz is now: {imgsz}")
    def on_line_thickness_spinbox_valueChanged(self, value):
        global line_thickness
        line_thickness = value
        print(f"line_thickness is now: {line_thickness}")
    def update_classes(self, text):
        global classes
        if text.strip():
            classes = [int(c) for c in text.split()]
        else:
            classes = None
        print(f"Classes are now: {classes}")
    def on_hide_labelscheckbox_checkbox_checkbox_stateChanged(self, state):
        global hide_labels
        hide_labels = state == 2
        print(f"hide_labels is now: {hide_labels}")
    def on_hide_confcheckbox_checkbox_checkbox_stateChanged(self, state):
        global hide_conf
        hide_conf = state == 2
        print(f"hide_conf is now: {hide_conf}")
    def on_browse_result_folder_button_clicked(self):
        # print("Function on_browse_result_folder_button_clicked called")
        global project
        # Get the button that triggered the slot
        button = self.sender()
        # Disconnect the button's clicked signal
        button.clicked.disconnect()
        # Browse the folder
        folder_path = QFileDialog.getExistingDirectory(self, "Select Result Folder", str(Path.home()))
        if folder_path:
            project = Path(folder_path)
            self.result_folder_lineEdit.setText(folder_path)
        # Reconnect the button's clicked signal
        button.clicked.connect(self.on_browse_result_folder_button_clicked)
        # print("on_browse_result_folder_button_clickedproject:",project)




    # 添加一个新的槽函数，用于根据给定的序号切换 stackedWidget 的当前页面
    def open_stackedWidget(self, index):
        self.stackedWidget.setCurrentIndex(index)





    def search_pt(self):
        pt_list = os.listdir('./pt')
        pt_list = [file for file in pt_list if file.endswith('.pt')]
        pt_list.sort(key=lambda x: os.path.getsize('./pt/' + x))

        if pt_list != self.pt_list:
            self.pt_list = pt_list
            self.comboBox.clear()
            self.comboBox.addItems(self.pt_list)

    # def is_save(self):
    #     if self.saveCheckBox.isChecked():
    #         # 选中时
    #         self.det_thread.save_fold = './result'
    #     else:
    #         self.det_thread.save_fold = None

    def is_save(self):
        global project
        if self.saveCheckBox.isChecked():
            # 选中时
            self.det_thread.save_fold = str(project)
        else:
            self.det_thread.save_fold = None

    def checkrate(self):
        if self.checkBox.isChecked():
            self.det_thread.rate_check = True
        else:
            self.det_thread.rate_check = False



    def chose_cam(self):
        try:
            self.stop()
            # MessageBox的作用：留出2秒，让上一次摄像头安全release
            MessageBox(
                self.closeButton, title='Tips', text='Loading camera', time=2000, auto=True).exec_()
            # 自动检测本机有哪些摄像头
            _, cams = Camera().get_cam_num()
            popMenu = QMenu()
            popMenu.setFixedWidth(self.cameraButton.width())
            popMenu.setStyleSheet('''
                                            QMenu {
                                            font-size: 16px;
                                            font-family: "Microsoft YaHei UI";
                                            font-weight: light;
                                            color:white;
                                            padding-left: 5px;
                                            padding-right: 5px;
                                            padding-top: 4px;
                                            padding-bottom: 4px;
                                            border-style: solid;
                                            border-width: 0px;
                                            border-color: rgba(255, 255, 255, 255);
                                            border-radius: 3px;
                                            background-color: rgba(200, 200, 200,50);}
                                            ''')

            for cam in cams:
                exec("action_%s = QAction('%s')" % (cam, cam))
                exec("popMenu.addAction(action_%s)" % cam)

            x = self.groupBox_5.mapToGlobal(self.cameraButton.pos()).x()
            y = self.groupBox_5.mapToGlobal(self.cameraButton.pos()).y()
            y = y + self.cameraButton.frameGeometry().height()
            pos = QPoint(x, y)
            action = popMenu.exec_(pos)
            if action:
                self.det_thread.source = action.text()
                self.statistic_msg('Loading camera：{}'.format(action.text()))
        except Exception as e:
            self.statistic_msg('%s' % e)
            print('%s' % e)
            print(type(e))  # 查看异常的具体类型
    # 导入配置文件
    def load_setting(self):
        config_file = 'config/setting.json'
        if not os.path.exists(config_file):
            iou = 0.26
            conf = 0.33
            rate = 10
            check = 0
            savecheck = 0
            new_config = {"iou": iou,
                          "conf": conf,
                          "rate": rate,
                          "check": check,
                          "savecheck": savecheck
                          }
            new_json = json.dumps(new_config, ensure_ascii=False, indent=2)
            with open(config_file, 'w', encoding='utf-8') as f:
                f.write(new_json)
        else:
            config = json.load(open(config_file, 'r', encoding='utf-8'))
            if len(config) != 5:
                iou = 0.26
                conf = 0.33
                rate = 10
                check = 0
                savecheck = 0
            else:
                iou = config['iou']
                conf = config['conf']
                rate = config['rate']
                check = config['check']
                savecheck = config['savecheck']
        self.confSpinBox.setValue(iou)
        self.iouSpinBox.setValue(conf)
        self.rateSpinBox.setValue(rate)
        self.checkBox.setCheckState(check)
        self.det_thread.rate_check = check   # 是否启用延时
        self.saveCheckBox.setCheckState(savecheck)
        self.is_save()    # 是否自动保存

    def change_val(self, x, flag):
        if flag == 'confSpinBox':
            self.confSlider.setValue(int(x*100))
        elif flag == 'confSlider':
            self.confSpinBox.setValue(x/100)
            self.det_thread.conf_thres = x/100
        elif flag == 'iouSpinBox':
            self.iouSlider.setValue(int(x*100))
        elif flag == 'iouSlider':
            self.iouSpinBox.setValue(x/100)
            self.det_thread.iou_thres = x/100
        elif flag == 'rateSpinBox':
            self.rateSlider.setValue(x)
        elif flag == 'rateSlider':
            self.rateSpinBox.setValue(x)
            self.det_thread.rate = x * 10
        else:
            pass

    def statistic_msg(self, msg):
        self.statistic_label.setText(msg)
        # self.qtimer.start(3000)   # 3秒后自动清除

    def show_msg(self, msg):
        self.runButton.setChecked(Qt.Unchecked)
        self.statistic_msg(msg)
        if msg == "Finished":
            self.saveCheckBox.setEnabled(True)

    def change_model(self, x):
        self.model_type = self.comboBox.currentText()
        self.det_thread.weights = "./pt/%s" % self.model_type
        self.statistic_msg('Change model to %s' % x)

    def open_file(self):

        config_file = 'config/fold.json'
        # config = json.load(open(config_file, 'r', encoding='utf-8'))
        config = json.load(open(config_file, 'r', encoding='utf-8'))
        open_fold = config['open_fold']
        if not os.path.exists(open_fold):
            open_fold = os.getcwd()
        name, _ = QFileDialog.getOpenFileName(self, 'Video/image', open_fold, "Pic File(*.mp4 *.mkv *.avi *.flv "
                                                                          "*.jpg *.png)")
        if name:
            self.det_thread.source = name
            self.statistic_msg('Loaded file：{}'.format(os.path.basename(name)))
            config['open_fold'] = os.path.dirname(name)
            config_json = json.dumps(config, ensure_ascii=False, indent=2)
            with open(config_file, 'w', encoding='utf-8') as f:
                f.write(config_json)
            # 切换文件后，上一次检测停止
            self.stop()

    def max_or_restore(self):
        if self.maxButton.isChecked():
            self.showMaximized()
        else:
            self.showNormal()

    # 继续/暂停
    def run_or_continue(self):
        self.det_thread.jump_out = False
        if self.runButton.isChecked():
            self.saveCheckBox.setEnabled(False)
            self.det_thread.is_continue = True
            if not self.det_thread.isRunning():
                self.det_thread.start()
            source = os.path.basename(self.det_thread.source)
            source = 'camera' if source.isnumeric() else source
            self.statistic_msg('Detecting >> model：{}，file：{}'.
                               format(os.path.basename(self.det_thread.weights),
                                      source))
        else:
            self.det_thread.is_continue = False
            self.statistic_msg('Pause')

    # 退出检测循环
    def stop(self):
        self.det_thread.jump_out = True
        self.saveCheckBox.setEnabled(True)

    def mousePressEvent(self, event):
        self.m_Position = event.pos()
        if event.button() == Qt.LeftButton:
            if 0 < self.m_Position.x() < self.groupBox.pos().x() + self.groupBox.width() and \
                    0 < self.m_Position.y() < self.groupBox.pos().y() + self.groupBox.height():
                self.m_flag = True

    def mouseMoveEvent(self, QMouseEvent):
        if Qt.LeftButton and self.m_flag:
            self.move(QMouseEvent.globalPos() - self.m_Position)

    def mouseReleaseEvent(self, QMouseEvent):
        self.m_flag = False

    @staticmethod
    def show_image(img_src, label):
        try:
            ih, iw, _ = img_src.shape
            w = label.geometry().width()
            h = label.geometry().height()
            # keep original aspect ratio
            # 保持纵横比
            # 找出长边
            if iw/w > ih/h:
                scal = w / iw
                nw = w
                nh = int(scal * ih)
                img_src_ = cv2.resize(img_src, (nw, nh))

            else:
                scal = h / ih
                nw = int(scal * iw)
                nh = h
                img_src_ = cv2.resize(img_src, (nw, nh))

            frame = cv2.cvtColor(img_src_, cv2.COLOR_BGR2RGB)
            img = QImage(frame.data, frame.shape[1], frame.shape[0], frame.shape[2] * frame.shape[1],
                         QImage.Format_RGB888)
            label.setPixmap(QPixmap.fromImage(img))

        except Exception as e:
            print(repr(e))


    # 实时统计
    def show_statistic(self, statistic_dic):
        try:
            self.resultWidget.clear()
            statistic_dic = sorted(statistic_dic.items(), key=lambda x: x[1], reverse=True)
            statistic_dic = [i for i in statistic_dic if i[1] > 0]
            results = [' '+str(i[0]) + '：' + str(i[1]) for i in statistic_dic]
            self.resultWidget.addItems(results)

        except Exception as e:
            print(repr(e))

    def closeEvent(self, event):
        # 如果摄像头开着，先把摄像头关了再退出，否则极大可能可能导致检测线程未退出
        self.det_thread.jump_out = True
        # 退出时，保存设置
        config_file = 'config/setting.json'
        config = dict()
        config['iou'] = self.confSpinBox.value()
        config['conf'] = self.iouSpinBox.value()
        config['rate'] = self.rateSpinBox.value()
        config['check'] = self.checkBox.checkState()
        config['savecheck'] = self.saveCheckBox.checkState()
        config_json = json.dumps(config, ensure_ascii=False, indent=2)
        with open(config_file, 'w', encoding='utf-8') as f:
            f.write(config_json)
        MessageBox(
            self.closeButton, title='Tips', text='Closing the program', time=2000, auto=True).exec_()
        sys.exit(0)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    myWin = MainWindow()
    myWin.show()
    sys.exit(app.exec_())
