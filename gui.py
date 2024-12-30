import os
import sys
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtWidgets import (QApplication, QWidget, QLabel, QFileDialog, QLineEdit, QPushButton, QVBoxLayout,
                             QTextEdit, QDialog)
from PyQt5.QtGui import QPixmap
from test import classify_image


class ImageViewer(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("图像识别CIFAR-10")
        self.setGeometry(800, 300, 1000, 1000)

        # 布局
        self.layout = QVBoxLayout()

        # 标签用于显示图片
        self.imageLabel = QLabel("请选择一张图片")
        self.imageLabel.setMinimumHeight(800)  # 设置最小高度
        self.imageLabel.setAlignment(Qt.AlignCenter)  # 居中显示
        self.imageLabel.setStyleSheet("font-size: 50px;")  # 设置字体大小
        self.layout.addWidget(self.imageLabel)

        # 文本框用于输出图片类型
        self.infoTextEdit = QLineEdit()
        self.infoTextEdit.setMinimumHeight(50)
        self.infoTextEdit.setReadOnly(True)
        self.infoTextEdit.setAlignment(Qt.AlignCenter)  # 居中显示
        self.infoTextEdit.setStyleSheet("font-size: 40px;")  # 设置字体大小
        self.layout.addWidget(self.infoTextEdit)

        # 按钮用于打开文件
        self.openButton = QPushButton("打开图片")
        self.openButton.clicked.connect(self.openImage)
        self.openButton.setStyleSheet("font-size: 30px;")  # 设置字体大小
        self.layout.addWidget(self.openButton)

        self.openButton2 = QPushButton("批量处理")
        self.openButton2.clicked.connect(self.batchprocessing)
        self.openButton2.setStyleSheet("font-size: 30px;")  # 设置字体大小
        self.layout.addWidget(self.openButton2)

        self.setLayout(self.layout)

    def openImage(self):
        try:
            # 打开文件对话框
            options = QFileDialog.Options()
            fileName, _ = QFileDialog.getOpenFileName(self, "打开图片", "",
                                                      "Images (*.png *.jpeg *.jpg *.bmp)",
                                                      options=options)
            if fileName:
                # 显示选择的图片
                pixmap = QPixmap(fileName)
                self.imageLabel.setPixmap(
                    pixmap.scaled(self.imageLabel.size(), aspectRatioMode=True, transformMode=True))

                # 显示“识别中”
                self.infoTextEdit.setText("识别中...")

                # 创建并启动线程
                self.classifier_thread = ImageClassifierThread(fileName)
                self.classifier_thread.result_ready.connect(self.infoTextEdit.setText)
                self.classifier_thread.start()

        except Exception as e:
            print("An error occurred:", e)

    def batchprocessing(self):
        try:
            # 打开文件对话框，允许多选
            options = QFileDialog.Options()
            files, _ = QFileDialog.getOpenFileNames(self, "选择图片", "",
                                                    "Images (*.png *.jpeg *.jpg *.bmp)", options=options)
            if files:
                self.processing_thread = ImageProcessingThread(files)
                self.processing_thread.progress_signal.connect(self.update_progress)
                self.processing_thread.finished_signal.connect(self.show_results)
                self.processing_thread.start()

        except Exception as e:
            print("An error occurred:", e)

    def update_progress(self, progress_text):
        self.infoTextEdit.setText(progress_text)

    def show_results(self, results):
        result_text = "\n".join(results)
        dialog = ResultDialog(self, result_text)
        dialog.exec_()


class ImageProcessingThread(QThread):
    progress_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(list)

    def __init__(self, files):
        super().__init__()
        self.files = files

    def run(self):
        results = []
        total = len(self.files)
        for i, file_path in enumerate(self.files, 1):
            class_name = get_class(file_path)
            file_name = os.path.basename(file_path)
            results.append(f"{file_name}: {class_name}")
            if i != total:
                message = f"处理进度: {i}/{total}"
            else:
                message = f"处理完成：{i}/{total}"
            self.progress_signal.emit(message)

        self.finished_signal.emit(results)


class ImageClassifierThread(QThread):
    result_ready = pyqtSignal(str)

    def __init__(self, file_path):
        super().__init__()
        self.file_path = file_path

    def run(self):
        class_name = get_class(self.file_path)
        self.result_ready.emit(class_name)


def get_class(file_path):
    class_names = {
        0: "飞机",
        1: "汽车",
        2: "鸟",
        3: "猫",
        4: "鹿",
        5: "狗",
        6: "青蛙",
        7: "马",
        8: "船",
        9: "卡车"
    }
    num = classify_image(file_path)
    return class_names.get(num, "未知类别")


class ResultDialog(QDialog):
    def __init__(self, parent=None, result_text=""):
        super().__init__(parent)
        self.setWindowTitle("批量处理结果")
        self.setGeometry(900, 330, 800, 800)

        layout = QVBoxLayout()

        # 创建文本编辑框并设置字体
        self.text_edit = QTextEdit()
        self.text_edit.setPlainText(result_text)
        self.text_edit.setReadOnly(True)
        self.text_edit.setStyleSheet("font-size: 40px;")
        layout.addWidget(self.text_edit)

        self.setLayout(layout)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ImageViewer()
    window.show()
    sys.exit(app.exec_())
