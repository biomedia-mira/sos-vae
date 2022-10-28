import warnings

from PySide6 import QtCore, QtWidgets, QtGui

from interactive.utils import torch_to_QImage


class ImageCanvas(QtWidgets.QScrollArea):
    def __init__(self, size):
        super().__init__()

        self.width, self.height = size
        self.scale = 1

        self.qImg_cache = None
        self.qImg_buffer_cache = None

        self.setWidgetResizable(False)
        self.setAlignment(QtCore.Qt.AlignCenter)
        self.setStyleSheet("QScrollArea {background-color: #303030}")

        self.label = QtWidgets.QLabel()
        self.label.setSizePolicy(
            QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Ignored
        )
        self.label.setScaledContents(True)
        self.setWidget(self.label)

        self.pixmap = QtGui.QPixmap(self.width, self.height)
        self.pixmap.fill(QtGui.QColor("white"))
        self.label.setPixmap(self.pixmap)

        self.painter = QtGui.QPainter(self.pixmap)

        self.action_zoom_in = QtGui.QAction("Zoom in", self)
        self.action_zoom_in.setShortcut(QtGui.QKeySequence.ZoomIn)
        self.action_zoom_in.triggered.connect(lambda: self.zoom_label(factor=2))

        self.action_zoom_out = QtGui.QAction("Zoom out", self)
        self.action_zoom_out.setShortcut(QtGui.QKeySequence.ZoomOut)
        self.action_zoom_out.triggered.connect(lambda: self.zoom_label(factor=0.5))

        self.zoom_label()

    def close(self):
        self.painter.end()
        super().close()

    def zoom_label(self, factor=None):
        if factor is not None:
            self.scale *= factor
        self.label.resize(self.width * self.scale, self.height * self.scale)

    def update_label(self):
        self.label.setPixmap(self.pixmap)

    def paint_image(self, img=None, cached_only=False):
        # Buffer must be kept in memory prior to usage

        if (
            cached_only
            and self.qImg_cache is not None
            and self.qImg_buffer_cache is not None
        ):
            if img is not None:
                warnings.warn("Using cached image, when new image is provided")

            self.painter.drawImage(0, 0, self.qImg_cache)
        else:
            if img is None:
                raise RuntimeError("Image to paint is not given")

            qImg, buffer = torch_to_QImage(img)

            self.qImg_cache = qImg
            self.qImg_buffer_cache = buffer

            self.painter.drawImage(0, 0, qImg)

        self.update_label()
