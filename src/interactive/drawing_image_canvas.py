from PySide6 import QtCore, QtWidgets, QtGui

from interactive.image_canvas import ImageCanvas
from interactive.utils import QImage_to_torch


class DrawingImageCanvas(ImageCanvas):
    def __init__(self, size, on_draw_complete):
        super().__init__(size)

        self.on_draw_complete = on_draw_complete

        self.pen_size = 1
        self.pen_color = QtGui.QColor("black")

        self.last_pos = None

        self.label.setCursor(QtGui.QCursor(QtCore.Qt.BlankCursor))
        self.label.setMouseTracking(True)
        self.label.mouseMoveEvent = self.labelMouseMoveEvent
        self.label.mouseReleaseEvent = self.labelMouseReleaseEvent

    def labelMouseMoveEvent(self, e):
        pos = e.pos()

        self.painter.setPen(QtCore.Qt.SolidLine)
        pen = self.painter.pen()
        pen.setWidth(self.pen_size)
        pen.setColor(self.pen_color)
        self.painter.setPen(pen)

        if not e.buttons() & QtCore.Qt.LeftButton:
            # Hover only
            self.paint_image(cached_only=True)
            self.painter.drawPoint(pos.x() // self.scale, pos.y() // self.scale)
            self.update_label()
            return

        if self.last_pos is None:
            self.last_pos = pos
            return

        self.painter.drawLine(
            self.last_pos.x() // self.scale,
            self.last_pos.y() // self.scale,
            pos.x() // self.scale,
            pos.y() // self.scale,
        )

        self.update_label()

        self.last_pos = pos

    def labelMouseReleaseEvent(self, e):
        if e.button() == QtCore.Qt.LeftButton:
            self.last_pos = None

            self.on_draw_complete(QImage_to_torch(self.pixmap.toImage()))
