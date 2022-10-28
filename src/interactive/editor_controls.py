from PySide6 import QtCore, QtWidgets, QtGui


class ControlsWidget(QtWidgets.QWidget):
    def __init__(self, update_pen_size, update_pen_color, grayscale=False):
        super().__init__()

        self.update_pen_size = update_pen_size
        self.update_pen_color = update_pen_color

        self.main_layout = QtWidgets.QVBoxLayout()
        self.main_layout.setSpacing(10)
        self.setLayout(self.main_layout)

        # Size control
        self.size_control_layout = QtWidgets.QHBoxLayout()
        self.size_control_layout.setSpacing(10)
        self.main_layout.addLayout(self.size_control_layout)

        self.size_control_label = QtWidgets.QLabel(
            "Pen size: 1" + " " * (len(str(10)) - 1) + "\t"
        )
        self.size_control_layout.addWidget(self.size_control_label)

        self.size_control_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.size_control_slider.setMinimum(1)
        self.size_control_slider.setMaximum(10)
        self.size_control_slider.setValue(1)
        self.size_control_slider.valueChanged.connect(self.pen_size_changed)
        self.size_control_layout.addWidget(self.size_control_slider)

        # Color control
        self.color_label = QtWidgets.QLabel("Pen colour")
        self.main_layout.addWidget(self.color_label)

        if grayscale:
            self.gray_control_layout = QtWidgets.QHBoxLayout()
            self.gray_control_layout.setSpacing(10)
            self.main_layout.addLayout(self.gray_control_layout)

            self.gray_control_label = QtWidgets.QLabel(
                "Gray: 0   " + " " * (len(str(255)) - 1) + "\t"
            )
            self.gray_control_layout.addWidget(self.gray_control_label)

            self.gray_control_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
            self.gray_control_slider.setMinimum(0)
            self.gray_control_slider.setMaximum(255)
            self.gray_control_slider.setValue(0)
            self.gray_control_slider.valueChanged.connect(self.gray_changed)
            self.gray_control_layout.addWidget(self.gray_control_slider)
        else:
            # Red
            self.red_control_layout = QtWidgets.QHBoxLayout()
            self.red_control_layout.setSpacing(10)
            self.main_layout.addLayout(self.red_control_layout)

            self.red_control_label = QtWidgets.QLabel(
                "Red: 0    " + " " * (len(str(255)) - 1) + "\t"
            )
            self.red_control_layout.addWidget(self.red_control_label)

            self.red_control_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
            self.red_control_slider.setMinimum(0)
            self.red_control_slider.setMaximum(255)
            self.red_control_slider.setValue(0)
            self.red_control_slider.valueChanged.connect(self.red_changed)
            self.red_control_layout.addWidget(self.red_control_slider)

            # Green
            self.green_control_layout = QtWidgets.QHBoxLayout()
            self.green_control_layout.setSpacing(10)
            self.main_layout.addLayout(self.green_control_layout)

            self.green_control_label = QtWidgets.QLabel(
                "Green: 0 " + " " * (len(str(255)) - 1) + "\t"
            )
            self.green_control_layout.addWidget(self.green_control_label)
            self.green_control_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
            self.green_control_slider.setMinimum(0)
            self.green_control_slider.setMaximum(255)
            self.green_control_slider.setValue(0)
            self.green_control_slider.valueChanged.connect(self.green_changed)
            self.green_control_layout.addWidget(self.green_control_slider)

            # Blue
            self.blue_control_layout = QtWidgets.QHBoxLayout()
            self.blue_control_layout.setSpacing(10)
            self.main_layout.addLayout(self.blue_control_layout)

            self.blue_control_label = QtWidgets.QLabel(
                "Blue: 0   " + " " * (len(str(255)) - 1) + "\t"
            )
            self.blue_control_layout.addWidget(self.blue_control_label)
            self.blue_control_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
            self.blue_control_slider.setMinimum(0)
            self.blue_control_slider.setMaximum(255)
            self.blue_control_slider.setValue(0)
            self.blue_control_slider.valueChanged.connect(self.blue_changed)
            self.blue_control_layout.addWidget(self.blue_control_slider)

    def pen_size_changed(self, value):
        self.size_control_label.setText(
            f"Pen size: {value}" + " " * (len(str(10)) - len(str(value))) + "\t"
        )
        self.update_pen_size(value)

    def gray_changed(self, value):
        self.gray_control_label.setText(
            f"Gray: {value}   " + " " * (len(str(255)) - len(str(value))) + "\t"
        )

        qcolor = QtGui.QColor(value, value, value)
        self.update_pen_color(qcolor)

    def red_changed(self, value):
        self.red_control_label.setText(
            f"Red: {value}    " + " " * (len(str(255)) - len(str(value))) + "\t"
        )

        qcolor = QtGui.QColor(
            value, self.green_control_slider.value(), self.blue_control_slider.value()
        )
        self.update_pen_color(qcolor)

    def green_changed(self, value):
        self.green_control_label.setText(
            f"Green: {value} " + " " * (len(str(255)) - len(str(value))) + "\t"
        )

        qcolor = QtGui.QColor(
            self.red_control_slider.value(), value, self.blue_control_slider.value()
        )
        self.update_pen_color(qcolor)

    def blue_changed(self, value):
        self.blue_control_label.setText(
            f"Blue: {value}   " + " " * (len(str(255)) - len(str(value))) + "\t"
        )

        qcolor = QtGui.QColor(
            self.red_control_slider.value(), self.green_control_slider.value(), value
        )
        self.update_pen_color(qcolor)
