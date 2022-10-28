from PySide6 import QtCore, QtWidgets, QtGui


class ControlsWidget(QtWidgets.QWidget):
    def __init__(
        self,
        num_sliders,
        on_slider_update=None,
        on_new_image_sample=None,
    ):
        super().__init__()

        self.num_sliders = num_sliders
        self.on_slider_update = on_slider_update
        self.on_new_image_sample = on_new_image_sample

        self.sliders = [None] * self.num_sliders
        self.labels = [None] * self.num_sliders
        self.slider_values = [0.0] * self.num_sliders

        self.main_layout = QtWidgets.QGridLayout()
        self.main_layout.setSpacing(10)
        self.setLayout(self.main_layout)

        # Buttons
        self.new_image_sample_button = QtWidgets.QPushButton("New image sample")
        self.new_image_sample_button.clicked.connect(self.on_new_image_sample)
        self.main_layout.addWidget(self.new_image_sample_button, 0, 0)

        # Sliders
        self.sliders_scroll_area = QtWidgets.QScrollArea()
        self.sliders_scroll_area.setWidgetResizable(True)
        self.sliders_scroll_area.setMaximumHeight(200)
        self.sliders_scroll_area.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)
        self.sliders_scroll_area.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarAlwaysOff
        )
        self.main_layout.addWidget(self.sliders_scroll_area, 1, 0)

        self.sliders_widget = QtWidgets.QWidget(self)

        self.sliders_layout = QtWidgets.QVBoxLayout()
        self.sliders_layout.setSpacing(10)
        self.sliders_widget.setLayout(self.sliders_layout)

        for i in range(self.num_sliders):
            slider_layout = QtWidgets.QHBoxLayout()
            slider_layout.setSpacing(10)
            self.sliders_layout.addLayout(slider_layout)

            label = QtWidgets.QLabel(f"{i+1}: 0.00\t")
            slider_layout.addWidget(label)

            slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
            slider.setMinimum(-50)
            slider.setMaximum(50)
            slider.setTickInterval(50)
            slider.setTickPosition(QtWidgets.QSlider.TicksBelow)
            slider.setValue(self.slider_values[i])
            slider.valueChanged.connect(
                lambda value, index=i: self.update_slider(index, value / 10)
            )
            slider.wheelEvent = lambda e: e.ignore()
            slider_layout.addWidget(slider)

            self.sliders[i] = slider
            self.labels[i] = label

        self.reset_button = QtWidgets.QPushButton("Reset all")
        self.reset_button.clicked.connect(self.reset_sliders)
        self.sliders_layout.addWidget(self.reset_button)

        self.sliders_scroll_area.setWidget(self.sliders_widget)

    def update_slider(self, index, value):
        self.slider_values[index] = value
        self.update_labels()

        self.on_slider_update(self.slider_values)

    def update_labels(self):
        for i, label in enumerate(self.labels):
            label.setText(f"{i+1}: {self.slider_values[i]:.2f}\t")

    def reset_sliders(self):
        self.slider_values = [0.0] * self.num_sliders

        for slider in self.sliders:
            slider.setValue(0)

        self.update_labels()

        self.on_slider_update(self.slider_values)
