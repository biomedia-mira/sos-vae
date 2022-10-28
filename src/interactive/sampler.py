from PySide6 import QtCore, QtWidgets, QtGui

import torch

import matplotlib.pyplot as plt

from torchvision.utils import save_image

from post.post import modify_dist

from visualiser.visualiser import (
    visualise_observation_space,
    visualise_sample_variation,
    visualise_scaled_pca_components,
)

from interactive.editor import launch as launch_editor

from interactive.sampler_controls import ControlsWidget
from interactive.image_canvas import ImageCanvas


class MainWindow(QtWidgets.QMainWindow):
    @torch.no_grad()
    def __init__(
        self,
        distribution,
        rank,
        shape,
        transform=lambda x: x,
        inverse_transform=lambda x: x,
    ):
        super().__init__()

        self.distribution = distribution
        self.rank = rank
        self.shape = shape
        self.transform = transform
        self.inverse_transform = inverse_transform

        self.temp_scales = torch.zeros(self.rank)
        self.omega_p = torch.randn(1, self.rank)

        # GUI
        self.setWindowTitle("Interactive Sampler")
        self.window_size = (600, 600)
        self.main_widget = QtWidgets.QWidget()
        self.setCentralWidget(self.main_widget)

        # Menu bar
        self.file_menu = self.menuBar().addMenu("File")
        self.view_menu = self.menuBar().addMenu("View")

        self.save_action = QtGui.QAction("Save current image", self)
        self.save_action.setShortcut(QtGui.QKeySequence.Save)
        self.save_action.triggered.connect(self.save_current_image)
        self.file_menu.addAction(self.save_action)

        self.editor_launch_action = QtGui.QAction(
            "Launch editor with current distribution", self
        )
        self.editor_launch_action.triggered.connect(
            lambda: launch_editor(
                self.modified_dist(),
                self.shape,
                transform=self.transform,
                inverse_transform=self.inverse_transform,
            )
        )
        self.file_menu.addAction(self.editor_launch_action)

        self.view_samples_action = QtGui.QAction("View multiple samples", self)
        self.view_samples_action.triggered.connect(self.view_multiple_samples)
        self.view_menu.addAction(self.view_samples_action)

        self.view_os_action = QtGui.QAction("View observation space slice", self)
        self.view_os_action.triggered.connect(self.view_observation_space)
        self.view_menu.addAction(self.view_os_action)

        self.view_scaled_pca_action = QtGui.QAction(
            "View individually scaled components", self
        )
        self.view_scaled_pca_action.triggered.connect(self.view_scaled_pca)
        self.view_menu.addAction(self.view_scaled_pca_action)

        # Status bar
        self.status_label = QtWidgets.QLabel()
        self.statusBar().addPermanentWidget(self.status_label)

        self.layout = QtWidgets.QGridLayout()
        self.layout.setSpacing(10)
        self.main_widget.setLayout(self.layout)

        self.image_canvas = ImageCanvas((self.shape[2], self.shape[1]))
        self.image_canvas.setSizePolicy(
            QtWidgets.QSizePolicy.MinimumExpanding,
            QtWidgets.QSizePolicy.MinimumExpanding,
        )
        self.layout.addWidget(self.image_canvas, 0, 0)

        self.view_menu.addAction(self.image_canvas.action_zoom_in)
        self.view_menu.addAction(self.image_canvas.action_zoom_out)

        self.controls = ControlsWidget(
            num_sliders=self.rank,
            on_slider_update=self.slider_change,
            on_new_image_sample=lambda: self.image_canvas.paint_image(
                self.new_image_sample()
            ),
        )
        self.controls.setSizePolicy(
            QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Maximum
        )
        self.layout.addWidget(self.controls, 1, 0)

        self.image_canvas.paint_image(self.new_image_sample())

        self.resize(*self.window_size)

    def modified_dist(self):
        return modify_dist(self.distribution, scales=self.temp_scales, pca=True)

    def closeEvent(self, e):
        self.image_canvas.close()
        e.accept()

    def new_image_sample(self, keep_state=False):
        assert self.distribution is not None

        if not keep_state:
            self.omega_p = torch.randn(1, self.rank)

        dist = self.modified_dist()

        # N.B: squeeze for removing batch dimension
        sample = dist.sample(omega_p=self.omega_p)
        sample = self.transform(sample.unflatten(1, self.shape).cpu()).squeeze(0)

        if sample.size(0) == 1 or sample.size(0) == 3:
            if sample.size(0) == 1:
                # Convert grayscale to RGB with repeat
                sample = sample.repeat(3, 1, 1)
        else:
            raise RuntimeError("Unsupported number of channels")

        return sample.clamp(min=0, max=1)

    def slider_change(self, scales):
        self.temp_scales = torch.tensor(scales)

        try:
            sample = self.new_image_sample(keep_state=True)
        except RuntimeError:
            # Cholesky crash
            self.status_label.setText("Distribition construction failed")
            return

        self.status_label.setText("")
        self.image_canvas.paint_image(sample)

    def view_multiple_samples(self):
        dist = self.modified_dist()
        visualise_sample_variation(
            dist, 1, 10, self.shape, cols_per_batch_item=10, transform=self.transform
        )
        plt.show()

    def view_observation_space(self):
        dist = self.modified_dist()
        visualise_observation_space(dist, 8, self.shape, transform=self.transform)
        plt.show()

    def view_scaled_pca(self):
        visualise_scaled_pca_components(
            self.distribution, self.shape, omega_p=self.omega_p
        )
        plt.show()

    def save_current_image(self):
        fd = QtWidgets.QFileDialog()
        fd.setWindowTitle("Save current image")
        fd.setAcceptMode(QtWidgets.QFileDialog.AcceptSave)
        fd.setNameFilter("Images (*.png *.jpg)")
        fd.setDefaultSuffix("png")
        if fd.exec_() == QtWidgets.QFileDialog.Accepted:
            filepath = fd.selectedFiles()[0]
            img = self.new_image_sample(keep_state=True)
            save_image(img, filepath)


def launch(distribution, *args, **kwargs):
    """
    Launch the interactive correction editor

    Parameters:
        distribution (LazyLowRankMultivariateNormalWrapper) distribution to be sampled
        rank (int) rank of the distribution
        shape (tuple of int): (CxHxW) tuple of image shape
        transform(function): transformation to be applied to convert network output into final RGB images (e.g. colour space conversion)
    """
    if not QtWidgets.QApplication.instance():
        app = QtWidgets.QApplication([])
    else:
        app = QtWidgets.QApplication.instance()

    window = MainWindow(distribution[0], *args, **kwargs)
    window.show()

    return app.exec_()
