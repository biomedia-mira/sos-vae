from PySide6 import QtCore, QtWidgets, QtGui

import torch

from torchvision.utils import save_image

from interactive.editor_controls import ControlsWidget
from interactive.drawing_image_canvas import DrawingImageCanvas

from corrections.corrections import propagated_correction


class MainWindow(QtWidgets.QMainWindow):
    @torch.no_grad()
    def __init__(
        self, distribution, shape, transform=lambda x: x, inverse_transform=lambda x: x
    ):
        super().__init__()

        self.distribution = distribution
        self.shape = shape
        self.transform = transform
        self.inverse_transform = inverse_transform

        # N.B: squeeze for removing batch dimension
        self.img = (
            self.transform(self.distribution.unflattened_mean(self.shape))
            .squeeze(0)
            .cpu()
            .clamp(min=0, max=1)
        )
        if self.img.size(0) == 1 or self.img.size(0) == 3:
            if self.img.size(0) == 1:
                # Convert grayscale to RGB with repeat
                self.img = self.img.repeat(3, 1, 1)
        else:
            raise RuntimeError("Unsupported number of channels")

        self.mask = torch.zeros_like(self.img, dtype=torch.bool)

        # GUI
        self.setWindowTitle("Interactive Editor")
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

        self.layout = QtWidgets.QGridLayout()
        self.layout.setSpacing(10)
        self.main_widget.setLayout(self.layout)

        self.image_canvas = DrawingImageCanvas(
            (self.img.size(2), self.img.size(1)), self.switch_background
        )
        self.view_menu.addAction(self.image_canvas.action_zoom_in)
        self.view_menu.addAction(self.image_canvas.action_zoom_out)

        self.layout.addWidget(self.image_canvas, 0, 0)

        self.controls = ControlsWidget(
            update_pen_size=lambda value: setattr(self.image_canvas, "pen_size", value),
            update_pen_color=lambda value: setattr(
                self.image_canvas, "pen_color", value
            ),
            grayscale=self.shape[0] == 1,
        )
        self.layout.addWidget(self.controls, 1, 0)

        self.image_canvas.paint_image(self.img)

        self.resize(*self.window_size)

    def closeEvent(self, e):
        self.image_canvas.close()
        e.accept()

    @torch.no_grad()
    def switch_background(self, current_img):

        pixel_channel_mask = ~torch.isclose(current_img, self.img, atol=1 / 255)
        # If any channel is corrected, all channels should be considered corrected
        pixel_mask = pixel_channel_mask.any(dim=0, keepdim=True)
        pixel_mask = pixel_mask.repeat(3, 1, 1)

        old_mask = self.mask.clone()
        self.mask |= pixel_mask

        if not torch.equal(self.mask, old_mask):
            # Handle grayscale case
            if self.shape[0] == 1:
                mask = self.mask[0].unsqueeze(0)
                ground_truth = current_img.mean(dim=0, keepdim=True)
            else:
                mask = self.mask
                ground_truth = current_img

            # Add batch dimension for propagated correction
            mask = mask.unsqueeze(0)
            ground_truth = ground_truth.unsqueeze(0)

            updated_means = propagated_correction(
                self.distribution, self.inverse_transform(ground_truth), mask
            ).cpu()

            new_img = self.transform(updated_means.unflatten(1, self.shape))
            # Remove batch dimension
            new_img = new_img.squeeze(0)

            if self.shape[0] == 1:
                new_img = new_img.repeat(3, 1, 1)

            self.img = new_img.clamp(min=0, max=1)
            self.image_canvas.paint_image(img=self.img)

    def save_current_image(self):
        fd = QtWidgets.QFileDialog()
        fd.setWindowTitle("Save current image")
        fd.setAcceptMode(QtWidgets.QFileDialog.AcceptSave)
        fd.setNameFilter("Images (*.png *.jpg)")
        fd.setDefaultSuffix("png")
        if fd.exec_() == QtWidgets.QFileDialog.Accepted:
            filepath = fd.selectedFiles()[0]
            save_image(self.img, filepath)


def launch(distribution, *args, **kwargs):
    """
    Launch the interactive correction editor

    Parameters:
        distribution (LazyLowRankMultivariateNormalWrapper): distribution with mean to be corrected (only first of batch will be used).
        shape (tuple of int): (CxHxW) tuple of image shape
        transform(function): transformation to be applied to convert network output into final RGB images (e.g. colour space conversion)
        inverse_transform(function): inverse of transform
    """
    if not QtWidgets.QApplication.instance():
        app = QtWidgets.QApplication([])
    else:
        app = QtWidgets.QApplication.instance()

    distribution = distribution[0]
    window = MainWindow(distribution, *args, **kwargs)
    window.show()

    return app.exec_()
