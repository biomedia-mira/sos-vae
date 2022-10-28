import warnings

from PySide6 import QtCore, QtWidgets

import torch

from model.vae import VAE
from model.conditional_vae import ConditionalVAE

from interactive.sampler import MainWindow as Sampler
from interactive.image_canvas import ImageCanvas

from post.lerp import slerp


class MainWindow(Sampler):
    @torch.no_grad()
    def __init__(
        self,
        model,
        latent_prior,
        rank,
        shape,
        label_value_map=None,
        transform=lambda x: x,
        inverse_transform=lambda x: x,
    ):
        if not isinstance(model, VAE):
            raise RuntimeError("VAE Sampler can only be used with VAE models")

        if label_value_map is not None and not isinstance(model, ConditionalVAE):
            warnings.warn(
                "label_value_map provided and non-conditional VAE model provided. Labels will be ignored"
            )
        if label_value_map is None and isinstance(model, ConditionalVAE):
            raise RuntimeError(
                "Condiional model passed, but no label_value_map provided"
            )

        self.model = model
        self.latent_prior = latent_prior
        self.label_value_map = label_value_map

        self.interpolate_t = 0.0
        self.left_sample = self.latent_prior.sample((1,))
        self.right_sample = self.latent_prior.sample((1,))

        if isinstance(self.model, ConditionalVAE):
            self.cond_label = torch.zeros(
                1, dtype=int, device=self.latent_prior.loc.device
            )
            dist = self.model.decode(self.left_sample, self.cond_label)
        else:
            dist = self.model.decode(self.left_sample)

        super().__init__(
            dist, rank, shape, transform=transform, inverse_transform=inverse_transform
        )
        self.setWindowTitle("Interactive VAE Sampler")

        # Remove old widgets for replacing
        self.layout.removeWidget(self.image_canvas)
        self.layout.removeWidget(self.controls)

        # Set up new widgets and reinstate old widgets
        self.left_image_canvas = ImageCanvas((self.shape[2], self.shape[1]))
        self.left_image_canvas.setSizePolicy(
            QtWidgets.QSizePolicy.MinimumExpanding,
            QtWidgets.QSizePolicy.MinimumExpanding,
        )
        self.layout.addWidget(self.left_image_canvas, 0, 0)

        self.layout.addWidget(self.image_canvas, 0, 1)

        self.right_image_canvas = ImageCanvas((self.shape[2], self.shape[1]))
        self.right_image_canvas.setSizePolicy(
            QtWidgets.QSizePolicy.MinimumExpanding,
            QtWidgets.QSizePolicy.MinimumExpanding,
        )
        self.layout.addWidget(self.right_image_canvas, 0, 2)

        self.layout.addWidget(self.controls, 1, 1)

        self.new_left_button = QtWidgets.QPushButton("New latent sample")
        self.new_left_button.clicked.connect(lambda: self.new_latent("left"))
        self.layout.addWidget(self.new_left_button, 1, 0)

        self.new_right_button = QtWidgets.QPushButton("New latent sample")
        self.new_right_button.clicked.connect(lambda: self.new_latent("right"))
        self.layout.addWidget(self.new_right_button, 1, 2)

        self.latent_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.latent_slider.setMinimum(0)
        self.latent_slider.setMaximum(100)
        self.latent_slider.setTickInterval(50)
        self.latent_slider.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.latent_slider.setValue(0)
        self.latent_slider.valueChanged.connect(self.latent_slider_update)
        self.layout.addWidget(self.latent_slider, 3, 0, 1, 3)

        # Add label buttons
        if isinstance(self.model, ConditionalVAE):
            self.label_button_layout = QtWidgets.QHBoxLayout()
            self.label_button_layout.setSpacing(10)
            self.layout.addLayout(self.label_button_layout, 4, 0, 1, 3)

            for label in self.label_value_map:
                button = QtWidgets.QPushButton(label)
                label_value = self.label_value_map[label]
                button.clicked[bool].connect(
                    lambda state, x=label_value: self.label_update(x)
                )
                self.label_button_layout.addWidget(button)

        # Connect zoom actions
        self.image_canvas.action_zoom_in.triggered.connect(
            self.left_image_canvas.action_zoom_in.trigger
        )
        self.image_canvas.action_zoom_in.triggered.connect(
            self.right_image_canvas.action_zoom_in.trigger
        )

        self.image_canvas.action_zoom_out.triggered.connect(
            self.left_image_canvas.action_zoom_out.trigger
        )
        self.image_canvas.action_zoom_out.triggered.connect(
            self.right_image_canvas.action_zoom_out.trigger
        )

        self.resize(1000, 600)
        self.update_all()

    def closeEvent(self, e):
        super().closeEvent(e)
        self.left_image_canvas.close()
        self.right_image_canvas.close()
        e.accept()

    @torch.no_grad()
    def update_all(self, keep_state=False):
        if isinstance(self.model, ConditionalVAE):
            left_dist = self.model.decode(self.left_sample, self.cond_label)
            right_dist = self.model.decode(self.right_sample, self.cond_label)
        else:
            left_dist = self.model.decode(self.left_sample)
            right_dist = self.model.decode(self.right_sample)

        self.left_image = self.transform(
            left_dist.unflattened_mean(self.shape).cpu()
        ).squeeze(0)
        self.right_image = self.transform(
            right_dist.unflattened_mean(self.shape).cpu()
        ).squeeze(0)

        self.left_image_canvas.paint_image(self.left_image)
        self.right_image_canvas.paint_image(self.right_image)

        inter_latent = slerp(self.left_sample, self.right_sample, self.interpolate_t)

        if isinstance(self.model, ConditionalVAE):
            self.distribution = self.model.decode(inter_latent, self.cond_label)
        else:
            self.distribution = self.model.decode(inter_latent)

        self.image_canvas.paint_image(self.new_image_sample(keep_state))

    def new_latent(self, side):
        if side == "left":
            self.left_sample = self.latent_prior.sample((1,))
        elif side == "right":
            self.right_sample = self.latent_prior.sample((1,))
        else:
            raise RuntimeError(f"Unknown side: {side}")

        self.update_all(keep_state=True)

    def latent_slider_update(self, value):
        self.interpolate_t = value / 100
        self.update_all(keep_state=True)

    def label_update(self, label_value):
        self.cond_label = torch.tensor(
            [label_value], dtype=int, device=self.latent_prior.loc.device
        )
        self.update_all(keep_state=True)


def launch(*args, **kwargs):
    """
    Launch the interactive correction editor

    Parameters:
        model (VAE)
        latent_prior (td.distribution) latent prior for samples to be generated from
        rank (int) rank of the distribution
        shape (tuple of int): (CxHxW) tuple of image shape
    """
    if not QtWidgets.QApplication.instance():
        app = QtWidgets.QApplication([])
    else:
        app = QtWidgets.QApplication.instance()

    window = MainWindow(*args, **kwargs)
    window.show()

    return app.exec_()
