from PySide6 import QtGui, QtCore

import torch
import numpy as np


@torch.no_grad()
def QImage_to_torch(qimg):
    """
    Convert RGB QImage to CxHxW torch array
    """
    qimg = qimg.convertToFormat(QtGui.QImage.Format.Format_RGB32)
    width = qimg.width()
    height = qimg.height()

    ptr = qimg.bits()
    np_img = np.frombuffer(ptr, np.uint8).reshape((height, width, 4))
    torch_img = torch.Tensor(np_img)
    torch_img = torch_img.permute(2, 0, 1)
    torch_img = torch_img[:-1, :, :]
    # Flip channel order to match expected
    torch_img = torch.flip(torch_img, (0,))
    torch_img /= 255
    return torch_img


@torch.no_grad()
def torch_to_QImage(torch_image):
    """
    Convert CxHxW torch array [0..1] to RGB QImage

    N.B: data buffer must remain in scope for lifetime of QImage
    """
    torch_image = torch_image * 255
    torch_image = torch_image.clamp(0, 255)
    torch_image = torch_image.permute(1, 2, 0)
    # Flip channel order to match expected
    torch_image = torch.flip(torch_image, (2,))
    ones = torch.ones(torch_image.size(0), torch_image.size(1), 4) * 255
    ones[:, :, :-1] = torch_image
    numpy_image = np.ascontiguousarray(ones.cpu().numpy().astype(np.uint8))

    height, width, channels = numpy_image.shape

    data = QtCore.QByteArray(numpy_image.data.tobytes())

    qImg = QtGui.QImage(
        data,
        width,
        height,
        QtGui.QImage.Format_RGB32,
    )

    return qImg, data
