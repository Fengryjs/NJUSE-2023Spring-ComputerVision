"""
2.在白纸上手写“我爱南大”四个字，拍照，转为与I1分辨率相同的二值图像（记为I2）。
"""
from PIL import Image
import cv2
i2_init = Image.open("img/i2_init.png")
i1_size = Image.open("img/hw01-I1.jpeg").size
print(i2_init.size)
print(i1_size)
i2_out = i2_init.resize(i1_size, Image.ANTIALIAS)
"""
This can be
           one of :py:data:`PIL.Image.NEAREST`, :py:data:`PIL.Image.BOX`,
           :py:data:`PIL.Image.BILINEAR`, :py:data:`PIL.Image.HAMMING`,
           :py:data:`PIL.Image.BICUBIC` or :py:data:`PIL.Image.LANCZOS`.
"""
i2_binary = i2_out.convert("L").convert("1")
"""
When translating a color image to greyscale (mode "L"),
the library uses the ITU-R 601-2 luma transform::

    L = R * 299/1000 + G * 587/1000 + B * 114/1000
"""
i2_binary.show()
i2_binary.save("./hw01-I2.png")
