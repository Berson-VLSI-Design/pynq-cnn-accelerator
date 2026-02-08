from pynq import Overlay
from pynq import allocate
import numpy as np
import time

overlay = Overlay("yolo_conv_core.bit")
overlay.download()

print("Overlay loaded")
