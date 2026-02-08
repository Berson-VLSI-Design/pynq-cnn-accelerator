from pynq import Overlay
from pynq import allocate
import numpy as np
import time

overlay = Overlay("design_1.bit")
overlay.download()

print("Overlay loaded")
