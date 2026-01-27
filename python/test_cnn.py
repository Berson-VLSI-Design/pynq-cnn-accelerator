from pynq import Overlay, allocate
import numpy as np
import time

# Load overlay
ol = Overlay("overlay/cnn_system_wrapper.bit")
cnn = ol.cnn_conv3x3_0

H = 5
W = 5

inp = allocate(shape=(H*W,), dtype=np.int8)
out = allocate(shape=((H-2)*(W-2),), dtype=np.int8)

inp[:] = np.arange(H*W, dtype=np.int8)

cnn.write(0x10, inp.physical_address)
cnn.write(0x18, out.physical_address)
cnn.write(0x20, H)
cnn.write(0x28, W)

cnn.write(0x00, 1)  # start
time.sleep(0.01)

print("Output:", out)

