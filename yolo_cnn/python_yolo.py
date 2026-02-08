from pynq import Overlay, allocate
import numpy as np
import cv2
import time

# ----------------------------
# Configuration (must match HLS)
# ----------------------------
IMG_H = 28
IMG_W = 28
IN_CH = 1
OUT_CH = 16
K = 3

OUT_H = IMG_H - 2
OUT_W = IMG_W - 2

# ----------------------------
# Load overlay
# ----------------------------
ol = Overlay("design_1.bit")
ip = ol.yolo_conv_core_0   # adjust name if needed

dma_fm = ol.axi_dma_fm     # feature map DMA
dma_wgt = ol.axi_dma_wgt   # weight DMA
dma_out = ol.axi_dma_out   # output DMA

# ----------------------------
# Load weights
# ----------------------------
W = np.load("weights_int8.npy").astype(np.int8)
B = np.load("bias_int8.npy").astype(np.int8)
scale = np.load("weight_scale.npy").item()

# Flatten weights in streaming order
w_stream = []

for oc in range(OUT_CH):
    for ic in range(IN_CH):
        for ky in range(K):
            for kx in range(K):
                w_stream.append(W[oc, ic, ky, kx])

# Append biases at end
for oc in range(OUT_CH):
    w_stream.append(B[oc])

w_stream = np.array(w_stream, dtype=np.int8)

# Allocate weight buffer
wgt_buf = allocate(shape=w_stream.shape, dtype=np.int8)
wgt_buf[:] = w_stream[:]

# ----------------------------
# Prepare input image
# ----------------------------
img = cv2.imread("digit.png", cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (IMG_W, IMG_H))

fm_stream = img.flatten().astype(np.uint8)

fm_buf = allocate(shape=fm_stream.shape, dtype=np.uint8)
fm_buf[:] = fm_stream[:]

# ----------------------------
# Output buffer
# ----------------------------
out_size = OUT_CH * OUT_H * OUT_W
out_buf = allocate(shape=(out_size,), dtype=np.int16)

# ----------------------------
# Configure HLS core
# ----------------------------
ip.write(0x10, IMG_W)
ip.write(0x18, IMG_H)
ip.write(0x20, IN_CH)
ip.write(0x28, 1)   # leaky = true
ip.write(0x30, 0)   # pooling disabled

# ----------------------------
# Start DMA transfers
# ----------------------------
dma_out.recvchannel.transfer(out_buf)
dma_wgt.sendchannel.transfer(wgt_buf)
dma_fm.sendchannel.transfer(fm_buf)

# Start accelerator
ip.write(0x00, 1)

# Wait for completion
dma_fm.sendchannel.wait()
dma_wgt.sendchannel.wait()
dma_out.recvchannel.wait()

# ----------------------------
# Post-processing
# ----------------------------
out = np.array(out_buf).reshape(OUT_CH, OUT_H, OUT_W)
out_real = out * scale

print("FPGA output mean:", out_real.mean())
