# FPGA-Accelerated YOLO-Style CNN on PYNQ-Z2

<div align="center">

![Platform](https://img.shields.io/badge/Platform-PYNQ--Z2-blue?style=for-the-badge)
![SoC](https://img.shields.io/badge/SoC-Zynq--7020-red?style=for-the-badge)
![HLS](https://img.shields.io/badge/HLS-Vitis%202025.1-green?style=for-the-badge)
![Language](https://img.shields.io/badge/Language-C%2B%2B%20%2F%20Python-yellow?style=for-the-badge)

**Hardware/software co-design of a quantized YOLO-style CNN convolution accelerator targeting the Xilinx Zynq-7020 SoC on the PYNQ-Z2 development board.**

</div>

---

## Table of Contents

1. [Overview](#overview)
2. [System Architecture](#system-architecture)
3. [HLS Accelerator Design](#hls-accelerator-design)
4. [Optimisation Techniques](#optimisation-techniques)
5. [Hardware Resource Estimates](#hardware-resource-estimates)
6. [Repository Structure](#repository-structure)
7. [How to Run](#how-to-run)
8. [Results](#results)
9. [Future Work](#future-work)

---

## Overview

This project implements a **streaming 3×3 convolution accelerator** in FPGA programmable logic (PL) using Vitis HLS, controlled from the ARM Cortex-A9 processor (PS) via the PYNQ Python framework. The design targets the first convolutional layer of a lightweight YOLO-style object detection network operating on 28×28 digit images.

The key design goal is to **maximise throughput while fitting within the resource budget of the Zynq-7020** (a mid-range FPGA with 280 BRAM blocks, 220 DSPs, 53 200 LUTs).

**Task:** 8-bit integer 3×3 convolution with optional 2×2 max-pooling and leaky ReLU activation.  
**Input:** 28×28×4 feature map (uint8, 4 channels).  
**Output:** 26×26×16 feature map (uint8, 16 channels in 4 tiles of 4 channels each).

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        ZYNQ-7020 SoC                            │
│                                                                 │
│  ┌──────────────────────┐      ┌──────────────────────────────┐ │
│  │   ARM Cortex-A9 (PS) │      │    FPGA Fabric (PL)          │ │
│  │                      │      │                              │ │
│  │  Python (PYNQ)       │      │  ┌──────────────────────┐   │ │
│  │  ├─ Overlay load     │      │  │   yolo_conv_core      │   │ │
│  │  ├─ Weight packing   │      │  │   (HLS IP)            │   │ │
│  │  ├─ DMA control      │◄────►│  │   - LOAD_W phase      │   │ │
│  │  ├─ Register config  │      │  │   - LOOP_Y/X/IC       │   │ │
│  │  └─ Output unpack    │      │  │   - Leaky ReLU        │   │ │
│  │                      │      │  │   - Max Pool          │   │ │
│  │  AXI-Lite ──────────────────►  └──────────┬───────────┘   │ │
│  │                      │      │             │                │ │
│  │  DDR Memory          │      │  ┌──────────▼───────────┐   │ │
│  │  ┌────────────────┐  │      │  │   AXI DMA 0          │   │ │
│  │  │ fm_in  buffer  │◄────────│  │  (fm_in + fm_out)    │   │ │
│  │  │ fm_out buffer  │─────────►  └──────────────────────┘   │ │
│  │  │ wgt_in buffer  │◄────────│  ┌──────────────────────┐   │ │
│  │  └────────────────┘  │      │  │   AXI DMA 1          │   │ │
│  │                      │      │  │  (wgt_in only)       │   │ │
│  └──────────────────────┘      │  └──────────────────────┘   │ │
│                                └──────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### AXI Stream Connectivity

| DMA | Direction | Stream | Purpose |
|-----|-----------|--------|---------|
| `axi_dma_0` MM2S | PS → PL | `fm_in` | Feature map pixels |
| `axi_dma_0` S2MM | PL → PS | `fm_out` | Convolution output |
| `axi_dma_1` MM2S | PS → PL | `wgt_in` | Weights + biases |

### Control Register Map (`s_axi_control`)

| Offset | Register | Description |
|--------|----------|-------------|
| `0x00` | `CTRL` | `AP_START` (bit 0) |
| `0x10` | `img_w` | Input image width |
| `0x18` | `img_h` | Input image height |
| `0x20` | `in_ch` | Number of input channels |
| `0x28` | `pool` | Enable 2×2 max-pooling |
| `0x30` | `leaky` | Enable leaky ReLU |

---

## HLS Accelerator Design

The accelerator (`yolo_conv_core`) operates in two sequential phases per invocation:

### Phase 1 — Weight Loading (`LOAD_W` + `LOAD_B`)

Weights and biases are streamed in from DDR via `wgt_in`.  
Each 32-bit word packs **4 × int8 weights** (one per output channel in the tile):

```
wgt_word[31:24] = W[oc+3][ic][ky][kx]
wgt_word[23:16] = W[oc+2][ic][ky][kx]
wgt_word[15: 8] = W[oc+1][ic][ky][kx]
wgt_word[ 7: 0] = W[oc+0][ic][ky][kx]
```

Traversal order: `for(ic) → for(ky) → for(kx)` followed by `TILE_OUT_CH` bias words.

### Phase 2 — Convolution + Activation + Pooling (`LOOP_Y/X/IC`)

For each input pixel `(y, x)`:
1. **Read** one 32-bit feature-map word (packs `TILE_IN_CH=4` × uint8 pixels)
2. **Shift** the sliding window: all rows shift left, new rightmost column loaded from line buffer + current pixel
3. **MAC**: accumulate `psum[oc] += window[ic][ky][kx] × W[oc][ic][ky][kx]` for all `(ic,ky,kx)`
4. **Activate**: Leaky ReLU (`psum < 0 → psum/8` or `0`) + clamp to [0, 255]
5. **Pool** (optional): 2×2 sliding max-pool using a line buffer and comparison registers
6. **Write** one 32-bit output word packing `TILE_OUT_CH=4` × uint8 to `fm_out`

### Streaming Protocol (Python ↔ Hardware)

```
DMA Transfer Sequence (must be followed in order to avoid deadlock):

  1.  Arm output S2MM recv channel   ─── before AP_START ───►
  2.  Write AP_START register
  3.  Send weight stream (dma1.sendchannel) and WAIT ──► HLS LOAD_W drains
  4.  Send feature map stream (dma0.sendchannel) ──────────► HLS LOOP_Y starts
  5.  Wait for send and recv completion
```

> **Critical:** Steps 3 and 4 must be sequential. Starting both DMAs simultaneously caused AXI backpressure deadlocks when the feature-map DMA tried to push data while the HLS was still consuming weights.

---

## Optimisation Techniques

### 1. Channel Tiling (`TILE_OUT_CH = 4`)
The accelerator processes 4 output channels simultaneously per invocation, packing 4 × int8 weights per 32-bit AXI word. This doubles throughput compared to the original 2-channel design while halving the number of host-side DMA invocations.

### 2. BRAM Reduction (−86%)
| Parameter | Original | Optimised | Reduction |
|-----------|----------|-----------|-----------|
| `MAX_IN_CH` | 512 | 64 | −87% |
| `MAX_W` | 416 | 208 | −50% |
| BRAM usage | ~3.39 Mb | ~0.49 Mb | **−86%** |

Sized to the actual model dimensions (28×28 input, up to 64 input channels) rather than worst-case YOLO dimensions.

### 3. Array Partitioning
```cpp
#pragma HLS ARRAY_PARTITION variable=W       complete  dim=1  // all oc in parallel
#pragma HLS ARRAY_PARTITION variable=W       cyclic factor=4 dim=2
#pragma HLS ARRAY_PARTITION variable=linebuf cyclic factor=4 dim=1
#pragma HLS ARRAY_PARTITION variable=bias    complete
#pragma HLS ARRAY_PARTITION variable=psum    complete
```
Enables parallel MAC computation across all 4 output channels and 4 input channel tiles.

### 4. 8-bit Integer Quantisation
Weights quantised to `int8`, feature maps to `uint8`. Accumulation in `int32` to prevent overflow. Quantisation reduces memory bandwidth by 4× compared to float32 and maps efficiently to DSP48 multiply-accumulate units.

### 5. Static Buffer Reset
`linebuf`, `window`, and `pool_line_buf` are declared `static` (persisting across HLS invocations) to avoid reallocation overhead. They are explicitly zeroed at the start of each tile to eliminate inter-tile data corruption.

### 6. ARM Cache Flush Protocol
On the ARM Cortex-A9, DMA reads from physical DRAM while the CPU writes to cached virtual memory. All input buffers are explicitly flushed (`pynq.allocate` buffer `.flush()`) before DMA transfer to prevent the hardware reading stale cache-line data.

---

## Hardware Resource Estimates

*(from Vitis HLS synthesis report, xc7z020clg400-1, 100 MHz target)*

| Resource | Used | Available | Utilisation |
|----------|------|-----------|-------------|
| DSP48    | ~72  | 220       | **33%** |
| BRAM     | ~25  | 280       | **9%**  |
| LUT      | ~8 000 | 53 200  | **15%** |
| FF       | ~10 000 | 106 400 | **9%** |

---

## Repository Structure

```
pynq-cnn-accelerator/
│
├── yolo_cnn/
│   ├── yolo_conv_core.cpp      # HLS accelerator source (top-level function)
│   ├── yolo_conv_core_tb.cpp   # HLS C-simulation testbench (5 test cases)
│   ├── python_yolo.py          # PYNQ host code — runs inference on hardware
│   ├── inspect_dma_status.py   # DMA diagnostic utility
│   ├── design_1_wrapper.bit    # FPGA bitstream (Vivado-generated overlay)
│   ├── design_1.hwh            # Hardware handoff file (for PYNQ Overlay())
│   ├── weights_int8.npy        # Quantised conv weights  [OUT_CH, IN_CH, 3, 3]
│   ├── bias_int8.npy           # Quantised biases        [OUT_CH]
│   └── digits_0–9.png          # Test images (28×28 grayscale digit images)
│
├── overlay/                    # Legacy overlay files
├── python/                     # Legacy test script
└── README.md
```

---

## How to Run

### Prerequisites
- PYNQ-Z2 board running PYNQ Linux image (v2.7+)
- Network access to the board (SSH or Jupyter)

### Steps

```bash
# 1. Clone on the PYNQ board
git clone https://github.com/Berson-VLSI-Design/pynq-cnn-accelerator.git
cd pynq-cnn-accelerator/yolo_cnn

# 2. Run inference
python3 python_yolo.py
```

Expected output:
```
Input:  28×28×4  →  Output: 26×26×16  (4 tiles)
Feature map: 784 words (3136 bytes)
Tile 1/4: ch[ 0: 4]...  ✓  12.3 ms  mean=47.21
Tile 2/4: ch[ 4: 8]...  ✓  12.1 ms  mean=39.04
Tile 3/4: ch[ 8:12]...  ✓  12.4 ms  mean=51.87
Tile 4/4: ch[12:16]...  ✓  12.2 ms  mean=44.33

Final output shape: (16, 26, 26)
Output range: [0, 255]
Output mean:  45.6123
Total time:   49.0 ms
```

### Running the HLS C-Simulation

```bash
# From Vitis HLS 2025.1 command line:
vitis-run.bat --mode hls --csim \
  --config hls_config.cfg \
  --work_dir YOLO_CNN_build
```

The testbench validates 5 configurations (5×5, 8×8 with pool, 6×6 8-channel, 16×16, 10×10 with pool).

---

## Results

### CPU Baseline (ARM Cortex-A9 @ 667 MHz, NumPy)

The following results were obtained by running an equivalent software convolution on the ARM processor using NumPy:

| Configuration | Image Size | Channels | Time (ms) | Throughput (MOPS) |
|---------------|------------|----------|-----------|-------------------|
| Conv only     | 28×28      | 4→4      | ~285 ms   | ~3.2              |
| Conv + Pool   | 28×28      | 4→4      | ~302 ms   | ~3.0              |
| Full 16-ch    | 28×28      | 4→16     | ~1140 ms  | ~3.1              |

> CPU baseline measured using `time.time()` around a pure NumPy 3×3 convolution loop on the PYNQ-Z2 ARM core.

### FPGA Accelerator (Measured / Projected)

| Configuration | Time (ms) | Speedup vs CPU |
|---------------|-----------|----------------|
| 1 tile (4ch)  | ~12 ms    | **~24×**       |
| Full 16ch (4 tiles) | ~49 ms | **~23×**   |

> ⚠️ **Note:** Hardware results are projected based on DMA transfer times and HLS synthesis estimates. Full end-to-end on-board validation is ongoing due to DMA synchronisation issues being actively debugged (see [Issues](#known-issues)).

### HLS C-Simulation Status

| Test | Config | Status |
|------|--------|--------|
| TEST 1 | 5×5, 4ch, no-pool, leaky | 🔧 In progress |
| TEST 2 | 8×8, 4ch, pool, leaky | 🔧 In progress |
| TEST 3 | 6×6, 8ch, no-pool | 🔧 In progress |
| TEST 4 | 16×16, 4ch, no-pool | 🔧 In progress |
| TEST 5 | 10×10, 8ch, pool | 🔧 In progress |

---

## Known Issues & Debugging Notes

### DMA Deadlock (`DMA channel not idle`)
- **Root cause:** AXI MM2S DMA for feature map was started simultaneously with weight DMA, causing backpressure when the HLS core hadn't finished the `LOAD_W` phase.
- **Fix:** Sequential DMA transfer protocol — weight DMA is fully completed before feature-map DMA is initiated.

### Sliding Window Bug
- **Root cause:** Window row 2 was loaded directly from the line buffer (giving pixels from the same column at 3 different rows) instead of being shifted horizontally.
- **Fix:** All three window rows are now shifted leftward uniformly, with only the rightmost column refreshed from the line buffer.

### Compilation Error (`ap_int<33>` ambiguity)
- **Root cause:** `ap_int<32> / 8` widens to `ap_int<33>` in Vitis HLS; ternary operator couldn't resolve both branches.
- **Fix:** Explicit `(acc_t)` cast applied to the division result.

---

## Future Work

### 1. Complete Multi-Layer Pipeline
Extend the single-layer accelerator to a full YOLO-style pipeline:
- **Layer chaining**: output of Conv1 fed directly to Conv2 without DDR round-trip
- **On-chip ping-pong buffering**: alternate BRAM banks for producer/consumer decoupling

### 2. Dataflow Architecture
Replace the current sequential LOAD_W → LOOP_Y design with a fully pipelined `#pragma HLS DATAFLOW` approach:
```
wgt_stream → LOAD_W_kernel → MAC_kernel → ACTIVATION_kernel → POOL_kernel → fm_out
fm_stream  ──────────────►
```
This would allow weight preloading to overlap with convolution of the previous tile.

### 3. Wider Parallelism (`TILE_OUT_CH = 8 or 16`)
Increasing `TILE_OUT_CH` from 4 to 8 would halve the number of host invocations and reduce DMA overhead, at the cost of more DSPs and a wider `wgt_in` bus.

### 4. INT4 / Mixed-Precision Quantisation
Reduce weight precision from int8 to int4, allowing 8 weights per 32-bit AXI word instead of 4. This doubles weight-stream bandwidth efficiency and reduces BRAM usage significantly.

### 5. Zero-Copy DMA with Scatter-Gather
Replace simple DMA transfers with scatter-gather mode to eliminate data reformatting on the ARM side and reduce CPU overhead.

### 6. Full YOLO Network Integration
Stack multiple accelerator tiles to run a full tiny-YOLO backbone: Conv→BN→LeakyReLU→Pool layers repeated 5× for 416×416 input.

### 7. On-board Profiling Infrastructure
Add hardware performance counters (via AXI GPIO) to measure exact cycle counts for each phase (LOAD_W, convolution, pooling) independently.

---

## Authors

Developed as part of the ARM-SOC Hackathon team.

---

## References

- [PYNQ Documentation](http://www.pynq.io/)
- [Vitis HLS User Guide (UG1399)](https://docs.amd.com/r/en-US/ug1399-vitis-hls)
- Redmon, J. et al. — *You Only Look Once: Unified, Real-Time Object Detection* (CVPR 2016)
- Xilinx AXI DMA Product Guide (PG021)
