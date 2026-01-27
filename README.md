# FPGA-Accelerated CNN on PYNQ-Z2

This repository contains a **hardware-accelerated 3×3 CNN convolution** implemented on a **Xilinx Zynq-7020 (PYNQ-Z2)** platform.

The design follows a **hardware/software co-design approach**:
- The **CNN convolution kernel** runs in FPGA fabric (PL)
- The **ARM Cortex-A9 processor (PS)** handles control, memory access, and orchestration
- Python (PYNQ framework) is used to control and test the accelerator

This repository provides a **ready-to-use FPGA overlay** and Python runtime code.  
No Vivado or Vitis installation is required on the PYNQ board.

---

## Target Platform

- **Board**: PYNQ-Z2  
- **SoC**: Xilinx Zynq-7020 (xc7z020clg400-1)  
- **FPGA Toolchain**: Vitis HLS + Vivado (used offline to generate bitstream)  
- **Runtime OS**: PYNQ Linux  
- **Interfaces**:
  - AXI-Lite (control)
  - AXI-Full (DDR memory access)

---

## Repository Structure
pynq-cnn-accelerator/
├── overlay/
│ ├── cnn_system_wrapper.bit # FPGA bitstream (overlay)
│ └── cnn_system.hwh # Hardware handoff file
│
├── python/
│ └── test_cnn.py # Python script to run CNN on FPGA
│
└── README.md


---

## What This Design Does

- Implements a **single 3×3 convolution kernel** in FPGA hardware
- Input feature map is stored in DDR memory
- Output feature map is written back to DDR
- The accelerator is controlled via AXI registers
- Python code:
  - Loads the FPGA overlay
  - Allocates buffers
  - Starts the hardware accelerator
  - Reads and prints the output

This design serves as a **base CNN accelerator block** that can be extended to:
- Multiple channels
- Multiple filters
- Full CNN pipelines (Conv → ReLU → Pool)

---

## How to Run on PYNQ-Z2

### 1. Boot the PYNQ-Z2

- Insert SD card with PYNQ image
- Power on the board
- Connect Ethernet
- Access the board via:
  - Jupyter Notebook **or**
  - SSH terminal

---

### 2. Clone this repository on the PYNQ board

git clone https://github.com/Berson-VLSI-Design/pynq-cnn-accelerator.git

cd pynq-cnn-accelerator


---

### 3. Run the CNN accelerator

python3 python/test_cnn.py


The script will:
1. Load the FPGA overlay (`.bit` + `.hwh`)
2. Allocate input/output buffers in DDR
3. Configure accelerator registers
4. Start the CNN hardware
5. Print the output feature map

If the script runs without error, the CNN is executing **on FPGA hardware**.

