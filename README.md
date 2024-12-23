# YOLO with Silicon

This repository contains my work on implementing YOLO (You Only Look Once) object detection on a Silicon MacBook. The project involves training a custom YOLOv8 model, converting it to ONNX format, and running it on a video dataset using MPS (Metal Performance Shaders).

## Background

While working on this project, I encountered challenges due to the hardware and software constraints of using a Silicon MacBook. Torch typically uses CUDA for GPU-based computations, which is only supported by Nvidia GPUs. Since my device uses an Apple Silicon processor, CUDA was not an option. After researching, I discovered MPS, which enables high-performance GPU computations for macOS devices using the Metal framework.

## Workflow

1. **Model Training:**
   - Initially trained a YOLOv5 model, but MPS support was limited to YOLOv8.
   - Retrained a YOLOv8 model using a custom dataset of 1000 images to achieve better accuracy.

2. **ONNX Conversion:**
   - Successfully converted the trained YOLOv8 model to ONNX format for compatibility.

3. **Video Processing:**
   - Used a video from the internet to test the trained model.
   - Overcame a `RuntimeError` caused by incompatible hardware and software configurations.
   - Implemented MPS to enable efficient video processing on the Silicon MacBook.

## Key Features

- **YOLOv8 Support:** Transitioned to YOLOv8 for compatibility with MPS.
- **MPS Integration:** Leveraged Apple’s Metal Performance Shaders for GPU computations.
- **ONNX Format:** Enabled model portability and compatibility.
- **Custom Dataset:** Utilized a dataset of 1000 images for improved results.

## Challenges and Solutions

- **Hardware Compatibility:**
  - Adapted the project for a Silicon MacBook by utilizing Metal programming framework.
