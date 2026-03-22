# MMPose Evaluation (Discarded)

## Goal

Evaluate MMPose as an alternative to MediaPipe for landmark estimation on LSA64 videos, and compare detection quality and throughput.

## Status

**Discarded.** MMPose ran at less than 1 FPS on a Quadro T1000, making it infeasible for both real-time inference and large-scale offline preprocessing of the training set.

## How to Run

```bash
conda env create -f environment.yml
conda activate mmpose-env
mim install mmengine
mim install "mmcv>=2.0.1"
mim install "mmdet>=3.1.0"
jupyter notebook
```
