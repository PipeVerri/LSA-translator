# Landmark Extraction Speed Benchmark

## Goal

MediaPipe offers two configurations for extracting the pose and hand landmarks used as model input:

1. **Holistic model** — a single unified model that processes pose, hands, and face simultaneously. The downside is that it computes facial landmarks that are not used in this project.
2. **Pose + Hands models** — two separate models run in combination. The advantage is that no facial landmarks are computed; the disadvantage is the overhead of running two separate inference passes.

This experiment benchmarks both configurations to determine which to use in the main pipeline.

## Results

| Method | Time per video |
|--------|---------------|
| Holistic | ~5.33 s |
| Pose + Hands (sequential) | ~6.51 s |
| Pose + Hands (multithreaded) | ~3.74 s |

Running Pose + Hands sequentially takes approximately twice as long as Holistic. Running them in parallel threads reduces this to roughly the same order as Holistic, but still slightly slower.

## Decision

**Holistic was selected** for the main pipeline. Despite the marginal speed disadvantage over multithreaded Pose+Hands, Holistic produces a consistently lower missing-landmark rate. For short training sequences (LSA64 clips range from ~20 to ~80 frames), missing landmarks have a disproportionate impact on sequence quality, making landmark reliability a higher priority than raw throughput.
