# LSA Sign Language Translator — ML Research Project

> **Status:** Phase 1 complete — resuming in ~2 months with a transformer-based architecture on LSA-T.

This repository documents an independent research project exploring the feasibility of **real-time Argentine Sign Language (LSA) recognition and translation** using deep learning. It covers the full pipeline — from raw video to text output — with a focus on understanding which representations, architectures, and training strategies are best suited to the constraints of a small, domain-specific sign language dataset.

The project is organized as a series of documented experiments, including dead ends and negative results, with analysis at each step.

---

## Background

**Argentine Sign Language (LSA)** is the primary language of the Deaf community in Argentina. Like all sign languages, LSA encodes meaning through hand shape, movement, spatial position, and facial expression — a high-dimensional, temporally structured signal.

**Sign Language Translation (SLT)** is the task of converting a sign language video into written or spoken natural language. It is technically distinct from, and harder than, **Sign Language Recognition (SLR)**, which classifies isolated, pre-segmented signs. Full translation requires:

1. **Temporal segmentation** — identifying where individual signs begin and end in a continuous stream.
2. **Sign recognition** — mapping each segmented clip to a gloss.
3. **Translation** — converting the gloss sequence into grammatically correct natural language.

Each step is a non-trivial research problem. This project focuses on steps 1 and 2, with exploratory analysis toward step 3.

---

## Datasets

### LSA64

LSA64 is an isolated sign language recognition dataset containing **64 LSA signs** recorded by **10 signers** (5 repetitions each, totaling 3,200 videos). Signs include 42 one-handed and 22 two-handed signs. Subjects wore colored gloves to facilitate hand tracking.

> [Paper](https://arxiv.org/abs/2310.17429) · [Project page](https://facundoq.github.io/datasets/lsa64/) · [GitHub](https://github.com/midusi/lsa64_nn)

---

## Approach

### Landmark Extraction

Raw video frames are converted into sequences of **normalized skeleton keypoints** using MediaPipe Holistic. Each frame produces a 144-dimensional feature vector representing upper-body and hand landmarks in a wrist-centric, rotation-normalized coordinate space.

This representation is:
- Compact and computationally cheap compared to raw pixel processing.
- Invariant to signer position and camera distance after normalization.
- Anatomically faithful for hand shape and motion, which carry the bulk of linguistic content in LSA.

> See [`src/lm_processing/landmarks.py`](src/lm_processing/landmarks.py) for normalization and coordinate transform details.

![](/docs/mediapipe_landmarks.png)

### Architecture — Phase 1

A **Recurrent Neural Network (RNN)** processes variable-length landmark sequences and outputs a distribution over 64 sign classes plus a background "no-sign" class.

| Component | Configuration |
|-----------|--------------|
| Input | 144-dim landmark vector per frame |
| Encoder | SimpleRNN (hidden\_dim=144, layers=5–7) |
| Output | Softmax over 65 classes (64 signs + no-sign) |
| Loss function | Cross-entropy |
| Optimizer | Adam (lr=1e-4, weight\_decay=1e-4) |
| LR scheduler | ReduceLROnPlateau (factor=0.5, patience=3) |
| Training framework | PyTorch Lightning |
| Experiment tracking | Weights & Biases |

The real-time inference pipeline uses a **sliding window** (2 s at 12 FPS, stride 0.1 s) over a live landmark stream, filtering predictions by a softmax confidence threshold (`no_sign_prob < 0.3`).

> See [`src/models/`](src/models/) and [`src/scripts/real_time_prediction.py`](src/scripts/real_time_prediction.py).

![](/docs/experiment1-diagram.png)

---

## Experiments

Full experiment documentation with state-of-the-art context: [`experiments/Docs/Experiments.md`](experiments/Docs/Experiments.md)

### Experiment 1 — Baseline RNN on LSA64 (Isolated Signs)

A vanilla RNN trained on pre-segmented LSA64 clips to establish an isolated-sign accuracy baseline and assess the model's capacity for sign representation.

| Metric | Value |
|--------|-------|
| Training accuracy | ~100% |
| Validation accuracy | ~98.27% |
| Real-world (continuous signing) | Fails — segmentation bottleneck |

**Key finding:** Near-perfect accuracy on isolated signs, but the model fails entirely when applied to continuous signing. The bottleneck is temporal segmentation, not sign recognition.

<!-- TODO: Add training loss/accuracy curves (docs/figures/exp1_training_curves.png) -->
<!-- TODO: Add confusion matrix on isolated test set (docs/figures/exp1_confusion_matrix.png) -->

### Experiment 2 — Two-Stage Detection

A two-stage architecture attempting to improve real-world robustness:
1. **Stage 1 (Hand Dominance Detector):** classifies which hand is dominant for the current sign.
2. **Stage 2 (Sign Classifier):** applies a hand-specific RNN classifier.

| Metric | Value |
|--------|-------|
| Validation accuracy | <!-- TODO: fill in --> |
| Improvement over Experiment 1 | <!-- TODO: fill in --> |

**Key finding:** The isolated-to-continuous gap persists regardless of classifier architecture. Temporal segmentation remains the limiting factor, pointing to a fundamental dataset problem rather than a model capacity problem.

<!-- TODO: Add softmax confidence distribution plot (docs/figures/exp2_confidence_distribution.png) -->
<!-- TODO: Add per-class accuracy bar chart (docs/figures/exp2_per_class_accuracy.png) -->

---

## Exploratory Work

The following experiments were conducted or partially implemented to probe specific sub-problems. They are not complete, but each produced findings that shaped the direction of the project.

### Landmark Extraction — MediaPipe Holistic vs. Pose + Hands

Evaluated two MediaPipe configurations for landmark extraction speed and quality.

| Method | Time per video |
|--------|---------------|
| Holistic | ~5.33 s |
| Pose + Hands (sequential) | ~6.51 s |
| Pose + Hands (multithreaded) | ~3.74 s |

**Finding:** Holistic was selected for the main pipeline. While multithreaded Pose+Hands is slightly faster, Holistic produces fewer missing landmarks, which is critical for short training sequences.

> See [`experiments/landmarks/01_processing_speed/`](experiments/landmarks/01_processing_speed/).

### MMPose Evaluation (Discarded)

MMPose was evaluated as an alternative to MediaPipe for landmark estimation. Discarded due to insufficient speed: less than 1 FPS on a Quadro T1000, making it infeasible for real-time or large-scale preprocessing.

> See [`experiments/landmarks/02_mmpose_test/`](experiments/landmarks/02_mmpose_test/).

### Acceleration-Based Segmentation (Discarded)

**Goal:** Use hand acceleration peaks as a heuristic to segment continuous signing into individual glosses.

**Finding:** Not viable due to **coarticulation** — in continuous LSA, there are no consistent motion pauses between signs. The final pose of one sign blends directly into the initial pose of the next, producing no discriminative acceleration discontinuity. Acceleration-based features may still be useful as *inputs to a learned segmentation model*, but not as a standalone threshold rule.

> See [`experiments/sign_analysis/01_hand_acceleration/`](experiments/sign_analysis/01_hand_acceleration/).

### Kinematic Noise Augmentation (Proposed)

**Motivation:** Small dataset size (~50 samples per sign) is a core constraint. This method proposes generating plausible synthetic variants through kinematic-constrained noise injection.

**Method:**
1. Decompose each sign into primitive directional motion vectors (e.g., "right hand: +10 cm vertical, +1 cm horizontal").
2. Inject noise proportional to the original direction: **v** + **v** ∘ **x**, where **x** ~ N(0, σ²I).
3. Propagate the perturbed motion to dependent landmarks via a forward kinematic chain.

This ensures that noise preserves anatomical plausibility: a predominantly upward motion perturbed slightly remains predominantly upward, rather than being deflected into an arbitrary direction.

> See [`experiments/data_augmentation/01_kinematic_augmentation/`](experiments/data_augmentation/01_kinematic_augmentation/).

### PCA-Based Augmentation (Proposed)

**Motivation:** Learn universal inter-signer variation directions that can be applied as data augmentation across all signs.

**Method:**
1. For each sign gloss, stack all signer samples into a matrix **X**.
2. Compute PCA; extract the principal variation direction **v₁** (eigenvector of the largest eigenvalue).
3. Normalize **v₁** to capture direction only, discarding scale.
4. Visualize pairwise similarity between sign variation directions via a heatmap.

The hypothesis: if multiple glosses share a dominant variation direction (e.g., "arm consistently goes slightly higher or lower across signers"), that direction generalizes across the lexicon and can be used as a sign-agnostic augmentation axis.

> See [`experiments/data_augmentation/02_PCA_augmentation/`](experiments/data_augmentation/02_PCA_augmentation/).

---

## Tech Stack

| Layer | Library / Tool |
|-------|---------------|
| Model definition | PyTorch |
| Training loop | PyTorch Lightning |
| Landmark extraction | MediaPipe |
| Experiment tracking | Weights & Biases |
| Numerical computing | NumPy · Pandas |
| Visualization | Matplotlib |
| Text-to-speech output | gTTS · pydub |
| Video acquisition | yt-dlp · OpenCV |

---

## Project Status

**Phase 1 is complete.** The project is currently paused and will resume in approximately 2 months.

**Phase 2** will transition from isolated sign recognition on LSA64 to continuous Sign Language Translation on **LSA-T**, using a transformer-based, semi-gloss-free architecture. The design will be based on:

> <!-- TODO: Add LSA-T translation paper link here -->

The core motivation for this transition: the isolated-to-continuous gap cannot be closed by improving the RNN or the segmentation heuristic — it requires an end-to-end sequence-to-sequence architecture that models the full signing stream without assuming pre-segmented inputs.

---

## How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Extract landmarks from LSA64 videos (configure your data path in the script)
python src/scripts/LSA64/lsa64_video_class.py

# Train the RNN
python src/train.py

# Run real-time inference (requires webcam)
python src/scripts/real_time_prediction.py

# Visualize processed landmarks
python src/scripts/visualize_landmarks_processing.py
```

---

## Attribution

Training data from [LSA64](https://facundoq.github.io/datasets/lsa64/) by MIDUSI, Instituto de Investigación en Informática LIDI, Universidad Nacional de La Plata, Argentina.
