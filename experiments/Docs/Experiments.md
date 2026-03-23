# How Feasible Is Real-Time LSA-to-Spanish Translation?

This document tracks the reasoning, experiments, results, and failures encountered throughout the project — including what each failure revealed and how it shaped the next step.

---

## Prior Work and State of the Art

### First-Generation Approach: RNN + Heuristic Segmentation

The earliest SLT systems followed a two-step pipeline:

1. **Train an RNN to classify individual glosses** from pre-segmented video clips.
2. **Segment the continuous stream heuristically** using acceleration peaks, fixed-size windows, or motion energy thresholds to decide when a sign has occurred.
3. Feed each detected segment to the RNN for classification.

This approach has been largely abandoned because heuristic segmentation **fails in practice**. The two main reasons are:

- **Coarticulation:** Signs do not occur as discrete, isolated units in continuous signing. If a signer produces "me" (pointing the index finger toward oneself) followed by "go" (pointing forward), the arm does not return to a rest position between them — it transitions directly from one configuration to the next. In isolation, each sign has a clear trajectory; in context, that trajectory is modified by adjacent signs. Acceleration-based methods cannot reliably detect these boundaries.
- **Speed variability:** The same sign may be executed at significantly different speeds depending on context, signer, and sentence structure. Fixed-window and threshold-based approaches have no way to adapt to this variability.

### Second Generation: Gloss-Free End-to-End Models

Gloss-free models address segmentation and translation simultaneously within a single transformer:

- A single network learns sign segmentation, gloss interpretation, and text generation jointly.
- They perform well with large datasets, but **degrade significantly on small datasets** like LSA-T, because they are being asked to learn three separate tasks simultaneously from a limited training signal.

### Current State of the Art: Semi-Gloss-Free Models

The most recent direction relaxes the gloss-free constraint while reducing dependence on dense gloss annotations:

- Sign recognition and text generation are still handled by a unified model, but the architecture uses **large language models (LLMs)** to handle the linguistic component.
- This allows the visual/temporal backbone to focus on motion representation, while the LLM handles text generation with strong language priors.
- Performance is better than pure gloss-free models on small datasets.

**This is the direction for Phase 2 of this project.**

---

## Experiment Summary

### Experiment 1 — Baseline RNN on LSA64 (Isolated Signs)

**Goal:** Train a vanilla RNN on pre-segmented LSA64 clips to establish a baseline and validate the landmark representation.

**Setup:**
- Dataset: LSA64, 2,548 training / 638 validation samples (80/20 split)
- Input: 144-dim normalized landmark vectors per frame
- Model: SimpleRNN (hidden\_dim=144, 7 layers, output over 65 classes); 5 layers was also tested but 7 was superior
- Negative class: ~150 "no-sign" clips extracted from TED Talk recordings
- Training: 100 epochs, lr=1e-4, L2=1e-3

**Why hidden\_dim=144:** Matches the input dimensionality. Each frame has 21 landmarks per hand (left + right = 42) plus 6 upper-body landmarks (indices 11, 12, 13, 14, 23, 24 from MediaPipe Pose), each with 3 coordinates (x, y, z): (42 + 6) × 3 = 144.

**Why SimpleRNN instead of GRU/LSTM:** The goal was to test the simplest possible architecture first. Temporal dependencies within a single sign span roughly 12 steps at 6 FPS — short enough that a vanilla RNN should be able to handle them without gating.

**Why TED talk clips for "no-sign":** Easy to scrape in bulk, and TED speakers produce a lot of non-sign hand movement (gesturing, pointing, resting) with a clearly visible speaker — a reasonable proxy for "hands moving but no sign happening".

**Results:**

| Metric | Value |
|--------|-------|
| Training accuracy | ~100% |
| Validation accuracy | ~98.27% |
| Real-world (continuous signing) | Fails — see below |

<!-- TODO: Add training accuracy / loss curves (docs/figures/exp1_training_curves.png) -->
<!-- TODO: Add confusion matrix on isolated test set (docs/figures/exp1_confusion_matrix.png) -->

**Real-time inference attempt:** A sliding window approach was tested at 6 FPS with 2-second windows (≈12 frames) and a softmax confidence threshold (`no_sign_prob < 0.3`, tuned manually) to distinguish "sign" from "no-sign". This produced two problems:
1. If a window boundary fell mid-sign, the prediction failed.
2. Softmax always commits to a class — there is no innate notion of "nothing happening".

**No-sign class attempt:** TED talk recordings were added as a "no-sign" class (class 65). The model learned to associate "no-sign" with TED speaker hand gestures specifically, and did not generalize to real signing transitions. Intermediate frames between signs — where the hand is mid-transition — did not match any class cleanly and were classified as some random sign.

**Analysis:** The model learns the isolated sign recognition task well, confirming that the 144-dim landmark representation is sufficient for sign discrimination. However, performance collapses on continuous input. The model has no mechanism to locate sign boundaries in a streaming context. This result establishes that the core problem in this pipeline is **temporal segmentation**, not representation or classifier capacity.

---

### Experiment 2 — Two-Stage Detection

**Goal:** Improve robustness by separating hand dominance classification from sign classification.

**Motivation:** In continuous signing, the hand used for the *previous* sign may still be moving (e.g., lowering back to rest) while the next sign begins. If the model sees both hands, it may treat the lingering motion of the irrelevant hand as part of the current sign — creating spurious correlations that hurt generalization. The right fix is more data, but since SLT datasets are small, the alternative is to split the task: first identify which hand is dominant for the current sign, then run a classifier that only has to answer "which sign is this hand making?" — a simpler problem with a cleaner input signal.

**Setup:**
- Dataset processing: 6 FPS, hand-normalized landmarks
- Negative class: 2-second TED talk clips for "no-sign"
- Stage 1: RNN that classifies which hand is dominant for the current sign (right, left, or both)
- Stage 2: Separate RNN classifiers conditioned on dominant hand
- Model width increased to 300 units per layer (wider helped; deeper caused overfitting without improvement)

**Results:**

| Metric | Value |
|--------|-------|
| Validation accuracy | <!-- TODO: fill in --> |
| Improvement over Experiment 1 | <!-- TODO: fill in --> |

<!-- TODO: Add per-class accuracy bar chart (docs/figures/exp2_per_class_accuracy.png) -->
<!-- TODO: Add softmax confidence distribution (docs/figures/exp2_confidence_distribution.png) -->

**Analysis:** The two-stage architecture provides marginal improvement on isolated recognition but does not resolve the continuous segmentation problem. The same failure mode from Experiment 1 persists: intermediate transition frames between signs are not represented in any training class, so the model misclassifies them. The bottleneck is architectural — a deeper fix requires replacing the heuristic segmentation stage entirely.

---

## Exploratory Experiments

### Landmark Extraction — MediaPipe Holistic vs. Pose+Hands

Benchmark comparing two MediaPipe configurations for speed and landmark quality. **Holistic was selected** despite being slower than multithreaded Pose+Hands, because it produces fewer missing landmarks — critical for short LSA64 clips (20–80 frames).

> See [`experiments/landmarks/01_processing_speed/`](../landmarks/01_processing_speed/README.md)

### MMPose Evaluation (Discarded)

MMPose was evaluated as an alternative landmark estimator. Discarded: less than 1 FPS on a Quadro T1000 — infeasible for real-time or large-scale preprocessing.

> See [`experiments/landmarks/02_mmpose_test/`](../landmarks/02_mmpose_test/README.md)

### Acceleration-Based Segmentation (Discarded)

Attempted to use hand acceleration peaks as a heuristic segmentation signal. Not viable due to coarticulation — in continuous LSA there are no consistent motion pauses between signs, so threshold-based methods produce too many false positives/negatives. Acceleration may still be useful as a *feature input* to a learned segmentation model, but not as a standalone rule.

> See [`experiments/sign_analysis/01_hand_acceleration/`](../sign_analysis/01_hand_acceleration/README.md)

---

## Proposed Methods (Phase 1.5 — Not Yet Implemented)

### Data Augmentation via Kinematic Noise

Addresses data scarcity by generating anatomically plausible synthetic sign variants. See [`experiments/data_augmentation/01_kinematic_augmentation/`](../data_augmentation/01_kinematic_augmentation/README.md).

### Data Augmentation via PCA

Learns inter-signer variation directions and applies them as sign-agnostic augmentation axes. See [`experiments/data_augmentation/02_PCA_augmentation/`](../data_augmentation/02_PCA_augmentation/README.md).

---

## Phase 2 — Continuous SLT on LSA-T

Phase 2 will move to the LSA-T dataset and adopt a semi-gloss-free transformer architecture. [LSA-X](https://github.com/PipeVerri/LSA-X) is an extended dataset built from this project.

**Motivation:** The isolated-to-continuous gap cannot be bridged by improving the RNN or the segmentation heuristic. A reliable system requires an end-to-end sequence-to-sequence model that treats the full signing stream as input, without assuming pre-segmented clips.

**Approach — weak segmentation labels from NLP:**
- Use LSA-T (continuous signing videos with Spanish subtitles).
- Convert subtitles to approximate gloss sequences using NLP.
- Use these as weak segmentation labels: no perfect alignment required, just an approximate temporal ordering.
- This allows training a model that understands temporal structure without requiring exact frame-level annotation.

**Planned architecture changes over the original LSA-T paper:**

The original paper uses 3 temporal convolutions with kernel size 1, producing only 3 features per frame — an aggressive reduction that likely discards useful temporal information. Proposed changes:

1. **More temporal features:** Use ~50 features per frame instead of 3, or pass landmarks directly to the sequential model without the convolution bottleneck.
2. **Dedicated facial expression module:** Rather than feeding raw facial landmarks directly (which would force the model to learn emotion recognition as a side task), use a pre-trained or dedicated model to embed facial expressions and pass the embedding to the main model.
3. **Hybrid RNN+transformer (exploratory idea):** Use a frozen RNN for individual gloss recognition, with sigmoid neurons indicating when to commit to the RNN's output and when to pass the sequence on. The transformer would learn to segment by knowing when the RNN has seen a complete sign. *Note: this likely still fails for the same underlying reasons — coarticulation and speed variability affect the RNN's input regardless of who does the segmenting.*

**Data sources:**
- [CN Sordos Argentina](https://www.youtube.com/@CNSORDOSARGENTINA/videos)
- [Canal Asociación Civil](https://www.youtube.com/c/CanalesAsociaci%C3%B3nCivil/videos)
- [Videolibros en señas](https://www.videolibros.org/video/106)

**Based on:** <!-- TODO: Link the target paper here -->

---

## Attribution

Training data from [LSA64](https://facundoq.github.io/datasets/lsa64/) by MIDUSI (Instituto de Investigación en Informática LIDI, Universidad Nacional de La Plata, Argentina).
