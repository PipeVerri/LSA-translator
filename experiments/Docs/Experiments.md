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
- Model: SimpleRNN (hidden\_dim=144, 5–7 layers, output over 65 classes)
- Negative class: ~150 "no-sign" clips extracted from TED Talk recordings

**Results:**

| Metric | Value |
|--------|-------|
| Training accuracy | ~100% |
| Validation accuracy | ~98.27% |
| Real-world (continuous signing) | Fails — see below |

<!-- TODO: Add training accuracy / loss curves (docs/figures/exp1_training_curves.png) -->
<!-- TODO: Add confusion matrix on isolated test set (docs/figures/exp1_confusion_matrix.png) -->

**Analysis:** The model learns the isolated sign recognition task well, confirming that the 144-dim landmark representation is sufficient for sign discrimination. However, performance collapses on continuous input. The model has no mechanism to locate sign boundaries in a streaming context, and the sliding window approach with a fixed confidence threshold is not a reliable substitute. This result establishes that the core problem in this pipeline is **temporal segmentation**, not representation or classifier capacity.

---

### Experiment 2 — Two-Stage Detection

**Goal:** Improve robustness by separating hand dominance classification from sign classification.

**Setup:**
- Stage 1: RNN that classifies which hand is dominant for the current sign (right, left, or both).
- Stage 2: Separate RNN classifiers conditioned on dominant hand.
- Both models: SimpleRNN, same architecture as Experiment 1.

**Results:**

| Metric | Value |
|--------|-------|
| Validation accuracy | <!-- TODO: fill in --> |
| Improvement over Experiment 1 | <!-- TODO: fill in --> |

<!-- TODO: Add per-class accuracy bar chart (docs/figures/exp2_per_class_accuracy.png) -->
<!-- TODO: Add softmax confidence distribution (docs/figures/exp2_confidence_distribution.png) -->

**Analysis:** The two-stage architecture provides marginal improvement on isolated recognition but does not resolve the continuous segmentation problem. The bottleneck is architectural, not classifier-specific — a deeper fix requires replacing the heuristic segmentation stage entirely.

---

## Proposed Methods (Phase 1.5 — Not Yet Implemented)

### Data Augmentation via Kinematic Noise

Addresses data scarcity by generating anatomically plausible synthetic sign variants. See [`experiments/data_augmentation/01_kinematic_augmentation/`](../data_augmentation/01_kinematic_augmentation/README.md).

### Data Augmentation via PCA

Learns inter-signer variation directions and applies them as sign-agnostic augmentation axes. See [`experiments/data_augmentation/02_PCA_augmentation/`](../data_augmentation/02_PCA_augmentation/README.md).

---

## Phase 2 — Continuous SLT on LSA-T

Phase 2 will move to the LSA-T dataset and adopt a semi-gloss-free transformer architecture.

**Motivation:** The isolated-to-continuous gap cannot be bridged by improving the RNN or the segmentation heuristic. A reliable system requires an end-to-end sequence-to-sequence model that treats the full signing stream as input, without assuming pre-segmented clips.

**Planned architecture:**
- Transformer-based encoder for landmark sequences.
- LLM-assisted decoding for Spanish text generation.
- Based on the approach described in: <!-- TODO: Link the target paper here -->

---

## Attribution

Training data from [LSA64](https://facundoq.github.io/datasets/lsa64/) by MIDUSI (Instituto de Investigación en Informática LIDI, Universidad Nacional de La Plata, Argentina).
