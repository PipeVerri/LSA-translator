# Kinematic Noise Augmentation (Proposed)

## Motivation

Data scarcity is one of the primary constraints in Sign Language Translation. LSA64 provides ~50 samples per sign across 10 signers — a very limited training set for a deep model. Standard augmentation approaches (random cropping, flipping, Gaussian noise on raw coordinates) risk producing anatomically implausible configurations.

This experiment proposes a structured augmentation strategy that respects the kinematic constraints of human motion.

## Method

1. **Decompose** each sign into a set of primitive directional motion vectors. For example: "right hand moves +10 cm vertically, +1 cm horizontally". This can be applied at the level of individual fingers as well as the palm.

2. **Represent** each primitive motion as a directional vector **v**.

3. **Inject noise** proportional to the original direction:

$$\mathbf{v}' = \mathbf{v} + \mathbf{v} \circ \mathbf{x}, \quad \mathbf{x} \sim \mathcal{N}(0, \sigma^2 I)$$

   The noise vector is scaled by the original direction, so larger motions receive proportionally larger perturbations. A sign involving a 10 cm upward movement perturbed slightly will remain predominantly upward (e.g., 12 cm up, 2 cm right) — not deflected into an orthogonal direction (e.g., 11 cm up, 7 cm right).

4. **Propagate** the perturbed motion through a forward kinematic chain: distal landmarks (finger tips, knuckles) are moved as a function of proximal joint positions, mimicking the bone-chain model used in skeletal animation.

## Status

Proposed — not yet implemented. Depends on a clean decomposition of MediaPipe landmarks into a kinematic tree structure.
