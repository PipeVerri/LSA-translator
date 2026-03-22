# PCA-Based Augmentation (Proposed)

## Motivation

Inter-signer variability is a significant challenge in sign language recognition: two people producing the same sign may differ in arm elevation, hand orientation, and movement amplitude. If some of these variation directions are consistent across the vocabulary (i.e., common to many different signs), they can be extracted once and used as a general-purpose augmentation strategy.

## Method

1. For each sign gloss, stack all signer samples (landmark sequences) into a matrix **X** ∈ ℝ^{n × d}.
2. Apply PCA to **X** and extract the principal variation direction **v₁** (the eigenvector corresponding to the largest eigenvalue).
3. Normalize **v₁** to unit length, retaining direction but discarding scale.
4. Repeat for each gloss to obtain a set of normalized principal variation vectors {**v₁**^(1), **v₁**^(2), ..., **v₁**^(G)}.
5. Construct a pairwise difference norm heatmap: entry (i, j) = ‖**v₁**^(i) − **v₁**^(j)‖. If two glosses share a similar dominant variation direction, this norm approaches 0.
6. Optionally cluster the variation vectors to identify groups of signs sharing a common variation axis.

## Hypothesis

If several glosses share a principal variation direction (e.g., the arm consistently goes slightly higher or lower across signers), that direction represents a signer-agnostic degree of freedom that can be applied as augmentation across all signs — not just the ones it was computed from. The heatmap and clustering step are designed to validate this hypothesis before applying the augmentation.

## Status

Proposed — not yet implemented.
