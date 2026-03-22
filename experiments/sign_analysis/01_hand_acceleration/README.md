# Acceleration-Based Sign Segmentation (Discarded)

## Goal

Visualize hand acceleration profiles from continuous LSA recordings and evaluate whether acceleration peaks can serve as a reliable heuristic to segment a continuous signing stream into individual glosses.

Two acceleration signals were measured:
- Landmark acceleration relative to the wrist (captures hand shape transitions).
- Wrist acceleration relative to the forearm (captures gross arm movement).

## Results

Acceleration peaks are not a reliable segmentation signal in continuous LSA. The fundamental issue is **coarticulation**: there are no consistent motion pauses between signs in fluent signing. The final configuration of one sign transitions directly into the initial configuration of the next, without a discriminative deceleration or rest phase.

Threshold-based approaches on acceleration profiles produce too many false positives (spurious peaks mid-sign) and false negatives (low-energy transitions between signs).

## Broader Context

Acceleration-based segmentation has been explored in the literature but has been superseded by learned approaches. Recent work still uses acceleration as a feature, but feeds it as an input to a trained segmentation model (e.g., a CRF or transformer) rather than using it as a standalone rule. See:

> [Sign Language Spotting with a Threshold Model Based on Conditional Random Fields](https://www.researchgate.net/publication/24428978_Sign_Language_Spotting_with_a_Threshold_Model_Based_on_Conditional_Random_Fields)

The current state of the art in segmentation uses end-to-end models that jointly learn segmentation and recognition, avoiding explicit boundary detection entirely.
