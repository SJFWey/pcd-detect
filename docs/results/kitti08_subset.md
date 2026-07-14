# SemanticKITTI Sequence 08 Development Subset

This result was generated from the detection-equivalent configuration in
`examples/kitti08_subset.yaml`. It is a development result, not a held-out
benchmark.

## Protocol

- Dataset: KITTI Odometry / SemanticKITTI
- Sequence: `08`
- Frames: `0-499`
- Targets: `car`, `person`
- Matching: BEV IoU >= 0.5, class-aware
- Prediction class source: box-size heuristic when no class metadata exists
- Reference-box source: PCA OBB fitted to visible SemanticKITTI instance points
  inside the same ROI used by the proposal pipeline
- Minimum reference support: `10` visible points after ROI cropping
- Benchmark status: custom proxy metrics, not KITTI 3D detection AP
- Ground RANSAC seed: `0` (fixed for repeatable runs)
- Output: `outputs/kitti08_subset/reports/det_metrics.json`
- Metrics source commit: `d2018e1e35b63cb59744ed99ec381a619fdf3809`
- Current reproduction-config SHA-256:
  `749a7a55991c4ae2a56cd5848610255133f5029289d450f60eaf9af6eb06d66e`
- Selection note: `max_box_volume: 10.0` won a controlled one-factor
  comparison on this same subset.

## Overall Metrics

| Metric | Value |
| --- | ---: |
| Frames | 500 |
| Predictions | 12812 |
| Instance-derived reference boxes | 1937 |
| True positives | 541 |
| False positives | 12271 |
| False negatives | 1396 |
| Precision | 0.0422 |
| Recall | 0.2793 |
| F1 | 0.0734 |
| Mean IoU of matched pairs | 0.7237 |

## Distance Metrics

| Range | Predictions | Reference | TP | FP | FN | Precision | Recall | F1 | Mean matched IoU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0-10m | 729 | 289 | 223 | 506 | 66 | 0.306 | 0.772 | 0.438 | 0.771 |
| 10-20m | 2879 | 578 | 234 | 2645 | 344 | 0.081 | 0.405 | 0.135 | 0.706 |
| 20-30m | 3538 | 495 | 58 | 3480 | 437 | 0.016 | 0.117 | 0.029 | 0.619 |
| 30-40m | 3309 | 325 | 16 | 3293 | 309 | 0.005 | 0.049 | 0.009 | 0.660 |
| 40-50m | 1817 | 250 | 10 | 1807 | 240 | 0.006 | 0.040 | 0.010 | 0.784 |
| 50m+ | 540 | 0 | 0 | 540 | 0 | 0.000 | 0.000 | 0.000 | — |

The original report used finite bins ending at 50 m. The 50m+ row is recovered
exactly from the overall totals minus the recorded finite bins. The evaluator
now adds this overflow bin automatically.

## Visual Checks
The images below use the same processing configuration as the measured subset.
`visualization.display_roi` crops only the rendered view, and panel 3 evaluates
only boxes centred within that visible ROI. It uses reference boxes fitted from
real SemanticKITTI instance labels and the same class-aware BEV-IoU matching rule
as the table above.

![KITTI BEV pipeline frame 30](../assets/kitti08-frame-000030-pipeline.png)

![KITTI BEV pipeline frame 120](../assets/kitti08-frame-000120-pipeline.png)

![KITTI BEV pipeline frame 450](../assets/kitti08-frame-000450-pipeline.png)

## Interpretation

This is a geometric proposal baseline. It is intentionally transparent but not
competitive with learned 3D detectors. The low precision is expected because the
pipeline treats many non-ground clusters as candidate objects and does not have
a learned objectness classifier. Mean matched IoU is conditional on the 541
pairs that already satisfy the class and IoU >= 0.5 rule. It describes
localization among successful matches only and should not be used as evidence
of overall detector quality.

The machine-readable transcription is
[`kitti08_subset.metrics.json`](kitti08_subset.metrics.json).
