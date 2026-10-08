# Dataset Rework: Synthetic Shapes → ModelNet10

## What changed and why

The project originally used a hand-drawn synthetic dataset (four shapes —
circle, square, triangle, star — rendered as small images, paired with
sampled 3D point clouds of the same shape). It was fast to build and got the
pipeline working end to end, but it read as an unconvincing toy: the shapes
were invented, not real objects, and that undermines the whole point of a
multimodal fusion project (nobody fuses sensors to classify hand-drawn stars).

Replaced with **ModelNet10**: a real, publicly available 3D CAD model dataset
(bathtub, bed, chair, desk, dresser, monitor, night_stand, sofa, table,
toilet — 10 real-world object categories), used in actual published robotics/
vision research (PointNet, VoxNet, 3DShapeNets). Directly downloadable with
no account/signup (~473MB from `3dvision.princeton.edu`), which matters
because nuScenes and KITTI — the "obvious" choices for a robotics-flavored
project — both require registration that would have blocked full automation.

## How both modalities are derived from one real object

This is the part worth being precise about, since it's easy to accidentally
reintroduce "fake-feeling" data even with a real source dataset:

- **Lidar point cloud**: sampled directly off the real mesh surface
  (area-weighted triangle sampling — the same technique PointNet's own
  preprocessing uses), then a sparse (256-point) subset is used as the
  "lidar" reading.
- **Camera image**: a 2D projection of a *denser* (6000-point) sample of
  that same mesh, from a random viewpoint, rasterized into a 40×40 grayscale
  image. Both the dense render-points and the sparse lidar-points are
  subsampled from the same normalized point set, in the same reference
  frame — not two independent generators that happen to share a label.

The deliberate asymmetry (dense points for the image, sparse for lidar)
mirrors something real: cameras produce dense, continuous-looking data;
lidar is inherently much lower-resolution. That's not a simplification bug,
it's the point.

## Known limitation, stated honestly

A single random viewpoint can be genuinely ambiguous for some real objects
(e.g. a monitor viewed edge-on looks like a thin sliver). This is not a
renderer bug — it's realistic (a single camera view is sometimes ambiguous
too) — but it does mean the lidar modality, which stays informative
regardless of viewing angle, carries real weight in the fusion. This is
actually a point worth making in an interview: it's a concrete example of
*why* sensor fusion helps, not just an assumption.

## What this did NOT fix by itself

Switching to real data surfaced a real overfitting problem (see
`docs/phase2_profiling_notes.md` and the STATUS block in `PLAN.md` for the
numbers and the fixes: augmentation, class weighting, weight decay, LR
decay). That's covered separately — this document is specifically about the
data source change, not the training fixes that followed it.
