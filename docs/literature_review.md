# Literature Review — AI Animal Image Recognition on Night-Vision Camera-Trap Images

**Module:** ACM 40960 (Summer 2026) · **Project 9**
**Authors:** Srivani Konda, Navya Sri Mungamuri · University College Dublin

## 1. Problem context

Camera traps are motion-triggered cameras deployed in the wild to photograph
passing animals. They are now a standard tool in ecology and conservation, but a
single study can accumulate **millions of images per year**, the large majority
of which contain no animal at all (triggered by wind, moving vegetation, or
changing light). Manual review therefore becomes the bottleneck: annotating a
large survey by hand can take months of expert time. Automating species
identification is what allows monitoring programmes to scale.

The problem is materially harder at **night**. Most traps switch to an infrared
(IR) flash after dark, producing grayscale, low-contrast frames in which animals
appear as rough, partly-lit outlines against a dark background. Models trained on
ordinary daytime colour photographs tend to degrade on this imagery. This project
studies how well a modern machine-learning classifier copes with exactly that
low-light regime, using an infrared, night-only subset of the **Caltech Camera
Traps** dataset.

## 2. Key prior work

### 2.1 Deep learning reaches expert-level accuracy — Norouzzadeh et al. (2018)
Working with the Snapshot Serengeti archive of **3.2 million** camera-trap
images, Norouzzadeh et al. showed that deep convolutional neural networks could
identify, count, and describe animals at accuracy **matching human volunteers
(>93%)** on common species. This is the headline result establishing that the
task is learnable end-to-end from images, and it is the primary *upper* baseline
this project measures itself against.
*Norouzzadeh, M.S., Nguyen, A., Kosmala, M., Swanson, A., Palmer, M.S., Packer,
C. & Clune, J. (2018). Automatically identifying, counting, and describing wild
animals in camera-trap images with deep learning. PNAS 115(25).*

### 2.2 Transfer learning across regions — Tabak et al. (2019)
A model trained on data from one geographic region does not have to be rebuilt
from scratch for another. Tabak et al. demonstrated that a network trained on
North-American species could be **adapted to new regions via transfer learning**,
dramatically reducing the labelled data needed for a new deployment. This
motivates the core methodological choice of this project: rather than train a
CNN from scratch, we take an ImageNet-pretrained backbone and retrain its later
layers on wildlife imagery.
*Tabak, M.A. et al. (2019). Machine learning to classify animal species in
camera-trap images: applications in ecology. Methods in Ecology and Evolution.*

### 2.3 The generalisation gap — Schneider et al. (2020)
The most important cautionary result for this project. Schneider et al. found
that when a model is evaluated on **camera locations it was not trained on**,
accuracy can collapse — in some cases from ~90% down to **below 70%**. Models
latch onto backgrounds and site-specific cues rather than the animal itself. This
"out-of-location" figure is the *lower*, worst-case baseline we compare against,
and it is the main reason our evaluation reports per-class precision/recall and a
confusion matrix rather than a single headline number.
*Schneider, S., Greenberg, S., Taylor, G.W. & Kremer, S.C. (2020). Three critical
factors affecting automated image species recognition in wildlife.*

### 2.4 Object detection as a front-end — Redmon et al. (2016), YOLO
Because camera-trap frames are mostly empty background with a small animal
somewhere in them, a detection-then-classification pipeline is common: an object
detector (the YOLO family, introduced by Redmon et al.) localises the animal, the
frame is cropped to it, and only then is it classified. This removes background
confounders and is a documented way to mitigate the generalisation gap above.
Our repository includes an optional YOLOv8 cropping stage (`src/detect.py`) for
this reason.
*Redmon, J., Divvala, S., Girshick, R. & Farhadi, A. (2016). You Only Look Once:
Unified, real-time object detection. CVPR.*

### 2.5 The dataset — Caltech Camera Traps
This project uses **Caltech Camera Traps (CCT)** (Beery et al., ECCV 2018):
~243k camera-trap frames from the U.S. Southwest, each with a species label and a
capture timestamp. At night the traps switch to an infrared flash, producing the
grayscale, low-contrast frames this project is about. We build an infrared-only,
night-only subset of CCT (six wild species: bobcat, coyote, raccoon, opossum,
rabbit, deer), pulled from the LILA BC Google-Cloud mirror, keeping only frames
verified to be grayscale and de-duplicating by capture sequence.
*Beery, S., Van Horn, G. & Perona, P. (2018). Recognition in Terra Incognita.
ECCV. Data: https://lila.science/datasets/caltech-camera-traps*

## 3. Where this project sits

| Study | Setting | Reported accuracy | Role here |
|-------|---------|-------------------|-----------|
| Norouzzadeh 2018 | Serengeti, in-distribution | >93% (expert-level) | upper baseline |
| Tabak 2019 | cross-region transfer | high with adaptation | motivates transfer learning |
| Schneider 2020 | out-of-location | can drop below 70% | worst-case baseline |
| **This project** | **infrared night-vision (CCT subset)** | **measured on held-out test set** | **contribution** |

The gap in the literature this project targets is the **specific low-light /
infrared regime**: prior benchmarks are dominated by daytime colour imagery, and
the works above show both that the task is solvable (2.1) and that it is fragile
under distribution shift (2.3). By training a transfer-learned CNN on an infrared
night-vision subset and reporting accuracy, precision, and recall against these
published baselines, we quantify how much of the expert-level performance
survives when the images are grayscale, low-contrast night-vision frames.

## 4. Method implied by the review

The reviewed work points to a concrete recipe, which is what this repository
implements:

1. **Preprocess** frames consistently — resize, normalise with ImageNet
   statistics, and route the single-channel IR frames through a grayscale→RGB
   step so the pretrained backbone sees a consistent input (see `src/data.py`).
2. **Transfer-learn** an ImageNet-pretrained ResNet, freezing the early
   (generic) convolutional layers and retraining the later, task-specific ones
   plus a new classification head (`src/model.py`), following Tabak (2.2).
3. **Optionally detect-and-crop** with YOLOv8 to remove background confounders
   (`src/detect.py`), following the detection literature (2.4) and mitigating the
   Schneider generalisation gap (2.3).
4. **Evaluate** with accuracy, macro/weighted precision and recall, and a
   confusion matrix, and plot the result against the Norouzzadeh and Schneider
   baselines (`src/evaluate.py`).
