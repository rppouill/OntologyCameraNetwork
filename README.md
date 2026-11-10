# OntologyCameraNetwork

MATLAB code for cross-camera person association: foreground extraction (Low
Processing), then eigen-target computation and learning of the inter-camera
transformations (Medium Processing).

All scripts are at the root of the repository and must be launched from it
(they use relative paths such as `./tools` and `./dataSet`).

## Pipeline

```
LowProcessing_Video.m   ─┐
                         ├─> dataSet/<NAME>/videoX/pY/*.png (+ D.mat)
LowProcessing_Images.m  ─┘
                                   │
                                   v
                         MediumProcessing.m ─> results/<NAME>/<METRIC>/*.png
                                           └─> results/<NAME>/<METRIC>/transformations.mat
                                                         │
                                                         v
                                                 transformation.m (confusion matrices)
```

## 1. Low Processing

Background subtraction and extraction of the foreground of each person, for
each camera.

| Script | Input | Output (variable to edit) |
|---|---|---|
| `LowProcessing_Video.m` | `.avi` videos named `<camera>-person<person>.avi` | `outpout_folder` (default `./dataSet/ImageSelectedwithD`) |
| `LowProcessing_Images.m` | folders `ESP*/<person>/` of images | `output_folder` (default `./NewDataset/`) |

Before running, set `DATASET_PATH` at the top of the script (and, for the video
version, `Nb_Camera`, `Nb_Person`).

```bash
matlab -nodesktop -nosplash -r "run('LowProcessing_Video.m'); exit"
# or
matlab -nodesktop -nosplash -r "run('LowProcessing_Images.m'); exit"
```

## 2. Medium Processing

Computes the eigen-targets (PCA) of each person per camera, learns the
transformations between each pair of cameras (Least Square, Least Square
Adaptive, Gradient Descent, Polynomial) and plots the accuracy as a function of
the number of eigenvectors.

At the top of `MediumProcessing.m`, set:

- `DATASET_PATH` / `DATASET_NAME`: the folder produced by the Low Processing
  (it must contain `video<i>/p<j>/*.png`, and optionally `D.mat`);
- `NB_CAMERA`, `NB_PERSON`, `NB_VECTOR`, `IMG_SIZE`.

```bash
matlab -nodesktop -nosplash -r "run('MediumProcessing.m'); exit"
```

Figures are saved in `results/<DATASET_NAME>/<METRIC>/` (`spectral` by default)
and the learned transformations in
`results/<DATASET_NAME>/<METRIC>/transformations.mat`.

## Tips

- To stay in MATLAB after the script (to inspect variables or figures), remove
  `; exit`: `matlab -nodesktop -nosplash -r "run('MediumProcessing.m')"`.
- Without a display (SSH, server), use `-nodisplay` instead of `-nodesktop`;
  figures are still saved with `saveas`.
- `matlab -batch "run('MediumProcessing.m')"` is equivalent and returns a
  non-zero exit code on error (useful in scripts).
