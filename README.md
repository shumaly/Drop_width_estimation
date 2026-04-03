# Drop-Width Estimation from Side-View Videos

<p align="center">
  <img src="The setup.png" alt="experimental setup" width="65%">
</p>



A Python toolkit to estimate the **front-view width** of a liquid drop sliding down an inclined surface using only **side-view** measurements.\
Implementation of Shumaly *et al.*, **Scientific Reports** (2024).

The repository supports two practical workflows:

- evaluating the trained model on videos already present in `data/dataset.xlsx`
- running inference on new measurement tables exported by the [4S-SROF toolkit](https://github.com/AK-Berger/4S-SROF)

> **Why is this useful?**\
> Drop width enters directly into both the *lateral adhesion* (Furmidge) and *dynamic friction* force equations.\
> Conventional approaches require a second camera or mirror that shrinks the observable area and complicates alignment.\
> Our model keeps the set-up minimal while still giving width data for the **entire 5 cm track**—even when the drop interacts with surface defects.

---

[https://github.com/AK-Berger/Drop\_width\_estimation/assets/57271994/85117226-5a3e-46f6-be1c-7ed3f92c787b](https://github.com/AK-Berger/Drop_width_estimation/assets/57271994/85117226-5a3e-46f6-be1c-7ed3f92c787b)

---

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
  - [Training](#training)
  - [Testing Existing Data](#testing-existing-data)
  - [Inference](#inference)
- [Data Structure](#data-structure)
- [Model Architecture](#model-architecture)
- [Output](#output)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [License](#license)
- [Citation](#citation)
- [Contact](#contact)

## Features

- Train an LSTM-based regression model on side-view drop features.
- Test the trained model on an existing dataset video for any `Video ID`.
- Run inference on a new Excel file exported by the [4S-SROF toolkit](https://github.com/AK-Berger/4S-SROF).
- Automatic normalization, sliding-window slicing, and result plotting.
- Example scripts: `train_model.py`, `test_on_existing_data.py`, and `inference.py`.

## Installation

```bash
git clone https://github.com/AK-Berger/Drop_width_estimation.git
cd Drop_width_estimation
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

### Training

```bash
python train_model.py \
  --data-file data/dataset.xlsx \
  --model-output models/lstm_weights.h5 \
  --epochs 2500 \
  --batch-size 16 \
  --window-size 20
```

- **--data-file**: Path to the Excel dataset (must include `status`).
- **--model-output**: Path to save trained model weights (.h5).
- **--epochs**, **--batch-size**, **--window-size**: Training parameters.

### Testing Existing Data

```bash
python test_on_existing_data.py \
  --data-file data/dataset.xlsx \
  --weights-file models/lstm_weights.h5 \
  --video-id 14 \
  --window-size 20
```

- **--data-file**: Source Excel (with `Video ID`, features, `status`).
- **--weights-file**: Trained LSTM weights (.h5).
- **--video-id**: Integer ID of the video to process.
- **--window-size**: Sliding window length (frames).
- **--output**: (Optional) Extra path to also save the plot (PNG).
- **Results Excel**: Saved automatically to `output/<datafile_stem>_<video_id>.xlsx`.
- **Results Plot**: Saved automatically to `output/<datafile_stem>_<video_id>.png` and displayed at the end.

### Inference

Use this script when you want to predict width for a new Excel file exported by the [4S-SROF toolkit](https://github.com/AK-Berger/4S-SROF). The script uses `data/dataset.xlsx` only as the training reference for normalization and model shape. It does not test by `Video ID`; instead, it runs prediction directly on the rows in the new input file.

```bash
python inference.py \
  --input-file data/srof_example.xlsx \
  --reference-data-file data/dataset.xlsx \
  --weights-file models/lstm_weights.h5 \
  --window-size 20
```

- **--input-file**: New SROF Excel file for inference.
- **--reference-data-file**: Reference dataset used for training normalization. Default: `data/dataset.xlsx`.
- **--weights-file**: Trained LSTM weights (.h5).
- **--window-size**: Sliding window length (frames).
- **--output**: (Optional) Extra path to also save the plot (PNG).
- **Results Excel**: Saved automatically to `output/inference_<input_file_stem>.xlsx`.
- **Results Plot**: Saved automatically to `output/inference_<input_file_stem>.png` and displayed at the end.

Inference workflow:

1. Extract side-view measurements with the [4S-SROF toolkit](https://github.com/AK-Berger/4S-SROF).
2. Export the resulting table to an Excel file with the standard column names listed below.
3. Run `inference.py` on that Excel file.
4. Collect the predicted width trace from the generated Excel file and plot in `output/`.

## Data Structure

### Input: `data/dataset.xlsx`

| Column                                       | Unit | Description                            |
| -------------------------------------------- | ---- | -------------------------------------- |
| `status`                                     | —    | `train`, `test`, or `final validation` |
| `Video ID`                                   | —    | Unique video identifier                |
| `Frame`                                      | —    | Frame number                           |
| `Advancing (degree)`                         | °    | Advancing contact angle                |
| `Receding (degree)`                          | °    | Receding contact angle                 |
| `Drop length (cm)`                           | cm   | Side-view drop length                  |
| `Drop height (cm)`                           | cm   | Drop centroid height                   |
| `Velocity (cm/s)`                            | cm/s | Drop sliding speed                     |
| `Middle line angle (degree)`                 | °    | Angle of drop midline                  |
| `Tilt angle (degree)`                        | °    | Inclination of the surface             |
| `Defect size [thickness,length,height] (μm)` | μm   | Surface defect dimensions              |
| `Drop width (cm)`                            | cm   | Ground-truth front-view width          |

### Inference Input: `data/srof_example.xlsx`

The new Excel file passed to `inference.py` should follow the same standard structure as `data/srof_example.xlsx`. This file represents the expected output format from the [4S-SROF toolkit](https://github.com/AK-Berger/4S-SROF).

The model input columns in `data/srof_example.xlsx` are aligned with the corresponding columns in `data/dataset.xlsx`, so no manual renaming is needed before inference.

| Column                          | Unit | Used as model input                    |
| ------------------------------- | ---- | -------------------------------------- |
| `Video ID`                      | —    | Preserved in output                    |
| `time (s)`                      | s    | Preserved in output                    |
| `x_center (cm)`                 | cm   | Preserved in output                    |
| `Advancing (degree)`            | °    | `Advancing (degree)`                   |
| `Receding (degree)`             | °    | `Receding (degree)`                    |
| `Drop length (cm)`              | cm   | `Drop length (cm)`                     |
| `Drop height (cm)`              | cm   | `Drop height (cm)`                     |
| `Middle line angle (degree)`    | °    | `Middle line angle (degree)`           |
| `Velocity (cm/s)`               | cm/s | `Velocity (cm/s)`                      |

Required columns for inference are:

- `Advancing (degree)`
- `Receding (degree)`
- `Drop length (cm)`
- `Drop height (cm)`
- `Middle line angle (degree)`
- `Velocity (cm/s)`

### Intermediate Outputs: `output/`

- `dataset_<video_id>.xlsx`: Testing output for an existing dataset video.
- `dataset_<video_id>.png`: Plot for the testing output above.
- `inference_<input_file_stem>.xlsx`: Inference result for a new 4S-SROF Excel file.
- `inference_<input_file_stem>.png`: Plot for the inferred widths.

## Model Architecture

- **LSTM layer**: 48 units, tanh activation, L2 regularization (λ=0.01)
- **Dropout**: 0.5
- **Dense output**: 1 unit, tanh activation
- **Input shape**: `(window_size, 6)` features per frame
- **Loss**: MSE, **Optimizer**: Adam
- **Label scaling**: Ground-truth width scaled from cm to μm

## Output

- **Excel**: `output/<datafile_stem>_<video_id>.xlsx` containing original features plus:
  - `Estimated Width (μm)`
  - `Measured Width (μm)`
- **Plot**: `output/<datafile_stem>_<video_id>.png` with estimated vs measured widths vs frame index
- **Inference Excel**: `output/inference_<input_file_stem>.xlsx` containing the trimmed original 4S-SROF rows plus `Estimated Width (μm)`
- **Inference Plot**: `output/inference_<input_file_stem>.png` with inferred widths vs frame index or exported row index

## Project Structure

```
Drop_width_estimation/
├── data/
│   ├── dataset.xlsx
│   └── srof_example.xlsx
├── models/
│   └── lstm_weights.h5
├── output/
│   ├── dataset_14.xlsx
│   ├── dataset_234.xlsx
│   ├── inference_srof_example.xlsx
│   └── inference_srof_example.png
├── src/
│   └── drop_width/
│       ├── preprocessing.py
│       ├── model.py
│       └── side_to_width.py
├── train_model.py
├── inference.py
├── test_on_existing_data.py
├── requirements.txt
├── The setup.png
└── README.md
```

## Requirements

See `requirements.txt` for full package versions.

## License

Licensed under the GNU General Public License. See [LICENSE](LICENSE) for details.

## Citation

```bibtex
@article{Shumaly2024,
  title   = {Estimating sliding drop width via side-view features using recurrent neural networks},
  author  = {Shumaly, Sajjad and Darvish, Fahimeh and Li, Xiaomei and Kukharenko, Oleksandra and Steffen, Werner and Guo, Yanhui and Butt, Hans-J{"u}rgen and Berger, R{"u}diger},
  journal = {Scientific Reports},
  volume  = {14},
  pages   = {12033},
  year    = {2024},
  doi     = {10.1038/s41598-024-62194-w}
}
```

## Contact

- **Technical**: Sajjad Shumaly — [shumalys@mpip-mainz.mpg.de](mailto\:shumalys@mpip-mainz.mpg.de)
