# VOTSp2026 Starter Kit

Starter kit for the [VOTSp2026 Point Tracking Challenge](https://www.votchallenge.net/vots2026/). Includes two ready-to-run tracker wrappers and a guide to submitting your own tracker.

## Quickstart

```bash
# 1. Install the VOT toolkit and download the dataset
pip install vot-toolkit
vot initialize vots2026/point --workspace ~/votsp_workspace

# 2. Clone this repo
git clone https://github.com/aharley/votsp-starter
cd votsp-starter

# 3. Copy trackers.ini into your workspace and update the paths field
cp trackers.ini ~/votsp_workspace/trackers.ini
# Edit trackers.ini: set paths = /path/to/votsp-starter

# 4. Run a tracker
vot evaluate --workspace ~/votsp_workspace AllTracker

# 5. Pack for submission
vot pack --workspace ~/votsp_workspace AllTracker
```

Then upload the resulting zip to the [evaluation server](https://eu.aihub.ml/competitions/263).

## Included trackers

Dependencies (required for both trackers):

```
pip install torch torchvision opencv-python
```

### AllTracker

[AllTracker](https://github.com/aharley/alltracker) (Harley et al., ICCV 2025) is the challenge baseline. Weights are downloaded automatically from HuggingFace on first run. Works on both CPU and GPU.

Wrapper: `alltracker_tracker.py`

### CoTracker3

[CoTracker3](https://github.com/facebookresearch/co-tracker) (Karaev et al., NeurIPS 2024) is another strong point tracker. Weights are downloaded automatically from HuggingFace on first run. Works on both CPU and GPU.

Wrapper: `cotracker3_tracker.py`

## Folder protocol

The VOT toolkit communicates with your tracker by reading and writing text files in a temporary directory. For each sequence, the toolkit writes:

- `frames_color.txt` — absolute paths to video frames, one per line
- `query_N.txt` — one file per tracked point:
  - Line 1: frame offset (the query frame index)
  - Line 2: query location as `x,y` in original image pixel coordinates

Your tracker must write:

- `output_N.txt` — one line per frame: `0` before the query frame, then `x,y` from the query frame onward

See `example/` for sample files illustrating the format.

## Integrating your own tracker

1. Copy one of the wrapper scripts and adapt the tracking logic
2. Add an entry to `trackers.ini`:

```ini
[MyTracker]
command = my_tracker
protocol = folderpython
paths = /path/to/my/tracker/code
```

3. Run evaluation and pack:

```bash
vot evaluate --workspace ~/votsp_workspace MyTracker
vot pack --workspace ~/votsp_workspace MyTracker
```

4. Register at [forms.gle/CWMNkgap61AS2Wuw6](https://forms.gle/CWMNkgap61AS2Wuw6) and submit the zip to the [evaluation server](https://eu.aihub.ml/competitions/263).

The tracker identifier (the `[section]` name in `trackers.ini`) must match the short name you register on the form.

## Training restrictions

Training on the following datasets is **not allowed**: EgoPoints, RoboTAP, AnimalTrack, BADJA, Cell Tracking Challenge, SpaceAnimal, DexYCB, CMU Panoptic, Horse-10, CTMCv1, CroHD, Ant Dataset, Tri-Mouse, I-MuPPET, Schooling-Fish.
