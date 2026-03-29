# Datasets

This directory is excluded from version control. You must obtain and place the dataset here yourself.

## Southampton Gait Database

This project was evaluated on the Southampton Gait Database (subject IDs starting with `021z` and `024z`).

**Training images** go in `datasets/training/` and follow the naming convention:
- `{SUBJECT_ID}f.jpg` for frontal view
- `{SUBJECT_ID}s.jpg` for side view

**Test images** go in `datasets/test/`.

The `img_mapping` dict in `run.py` maps test image filenames to their corresponding training image filenames (ground truth labels). Update this dict to match your own dataset if using different images.

## Directory structure

```
datasets/
  training/
    021z001pf.jpg
    021z001ps.jpg
    ...
  test/
    DSC00165.JPG
    DSC00166.JPG
    ...
```
