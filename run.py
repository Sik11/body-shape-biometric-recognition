import os
import argparse
import glob
import time

import cv2 as cv
import numpy as np
import torch
import matplotlib
from tqdm import tqdm
from typing import Dict, List, Tuple
from scipy.spatial.distance import cdist
from ultralytics import YOLO, SAM

from src.analyser import HumanBodyAnalyser, draw_keypoint_line
from src.classifier import KNNClassifiers, SVMClassifiers, UnifiedSVMClassifier
from src.utils import (
    get_prediction_dict,
    get_scaled_embeddings_and_labels,
    write_predictions_to_file,
    calculate_ccr,
    calculate_eer_from_distances,
    calculate_verification_ccr_at_threshold,
    extract_distance_label_pairs,
    normalize_distance_matrix,
    plot_and_save_eer_curve,
    plot_and_save_all_intra_inter_histograms,
)


# Dataset-specific mapping: test image filename -> training image filename.
# Update this dict when using a different dataset.
img_mapping = {
    "DSC00165.JPG": "021z001ps.jpg",
    "DSC00166.JPG": "021z001pf.jpg",
    "DSC00167.JPG": "021z002ps.jpg",
    "DSC00168.JPG": "021z002pf.jpg",
    "DSC00169.JPG": "021z003ps.jpg",
    "DSC00170.JPG": "021z003pf.jpg",
    "DSC00171.JPG": "021z004ps.jpg",
    "DSC00172.JPG": "021z004pf.jpg",
    "DSC00173.JPG": "021z005ps.jpg",
    "DSC00174.JPG": "021z005pf.jpg",
    "DSC00175.JPG": "021z006ps.jpg",
    "DSC00176.JPG": "021z006pf.jpg",
    "DSC00177.JPG": "021z007ps.jpg",
    "DSC00178.JPG": "021z007pf.jpg",
    "DSC00179.JPG": "021z008ps.jpg",
    "DSC00180.JPG": "021z008pf.jpg",
    "DSC00181.JPG": "021z009ps.jpg",
    "DSC00182.JPG": "021z009pf.jpg",
    "DSC00183.JPG": "021z010ps.jpg",
    "DSC00184.JPG": "021z010pf.jpg",
    "DSC00185.JPG": "024z011pf.jpg",
    "DSC00186.JPG": "024z011ps.jpg",
}


def load_test_images(image_dir: str) -> List[str]:
    extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff",
                  "*.JPG", "*.JPEG", "*.PNG", "*.BMP", "*.TIFF"]
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(image_dir, ext)))
    return files


def load_train_images(training_dir: str) -> Tuple[List[str], List[str]]:
    side = glob.glob(os.path.join(training_dir, "*s.jpg"))
    front = glob.glob(os.path.join(training_dir, "*f.jpg"))
    return side, front


def split_test_images(paths: List[str], reference: Dict) -> Tuple[List[str], List[str]]:
    side_images, front_images = [], []
    for path in paths:
        ref = reference[os.path.basename(path)]
        if ref.endswith("s.jpg"):
            side_images.append(path)
        elif ref.endswith("f.jpg"):
            front_images.append(path)
        else:
            raise RuntimeError(f"Unknown file suffix for: {path}")
    return side_images, front_images


def train_and_classify(
    fe_pickle_path: str,
    train_dir: str,
    test_dir: str,
    dlib_detector: str = "mmod_human_face_detector.dat",
    dlib_predictor: str = "shape_predictor_68_face_landmarks.dat",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # --- Initialise models --- #
    yolo_model = YOLO("yolov8m.pt")
    sam_model = SAM("sam2.1_b.pt")

    if os.path.exists(fe_pickle_path):
        fe = torch.load(fe_pickle_path)
    else:
        fe = HumanBodyAnalyser(
            input_height=600,
            sam_model=sam_model,
            yolo_model=yolo_model,
            dlib_face_detector_path=dlib_detector,
            dlib_face_predictor_path=dlib_predictor,
        )

    # --- Load data --- #
    def get_class_label(path):
        return os.path.basename(path)[:-5]

    train_side, train_front = load_train_images(train_dir)
    train_images = train_side + train_front
    print("Number of train images:", len(train_images))
    if not train_images:
        raise ValueError("No train images found in: " + train_dir)

    train_labels = [get_class_label(p) for p in train_front]

    test_images = load_test_images(test_dir)
    print("Number of test images:", len(test_images))
    if not test_images:
        raise ValueError("No test images found in: " + test_dir)

    test_side, test_front = split_test_images(test_images, img_mapping)
    test_labels = [
        get_class_label(os.path.join(train_dir, img_mapping[os.path.basename(p)]))
        for p in test_front
    ]
    assert len(train_side) == len(train_front) and len(test_side) == len(test_front)

    # --- Preprocessing --- #
    for i, path in enumerate(tqdm(train_images + test_images, desc="Preprocessing")):
        t0 = time.time()
        fe.generate_silhouette(path)
        print(f"Processed mask {i+1} in {time.time() - t0:.2f}s")
        t0 = time.time()
        fe.extract_keypoints(path)
        print(f"Processed keypoints {i+1} in {time.time() - t0:.2f}s")

    # --- Feature sets (best performing configuration) --- #
    sf = ["measurements", "silhouette_height", "hu_silhouette", "angles", "hu_head", "pca"]
    ff = ["measurements", "silhouette_height"]

    # --- Train classifier --- #
    classifier = KNNClassifiers(fe, neighbors_per_view={"front": 1, "side": 1})
    classifier.add_samples(train_images, ff, sf)

    # --- Predict on test set --- #
    for path in test_images:
        classifier.predict(path)

    # --- Extract embeddings --- #
    gallery_front_emb, gallery_front_labels = classifier.get_gallery_embeddings_and_labels("front")
    gallery_side_emb, gallery_side_labels = classifier.get_gallery_embeddings_and_labels("side")
    probe_front_emb, probe_front_labels = get_scaled_embeddings_and_labels(test_front, classifier, img_mapping)
    probe_side_emb, probe_side_labels = get_scaled_embeddings_and_labels(test_side, classifier, img_mapping)

    # --- Distance matrices --- #
    distances_front = cdist(gallery_front_emb, probe_front_emb, metric="euclidean")
    distances_side = cdist(gallery_side_emb, probe_side_emb, metric="euclidean")

    # --- Evaluate --- #
    ccr_front = calculate_ccr(get_prediction_dict(test_front, img_mapping, classifier))
    eer_front, t_front = calculate_eer_from_distances(distances_front, gallery_front_labels, probe_front_labels)
    ccr_eer_front = calculate_verification_ccr_at_threshold(distances_front, gallery_front_labels, probe_front_labels, t_front)

    ccr_side = calculate_ccr(get_prediction_dict(test_side, img_mapping, classifier))
    eer_side, t_side = calculate_eer_from_distances(distances_side, gallery_side_labels, probe_side_labels)
    ccr_eer_side = calculate_verification_ccr_at_threshold(distances_side, gallery_side_labels, probe_side_labels, t_side)

    n_front, n_side = len(probe_front_labels), len(probe_side_labels)
    n_total = n_front + n_side
    ccr_weighted = (ccr_front * n_front + ccr_side * n_side) / n_total
    eer_weighted = (eer_front * n_front + eer_side * n_side) / n_total
    ccr_eer_weighted = (ccr_eer_front * n_front + ccr_eer_side * n_side) / n_total

    # --- Save outputs --- #
    write_predictions_to_file(test_images, classifier, img_mapping)

    plot_and_save_all_intra_inter_histograms(
        gallery_front_emb, probe_front_emb, gallery_front_labels, probe_front_labels,
        gallery_side_emb, probe_side_emb, gallery_side_labels, probe_side_labels,
    )

    global_min = min(distances_front.min(), distances_side.min())
    global_max = max(distances_front.max(), distances_side.max())
    front_pairs = extract_distance_label_pairs(
        normalize_distance_matrix(distances_front, global_min, global_max),
        gallery_front_labels, probe_front_labels,
    )
    side_pairs = extract_distance_label_pairs(
        normalize_distance_matrix(distances_side, global_min, global_max),
        gallery_side_labels, probe_side_labels,
    )

    plot_and_save_eer_curve(distances_front, gallery_front_labels, probe_front_labels, view_name="Frontal")
    plot_and_save_eer_curve(distances_side, gallery_side_labels, probe_side_labels, view_name="Side")
    plot_and_save_eer_curve(front_pairs + side_pairs, view_name="Combined")

    # --- Print results --- #
    print(f"\nFront Evaluation:"
          f"\n  CCR:          {ccr_front * 100:.2f}%"
          f"\n  EER:          {eer_front * 100:.2f}%  (threshold: {t_front:.4f})"
          f"\n  CCR @ EER:    {ccr_eer_front * 100:.2f}%")

    print(f"\nSide Evaluation:"
          f"\n  CCR:          {ccr_side * 100:.2f}%"
          f"\n  EER:          {eer_side * 100:.2f}%  (threshold: {t_side:.4f})"
          f"\n  CCR @ EER:    {ccr_eer_side * 100:.2f}%")

    print(f"\nWeighted Combined:"
          f"\n  CCR:          {ccr_weighted * 100:.2f}%"
          f"\n  EER:          {eer_weighted * 100:.2f}%"
          f"\n  CCR @ EER:    {ccr_eer_weighted * 100:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Body Shape Biometric Recognition: train and evaluate on a dataset."
    )
    parser.add_argument("--train", default="datasets/training",
                        help="Path to training images directory (default: datasets/training)")
    parser.add_argument("--test", default="datasets/test",
                        help="Path to test images directory (default: datasets/test)")
    parser.add_argument("--cache", default="feature_extractor_dl_600.pickle",
                        help="Path to cached feature extractor pickle")
    parser.add_argument("--dlib-detector", default="mmod_human_face_detector.dat",
                        help="Path to dlib CNN face detector .dat file")
    parser.add_argument("--dlib-predictor", default="shape_predictor_68_face_landmarks.dat",
                        help="Path to dlib 68-point shape predictor .dat file")
    parser.add_argument("--latex-plots", action="store_true",
                        help="Render plots as PGF for LaTeX inclusion")
    args = parser.parse_args()

    if args.latex_plots:
        matplotlib.use("pgf")
        matplotlib.rcParams.update({
            "pgf.texsystem": "pdflatex",
            "font.family": "serif",
            "text.usetex": True,
            "pgf.rcfonts": False,
        })
    else:
        matplotlib.use("Agg")

    train_and_classify(
        fe_pickle_path=args.cache,
        train_dir=args.train,
        test_dir=args.test,
        dlib_detector=args.dlib_detector,
        dlib_predictor=args.dlib_predictor,
    )
