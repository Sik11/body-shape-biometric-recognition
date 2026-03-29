"""Evaluation and visualization utilities for the body shape biometric pipeline."""

import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from typing import Dict, List, Tuple


# ---------------------------------------------------------------------------
# Prediction helpers
# ---------------------------------------------------------------------------

def get_prediction_dict(paths: List[str], img_mapping: Dict[str, str], classifier) -> Dict[str, Dict]:
    """Map each test path to its predicted and true label."""
    prediction_data = {}
    for path in paths:
        filename = os.path.basename(path)
        predicted_label = classifier.data[path]["predicted_label"]
        predicted_prob = classifier.data[path]["class_probabilities"].get(predicted_label, 0.0)

        if filename in img_mapping:
            true_label = os.path.basename(img_mapping[filename])[:-5]
        else:
            true_label = "UNKNOWN"

        prediction_data[path] = {
            "predicted_label": predicted_label,
            "predicted_prob": predicted_prob,
            "true_label": true_label,
        }
    return prediction_data


def get_scaled_embeddings_and_labels(
    paths: List[str], classifier, img_mapping: Dict[str, str]
) -> Tuple[np.ndarray, List[str]]:
    """Return scaled embeddings and ground truth labels for a list of image paths."""
    embeddings = []
    labels = []
    prediction_dict = get_prediction_dict(paths, img_mapping, classifier)
    for path in paths:
        if path not in classifier.data:
            raise ValueError(f"Missing prediction for path: {path}")
        if "X_scaled" not in classifier.data[path]:
            raise ValueError(f"Missing scaled embedding for path: {path}")
        embeddings.append(classifier.data[path]["X_scaled"])
        labels.append(prediction_dict[path]["true_label"])
    return np.vstack(embeddings), labels


def write_predictions_to_file(
    test_paths: List[str],
    classifier,
    img_mapping: Dict[str, str],
    output_path: str = "outputs/predictions.txt",
):
    """Write predicted and ground truth labels for each test image to a file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write("Image\tGround Truth\tPrediction\n")
        f.write("=" * 50 + "\n")
        for path in test_paths:
            filename = os.path.basename(path)
            predicted_class = classifier.data[path]["predicted_label"]
            ground_truth = os.path.basename(img_mapping[filename])[:-5]
            f.write(f"{filename}\t{ground_truth}\t{predicted_class}\n")
    print(f"Predictions written to {output_path}")


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------

def calculate_ccr(predictions: Dict[str, Dict]) -> float:
    """Correct Classification Rate from a prediction dict."""
    correct = sum(1 for p in predictions.values() if p["predicted_label"] == p["true_label"])
    total = len(predictions)
    return correct / total if total > 0 else 0.0


def calculate_eer_from_distances(
    distances: np.ndarray, labels1: List[str], labels2: List[str]
) -> Tuple[float, float]:
    """Equal Error Rate and threshold from a pairwise distance matrix."""
    thresholds = np.linspace(np.min(distances), np.max(distances), 10000)
    fars, frrs = [], []
    for threshold in thresholds:
        fa = fr = genuine = impostor = 0
        for i in range(len(labels1)):
            for j in range(len(labels2)):
                is_genuine = labels1[i] == labels2[j]
                dist = distances[i, j]
                if is_genuine:
                    fr += dist > threshold
                    genuine += 1
                else:
                    fa += dist <= threshold
                    impostor += 1
        fars.append(fa / impostor if impostor else 0)
        frrs.append(fr / genuine if genuine else 0)
    fars = np.array(fars)
    frrs = np.array(frrs)
    idx = np.argmin(np.abs(fars - frrs))
    eer = (fars[idx] + frrs[idx]) / 2
    return eer, thresholds[idx]


def calculate_verification_ccr_at_threshold(
    distances: np.ndarray, labels1: List[str], labels2: List[str], threshold: float
) -> float:
    """CCR at a given distance threshold for verification."""
    correct = total = 0
    for i in range(len(labels1)):
        for j in range(len(labels2)):
            same = labels1[i] == labels2[j]
            accept = distances[i, j] <= threshold
            if (same and accept) or (not same and not accept):
                correct += 1
            total += 1
    return correct / total if total > 0 else 0.0


def extract_distance_label_pairs(
    distances: np.ndarray, gallery_labels: List[str], probe_labels: List[str]
) -> List[Tuple[float, int]]:
    """Flatten a distance matrix into (distance, genuine_flag) pairs."""
    pairs = []
    for i in range(len(gallery_labels)):
        for j in range(len(probe_labels)):
            pairs.append((distances[i, j], int(gallery_labels[i] == probe_labels[j])))
    return pairs


def normalize_distance_matrix(dists: np.ndarray, d_min: float, d_max: float) -> np.ndarray:
    return (dists - d_min) / (d_max - d_min)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_and_save_eer_curve(
    distances,
    gallery_labels: List[str] = None,
    probe_labels: List[str] = None,
    view_name: str = "Combined",
    output_dir: str = "outputs/eer_curves",
):
    """
    Plot and save a FAR/FRR EER curve.

    Accepts either a 2D distance matrix + label lists (frontal/side),
    or a flat list of (distance, genuine_flag) pairs (combined view).
    """
    os.makedirs(output_dir, exist_ok=True)

    if isinstance(distances, list) and isinstance(distances[0], tuple):
        dists = np.array([d for d, _ in distances])
        labels = np.array([l for _, l in distances])
        thresholds = np.linspace(dists.min(), dists.max(), 10000)
        fars, frrs = [], []
        for t in thresholds:
            preds = dists <= t
            fars.append(np.sum((preds == 1) & (labels == 0)) / np.sum(labels == 0))
            frrs.append(np.sum((preds == 0) & (labels == 1)) / np.sum(labels == 1))
        eer_idx = np.argmin(np.abs(np.array(fars) - np.array(frrs)))
        eer = (fars[eer_idx] + frrs[eer_idx]) / 2
        eer_threshold = thresholds[eer_idx]
        ccr_at_eer = np.mean((dists <= eer_threshold) == labels)
    else:
        thresholds = np.linspace(distances.min(), distances.max(), 10000)
        fars, frrs = [], []
        for t in thresholds:
            fa = fr = genuine = impostor = 0
            for i in range(len(gallery_labels)):
                for j in range(len(probe_labels)):
                    d = distances[i, j]
                    same = gallery_labels[i] == probe_labels[j]
                    if same:
                        fr += d > t
                        genuine += 1
                    else:
                        fa += d <= t
                        impostor += 1
            fars.append(fa / impostor if impostor else 0)
            frrs.append(fr / genuine if genuine else 0)
        fars = np.array(fars)
        frrs = np.array(frrs)
        eer_idx = np.argmin(np.abs(fars - frrs))
        eer = (fars[eer_idx] + frrs[eer_idx]) / 2
        eer_threshold = thresholds[eer_idx]
        correct = total = 0
        for i in range(len(gallery_labels)):
            for j in range(len(probe_labels)):
                d = distances[i, j]
                same = gallery_labels[i] == probe_labels[j]
                match = d <= eer_threshold
                if (same and match) or (not same and not match):
                    correct += 1
                total += 1
        ccr_at_eer = correct / total

    plt.figure(figsize=(7, 5))
    plt.plot(thresholds, fars, label="FAR", color="red")
    plt.plot(thresholds, frrs, label="FRR", color="blue")
    plt.axvline(eer_threshold, color="black", linestyle="--",
                label=f"EER Threshold ({eer_threshold:.3f})")
    plt.title(f"{view_name} View EER\nEER = {eer*100:.2f}% | CCR @ EER = {ccr_at_eer*100:.2f}%")
    plt.xlabel("Distance Threshold")
    plt.ylabel("Rate")
    plt.legend()
    plt.tight_layout()
    out_path = os.path.join(output_dir, f"{view_name.lower()}_eer_curve.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Saved EER curve: {out_path}")


def plot_and_save_all_intra_inter_histograms(
    gallery_front, probe_front, labels_g_front, labels_p_front,
    gallery_side, probe_side, labels_g_side, labels_p_side,
    output_dir: str = "outputs/histograms",
):
    """Plot intra/inter-class distance histograms for frontal, side, and combined views."""
    def _intra_inter(gallery, probe, labels_g, labels_p):
        dists = cdist(gallery, probe, metric="euclidean")
        intra, inter = [], []
        for i, gl in enumerate(labels_g):
            for j, pl in enumerate(labels_p):
                (intra if gl == pl else inter).append(dists[i, j])
        return np.array(intra), np.array(inter)

    def _norm(arr, vmin, vmax):
        return (arr - vmin) / (vmax - vmin)

    def _save_hist(intra, inter, title, filename):
        plt.figure(figsize=(7, 5))
        plt.hist(intra, bins=20, alpha=0.6, color="blue", label="Intra-class", density=True)
        plt.hist(inter, bins=20, alpha=0.6, color="red", label="Inter-class", density=True)
        plt.title(title)
        plt.xlabel("Normalized Distance")
        plt.ylabel("Probability Density")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, filename))
        plt.close()
        print(f"Saved: {filename}")

    os.makedirs(output_dir, exist_ok=True)

    intra_f, inter_f = _intra_inter(gallery_front, probe_front, labels_g_front, labels_p_front)
    all_f = np.concatenate([intra_f, inter_f])
    _save_hist(_norm(intra_f, all_f.min(), all_f.max()),
               _norm(inter_f, all_f.min(), all_f.max()),
               "Frontal View - Intra vs. Inter Class Distances",
               "frontal_intra_inter_histogram.png")

    intra_s, inter_s = _intra_inter(gallery_side, probe_side, labels_g_side, labels_p_side)
    all_s = np.concatenate([intra_s, inter_s])
    _save_hist(_norm(intra_s, all_s.min(), all_s.max()),
               _norm(inter_s, all_s.min(), all_s.max()),
               "Side View - Intra vs. Inter Class Distances",
               "side_intra_inter_histogram.png")

    intra_combined = np.concatenate([_norm(intra_f, intra_f.min(), intra_f.max()),
                                     _norm(intra_s, intra_s.min(), intra_s.max())])
    inter_combined = np.concatenate([_norm(inter_f, inter_f.min(), inter_f.max()),
                                     _norm(inter_s, inter_s.min(), inter_s.max())])
    _save_hist(intra_combined, inter_combined,
               "Combined View - Intra vs. Inter Class Distances",
               "combined_unified_intra_inter_histogram.png")


# ---------------------------------------------------------------------------
# Image export helpers
# ---------------------------------------------------------------------------

def export_image_outputs(paths: List[str], output_subdir: str, img_fn):
    """Save processed images (masks, keypoints, etc.) to outputs/."""
    os.makedirs(f"outputs/{output_subdir}", exist_ok=True)
    for path in paths:
        img = img_fn(path)
        cv.imwrite(f"outputs/{output_subdir}/{os.path.basename(path)}", img)


def draw_keypoints(path: str, fe, keypoints=None):
    """Render OpenPose keypoints and skeleton onto an image and save to outputs/keypoints/."""
    debug_image = fe.fetch_image(path).copy()
    openpose_keypoints = keypoints if keypoints else fe.extract_keypoints(path)
    debug_image = cv.cvtColor(np.array(debug_image), cv.COLOR_RGB2BGR)

    point_color = (0, 255, 0)
    line_color = (255, 0, 0)
    rectangle_color = (0, 255, 255)
    radius, thickness = 5, 2

    for x, y, p in openpose_keypoints:
        if p > 0:
            cv.circle(debug_image, (int(x), int(y)), radius, point_color, -1)

    skeleton_pairs = [
        (2, 3), (3, 4), (5, 6), (6, 7),
        (8, 9), (9, 10), (11, 12), (12, 13),
        (0, 1), (14, 16), (15, 17), (0, 14), (0, 15),
    ]
    for i, j in skeleton_pairs:
        if openpose_keypoints[i][2] > 0 and openpose_keypoints[j][2] > 0:
            cv.line(debug_image,
                    (int(openpose_keypoints[i][0]), int(openpose_keypoints[i][1])),
                    (int(openpose_keypoints[j][0]), int(openpose_keypoints[j][1])),
                    line_color, thickness)

    kp = openpose_keypoints
    if all(kp[idx][2] > 0 for idx in [1, 5, 2, 11, 8]):
        chest_pts = np.array([[kp[5][0], kp[5][1]], [kp[2][0], kp[2][1]],
                               [kp[8][0], kp[8][1]], [kp[11][0], kp[11][1]]], np.int32)
        cv.polylines(debug_image, [chest_pts], isClosed=True, color=rectangle_color, thickness=thickness)

    os.makedirs("outputs/keypoints", exist_ok=True)
    out = f"outputs/keypoints/{os.path.basename(path)}"
    cv.imwrite(out, debug_image)
    print(f"Keypoint image saved: {out}")
