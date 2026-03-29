import os
import errno
import tempfile
import cv2 as cv
import numpy as np
import math
import matplotlib.pyplot as plt
import scipy.signal
import scipy.stats

import torch
import torch.nn as nn
import torchvision.models.segmentation as segmentation
import torchvision.models.detection as detection
from scipy.ndimage import gaussian_filter1d


from PIL import Image
from collections import OrderedDict
from torchvision import transforms
from typing import List, Dict, Optional, Tuple
from skimage.feature import hog
from skimage import exposure
from sklearn.decomposition import PCA
from math import atan2, degrees
import dlib 
from sklearn.linear_model import LinearRegression

COCO_PERSON_KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]

OPENPOSE_PERSON_KEYPOINT_NAMES = [
    "nose", "chest",
    "right_shoulder", "right_elbow", "right_wrist",
    "left_shoulder", "left_elbow", "left_wrist",
    "right_hip", "right_knee", "right_ankle",
    "left_hip", "left_knee", "left_ankle",
    "right_eye", "left_eye", "right_ear", "left_ear"
]

# Define common keypoints (only essential ones)
BASE_KEYPOINT_FEATURES = {
    "y_neck": [1, 0],  # Vertical distance from neck to nose
}

# View-specific feature dictionaries
KEYPOINT_FEATURES = {
    "front": {
        **BASE_KEYPOINT_FEATURES,  # Inherit common keypoints
        "head_width": [16, 17],  # Width between ears
        "shoulders": [2, 5],  # Shoulder width
        #legs
        # "right_leg": [8, 9],  # Length of right leg
        # "left_leg": [11, 12],  # Length of left leg
    },
    "left": {
        **BASE_KEYPOINT_FEATURES,  # Inherit only y_neck
        "left_leg": [11, 12],  # Length of left leg
        "angle_shoulder_nose": [5, 0],  # Shoulder-to-nose angle
        "eye_ear": [15, 17],  # Distance between eye and ear (left view specific)
        "nose_ear": [0, 17],  # Distance from nose to ear (left view specific)
    },
    "right": {
        **BASE_KEYPOINT_FEATURES,  # Inherit only y_neck
        "right_leg": [8, 9],  # Length of right leg
        "angle_shoulder_nose": [2, 0],  # Shoulder-to-nose angle
        "neck": [0, 1],  # Distance from nose to neck (only right)
        "eye_ear": [15, 17],  # Distance between eye and ear (right view specific)
        "nose_ear": [0, 17],  # Distance from nose to ear (right view specific)
    }
}

def draw_keypoint_line(
    image: np.array,
    point1: Tuple[int, int, int],
    point2: Tuple[int, int, int],
    color: Tuple[int, int, int],
    thickness: int = 2,
    alpha: float = 1.0,  # Transparency: 1.0 = solid, 0.0 = fully transparent
):
    """
    Draws a line connecting two keypoints on an image, if both are valid.

    - Uses anti-aliasing for smooth lines.
    - Adds optional transparency for overlays.

    Args:
        image (np.array): The image on which to draw.
        point1 (Tuple[int, int, int]): First keypoint (x, y, visibility flag).
        point2 (Tuple[int, int, int]): Second keypoint (x, y, visibility flag).
        color (Tuple[int, int, int]): BGR color of the line (e.g., (255, 0, 0) for blue).
        thickness (int, optional): Line thickness. Defaults to 2.
        alpha (float, optional): Line transparency (1.0 = solid, 0.5 = semi-transparent). Defaults to 1.0.

    Returns:
        np.array: The image with the drawn connection.
    """

    x1, y1, v1 = point1
    x2, y2, v2 = point2

    # Ensure visibility flag is set for both keypoints
    if v1 > 0 and v2 > 0 and min(x1, y1, x2, y2) >= 0:
            overlay = image.copy()  # Create overlay for transparency
            cv.line(overlay, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness, cv.LINE_AA)
            return cv.addWeighted(overlay, alpha, image, 1 - alpha, 0)
    return image

def resize_image(image: np.array, width: int = None, height: int = None, inter: int = cv.INTER_AREA) -> np.array:
    """
        Resizes an image while preserving aspect ratio if only one dimension is given.
        
        - If both `width` and `height` are provided, resizes normally.
        - If only one is provided, the other dimension is **calculated** to maintain aspect ratio.
        - If neither is provided, **returns the original image**.

        Args:
            image (np.array): Input image to resize (supports grayscale or color images).
            width (int, optional): Desired width. Defaults to None.
            height (int, optional): Desired height. Defaults to None.
            inter (int, optional): Interpolation method. Defaults to cv.INTER_AREA.

        Returns:
            np.array: Resized image.
    """

    # Validate input image
    
    #Get instance of image
    if image is None or not isinstance(image, np.ndarray):
        raise ValueError(f"Error: resize_image() expected a NumPy array, but got {type(image)}")

    # Get original dimensions
    (h, w) = image.shape[:2]

    # If neither width nor height is provided, return the original image
    if width is None and height is None:
        return image

    # Compute new dimensions while maintaining aspect ratio
    if width is None:  # Height is provided, calculate width
        aspect_ratio = height / float(h)
        new_dim = (round(w * aspect_ratio), height)
    elif height is None:  # Width is provided, calculate height
        aspect_ratio = width / float(w)
        new_dim = (width, round(h * aspect_ratio))
    else:  # Both width and height are provided, resize normally
        new_dim = (width, height)

    # Resize the image
    return cv.resize(image, new_dim, interpolation=inter)

def find_edge_threshold(pixel_sums: np.array, intensity_threshold: int, consecutive_required: int, from_right: bool = False) -> int:
    """
    Identifies the first position where pixel intensity exceeds a threshold 
    for a required number of consecutive values.

    This helps locate edges in an image, such as detecting the top/bottom 
    of a silhouette.

    Args:
        pixel_sums (np.array): Array representing summed pixel intensities along an axis.
        intensity_threshold (int): Minimum pixel intensity required to be considered an edge.
        consecutive_required (int): Number of consecutive values that must exceed the threshold.
        from_right (bool, optional): If True, searches from right to left. Defaults to False.

    Returns:
        int: Index where the threshold is first exceeded, or None if no edge is found.
    """
    ongoing_count = 0  # Tracks consecutive points above the threshold
    edge_start = 0  # Stores the starting index of the detected edge

    # Iterate over the pixel_sums array
    for idx, value in enumerate(reversed(pixel_sums) if from_right else pixel_sums):
        position = len(pixel_sums) - 1 - idx if from_right else idx

        if value > intensity_threshold:
            if ongoing_count == 0:
                edge_start = position  # Mark where threshold is first exceeded
            ongoing_count += 1
        else:
            ongoing_count = 0  # Reset if consecutive count is broken

        # If enough consecutive values exceed the threshold, return the starting position
        if ongoing_count >= consecutive_required:
            return edge_start

    return None  # Return None if no valid edge is found

def compute_mean_keypoint(keypoints: List[List[int]], indexes: List[int]) -> List[int]:
    """
    Computes the mean of the specified keypoints.

    Args:
        keypoints (List[List[int]]): The detected keypoints in COCO format.
        indexes (List[int]): Indexes of keypoints to be averaged.

    Returns:
        List[int]: The averaged (x, y, p) keypoint or [0, 0, 0] if no valid keypoints exist.
    """
    valid_points = [keypoints[i] for i in indexes if keypoints[i][2] > 0]  # p > 0 means the keypoint is detected
    if valid_points:
        averaged_point = np.mean(valid_points, axis=0, dtype=int).tolist()
        return averaged_point  # Keep (x, y, p)
    return [0, 0, 0]  # Default to missing keypoint

def convert_coco_to_openpose(keypoints: np.array) -> List[List[int]]:
    """
    Convert COCO keypoints to OpenPose format by mapping existing points and adding a chest keypoint.

    Args:
        keypoints (np.array): COCO keypoints (x, y, p) format.

    Returns:
        List[List[int]]: OpenPose keypoints format with the added chest keypoint.
    """
    from_order = COCO_PERSON_KEYPOINT_NAMES
    to_order = OPENPOSE_PERSON_KEYPOINT_NAMES
    
    mapped = [[0, 0, 0]] * len(to_order)

    # Map all available keypoints
    for i, key in enumerate(to_order):
        if key in from_order:
            mapped[i] = keypoints[from_order.index(key)]

    # Compute chest keypoint (midpoint between left and right shoulders)
    left_shoulder = keypoints[from_order.index("left_shoulder")]
    right_shoulder = keypoints[from_order.index("right_shoulder")]

    if left_shoulder[2] > 0 and right_shoulder[2] > 0:  # If both shoulders are detected
        chest_x = int((left_shoulder[0] + right_shoulder[0]) / 2)
        chest_y = int((left_shoulder[1] + right_shoulder[1]) / 2)
        mapped[to_order.index("chest")] = [chest_x, chest_y, 1]  # Add chest keypoint

    return mapped


class HumanBodyAnalyser:
    def __init__(self,
        keypoint_estimator: Optional[nn.Module] = None,
        input_height: int = 600,
        debug_mode: bool = False,
        sam_model: str = None,
        yolo_model: str = None,
        dlib_face_detector_path: str = "mmod_human_face_detector.dat",
        dlib_face_predictor_path: str = "shape_predictor_68_face_landmarks.dat",
        # pca_model: str = None,
    ):
        """
        Initializes the HumanBodyAnalyser with a segmentation and keypoint model.

        Args:
            keypoint_estimator (torch.nn.Module, optional): A keypoint detection model. Defaults to Keypoint R-CNN.
            input_height (int): The target image height for processing.
            sam_model: Loaded SAM model instance.
            yolo_model: Loaded YOLO model instance.
            dlib_face_detector_path (str): Path to mmod_human_face_detector.dat.
                Download from: http://dlib.net/files/mmod_human_face_detector.dat.bz2
            dlib_face_predictor_path (str): Path to shape_predictor_68_face_landmarks.dat.
                Download from: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
        """
        # Assign device (GPU if available)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")  # Log device assignment

        # Initialize keypoint detection model if not provided
        self.keypoint_estimator = (
            keypoint_estimator
            if keypoint_estimator
            else detection.keypointrcnn_resnet50_fpn(pretrained=True)
        )
        self.keypoint_estimator.to(self.device)
        self.keypoint_estimator.eval()  # Set to inference mode

        # Store image processing parameters
        self.input_height = input_height

        # Initialize caching for efficient processing
        self.cache = {"keypoints": {}, "masks": {}, "images": {}}

        # Debugging mode
        self.debug_mode = debug_mode

        # Initialize SAM model
        self.sam_model = sam_model

        # Initialize YOLO model
        self.yolo_model = yolo_model

        if not os.path.exists(dlib_face_detector_path):
            raise FileNotFoundError(
                f"dlib face detector model not found at '{dlib_face_detector_path}'. "
                "Download from: http://dlib.net/files/mmod_human_face_detector.dat.bz2 "
                "and pass the path via --dlib-detector."
            )
        if not os.path.exists(dlib_face_predictor_path):
            raise FileNotFoundError(
                f"dlib shape predictor model not found at '{dlib_face_predictor_path}'. "
                "Download from: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2 "
                "and pass the path via --dlib-predictor."
            )
        self.face_detector = dlib.cnn_face_detection_model_v1(dlib_face_detector_path)
        self.face_predictor = dlib.shape_predictor(dlib_face_predictor_path)

        # # Initialize PCA model
        # self.pca_model = pca_model
    
    def fetch_image(self, path: str) -> np.array:
        """
        Loads an image from the specified path, resizes it, and caches it.
        
        - Uses caching to avoid redundant file reads.
        - Raises a clear error if the image file is not found.

        Args:
            path (str): Path to the image file.

        Returns:
            np.array: Processed image in OpenCV format.
        """

        # If image is already cached, return it
        if path in self.cache["images"]:
            return self.cache["images"][path].copy()

        # Load the image using OpenCV
        image = cv.imread(path)
        if image is None:
            raise FileNotFoundError(f"Error: Image file not found -> {path}")

        # Resize image
        image = resize_image(image, height=self.input_height)

        # Add to cache
        self.cache["images"][path] = image

        return image.copy()

    # def generate_silhouette(self, path: str) -> np.array:
    #     """
    #     Generates a binary mask of the human subject (including held objects like signs) using YOLO and SAM.

    #     - Uses YOLO to detect persons and signs.
    #     - Extracts only the "person" and "sign" class from YOLO results.
    #     - Uses SAM to generate refined masks for detected persons and signs.
    #     - Merges masks into a single silhouette.
    #     - Caches the final mask to avoid recomputation.

    #     Args:
    #         path (str): Path to the image.

    #     Returns:
    #         np.array: Binary mask where `255` = person + sign, `0` = background.
    #     """

    #     # Return cached mask if available
    #     if path in self.cache["masks"]:
    #         return self.cache["masks"][path]


    #     #Use resized image
    #     image = self.fetch_image(path)

    #     #Save temporary resized version (For models that take file paths)
    #     temp_path = "temp_resized_image.jpg"
    #     cv.imwrite(temp_path, image)

    #     # Step 1: Run YOLO to detect objects
    #     yolo_results = self.yolo_model(temp_path,verbose=False)  # Run YOLO on the image
    #     yolo_boxes = yolo_results[0].boxes.data.cpu().numpy()  # Get bounding boxes
    #     yolo_classes = yolo_results[0].names  # Get class names

    #     person_bboxes = [list(box[:4]) for box in yolo_boxes if yolo_classes[int(box[5])] == "person"]
    #     if not person_bboxes:
    #         raise ValueError("No person detected by YOLO.")
        
    #     # Step 2: Find the bounding box with the largest area
    #     largest_bbox = None
    #     largest_area = 0

    #     for bbox in person_bboxes:
    #         # Calculate the area of the bounding box (width * height)
    #         width = bbox[2] - bbox[0]
    #         height = bbox[3] - bbox[1]
    #         area = width * height

    #         # If this box is the largest so far, store it
    #         if area > largest_area:
    #             largest_area = area
    #             largest_bbox = bbox

    #     if largest_bbox is None:
    #         raise ValueError("No valid bounding box found for the largest person.")
        
    #     person_bboxes = [largest_bbox]  # Use only the largest bounding box
        
    #     # Step 1: Run SAM with the YOLO bounding box
    #     sam_yresults = self.sam_model(temp_path, bboxes=person_bboxes,verbose=False)  # Use YOLO bbox in SAM
    #     sam_ymasks = sam_yresults[0].masks
    #     # sam_yresults[0].save("sam_masks_with_yolo.png")

    #     # Step 2: Find the largest mask inside the YOLO bounding box
    #     best_person_mask = None
    #     largest_area = 0
    #     # best_person_mask = (sam_ymasks[0].data.squeeze().cpu().numpy() > 0).astype(np.uint8)

    #     # Only do this to get the Largest SAM MASK if having issues with first one
    #     for mask_obj in sam_ymasks:
    #         mask_tensor = mask_obj.data
    #         mask_np = mask_tensor.squeeze().cpu().numpy()
    #         mask_binary = (mask_np > 0).astype(np.uint8)

    #         mask_area = mask_binary.sum()
    #         if mask_area > largest_area:  # Keep the largest mask
    #             largest_area = mask_area
    #             best_person_mask = mask_binary

    #     if best_person_mask is None:
    #         raise ValueError("SAM failed to generate a valid person mask.")

    #     # Step 3: Expand the best_person_mask (Padding)
    #     kernel = np.ones((100, 100), np.uint8)  # 5x5 kernel for slight expansion
    #     padded_best_person_mask = cv.dilate(best_person_mask, kernel, iterations=2)  # Increase iterations for larger padding

    #     # Initialize output silhouette mask
    #     person_mask = np.zeros(image.shape[:2], dtype=np.uint8)

    #     # Assign to final mask
    #     person_mask[best_person_mask > 0] = 255   

    #     # Step 4: Run SAM again to get all masks
    #     sam_results = self.sam_model(temp_path,verbose=False)  # Run SAM on the image   
    #     sam_masks = sam_results[0].masks
    #     # sam_results[0].save("sam_masks.png")

    #     # Step 5: Find masks inside the best_person_mask
    #     for j, mask_obj in enumerate(sam_masks):
    #         mask_tensor = mask_obj.data
    #         mask_np = mask_tensor.squeeze().cpu().numpy()
    #         mask_binary = (mask_np > 0).astype(np.uint8)

    #         # Get indices of nonzero pixels in the current mask
    #         nonzero_y, nonzero_x = np.where(mask_binary > 0)

    #         # Ensure indices are within image bounds before checking
    #         valid_indices = (nonzero_y < padded_best_person_mask.shape[0]) & (nonzero_x < padded_best_person_mask.shape[1])

    #         # Check if all pixels are inside the padded best_person_mask
    #         if np.all(padded_best_person_mask[nonzero_y[valid_indices], nonzero_x[valid_indices]] == 1):
    #             person_mask[mask_np > 0] = 255  # Add mask to final output
        
    #     # Cache and return the mask
    #     self.cache["masks"][path] = person_mask
    #     print(f"Generated silhouette mask for {path}")
    #     return person_mask

    def generate_silhouette(self, path: str, visualise: bool = False) -> np.array:
        """
        Generates a binary mask of the human subject (including held objects like signs) using YOLO and SAM.

        - Uses YOLO to detect persons and signs.
        - Extracts only the "person" and "sign" class from YOLO results.
        - Uses SAM to generate refined masks for detected persons and signs.
        - Merges masks into a single silhouette.
        - Caches the final mask to avoid recomputation.

        Args:
            path (str): Path to the image.
            visualise (bool): Flag to visualize intermediate results for debugging.

        Returns:
            np.array: Binary mask where `255` = person + sign, `0` = background.
        """
        
        # Return cached mask if available
        if path in self.cache["masks"]:
            return self.cache["masks"][path]

        # Use resized image
        image = self.fetch_image(path)

        # Save temporary resized version (For models that take file paths)
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            temp_path = tmp.name
        cv.imwrite(temp_path, image)

        # Step 1: Run YOLO to detect objects
        yolo_results = self.yolo_model(temp_path, verbose=False)  # Run YOLO on the image
        yolo_boxes = yolo_results[0].boxes.data.cpu().numpy()  # Get bounding boxes
        yolo_classes = yolo_results[0].names  # Get class names

        # Extract bounding boxes for persons
        person_bboxes = [list(box[:4]) for box in yolo_boxes if yolo_classes[int(box[5])] == "person"]
        if not person_bboxes:
            raise ValueError("No person detected by YOLO.")

        # Step 2: Find the bounding box with the largest area
        largest_bbox = None
        largest_area = 0

        for bbox in person_bboxes:
            # Calculate the area of the bounding box (width * height)
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            area = width * height

            # If this box is the largest so far, store it
            if area > largest_area:
                largest_area = area
                largest_bbox = bbox

        if largest_bbox is None:
            raise ValueError("No valid bounding box found for the largest person.")
        
        person_bboxes = [largest_bbox]  # Use only the largest bounding box

        # Step 3: Run SAM with the YOLO bounding box
        sam_yresults = self.sam_model(temp_path, bboxes=person_bboxes, verbose=False)  # Use YOLO bbox in SAM
        sam_ymasks = sam_yresults[0].masks

        # Find the largest mask inside the YOLO bounding box
        best_person_mask = None
        largest_area = 0
        for mask_obj in sam_ymasks:
            mask_tensor = mask_obj.data
            mask_np = mask_tensor.squeeze().cpu().numpy()
            mask_binary = (mask_np > 0).astype(np.uint8)

            mask_area = mask_binary.sum()
            if mask_area > largest_area:  # Keep the largest mask
                largest_area = mask_area
                best_person_mask = mask_binary

        if best_person_mask is None:
            raise ValueError("SAM failed to generate a valid person mask.")

        # Step 4: Expand the best_person_mask (Padding)
        kernel = np.ones((50, 50), np.uint8)  # 75x75 kernel for slight expansion
        padded_best_person_mask = cv.dilate(best_person_mask, kernel, iterations=2)  # Increase iterations for larger padding

        # Initialize output silhouette mask
        person_mask = np.zeros(image.shape[:2], dtype=np.uint8)

        # Assign to final mask
        person_mask[best_person_mask > 0] = 255   

        # Step 5: Run SAM again to get all masks
        sam_results = self.sam_model(temp_path, verbose=False)  # Run SAM on the image   
        sam_masks = sam_results[0].masks

        # Step 6: Find masks inside the best_person_mask
        for j, mask_obj in enumerate(sam_masks):
            mask_tensor = mask_obj.data
            mask_np = mask_tensor.squeeze().cpu().numpy()
            mask_binary = (mask_np > 0).astype(np.uint8)

            # Get indices of nonzero pixels in the current mask
            nonzero_y, nonzero_x = np.where(mask_binary > 0)

            # Ensure indices are within image bounds before checking
            valid_indices = (nonzero_y < padded_best_person_mask.shape[0]) & (nonzero_x < padded_best_person_mask.shape[1])

            # Check if all pixels are inside the padded best_person_mask
            if np.all(padded_best_person_mask[nonzero_y[valid_indices], nonzero_x[valid_indices]] == 1):
                person_mask[mask_np > 0] = 255  # Add mask to final output

        # Debug Visualization
        if visualise:
            os.makedirs("outputs", exist_ok=True)

            # Visualize YOLO bounding boxes on the image
            vis_yolo = image.copy()
            for bbox in person_bboxes:
                cv.rectangle(vis_yolo, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
            cv.imwrite(f"outputs/yolo_bboxes_{os.path.basename(path)}", vis_yolo)

            # Visualize the best SAM mask
            vis_best_mask = image.copy()
            vis_best_mask[best_person_mask == 1] = [255, 0, 0]  # Blue for the best SAM mask
            cv.imwrite(f"outputs/best_sam_mask_{os.path.basename(path)}", vis_best_mask)


            # Visualize the padded SAM mask
            vis_mask = image.copy()
            vis_mask[padded_best_person_mask == 1] = [0, 0, 255]  # Red for the padded mask
            cv.imwrite(f"outputs/padded_sam_mask_{os.path.basename(path)}", vis_mask)

            # Visualize the final result (including areas inside the mask)
            vis_final = image.copy()
            vis_final[person_mask == 255] = [0, 255, 0]  # Green for the final person mask
            cv.imwrite(f"outputs/final_mask_{os.path.basename(path)}", vis_final)

            binary_mask = (person_mask > 0).astype(np.uint8)  # Convert to binary mask
            # Save the final silhouette mask
            cv.imwrite(f"outputs/silhouette_mask_{os.path.basename(path)}", binary_mask * 255)

        # Clean up temp file
        try:
            os.remove(temp_path)
        except OSError:
            pass

        # Cache and return the mask
        self.cache["masks"][path] = person_mask
        print(f"Generated silhouette mask for {path}")
        return person_mask

    def get_height(self, path: str):
        """
        Computes the height of the silhouette by detecting the highest (top) 
        and lowest (bottom) non-zero pixel rows in the binary mask.

        - Uses `find_edge_threshold` to locate the top and bottom edges.
        - Returns `None` if no valid silhouette is detected.
        
        Args:
            path (str): Path to the image.
            debug (bool, optional): If True, visualizes detected edges.

        Returns:
            int or None: Height of the silhouette (y1 - y0), or None if not detected.
        """
        mask = self.generate_silhouette(path)  # Get the silhouette mask
        pixel_sums = np.sum(mask != 0, axis=1)  # Count non-zero pixels per row

        # Find the top and bottom edges of the silhouette
        y0 = find_edge_threshold(pixel_sums, intensity_threshold=1, consecutive_required=20)
        y1 = find_edge_threshold(pixel_sums, intensity_threshold=1, consecutive_required=20, from_right=True)

        if y0 is None or y1 is None:
            if self.debug_mode:
                print(f"⚠ Warning: Height calculation failed for {path}. Detected top: {y0}, bottom: {y1}")
            #Raise exception if top or bottom edge is not detected
            raise ValueError(f"Error: Height calculation failed for {path}. Detected top: {y0}, bottom: {y1}")

        height = y1 - y0  # Compute silhouette height

        # Optional Debugging: Show edges on the mask
        if self.debug_mode:
            # Print the height of the silhouette
            print(f"Height of the silhouette: {height} pixels")

            # Convert mask to a 3-channel (RGB) image for colored visualization
            debug_mask = cv.cvtColor(mask.copy(), cv.COLOR_GRAY2BGR)

            # Mark the top and bottom edges in gray
            debug_mask[y0:y0+2, :] = (128, 128, 128)  # Top edge
            debug_mask[y1:y1+2, :] = (128, 128, 128)  # Bottom edge

            # Find the center X-coordinate of the silhouette
            center_x = mask.shape[1] // 2

            # Draw a vertical red line from y0 (top) to y1 (bottom)
            cv.line(debug_mask, (center_x, y0), (center_x, y1), (0, 0, 255), 2)

            # Save the debug image
            cv.imwrite(f"debug_silhouette_height_{os.path.basename(path)}.png", debug_mask)


        return height

    def extract_keypoints(self, path: str) -> List[np.array]:
        """
        Extracts keypoints from an image using a pretrained Keypoint R-CNN model.

        - Returns COCO 18-point keypoints format: (x, y, p), where:
            - `x, y` → Keypoint coordinates
            - `p` → Visibility confidence score
        - Caches results to avoid redundant processing.
        - Handles multiple people but only returns the first detected person.
        - Uses GPU acceleration.

        Args:
            path (str): Path to the input image.

        Returns:
            List[np.array]: List of 18 keypoints for the detected person.
        """
        
        # Return cached keypoints if available
        if path in self.cache["keypoints"]:
            return self.cache["keypoints"][path]

        # Load and preprocess the image
        image = self.fetch_image(path)
        image = Image.fromarray(cv.cvtColor(image, cv.COLOR_BGR2RGB))

        preprocess = transforms.Compose([transforms.ToTensor()])
        input_tensor = preprocess(image).unsqueeze(0).to(self.device)  # Move to GPU/CPU

        # Perform inference
        with torch.no_grad():
            output = self.keypoint_estimator(input_tensor)
            keypoints = output[0]["keypoints"]

        # Handle cases with no detection or multiple people
        if keypoints.shape[0] == 0:
            raise ValueError(f"Keypoint detection failed: No person detected in {path}.")
        elif keypoints.shape[0] > 1:
            print(f"⚠ Warning: Multiple people detected in {path}. Using the first person.")

        # Convert to OpenPose format
        keypoints_np = keypoints[0].cpu().numpy()
        openpose_keypoints = convert_coco_to_openpose(keypoints_np)
        self.cache["keypoints"][path] = openpose_keypoints


        if self.debug_mode:
            debug_image = image.copy()
            debug_image = np.array(debug_image)
            debug_image = cv.cvtColor(debug_image, cv.COLOR_RGB2BGR)  # Convert to BGR for OpenCV

            # Define colors and parameters
            point_color = (0, 255, 0)  # Green keypoints
            line_color = (255, 0, 0)   # Blue lines for skeleton
            rectangle_color = (0, 255, 255)  # Yellow for chest rectangle
            radius = 5
            thickness = 2

            # Draw keypoints
            for i, (x, y, p) in enumerate(openpose_keypoints):
                if p > 0:  # Only draw visible keypoints
                    cv.circle(debug_image, (int(x), int(y)), radius, point_color, -1)

            # Draw skeleton connections (EXCLUDING chest-to-hips and chest-to-shoulders)
            skeleton_pairs = [
                (2, 3), (3, 4),  # Right arm
                (5, 6), (6, 7),  # Left arm
                (8, 9), (9, 10),  # Right leg
                (11, 12), (12, 13),  # Left leg
                (0, 1),  # Connect nose to chest

                # Face keypoints
                (14, 16),  # Right eye to right ear
                (15, 17),  # Left eye to left ear
                (0, 14),  # Nose to right eye
                (0, 15)   # Nose to left eye
            ]
            
            for i, j in skeleton_pairs:
                if openpose_keypoints[i][2] > 0 and openpose_keypoints[j][2] > 0:
                    cv.line(debug_image, (int(openpose_keypoints[i][0]), int(openpose_keypoints[i][1])),
                            (int(openpose_keypoints[j][0]), int(openpose_keypoints[j][1])), line_color, thickness)

            # Draw chest rectangle (shoulders to hips)
            chest_idx = 1
            left_shoulder = 5
            right_shoulder = 2
            left_hip = 11
            right_hip = 8

            if (openpose_keypoints[chest_idx][2] > 0 and openpose_keypoints[left_shoulder][2] > 0 and
                    openpose_keypoints[right_shoulder][2] > 0 and openpose_keypoints[left_hip][2] > 0 and
                    openpose_keypoints[right_hip][2] > 0):
                chest_pts = np.array([
                    [openpose_keypoints[left_shoulder][0], openpose_keypoints[left_shoulder][1]],
                    [openpose_keypoints[right_shoulder][0], openpose_keypoints[right_shoulder][1]],
                    [openpose_keypoints[right_hip][0], openpose_keypoints[right_hip][1]],
                    [openpose_keypoints[left_hip][0], openpose_keypoints[left_hip][1]]
                ], np.int32)
                cv.polylines(debug_image, [chest_pts], isClosed=True, color=rectangle_color, thickness=thickness)

            # Save the debug image
            debug_path = f"debug_keypoints_{os.path.basename(path)}.png"
            cv.imwrite(debug_path, debug_image)
            print(f"✅ Debug image saved: {debug_path}")


        return openpose_keypoints

    def get_view_direction(self, path: str) -> str:
        """
        Determines the viewing direction of a person using keypoints.

        - If both shoulders are detected and far apart, the person is facing **front**.
        - If shoulders are close together, the direction is determined by the **relative position of the nose to the neck**.

        Args:
            path (str): Path to the image.

        Returns:
            str: One of {"front", "left", "right"} indicating the viewing direction.
        """
        keypoints = self.extract_keypoints(path)

        # Extract keypoint confidence and positions
        right_shoulder = keypoints[2]
        left_shoulder = keypoints[5]
        nose = keypoints[0]
        neck = keypoints[1]  # Midpoint between shoulders (computed in OpenPose format)

        # Check if both shoulders are detected
        if right_shoulder[2] > 0 and left_shoulder[2] > 0:
            shoulder_distance = np.linalg.norm(np.array(right_shoulder[:2]) - np.array(left_shoulder[:2]))

            if shoulder_distance >= 50:  # Threshold for determining front view
                return "front"

        # If shoulders are close, determine direction based on nose position
        if nose[2] > 0 and neck[2] > 0:
            return "left" if nose[0] > neck[0] else "right"

        # Default fallback in case of missing keypoints
        return "unknown"

    def extract_measurements(self, path: str, visualize: bool = False) -> Tuple[Dict[str, float], np.array]:
        """
        Computes keypoint measurements including distances and angles, and optionally visualizes them.

        Args:
            path (str): Path to the image.
            visualize (bool): If True, draws the measurements on the image.

        Returns:
            Tuple[Dict[str, float], np.array]: 
                - Dictionary of computed keypoint features.
                - Image with drawn keypoint measurements (if visualize=True).
        """
        image = self.fetch_image(path).copy()
        keypoints = self.extract_keypoints(path)
        direction = self.get_view_direction(path)

        distances = OrderedDict()

        colors = {
            "angle": (255, 0, 0),         # Blue for angles
            "y": (255, 255, 255),         # White for horizontal
            "y_vertical": (255, 0, 255),  # Magenta for vertical
            "default": (0, 0, 255),       # Red for other distances
            "angle_line": (0, 255, 255) ,  # Yellow for custom angle lines
            "leg": (0, 255, 0)            # Green for leg lines
        }

        def calculate_distance(i1, i2, metric):
            (x1, y1, p1), (x2, y2, p2) = keypoints[i1], keypoints[i2]
            if p1 == 0 or p2 == 0:
                return None
            if metric == "angle":
                return np.degrees(np.arctan2(y2 - y1, x2 - x1))
            elif metric == "y":
                return abs(y1 - y2)
            elif metric == "x":
                return abs(x1 - x2)
            return np.linalg.norm([x2 - x1, y2 - y1])

        # --- Standard Measurements from KEYPOINT_FEATURES ---
        for measurement, keys in KEYPOINT_FEATURES[direction].items():

            metric_type = measurement.split("_")[0]
            total_distance = 0

            for i1, i2 in zip(keys, keys[1:]):
                distance = calculate_distance(i1, i2, metric_type)
                if distance is None:
                    distances[measurement] = None
                    break
                total_distance += distance

                if visualize:
                    color = colors.get(metric_type, colors["default"])

                    if metric_type == "y":
                        image = draw_keypoint_line(image, (keypoints[i1][0], keypoints[i1][1], 1),
                                                (keypoints[i2][0], keypoints[i1][1], 1), colors["y"], 1)
                        image = draw_keypoint_line(image, (keypoints[i2][0], keypoints[i1][1], 1),
                                                keypoints[i2], colors["y_vertical"], 2)
                    else:
                        image = draw_keypoint_line(image, keypoints[i1], keypoints[i2], color, 2)

            distances[measurement] = total_distance

        return distances, image

    def extract_keypoint_ratios(self, path: str) -> Dict[str, float]:
        """
        Computes normalized keypoint ratios from distances (excluding angles), 
        dividing by silhouette height for scale invariance.

        Args:
            path (str): Path to the image.

        Returns:
            Dict[str, float]: Dictionary of normalized distance-based keypoint ratios.
        """
        try:
            height = self.get_height(path)
            measurements, _ = self.extract_measurements(path)
        except Exception as e:
            print(f"⚠ Error extracting ratios for {path}: {e}")
            return {}
        
        #Pre-fetch normalizers if available
        shoulder_width = measurements.get("shoulders", None)
        head_width = measurements.get("head_width", None)
        # print(f"Shoulder Width: {shoulder_width}, Head Width: {head_width}")

        ratios = {}
        for key, value in measurements.items():
            # Ignore angle-based measurements (e.g., those prefixed with 'angle_')
            if value is None or key.startswith("angle_"):
                continue

            # Choose normalization strategy
            # if key in ["y_neck", "eye_ear", "nose_ear"] and head_width is not None:
            #     normalizer = head_width
            # else:
            normalizer = height

            if normalizer and normalizer > 0:
                ratios[key + "_ratio"] = np.float32(value / normalizer)
            else:
                ratios[key + "_ratio"] = None

        return {k: v for k, v in ratios.items() if v is not None}

    def extract_angles(self, path: str, visualize: bool = False):
        image = self.fetch_image(path).copy()
        keypoints = self.extract_keypoints(path)
        angles = OrderedDict()
        colors = {
            "angle": (255, 0, 0),         # Blue for angles
            "y": (255, 255, 255),         # White for horizontal
            "y_vertical": (255, 0, 255),  # Magenta for vertical
            "default": (0, 0, 255),       # Red for other distances
            "angle_line": (0, 255, 255) ,  # Yellow for custom angle lines
            "leg": (0, 255, 0)            # Green for leg lines
        }

        # --- Angle Utility Function ---
        def compute_angle(p1, p2, p3):
            if any(k[2] == 0 for k in (p1, p2, p3)):
                return None
            a = np.array([p1[0] - p2[0], p1[1] - p2[1]])
            b = np.array([p3[0] - p2[0], p3[1] - p2[1]])
            
            norm_a = np.linalg.norm(a)
            norm_b = np.linalg.norm(b)
            
            # Prevent division by zero
            if norm_a < 1e-6 or norm_b < 1e-6:
                return None

            dot = np.dot(a, b)
            cos_angle = np.clip(dot / (norm_a * norm_b), -1.0, 1.0)
            angle = np.degrees(np.arccos(cos_angle))
            return angle


        # --- Custom Angle-Based Features ---
        angle_features = {
            "elbow_left_angle": (5, 6, 7),     # shoulder, elbow, wrist
            "elbow_right_angle": (2, 3, 4),
            "shoulder_slope": (2, 1, 5),       # right_shoulder, chest, left_shoulder
            "torso_lean": (8, 1, 11),          # right_hip, chest, left_hip
            "head_tilt": (14, 0, 15),          # right_eye, nose, left_eye
        }

        for name, (i1, i2, i3) in angle_features.items():
            angle = compute_angle(keypoints[i1], keypoints[i2], keypoints[i3])
            if angle is None:
                angle = 0
            angles[name] = angle

            if visualize and angle is not None:
                image = draw_keypoint_line(image, keypoints[i2], keypoints[i1], colors["angle_line"], 2)
                image = draw_keypoint_line(image, keypoints[i2], keypoints[i3], colors["angle_line"], 2)
                cv.putText(image, f"{int(angle)}°",
                        (int(keypoints[i2][0]), int(keypoints[i2][1]) - 10),
                        cv.FONT_HERSHEY_SIMPLEX, 0.4, colors["angle_line"], 1, cv.LINE_AA)
        
        # --- LEG SPREAD ANGLE (Corrected) ---
        left_hip, right_hip = keypoints[11], keypoints[8]
        left_knee, right_knee = keypoints[12], keypoints[9]

        if all(k[2] > 0 for k in [left_hip, right_hip, left_knee, right_knee]):
            hip_center = 0.5 * (np.array(left_hip[:2]) + np.array(right_hip[:2]))
            left_vec = np.array(left_knee[:2]) - hip_center
            right_vec = np.array(right_knee[:2]) - hip_center

            dot = np.dot(left_vec, right_vec)
            norm_product = np.linalg.norm(left_vec) * np.linalg.norm(right_vec)
            leg_angle = np.degrees(np.arccos(np.clip(dot / norm_product, -1.0, 1.0)))
            angles["leg_spread_angle"] = leg_angle

            if visualize:
                green = (0, 255, 0)
                # Draw from hip center to knees
                cv.line(image, tuple(map(int, hip_center)), tuple(map(int, left_knee[:2])), green, 2)
                cv.line(image, tuple(map(int, hip_center)), tuple(map(int, right_knee[:2])), green, 2)
                cv.putText(image, f"{leg_angle:.1f}°", (int(hip_center[0]), int(hip_center[1]) - 10),
                        cv.FONT_HERSHEY_SIMPLEX, 0.6, green, 2)
        else:
            angles["leg_spread_angle"] = 0

        return angles, image

    def extract_contour(self, path: str, head: bool = False):
        """
        Extracts the largest contour from a binary mask of either:
        - The entire body silhouette (`mask(path)`) OR
        - Just the head (`head_mask(path)`, if `head=True`).

        Args:
            path (str): Path to the input image.
            head (bool, optional): If True, extracts only the head contour. Defaults to False.

        Returns:
            np.array or None: The largest contour (Nx2 array of (x, y) points) or None if no contour found.
        """
        mask = self.extract_head_mask(path) if head else self.generate_silhouette(path)
        # mask = self.generate_silhouette(path)
        # print(f"Extracted Silhoette: {mask}")
        np.savetxt("silhoette.txt", mask, fmt="%d", delimiter=",") 

        # Ensure mask is valid
        if mask is None or np.count_nonzero(mask) == 0:
            print("⚠ Warning: No valid mask found, returning None.")
            return None

        contours, hierarchy = cv.findContours(
            mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE
        )

        if not contours:  # No contours found
            print("⚠ Warning: No contours detected, returning None.")
            return None

        # Extract the largest contour
        largest_contour = max(contours, key=cv.contourArea)  # Use contourArea instead of len()
        return np.squeeze(largest_contour)
    
    def extract_fneck_points(self, path: str, plot: bool = False):
        keypoints = self.extract_keypoints(path)
        contour = self.extract_contour(path)
        direction = self.get_view_direction(path)
        if direction in ["left", "right"]:
            left_mask = contour[:, 0] <= keypoints[1][0]
            right_mask = contour[:, 0] > keypoints[1][0]
            kpl, kpr = 1, 1
            ax = 1
        elif direction in ["front"]:
            left_mask = contour[:, 1] <= keypoints[2][1]
            right_mask = contour[:, 1] <= keypoints[5][1]
            kpl, kpr = 2, 5
            ax = 0
        else:
            raise NotImplementedError("Direction not supported")
        # Start at the closest Y position on each side
        # Use flipped array for left points so can move up by increasing index
        index_left = np.argmin(np.abs(contour[left_mask, ax] - keypoints[kpl][ax])) - 1
        index_left = np.flip(np.arange(contour.shape[0]))[left_mask][index_left]
        points_left = np.flip(contour, axis=0)
        # ---
        index_right = np.argmin(np.abs(contour[right_mask, ax] - keypoints[kpr][ax]))
        index_right = np.arange(contour.shape[0])[right_mask][index_right]
        points_right = contour
        # Nudge index back in case on turning point
        nudge = 10
        index_left -= nudge
        index_right -= nudge
        # Find curving point of neck
        peaks_left = scipy.signal.find_peaks(points_left[:, 0])[0]
        # Negate X dimension of right points so can find peak instead of trough
        peaks_right = scipy.signal.find_peaks(-points_right[:, 0])[0]
        finish_index_left, finish_index_right = -1, -1
        # Sorting function to handle wrapping of peaks
        if len(peaks_left) > 0:
            key = lambda x: x if x > index_left else x + len(points_left)  # noqa
            finish_index_left = sorted(peaks_left, key=key)[0]
        if len(peaks_right) > 0:
            key = lambda x: x if x > index_right else x + len(points_right)  # noqa
            finish_index_right = sorted(peaks_right, key=key)[0]

        if plot:
            plt.figure("Neck Left")
            plt.plot(points_left[:, 0])
            plt.scatter(peaks_left, points_left[peaks_left, 0], color="r")
            plt.scatter(index_left, points_left[index_left, 0], color="b")
            plt.scatter(finish_index_left, points_left[finish_index_left, 0], color="g")
            plt.figure("Neck Right")
            plt.plot(points_right[:, 0])
            plt.scatter(index_right, points_right[index_right, 0], color="b")
            plt.scatter(peaks_right, points_right[peaks_right, 0], color="r")
            plt.scatter(
                finish_index_right, points_right[finish_index_right, 0], color="g"
            )

        # Got finish point
        finish_point_left = points_left[finish_index_left]
        finish_point_right = points_right[finish_index_right]

        if plot:
            image = self.fetch_image(path).copy()
            max_x, max_y = image.shape[1], image.shape[0]
            contour = np.array(
                [
                    finish_point_left,
                    finish_point_right,
                    (max_x, finish_point_right[1]),
                    (max_x, max_y),
                    (0, max_y),
                    (0, finish_point_left[1]),
                ]
            )
            cv.drawContours(image, [contour], -1, (0, 0, 255), 1)
            cv.circle(
                image, tuple(points_left[index_left]), 3, (0, 0, 255), -1, cv.LINE_AA
            )
            cv.circle(
                image, tuple(points_right[index_right]), 3, (0, 0, 255), -1, cv.LINE_AA
            )
            cv.circle(image, tuple(finish_point_left), 3, (0, 255, 0), -1, cv.LINE_AA)
            cv.circle(image, tuple(finish_point_right), 3, (0, 255, 0), -1, cv.LINE_AA)
            plt.figure()
            plt.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))
            #Save the image
            plt.savefig("Neck_Curvature.png")

        return (finish_point_left, finish_point_right)
    
    def extract_sneck_points(self, path: str, plot: bool = False):
        keypoints = self.extract_keypoints(path)
        contour = self.extract_contour(path)
        direction = self.get_view_direction(path)

        chest_x, chest_y = keypoints[1][:2]
        vertical_margin = 50
        y_nudge = -12  # upward adjustment

        # --- Slice contour near chest height ---
        vertical_mask = (contour[:, 1] >= chest_y - vertical_margin) & (contour[:, 1] <= chest_y + vertical_margin)
        neck_region = contour[vertical_mask]

        if neck_region.size == 0:
            print(f"Warning: No contour found near neck for {path}")
            return keypoints[1][:2], keypoints[1][:2]

        if direction in ["left", "right"]:
            left_side = neck_region[neck_region[:, 0] <= chest_x]
            right_side = neck_region[neck_region[:, 0] > chest_x]

            if left_side.size > 0:
                left_y_diff = np.abs(left_side[:, 1] - chest_y)
                point_left = left_side[np.argmin(left_y_diff)]
            else:
                point_left = keypoints[1][:2]

            if right_side.size > 0:
                right_y_diff = np.abs(right_side[:, 1] - chest_y)
                point_right = right_side[np.argmin(right_y_diff)]
            else:
                point_right = keypoints[1][:2]

        elif direction == "front":
            shoulder_left = keypoints[5]
            shoulder_right = keypoints[2]

            min_x = min(shoulder_left[0], shoulder_right[0])
            max_x = max(shoulder_left[0], shoulder_right[0])
            horizontal_mask = (neck_region[:, 0] >= min_x) & (neck_region[:, 0] <= max_x)
            region = neck_region[horizontal_mask]

            if region.size == 0:
                print(f"Warning: No front neck points found for {path}")
                return keypoints[1][:2], keypoints[1][:2]

            point_left = region[np.argmin(region[:, 0])]
            point_right = region[np.argmax(region[:, 0])]

        else:
            raise NotImplementedError("Unsupported view direction.")

        # --- Nudge points upward to hit base of neck ---
        point_left[1] += y_nudge
        point_right[1] += y_nudge

        # --- Plot ---
        if plot:
            image = self.fetch_image(path).copy()
            cv.circle(image, tuple(point_left), 5, (0, 255, 0), -1)
            cv.circle(image, tuple(point_right), 5, (0, 0, 255), -1)
            cv.line(image, tuple(point_left), tuple(point_right), (255, 0, 0), 2)

            os.makedirs("outputs/neck_points_debug", exist_ok=True)
            save_path = os.path.join("outputs/neck_points_debug", os.path.basename(path))
            cv.imwrite(save_path, image)
            print(f"Saved neck plot to {save_path}")

        return point_left, point_right

    def extract_head_mask(self, path: str) -> np.ndarray:
        """
        Extracts a binary mask of the head region based on the silhouette contour and neck curvature points.

        Args:
            path (str): Path to the image.

        Returns:
            np.ndarray: Binary mask of the head region (255 = head, 0 = background).
        """
        head_mask = np.zeros_like(self.generate_silhouette(path))
        contour = self.extract_contour(path)
        #Check view direction
        direction = self.get_view_direction(path)
        if direction in ["left", "right"]:
            neck_left, neck_right = self.extract_fneck_points(path)
        elif direction == "front":
            neck_left, neck_right = self.extract_fneck_points(path)

        # Find closest contour points to neck markers
        left_index = np.linalg.norm(contour - neck_left, axis=1).argmin()
        right_index = np.linalg.norm(contour - neck_right, axis=1).argmin()

        # Ensure indices are ordered correctly
        if left_index > right_index:
            left_index, right_index = right_index, left_index

        # Remove contour points between the neck points (below head)
        keep_mask = np.ones(contour.shape[0], dtype=bool)
        keep_mask[left_index + 1 : right_index] = False
        head_contour = contour[keep_mask]

        # Draw and return head region mask
        cv.drawContours(head_mask, [head_contour], -1, color=255, thickness=cv.FILLED)
        return head_mask  

    def extract_hog_features(self, path: str, visualize: bool = False) -> np.ndarray:
        """
        Extracts HOG (Histogram of Oriented Gradients) features from the silhouette binary mask.

        Args:
            path (str): Path to the image.
            visualize (bool): If True, returns a visualization of the HOG image.

        Returns:
            np.ndarray: 1D HOG feature vector (optionally also returns the HOG image if visualize=True).
        """
        # 1. Get the silhouette mask (binary image)
        mask = self.generate_silhouette(path)  # 255 = foreground
        # mask = self.extract_head_mask(path)

        # 2. Normalize to [0, 1] for HOG
        binary_mask = (mask > 0).astype(float)

        # 3. Extract HOG
        features, hog_image = hog(
            binary_mask,
            orientations=9,
            pixels_per_cell=(16, 16),
            cells_per_block=(2, 2),
            block_norm='L2-Hys',
            visualize=True,
            feature_vector=True
        )

        if visualize:
            import matplotlib.pyplot as plt
            plt.imshow(exposure.rescale_intensity(hog_image, in_range=(0, 10)), cmap='gray')
            plt.title("HOG from Silhouette")
            plt.axis("off")
            #Save the image
            plt.savefig("HOG_Silhouette.png")

        return features

    def extract_feature_vector(self, path: str, features_to_include=None):
        features_to_include = features_to_include or []
        features = []

        if "measurements" in features_to_include:
            measurements, _ = self.extract_measurements(path)
            features.extend([v if v is not None else 0 for v in measurements.values()])

        if "silhouette_height" in features_to_include:
            features.append(self.get_height(path))

        if "hu_silhouette" in features_to_include:
            silhouette = self.generate_silhouette(path)
            features.extend(self._compute_hu_moments(silhouette))

        if "angles" in features_to_include:
            angles, _ = self.extract_angles(path)
            features.extend([v if v is not None else 0 for v in angles.values()])

        if "hu_head" in features_to_include:
            head_mask = self.extract_head_mask(path)
            # head_mask = self.generate_head_mask_dlib(path)
            features.extend(self._compute_hu_moments(head_mask))

        if "hog" in features_to_include:
            hog_features = self.extract_hog_features(path)
            features.extend(hog_features)
        
        if "ratios" in features_to_include:
            ratios = self.extract_keypoint_ratios(path)
            features.extend([v if v is not None else 0 for v in ratios.values()])

        if "pca" in features_to_include:
            pca_features = self.extract_pca_features(path)
            features.extend([v if v is not None else 0 for v in pca_features])

        if "raycast" in features_to_include:
            raycast_signature = self.extract_rotation_corrected_raycast(path)
            features.extend([v if v is not None else 0 for v in raycast_signature])
        
            
        return np.array(features, dtype=np.float64)


    def extract_raycast_signature(self, path: str, num_rays: int = 64, normalize: bool = True, visualize: bool = False) -> np.ndarray:
        """
        Extracts a raycasting signature from the silhouette of a person.

        Args:
            path (str): Path to the image.
            num_rays (int): Number of rays to cast from the center.
            normalize (bool): Normalize distances to unit scale.
            visualize (bool): If True, saves debug image with rays drawn.

        Returns:
            np.ndarray: Signature vector of distances from centroid to contour along each ray.
        """
        mask = self.generate_silhouette(path)
        h, w = mask.shape

        # Get center of mass
        moments = cv.moments(mask)
        if moments["m00"] == 0:
            return np.zeros(num_rays)
        cx = int(moments["m10"] / moments["m00"])
        cy = int(moments["m01"] / moments["m00"])

        angles = np.linspace(0, 2 * np.pi, num_rays, endpoint=False)
        max_radius = int(np.hypot(h, w))
        signature = []

        endpoints = []  # for visualization
        for theta in angles:
            for r in range(1, max_radius):
                x = int(cx + r * np.cos(theta))
                y = int(cy + r * np.sin(theta))

                if x < 0 or x >= w or y < 0 or y >= h:
                    signature.append(r)
                    endpoints.append((x, y))
                    break
                if mask[y, x] == 0:
                    signature.append(r)
                    endpoints.append((x, y))
                    break
            else:
                signature.append(max_radius)
                endpoints.append((x, y))

        signature = np.array(signature, dtype=np.float32)
        if normalize and signature.max() > 0:
            signature /= signature.max()

        # --- Visualization ---
        if visualize:
            vis = cv.cvtColor(mask.copy(), cv.COLOR_GRAY2BGR)
            for (x, y) in endpoints:
                cv.line(vis, (cx, cy), (x, y), (0, 255, 0), 1)
            cv.circle(vis, (cx, cy), 2, (0, 0, 255), -1)

            os.makedirs("outputs/raycast_debug", exist_ok=True)
            out_path = os.path.join("outputs/raycast_debug", os.path.basename(path))
            cv.imwrite(out_path, vis)
            print(f"✅ Saved raycast visualisation to: {out_path}")

        return signature

    def extract_pca_features(self, path: str, n_components: int = 2, visualise: bool = False) -> np.ndarray:
        """
        Extract PCA-based features from the silhouette contour of a person and optionally visualize.

        Args:
            path (str): Path to the image.
            n_components (int): Number of principal components to retain.
            visualise (bool): If True, generates a plot with PCA axes overlaid on contour.

        Returns:
            np.ndarray: PCA variance ratios (or zeros if not enough data).
        """
        contour = self.extract_contour(path)  # Shape (N, 2)

        if contour is None or len(contour) < n_components:
            print(f"⚠ Not enough contour points for PCA in {path}")
            return np.zeros(n_components)

        # Center and normalize contour
        mean = np.mean(contour, axis=0)
        contour_centered = contour - mean
        contour_normalized = contour_centered / np.std(contour_centered)

        # Fit PCA
        pca = PCA(n_components=n_components)
        pca.fit(contour_normalized)

        # --- Optional Visualization ---
        if visualise:
            # Base image
            img_base = self.fetch_image(path).copy()
            img_base = cv.cvtColor(img_base, cv.COLOR_RGB2BGR)

            # --- Contour-Only Image ---
            contour_img = img_base.copy()
            for point in contour.astype(int):
                cv.circle(contour_img, tuple(point), 1, (0, 255, 0), -1)

            # --- PCA-Overlaid Image ---
            pca_img = contour_img.copy()
            scale = 100  # for visualization length
            for i in range(min(2, n_components)):  # only draw first two components
                vec = pca.components_[i]
                start = tuple(mean.astype(int))
                end = tuple((mean + vec * scale).astype(int))
                color = (0, 0, 255) if i == 0 else (255, 0, 0)
                cv.arrowedLine(pca_img, start, end, color, 2, tipLength=0.1)

            # --- Save Both Images ---
            os.makedirs("outputs/pca_debug", exist_ok=True)
            filename = os.path.splitext(os.path.basename(path))[0]
            contour_path = os.path.join("outputs/pca_debug", f"{filename}_contour.png")
            pca_path = os.path.join("outputs/pca_debug", f"{filename}_pca.png")

            cv.imwrite(contour_path, contour_img)
            cv.imwrite(pca_path, pca_img)
            print(f"✅ Saved contour to: {contour_path}")
            print(f"✅ Saved PCA overlay to: {pca_path}")
        

        return pca.explained_variance_ratio_

    def generate_head_mask_dlib(self, path: str, visualise: bool = False) -> np.ndarray:
        """
        Crops the head above the jawline using Dlib landmarks and silhouette.
        Returns a binary mask with the head region = 1, everything else = 0.
        """
        # Load image + silhouette
        image = self.fetch_image(path)
        silhouette_mask = self.generate_silhouette(path)
        silhouette_mask = (silhouette_mask > 0).astype(np.uint8)

        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

        # Detect face
        faces = self.face_detector(gray, 1)
        if len(faces) == 0:
            print(f"No faces found in {path}")
            return np.zeros_like(silhouette_mask, dtype=np.uint8)

        face = faces[0].rect if hasattr(faces[0], "rect") else faces[0]
        shape = self.face_predictor(gray, face)

        # Get jawline points
        jaw = np.array([(shape.part(i).x, shape.part(i).y) for i in range(17)], dtype=np.int32)

        # Find the lowest jaw point (usually index 8)
        lowest_jaw_point_y = jaw[8][1]
        lowest_jaw_point_y += 5

        # Clip everything below the lowest jawline point (horizontal line)
        h, w = silhouette_mask.shape
        mask_to_zero = np.zeros_like(silhouette_mask, dtype=np.uint8)

        # Set the horizontal line at the lowest jawline point
        mask_to_zero[lowest_jaw_point_y:, :] = 1  # Everything below the lowest jaw point

        # Crop the head
        head_only = silhouette_mask.copy()
        head_only[mask_to_zero == 1] = 0

        if visualise:
            os.makedirs("outputs", exist_ok=True)

            # Save silhouette
            cv.imwrite(f"outputs/silhouette_mask_{os.path.basename(path)}", silhouette_mask * 255)

            # Save jaw mask
            mask_vis = np.stack([silhouette_mask * 255] * 3, axis=-1)
            cv.polylines(mask_vis, [jaw], isClosed=False, color=(0, 0, 255), thickness=2)
            cv.fillPoly(mask_vis, [jaw], color=(0, 0, 128))
            cv.imwrite(f"outputs/vis_clip_polygon_{os.path.basename(path)}", mask_vis)

            # Save landmarks
            vis = image.copy()
            for i in range(68):
                x, y = shape.part(i).x, shape.part(i).y
                cv.circle(vis, (x, y), 2, (0, 255, 0), -1)
            cv.polylines(vis, [jaw], isClosed=False, color=(0, 0, 255), thickness=2)
            cv.imwrite(f"outputs/vis_head_landmarks_{os.path.basename(path)}", vis)

            # Final output
            cv.imwrite(f"outputs/mask_head_{os.path.basename(path)}", head_only * 255)

        return head_only

    # def extract_rotation_corrected_raycast(self, path: str, num_rays: int = 360, normalize: bool = True, visualise: bool = False) -> np.ndarray:
    #     # Fetch the base image and the head mask
    #     base_img = self.fetch_image(path)
    #     head_mask = self.extract_head_mask(path)
    #     # head_mask = self.generate_head_mask_dlib(path)
        
    #     if head_mask is None or np.count_nonzero(head_mask) < 10:
    #         return np.zeros(num_rays)

    #     # Detect keypoints (landmarks)
    #     keypoints = self.extract_keypoints(path)
    #     if not keypoints or len(keypoints) < 18:
    #         return np.zeros(num_rays)

    #     # Utility function to safely get keypoints
    #     def safe_get(k):
    #         x, y, p = keypoints[k]
    #         return (x, y) if p > 0 else (0, 0)

    #     # Determine the view direction (either 'side' or 'front')
    #     view_direction = self.get_view_direction(path)
    #     view_direction = "side" if view_direction in ["left", "right"] else "front"

    #     # Initialize variables
    #     distances = []
    #     rays = []
    #     angles = np.linspace(0, 2 * np.pi, num_rays, endpoint=False)  # Full 360° angles for raycasting
    #     max_radius = int(np.hypot(head_mask.shape[1], head_mask.shape[0]))  # Maximum possible distance for raycasting

    #     if view_direction == "side":
    #         # Side View Logic:
    #         eye = np.array(safe_get(15))  # Left eye
    #         ear = np.array(safe_get(17))  # Left ear
    #         origin = np.array(ear)

    #         dx, dy = eye - ear
    #         angle = -degrees(atan2(dy, dx))  # Tilt correction

    #         def rotate_point(p, center, angle_deg):
    #             angle_rad = np.radians(angle_deg)
    #             R = np.array([[np.cos(angle_rad), -np.sin(angle_rad)],
    #                         [np.sin(angle_rad),  np.cos(angle_rad)]])
    #             return R @ (p - center) + center

    #         head_mask = (head_mask > 0).astype(np.uint8)
    #         h, w = head_mask.shape
    #         M = cv.getRotationMatrix2D(center=tuple(eye), angle=angle, scale=1.0)
    #         rotated_mask = cv.warpAffine(head_mask, M, (w, h), flags=cv.INTER_NEAREST)

    #         rotated_origin = rotate_point(origin, eye, angle).astype(int)
    #         rotated_eye = rotate_point(eye, eye, angle)

    #         # Filtering bounds based on non-zero head pixels
    #         ys, xs = np.nonzero(rotated_mask)
    #         ymin, ymax = ys.min(), ys.max()

    #         # Vertical boundary function for side view (rotated)
    #         def rotated_vertical_boundary(y_val):
    #             angle_rad = np.radians(angle)
    #             return rotated_eye[0] - 5 * np.cos(angle_rad) - (y_val - rotated_eye[1]) * np.tan(angle_rad)

    #         # Raycasting for side view
    #         for theta in angles:
    #             for d in range(1, max_radius):
    #                 x = int(rotated_origin[0] + d * np.cos(theta))
    #                 y = int(rotated_origin[1] + d * np.sin(theta))

    #                 if 0 <= x < w and 0 <= y < h:
    #                     if rotated_mask[y, x] == 0:  # Background hit
    #                         if ymin <= y <= ymax and x <= rotated_vertical_boundary(y):
    #                             distances.append(d)
    #                             rays.append(((rotated_origin[0], rotated_origin[1]), (x, y)))
    #                         else:
    #                             distances.append(0)  # Filtered out
    #                             rays.append((rotated_origin, rotated_origin))
    #                         break
    #             else:
    #                 distances.append(0)
    #                 rays.append((rotated_origin, rotated_origin))

    #     elif view_direction == "front":
    #         # Front View Logic:
    #         eye_left = np.array(safe_get(15))  # Left eye
    #         eye_right = np.array(safe_get(16))  # Right eye
    #         nose = np.array(safe_get(0))  # Nose

    #         # Midpoint between the eyes (for front view)
    #         origin = (eye_left + eye_right) / 2.0

    #         head_mask = (head_mask > 0).astype(np.uint8)
    #         h, w = head_mask.shape

    #         # Raycasting for front view (no boundary filtering)
    #         for theta in angles:
    #             for d in range(1, max_radius):
    #                 x = int(origin[0] + d * np.cos(theta))
    #                 y = int(origin[1] + d * np.sin(theta))

    #                 if 0 <= x < w and 0 <= y < h:
    #                     if head_mask[y, x] == 0:  # Background hit
    #                         distances.append(d)
    #                         rays.append(((origin[0], origin[1]), (x, y)))
    #                         break
    #             else:
    #                 distances.append(0)
    #                 rays.append((origin, origin))

    #     # Normalize distances if required
    #     distances = np.array(distances, dtype=np.float32)
    #     if normalize and distances.max() > 0:
    #         distances = (distances - distances.min()) / (distances.max() - distances.min())

    #     # Visualisation
    #     if visualise:
    #         os.makedirs("outputs/raycast_debug", exist_ok=True)

    #         # Visualize the rays on the mask
    #         vis_mask = cv.cvtColor((rotated_mask * 255).astype(np.uint8), cv.COLOR_GRAY2BGR) if view_direction == "side" else cv.cvtColor((head_mask * 255).astype(np.uint8), cv.COLOR_GRAY2BGR)
        
    #         for o, p in rays:
    #             cv.line(vis_mask, tuple(map(int, o)), tuple(map(int, p)), (0, 128, 255), 1)
    #         if view_direction == "side":
    #             cv.circle(vis_mask, tuple(rotated_origin.astype(int)), 3, (0, 255, 0), -1)
    #         else:
    #             cv.circle(vis_mask, tuple(origin.astype(int)), 3, (0, 255, 0), -1)
    #         cv.imwrite(f"outputs/raycast_debug/{os.path.basename(path)}_rays_on_mask.png", vis_mask)

    #         # Plot the raycast signature
    #         angles_deg = np.degrees(angles)
    #         plt.figure()
    #         plt.plot(angles_deg, distances)
    #         plt.title(f"Rotation-Corrected Raycast Signature ({view_direction.capitalize()} View)")
    #         plt.xlabel("Angle (°)")
    #         plt.ylabel("Normalized Distance")
    #         plt.tight_layout()
    #         plt.savefig(f"outputs/raycast_debug/{os.path.basename(path)}_signature_{view_direction}.png")
    #         plt.close()

    #     return distances


    def extract_rotation_corrected_raycast(self, path: str, num_rays: int = 360, normalize: bool = True, visualise: bool = False,num_sectors: int = 24) -> np.ndarray:
        base_img = self.fetch_image(path)
        head_mask = self.extract_head_mask(path)

        if head_mask is None or np.count_nonzero(head_mask) < 10:
            return np.zeros(num_rays)

        keypoints = self.extract_keypoints(path)
        if not keypoints or len(keypoints) < 18:
            return np.zeros(num_rays)

        def safe_get(k):
            x, y, p = keypoints[k]
            return (x, y) if p > 0 else (0, 0)

        view_direction = self.get_view_direction(path)
        view_direction = "side" if view_direction in ["left", "right"] else "front"

        distances = []
        rays = []
        angles = np.linspace(0, 2 * np.pi, num_rays, endpoint=False)
        max_radius = int(np.hypot(head_mask.shape[1], head_mask.shape[0]))

        if view_direction == "side":
            eye = np.array(safe_get(15))  # Left eye
            ear = np.array(safe_get(17))  # Left ear
            origin = np.array(ear)

            dx, dy = eye - ear
            angle = -degrees(atan2(dy, dx))

            def rotate_point(p, center, angle_deg):
                angle_rad = np.radians(angle_deg)
                R = np.array([[np.cos(angle_rad), -np.sin(angle_rad)],
                            [np.sin(angle_rad),  np.cos(angle_rad)]])
                return R @ (p - center) + center

            head_mask = (head_mask > 0).astype(np.uint8)
            h, w = head_mask.shape
            M = cv.getRotationMatrix2D(center=tuple(origin), angle=angle, scale=1.0)
            rotated_mask = cv.warpAffine(head_mask, M, (w, h), flags=cv.INTER_NEAREST)

            rotated_origin = rotate_point(origin, origin, angle).astype(int)
            rotated_eye = rotate_point(eye, origin, angle)

            ys, xs = np.nonzero(rotated_mask)
            ymin, ymax = ys.min(), ys.max()

            def rotated_vertical_boundary(y_val):
                angle_rad = np.radians(angle)
                return rotated_eye[0] - 5 * np.cos(angle_rad) - (y_val - rotated_eye[1]) * np.tan(angle_rad)

            for theta in angles:
                for d in range(1, max_radius):
                    x = int(rotated_origin[0] + d * np.cos(theta))
                    y = int(rotated_origin[1] + d * np.sin(theta))

                    if 0 <= x < w and 0 <= y < h:
                        if rotated_mask[y, x] == 0:
                            if ymin <= y <= ymax and x <= rotated_vertical_boundary(y):
                                distances.append(d)
                                rays.append(((rotated_origin[0], rotated_origin[1]), (x, y)))
                            else:
                                distances.append(0)
                                rays.append((rotated_origin, rotated_origin))
                            break
                else:
                    distances.append(0)
                    rays.append((rotated_origin, rotated_origin))

            # Angular correction: align forward-facing direction to 0°
            forward_angle = atan2(rotated_eye[1] - rotated_origin[1], rotated_eye[0] - rotated_origin[0])
            angle_offset = int((np.degrees(forward_angle) % 360) / 360 * num_rays)
            distances = np.roll(distances, -angle_offset)

            # Normalize using anatomical distance (ear to eye)
            ref_length = np.linalg.norm(eye - ear)
            distances = np.array(distances, dtype=np.float32)
            if normalize and ref_length > 0:
                distances /= ref_length

            # Optional smoothing
            distances = gaussian_filter1d(distances, sigma=1)

        elif view_direction == "front":
            eye_left = np.array(safe_get(15))
            eye_right = np.array(safe_get(16))
            origin = (eye_left + eye_right) / 2.0
            head_mask = (head_mask > 0).astype(np.uint8)
            h, w = head_mask.shape

            for theta in angles:
                for d in range(1, max_radius):
                    x = int(origin[0] + d * np.cos(theta))
                    y = int(origin[1] + d * np.sin(theta))

                    if 0 <= x < w and 0 <= y < h:
                        if head_mask[y, x] == 0:
                            distances.append(d)
                            rays.append(((origin[0], origin[1]), (x, y)))
                            break
                else:
                    distances.append(0)
                    rays.append((origin, origin))

            # Normalize by inter-eye distance
            ref_length = np.linalg.norm(eye_left - eye_right)
            distances = np.array(distances, dtype=np.float32)
            if normalize and ref_length > 0:
                distances /= ref_length

            distances = gaussian_filter1d(distances, sigma=1)
        
        # ↓↓↓ DOWN-SAMPLING BY SECTOR AVERAGING ↓↓↓
        # def downsample_by_sector(arr: np.ndarray, sectors: int) -> np.ndarray:
        #     sector_size = len(arr) // sectors
        #     return np.array([
        #         arr[i * sector_size : (i + 1) * sector_size].mean()
        #         for i in range(sectors)
        #     ])

        # distances = downsample_by_sector(distances, num_sectors)

        if visualise:
            os.makedirs("outputs/raycast_debug", exist_ok=True)
            vis_mask = cv.cvtColor((rotated_mask * 255).astype(np.uint8), cv.COLOR_GRAY2BGR) if view_direction == "side" else cv.cvtColor((head_mask * 255).astype(np.uint8), cv.COLOR_GRAY2BGR)
            for o, p in rays:
                cv.line(vis_mask, tuple(map(int, o)), tuple(map(int, p)), (0, 128, 255), 1)
            marker = tuple(rotated_origin.astype(int)) if view_direction == "side" else tuple(origin.astype(int))
            cv.circle(vis_mask, marker, 3, (0, 255, 0), -1)
            cv.imwrite(f"outputs/raycast_debug/{os.path.basename(path)}_rays_on_mask.png", vis_mask)

            angles_deg = np.degrees(angles)
            plt.figure()
            plt.plot(angles_deg, distances)
            plt.title(f"Rotation-Corrected Raycast Signature ({view_direction.capitalize()} View)")
            plt.xlabel("Angle (°)")
            plt.ylabel("Normalized Distance")
            plt.tight_layout()
            plt.savefig(f"outputs/raycast_debug/{os.path.basename(path)}_signature_{view_direction}.png")
            plt.close()

        return distances

    # def extract_pca_from_hog(self, path: str) -> np.ndarray:
    #     hog_vec = self.extract_hog_features(path)
    #     return self.pca_model.transform(hog_vec.reshape(1, -1)).flatten()

    def _compute_hu_moments(self, binary_mask: np.ndarray) -> list:
        """Helper to compute Hu Moments from a binary mask."""
        moments = cv.moments(binary_mask)
        return cv.HuMoments(moments).flatten().tolist()

    def extract_all_features(self, path: str,features) -> Dict:
        """
        Extracts all relevant biometric features from the given image.

        Returns a dictionary containing:
        - path: File path to the image
        - keypoints: List of 18 (x, y, confidence) keypoints in OpenPose format
        - direction: View direction ("front", "left", "right")
        - measurements: Keypoint-based distances (e.g. head width, shoulder width)
        - feature: Flattened feature vector (includes measurements, height, Hu moments)
        """

        keypoints = self.extract_keypoints(path)
        direction = self.get_view_direction(path)
        measurements, _ = self.extract_measurements(path)
        feature_vector = self.extract_feature_vector(path, features)

        return {
            "path": path,
            "keypoints": keypoints,
            "direction": direction,
            "measurements": measurements,
            "feature": feature_vector
        }



    





