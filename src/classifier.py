from typing import List, Dict, Tuple
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
import numpy as np
import os


class KNNClassifiers:
    def __init__(self, feature_extractor, neighbors_per_view: Dict[str, int]):
        """
        Hybrid KNN classifier: uses KNeighborsClassifier for prediction 
        and manual pairwise distances for evaluation.

        Args:
            feature_extractor: Instance of HumanBodyAnalyser
            neighbors_per_view: Dict like {"front": 3, "side": 1}
        """
        self.fe = feature_extractor
        self.views = ["front", "side"]
        self.n_neighbors = neighbors_per_view

        self.models: Dict[str, KNeighborsClassifier] = {}
        self.scalers: Dict[str, StandardScaler] = {}
        self.features: Dict[str, List[np.ndarray]] = {v: [] for v in self.views}
        self.labels: Dict[str, List[str]] = {v: [] for v in self.views}
        self.data: Dict[str, Dict] = {}
        self.front_features = []
        self.side_features = []

    def reset(self):
        self.models.clear()
        self.scalers.clear()
        self.features = {v: [] for v in self.views}
        self.labels = {v: [] for v in self.views}
        self.data.clear()
        self.front_features = []
        self.side_features = []

    def _get_label(self, path: str) -> str:
        return os.path.basename(path)[:-5]

    def add_samples(self, image_paths: List[str],front_features: List[str],side_features: List[str]):
        self.front_features = front_features
        self.side_features = side_features
        for path in image_paths:
            direction = self.fe.get_view_direction(path)
            features = self.front_features if direction == "front" else self.side_features
            sample = self.fe.extract_all_features(path, features)
            view = "side" if sample["direction"] in ["left", "right"] else "front"
            label = self._get_label(path)
            self.features[view].append(sample["feature"])
            self.labels[view].append(label)
            self.data[path] = sample
            

        self._fit_models()

    def _fit_models(self):
        for view in self.views:
            if self.features[view]:
                X_raw = np.stack(self.features[view])
                y = self.labels[view]

                # Normalize features
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X_raw)
                self.scalers[view] = scaler

                # Train KNN model
                knn = KNeighborsClassifier(n_neighbors=self.n_neighbors[view])
                knn.fit(X_scaled, y)
                self.models[view] = knn

    def get_gallery_embeddings_and_labels(self, view: str) -> Tuple[np.ndarray, List[str]]:
        """
        Returns scaled gallery embeddings and labels for a given view ('front' or 'side').

        Args:
            view (str): Either 'front' or 'side'

        Returns:
            Tuple[np.ndarray, List[str]]: (scaled embeddings, labels)
        """
        if view not in self.views:
            raise ValueError(f"View '{view}' is invalid. Must be 'front' or 'side'.")

        if view not in self.scalers or view not in self.features:
            raise ValueError(f"Missing data for view: '{view}'")

        X_raw = np.stack(self.features[view])
        X_scaled = self.scalers[view].transform(X_raw)
        y_labels = self.labels[view]

        return X_scaled, y_labels


    def predict(self, path: str) -> Dict:
        direction = self.fe.get_view_direction(path)
        features = self.side_features if direction in ["left", "right"] else self.front_features

        sample = self.fe.extract_all_features(path, features)
        view = "side" if sample["direction"] in ["left", "right"] else "front"

        X = sample["feature"].reshape(1, -1)

        if view not in self.models or self.models[view] is None:
            raise ValueError(f"No trained model available for view: '{view}'")

        # Normalize using the same scaler
        X_scaled = self.scalers[view].transform(X)
        y_train_scaled = self.scalers[view].transform(np.stack(self.features[view]))

        model = self.models[view]

        # KNN prediction
        predicted_label = model.predict(X_scaled)[0]
        _, indices = model.kneighbors(X_scaled)
        proba = model.predict_proba(X_scaled)[0]

        # Manual full pairwise distances (for evaluation, not prediction)
        full_distances = cdist(X_scaled, y_train_scaled, metric="euclidean")[0]

        sample.update({
            "predicted_label": predicted_label,
            "feature_dists": full_distances,
            "pred_indexes": indices[0][0],
            "class_probabilities": dict(zip(model.classes_, proba)),
            "X_scaled": X_scaled.flatten(),
        })

        self.data[path] = sample
        return sample

class SVMClassifiers:
    def __init__(self, feature_extractor):
        """
        SVM classifier: uses SVC with probability=True 
        and manual pairwise distances for evaluation.

        Args:
            feature_extractor: Instance of HumanBodyAnalyser
        """
        self.fe = feature_extractor
        self.views = ["front", "side"]
        self.models: Dict[str, SVC] = {}
        self.scalers: Dict[str, StandardScaler] = {}
        self.features: Dict[str, List[np.ndarray]] = {v: [] for v in self.views}
        self.labels: Dict[str, List[str]] = {v: [] for v in self.views}
        self.data: Dict[str, Dict] = {}
        self.front_features = []
        self.side_features = []
        

    def reset(self):
        self.models.clear()
        self.scalers.clear()
        self.features = {v: [] for v in self.views}
        self.labels = {v: [] for v in self.views}
        self.data.clear()
        self.front_features = []
        self.side_features = []

    def _get_label(self, path: str) -> str:
        return os.path.basename(path)[:-5]

    def add_samples(self, image_paths: List[str], front_features: List[str], side_features: List[str]):
        self.front_features = front_features
        self.side_features = side_features
        for path in image_paths:
            direction = self.fe.get_view_direction(path)
            features = self.front_features if direction == "front" else self.side_features
            sample = self.fe.extract_all_features(path, features)
            view = "side" if sample["direction"] in ["left", "right"] else "front"
            label = self._get_label(path)
            self.features[view].append(sample["feature"])
            self.labels[view].append(label)
            self.data[path] = sample

        self._fit_models()

    def _fit_models(self):
        for view in self.views:
            if self.features[view]:
                X_raw = np.stack(self.features[view])
                y = self.labels[view]

                # Normalize features
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X_raw)
                self.scalers[view] = scaler

                # Train SVM with probabilities enabled
                model = SVC(probability=True, kernel="rbf", C=1.0, gamma="scale")
                model.fit(X_scaled, y)
                self.models[view] = model

    def predict(self, path: str) -> Dict:
        direction = self.fe.get_view_direction(path)
        features = self.side_features if direction in ["left", "right"] else self.front_features

        sample = self.fe.extract_all_features(path, features)
        view = "side" if sample["direction"] in ["left", "right"] else "front"

        X = sample["feature"].reshape(1, -1)

        if view not in self.models or self.models[view] is None:
            raise ValueError(f"No trained model available for view: '{view}'")

        # Normalize using trained scaler
        X_scaled = self.scalers[view].transform(X)
        y_train_scaled = self.scalers[view].transform(np.stack(self.features[view]))

        model = self.models[view]
        predicted_label = model.predict(X_scaled)[0]
        proba = model.predict_proba(X_scaled)[0]

        # Compute pairwise distances (still used in evaluation like EER/CCR)
        full_distances = cdist(X_scaled, y_train_scaled, metric="euclidean")[0]

        sample.update({
            "predicted_label": predicted_label,
            "feature_dists": full_distances,
            "pred_indexes": None,  # SVM doesn't provide this
            "class_probabilities": dict(zip(model.classes_, proba))
        })

        self.data[path] = sample
        return sample

class UnifiedSVMClassifier:
    def __init__(self, feature_extractor):
        """
        Unified SVM classifier across all views (front + side).
        """
        self.fe = feature_extractor
        self.model: SVC = None
        self.scaler: StandardScaler = None
        self.features: List[np.ndarray] = []
        self.labels: List[str] = []
        self.data: Dict[str, Dict] = {}
        self.feature_set: List[str] = []

    def reset(self):
        self.model = None
        self.scaler = None
        self.features = []
        self.labels = []
        self.data.clear()
        self.feature_set = []

    def _get_label(self, path: str) -> str:
        return os.path.basename(path)[:-5]

    def add_samples(self, image_paths: List[str], feature_set: List[str]):
        """
        Collect training data with a unified feature set (used for all images).
        """
        self.feature_set = feature_set
        for path in image_paths:
            features = self.fe.extract_feature_vector(path, self.feature_set)
            label = self._get_label(path)

            self.features.append(features)
            self.labels.append(label)

            self.data[path] = {
                "feature": features,
                "label": label
            }

        self._fit_model()

    def _fit_model(self):
        """
        Train a single SVM model on the combined feature set.
        """
        X_raw = np.stack(self.features)
        y = self.labels

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_raw)

        self.model = SVC(probability=True, kernel="rbf", C=1.0, gamma="scale")
        self.model.fit(X_scaled, y)

    def predict(self, path: str) -> Dict:
        """
        Predict label and return distances and probabilities.
        """
        features = self.fe.extract_feature_vector(path, self.feature_set).reshape(1, -1)
        X_scaled = self.scaler.transform(features)
        y_train_scaled = self.scaler.transform(np.stack(self.features))

        predicted_label = self.model.predict(X_scaled)[0]
        proba = self.model.predict_proba(X_scaled)[0]
        distances = cdist(X_scaled, y_train_scaled, metric="euclidean")[0]

        result = {
            "predicted_label": predicted_label,
            "class_probabilities": dict(zip(self.model.classes_, proba)),
            "feature_dists": distances,
            "feature": features.flatten()
        }

        self.data[path] = result
        return result
