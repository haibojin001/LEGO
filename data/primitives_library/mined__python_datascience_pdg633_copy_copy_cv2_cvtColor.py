# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg633::copy.copy+cv2.cvtColor
# name: copy_cv2_primitive
# summary: Uses copy.copy, cv2.cvtColor across 2 repos
# anchor_symbols: ['copy.copy', 'cv2.cvtColor']
# observed in 2 repos: ['sb-ai-lab__LightAutoML', 'sberbank-ai-lab__LightAutoML']...

# --- from sb-ai-lab__LightAutoML::lightautoml/image/image.py::ColorFeatures.get_features ---
def get_features(self, img: np.ndarray) -> List[numeric]:
        """Calculate normalized color histogram for rgb or hsv image.

        Args:
            img: Image with shape ``(h, w)``.

        Returns:
            List of histogram values.

        """
        img = copy(img)
        if self.is_hsv:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

        res = []
        for i in range(3):
            channel = img[:, :, i]
            res.extend(self.compute_histogram(channel))

        return res

# --- from sberbank-ai-lab__LightAutoML::lightautoml/image/image.py::ColorFeatures.get_features ---
def get_features(self, img: np.ndarray) -> List[numeric]:
        """Calculate normalized color histogram for rgb or hsv image.

        Args:
            img: Image with shape ``(h, w)``.

        Returns:
            List of histogram values.

        """
        img = copy(img)
        if self.is_hsv:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

        res = []
        for i in range(3):
            channel = img[:, :, i]
            res.extend(self.compute_histogram(channel))

        return res
