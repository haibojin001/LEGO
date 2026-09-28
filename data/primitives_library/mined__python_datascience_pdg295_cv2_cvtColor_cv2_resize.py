# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg295::cv2.cvtColor+cv2.resize
# name: cv2_primitive
# summary: Uses cv2.cvtColor, cv2.resize across 2 repos
# anchor_symbols: ['cv2.cvtColor', 'cv2.resize']
# observed in 2 repos: ['deepchecks__deepchecks', 'lazyprogrammer__machine_learning_examples']...

# --- from lazyprogrammer__machine_learning_examples::rl3/a2c/atari_wrappers.py::WarpFrame.observation ---
def observation(self, frame):
        if self.grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        if self.grayscale:
            frame = np.expand_dims(frame, -1)
        return frame

# --- from deepchecks__deepchecks::deepchecks/vision/checks/train_test_validation/heatmap_comparison.py::HeatmapComparison._grayscale_sum_image ---
def _grayscale_sum_image(self, batch: Iterable[np.ndarray]) -> np.ndarray:
        """Sum all images in batch to one grayscale image of shape target_shape.

        Parameters
        ----------
        batch: np.ndarray
            batch of images.

        Returns
        -------
        np.ndarray
            summed image.
        """
        summed_image = None

        # Iterate over all images in batch, using the first image as the target shape if target_shape is None.
        # All subsequent images will be resized to the target shape and their gray values will be added to the
        # summed image.
        for img in batch:
            # Cast to grayscale
            if img.shape[2] == 1:
                resized_img = img
            elif img.shape[2] == 3:
                resized_img = cv2.cvtColor(img.astype('uint8'), cv2.COLOR_RGB2GRAY)
            else:
                raise NotImplementedError('Images must be RGB or grayscale')

            # reshape to one shape
            if self._shape is None:
                self._shape = resized_img.shape[:2][::-1]
            resized_img = cv2.resize(resized_img.astype('uint8'), self._shape, interpolation=cv2.INTER_AREA)

            # sum images
            if summed_image is None:
                summed_image = resized_img.squeeze().astype(np.int64)
            else:
                summed_image += resized_img.squeeze().astype(np.int64)

        return summed_image
