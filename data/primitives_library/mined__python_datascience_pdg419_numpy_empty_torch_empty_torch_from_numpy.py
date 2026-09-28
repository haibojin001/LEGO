# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg419::numpy.empty+torch.empty+torch.from_numpy
# name: numpy_torch_primitive
# summary: Uses numpy.empty, torch.empty, torch.from_numpy across 3 repos
# anchor_symbols: ['numpy.empty', 'torch.empty', 'torch.from_numpy']
# observed in 3 repos: ['OML-Team__open-metric-learning', 'deepchecks__deepchecks', 'libffcv__ffcv']...

# --- from libffcv__ffcv::ffcv/pipeline/allocation_query.py::allocate_query ---
def allocate_query(memory_allocation: AllocationQuery, batch_size: int, batches_ahead: int):
    # We compute the total amount of memory needed for this
    # operation
    final_shape = [batches_ahead,
                   batch_size, *memory_allocation.shape]
    if isinstance(memory_allocation.dtype, ch.dtype):
        result = []
        for _ in range(final_shape[0]):
            partial = ch.empty(*final_shape[1:],
                              dtype=memory_allocation.dtype,
                              device=memory_allocation.device)
            try:
                partial = partial.pin_memory()
            except:
                pass
            result.append(partial)
    else:
        ch_dtype = ch.from_numpy(np.empty(0, dtype=memory_allocation.dtype)).dtype
        result = ch.empty(*final_shape,
                          dtype=ch_dtype)
        try:
            result = result.pin_memory()
        except:
            pass
        result = result.numpy()
    return result

# --- from OML-Team__open-metric-learning::oml/metrics/accumulation.py::Accumulator._allocate_memory_if_need ---
def _allocate_memory_if_need(self, key: str, batch_value: Any) -> None:
        if self.num_samples is None:
            raise ValueError(
                f"The parameter for memory allocation has not been set up."
                f"Are you sure you've called {self.refresh.__name__}?"
            )

        if key not in self._storage:
            if isinstance(batch_value, torch.Tensor):
                self._storage[key] = torch.empty(
                    (self.num_samples, *batch_value.shape[1:]),
                    dtype=batch_value.dtype,
                    device="cpu",
                    requires_grad=False,
                )
            elif isinstance(batch_value, np.ndarray):
                self._storage[key] = np.empty((self.num_samples, *batch_value.shape[1:]), dtype=batch_value.dtype)
            elif isinstance(batch_value, (list, tuple)):
                self._storage[key] = []
            else:
                raise TypeError(f"Type '{type(batch_value)}' is not available for accumulating")

# --- from libffcv__ffcv::ffcv/pipeline/pipeline.py::Pipeline.allocate_query ---
def allocate_query(self, memory_allocation: AllocationQuery, batch_size: int, batches_ahead: int):
        # We compute the total amount of memory needed for this
        # operation
        final_shape = [batches_ahead,
                       batch_size, *memory_allocation.shape]
        if isinstance(memory_allocation.dtype, ch.dtype):
            result = []
            for _ in range(final_shape[0]):
                partial = ch.empty(*final_shape[1:],
                                  dtype=memory_allocation.dtype,
                                  device=memory_allocation.device)
                try:
                    partial = partial.pin_memory()
                except:
                    pass
                result.append(partial)
        else:
            ch_dtype = ch.from_numpy(np.empty(0, dtype=memory_allocation.dtype)).dtype
            result = ch.empty(*final_shape,
                              dtype=ch_dtype)
            try:
                result = result.pin_memory()
            except:
                pass
            result = result.numpy()
        return result

# --- from deepchecks__deepchecks::docs/source/vision/tutorials/other/plot_custom_task_tutorial.py::CocoInstanceSegmentationDataset.__getitem__ ---
def __getitem__(self, idx: int) -> t.Tuple[Image.Image, np.ndarray]:
        """Get the image and label at the given index."""
        image = Image.open(str(self.images[idx]))
        label_file = self.labels[idx]

        masks = []
        if label_file is not None:
            for label_str in label_file.open('r').read().strip().splitlines():
                label = np.array(label_str.split(), dtype=np.float32)
                class_id = int(label[0])
                # Transform normalized coordinates to un-normalized
                coordinates = (label[1:].reshape(-1, 2) * np.array([image.width, image.height])).reshape(-1).tolist()
                # Create mask image
                mask = Image.new('L', (image.width, image.height), 0)
                ImageDraw.Draw(mask).polygon(coordinates, outline=1, fill=1)
                # Add to list
                masks.append(np.array(mask, dtype=bool))

        if self.transforms is not None:
            # Albumentations accepts images as numpy
            transformed = self.transforms(image=np.array(image), masks=masks if masks else None)
            image = transformed['image']
            masks = transformed['masks']
            # Transform masks to tensor of (num_masks, H, W)
            if masks:
                if isinstance(masks[0], np.ndarray):
                    masks = [torch.from_numpy(m) for m in masks]
                masks = torch.stack(masks)
            else:
                masks = torch.empty((0, 3))

        return image, masks
