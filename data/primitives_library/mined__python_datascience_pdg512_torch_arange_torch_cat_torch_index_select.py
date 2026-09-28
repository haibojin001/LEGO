# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg512::torch.arange+torch.cat+torch.index_select
# name: torch_primitive
# summary: Uses torch.arange, torch.cat, torch.index_select, torch.stack across 2 repos
# anchor_symbols: ['torch.arange', 'torch.cat', 'torch.index_select', 'torch.stack']
# observed in 2 repos: ['HazyResearch__meerkat', 'Lightning-AI__torchmetrics']...

# --- from HazyResearch__meerkat::meerkat/datasets/video_corruptions/transforms.py::TemporalDownsampling.__call__ ---
def __call__(self, video: "torch.Tensor") -> "torch.Tensor":
        video_length = video.size(self.time_dim)
        downsampled_indices = torch.arange(
            0, video_length, self.downsample_factor
        ).long()
        frames = torch.index_select(video, self.time_dim, downsampled_indices)
        return frames

# --- from Lightning-AI__torchmetrics::src/torchmetrics/functional/audio/pit.py::pit_permutate ---
def pit_permutate(preds: Tensor, perm: Tensor) -> Tensor:
    """Permutate estimate according to perm.

    Args:
        preds: the estimates you want to permutate, shape [batch, spk, ...]
        perm: the permutation returned from permutation_invariant_training, shape [batch, spk]

    Returns:
        Tensor: the permutated version of estimate

    """
    return torch.stack([torch.index_select(pred, 0, p) for pred, p in zip(preds, perm)])

# --- from Lightning-AI__torchmetrics::src/torchmetrics/functional/image/utils.py::_single_dimension_pad ---
def _single_dimension_pad(inputs: Tensor, dim: int, pad: int, outer_pad: int = 0) -> Tensor:
    """Apply single-dimension reflection padding to match scipy implementation.

    Args:
        inputs: Input image
        dim: A dimension the image should be padded over
        pad: Number of pads
        outer_pad: Number of outer pads

    Return:
        Image padded over a single dimension

    """
    _max = inputs.shape[dim]
    x = torch.index_select(inputs, dim, torch.arange(pad - 1, -1, -1).to(inputs.device))
    y = torch.index_select(inputs, dim, torch.arange(_max - 1, _max - pad - outer_pad, -1).to(inputs.device))
    return torch.cat((x, inputs, y), dim)

# --- from HazyResearch__meerkat::meerkat/datasets/video_corruptions/transforms.py::TemporalCrop.__call__ ---
def __call__(self, video: "torch.Tensor") -> "torch.Tensor":
        video_length = video.size(self.time_dim)
        clips = []
        for clip_number in range(self.n_clips):
            start, end = self._get_sampling_boundaries(video_length, clip_number)
            if self.sample_starting_location:
                first_frame = random.randint(start, end)
            else:
                first_frame = start
            indices = self._build_indices(first_frame, video_length)
            clip = torch.index_select(video, self.time_dim, indices)
            clips.append(clip)
        if self.stack_clips:  # new dim for clips (n_clips, n_channels, duration, h, w)
            all_clips = torch.stack(clips, dim=0)
        else:  # concat clips in time dimension (n_channels, n_clips * duration, h, w)
            all_clips = torch.cat(clips, dim=self.time_dim)

        return all_clips
