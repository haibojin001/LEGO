# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg153::numpy.clip+numpy.searchsorted
# name: numpy_primitive
# summary: Uses numpy.clip, numpy.searchsorted across 2 repos
# anchor_symbols: ['numpy.clip', 'numpy.searchsorted']
# observed in 2 repos: ['libffcv__ffcv', 'microsoft__nni']...

# --- from libffcv__ffcv::ffcv/transforms/poisoning.py::Poison.generate_code.poison ---
def poison(images, temp_array, indices):
            for i in my_range(images.shape[0]):
                sample_ix = indices[i]
                # We check if the index is in the list of indices
                # to poison
                position = np.searchsorted(to_poison, sample_ix)
                if position < len(to_poison) and to_poison[position] == sample_ix:
                    temp = temp_array[i]
                    temp[:] = images[i]
                    temp *= 1 - alpha
                    temp += mask
                    np.clip(temp, clamp[0], clamp[1], out=temp)
                    images[i] = temp
            return images

# --- from libffcv__ffcv::ffcv/transforms/poisoning.py::Poison.generate_code ---
def generate_code(self) -> Callable:

        alpha = np.repeat(self.alpha[:, :, None], 3, axis=2)
        mask = self.mask.astype('float') * alpha
        to_poison = self.indices
        clamp = self.clamp
        my_range = Compiler.get_iterator()

        def poison(images, temp_array, indices):
            for i in my_range(images.shape[0]):
                sample_ix = indices[i]
                # We check if the index is in the list of indices
                # to poison
                position = np.searchsorted(to_poison, sample_ix)
                if position < len(to_poison) and to_poison[position] == sample_ix:
                    temp = temp_array[i]
                    temp[:] = images[i]
                    temp *= 1 - alpha
                    temp += mask
                    np.clip(temp, clamp[0], clamp[1], out=temp)
                    images[i] = temp
            return images

        poison.is_parallel = True
        poison.with_indices = True

        return poison

# --- from microsoft__nni::nni/algorithms/hpo/tpe_tuner.py::adaptive_parzen_normal ---
def adaptive_parzen_normal(args, history_mus, prior_mu, prior_sigma):
    """
    The "Adaptive Parzen Estimator" described in paper section 4.2, for normal distribution.

    Because TPE internally only supports categorical and normal distributed space (domain),
    this function is used for everything other than "choice" and "randint".

    Parameters
    ----------
    args: TpeArguments
        Algorithm arguments.
    history_mus: 1-d array of float
        Parameter values evaluated in history.
        These are the "observations" in paper section 4.2. ("placing density in the vicinity of K observations")
    prior_mu: float
        µ value of normal search space.
    piror_sigma: float
        σ value of normal search space.

    Returns
    -------
    Tuple of three 1-d float arrays: (weight, µ, σ).

    The tuple represents N+1 "vicinity of observations" and each one's weight,
    calculated from "N" history and "1" user provided prior.

    The result is sorted by µ.
    """
    mus = np.append(history_mus, prior_mu)
    order = np.argsort(mus)
    mus = mus[order]
    prior_index = np.searchsorted(mus, prior_mu)

    if len(mus) == 1:
        sigmas = np.asarray([prior_sigma])
    elif len(mus) == 2:
        sigmas = np.asarray([prior_sigma * 0.5, prior_sigma * 0.5])
        sigmas[prior_index] = prior_sigma
    else:
        l_delta = mus[1:-1] - mus[:-2]
        r_delta = mus[2:] - mus[1:-1]
        sigmas_mid = np.maximum(l_delta, r_delta)
        sigmas = np.concatenate([[mus[1] - mus[0]], sigmas_mid, [mus[-1] - mus[-2]]])
        sigmas[prior_index] = prior_sigma
    # "magic formula" in official implementation
    n = min(100, len(mus) + 1)
    sigmas = np.clip(sigmas, prior_sigma / n, prior_sigma)

    weights = np.append(linear_forgetting_weights(args, len(mus) - 1), args.prior_weight)
    weights = weights[order]

    return weights / np.sum(weights), mus, sigmas
