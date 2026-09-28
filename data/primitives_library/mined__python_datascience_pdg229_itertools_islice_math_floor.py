# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg229::itertools.islice+math.floor
# name: itertools_math_primitive
# summary: Uses itertools.islice, math.floor across 2 repos
# anchor_symbols: ['itertools.islice', 'math.floor']
# observed in 2 repos: ['allenai__allennlp', 'capitalone__DataProfiler']...

# --- from allenai__allennlp::allennlp/data/data_loaders/multitask_data_loader.py::MultiTaskDataLoader._get_instances_for_epoch ---
def _get_instances_for_epoch(self) -> Dict[str, Iterable[Instance]]:
        if self._instances_per_epoch is None:
            return {
                key: maybe_shuffle_instances(loader, self._shuffle)
                for key, loader in self._loaders.items()
            }
        if self.sampler is None:
            # We already checked for this in the constructor, so this should never happen unless you
            # modified the object after creation. But mypy is complaining, so here's another check.
            raise ValueError(
                "You must specify an EpochSampler if self._instances_per_epoch is not None."
            )
        dataset_proportions = self.sampler.get_task_proportions(self._loaders)
        proportion_sum = sum(dataset_proportions.values())
        num_instances_per_dataset = {
            key: math.floor(proportion * self._instances_per_epoch / proportion_sum)
            for key, proportion in dataset_proportions.items()
        }
        return {
            key: itertools.islice(self._iterators[key], num_instances)
            for key, num_instances in num_instances_per_dataset.items()
        }

# --- from capitalone__DataProfiler::dataprofiler/data_readers/data_utils.py::reservoir ---
def reservoir(file: TextIOWrapper, sample_nrows: int) -> list:
    """
    Implement the mathematical logic of Reservoir sampling.

    :param file: wrapper of the opened csv file
    :type file: TextIOWrapper
    :param sample_nrows: number of rows to sample
    :type sample_nrows: int

    :raises: ValueError()

    :return: sampled values
    :rtype: list
    """
    # Copyright 2021 Oscar Benjamin
    #
    # Permission is hereby granted, free of charge, to any person obtaining a copy
    # of this software and associated documentation files (the "Software"), to deal
    # in the Software without restriction, including without limitation the rights
    # to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    # copies of the Software, and to permit persons to whom the Software is
    # furnished to do so, subject to the following conditions:
    #
    # The above copyright notice and this permission notice shall be included in
    # all copies or substantial portions of the Software.
    #
    # THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    # IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    # FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    # AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    # LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    # OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    # SOFTWARE.
    # https://gist.github.com/oscarbenjamin/4c1b977181f34414a425f68589e895d1

    iterator = iter(file)
    values = list(islice(iterator, sample_nrows))

    irange = range(len(values))
    indices = dict(zip(irange, irange))

    kinv = 1 / sample_nrows
    W = 1.0
    rng = rng_utils.get_random_number_generator()

    while True:
        W *= rng.random() ** kinv
        # random() < 1.0 but random() ** kinv might not be
        # W == 1.0 implies "infinite" skips
        if W == 1.0:
            break
        # skip is geometrically distributed with parameter W
        skip = floor(log(rng.random()) / log1p(-W))
        try:
            newval = next(islice(iterator, skip, skip + 1))
        except StopIteration:
            break
        # Append new, replace old with dummy, and keep track of order
        remove_index = rng.integers(0, sample_nrows)
        values[indices[remove_index]] = str(None)
        indices[remove_index] = len(values)
        values.append(newval)

    values = [values[indices[i]] for i in irange]
    return values
