# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg123::collections.defaultdict+random.shuffle+torch.split
# name: collections_random_primitive
# summary: Uses collections.defaultdict, random.shuffle, torch.split, torch.tensor across 3 repos
# anchor_symbols: ['collections.defaultdict', 'random.shuffle', 'torch.split', 'torch.tensor']
# observed in 3 repos: ['microsoft__nni', 'sb-ai-lab__LightAutoML', 'sberbank-ai-lab__LightAutoML']...

# --- from microsoft__nni::test/algo/nas/graph_converter/test_convert_operators.py::TestOperators.test_basic_split ---
def test_basic_split(self):
        class SimpleOp(nn.Module):
            def forward(self, x):
                out = torch.split(x, 2, 1)
                return out
        x = torch.tensor([[0.0, 1.0, 1.0, 0.0, 2.0, 2.0], [2.0, 3.0, 3.0, 2.0, 1.0, 1.0]])
        self.checkExportImport(SimpleOp(), (x, ))

# --- from microsoft__nni::test/algo/nas/graph_converter/test_convert_operators.py::TestOperators.test_basic_split_with_sizes ---
def test_basic_split_with_sizes(self):
        class SimpleOp(nn.Module):
            def forward(self, x):
                out = torch.split(x, [2, 1, 3], 1)
                return out
        x = torch.tensor([[0.0, 1.0, 1.0, 0.0, 2.0, 2.0], [2.0, 3.0, 3.0, 2.0, 1.0, 1.0]])
        self.checkExportImport(SimpleOp(), (x, ))

# --- from sb-ai-lab__LightAutoML::lightautoml/addons/interpretation/data_process.py::BySequenceLengthSampler.__iter__ ---
def __iter__(self):
        data_buckets = defaultdict(list)
        for i, slen in self.ind_n_len:
            pid = self.to_bucket(slen)
            data_buckets[pid].append(i)

        for k in data_buckets.keys():
            data_buckets[k] = torch.tensor(data_buckets[k])

        iter_list = []
        for k in data_buckets.keys():
            t = self.shuffle_tensor(data_buckets[k])
            batch = torch.split(t, self.batch_size, dim=0)

            if self.drop_last and len(batch[-1]) != self.batch_size:
                batch = batch[:-1]

            iter_list += batch

        if self.shuffle:
            shuffle(iter_list)

        for i in iter_list:
            yield i.numpy().tolist()

# --- from sberbank-ai-lab__LightAutoML::lightautoml/addons/interpretation/data_process.py::BySequenceLengthSampler.__iter__ ---
def __iter__(self):
        data_buckets = defaultdict(list)
        for i, slen in self.ind_n_len:
            pid = self.to_bucket(slen)
            data_buckets[pid].append(i)

        for k in data_buckets.keys():
            data_buckets[k] = torch.tensor(data_buckets[k])

        iter_list = []
        for k in data_buckets.keys():
            t = self.shuffle_tensor(data_buckets[k])
            batch = torch.split(t, self.batch_size, dim=0)

            if self.drop_last and len(batch[-1]) != self.batch_size:
                batch = batch[:-1]

            iter_list += batch

        if self.shuffle:
            shuffle(iter_list)

        for i in iter_list:
            yield i.numpy().tolist()
