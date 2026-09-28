# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg631::os.cpu_count+torch.set_num_threads
# name: os_torch_primitive
# summary: Uses os.cpu_count, torch.set_num_threads across 2 repos
# anchor_symbols: ['os.cpu_count', 'torch.set_num_threads']
# observed in 2 repos: ['sb-ai-lab__LightAutoML', 'sberbank-ai-lab__LightAutoML']...

# --- from sberbank-ai-lab__LightAutoML::lightautoml/automl/presets/image_presets.py::TabularCVAutoML.infer_auto_params ---
def infer_auto_params(self, train_data: DataFrame, multilevel_avail: bool = False):
        """Infer automatic parameters."""
        # infer gpu params
        gpu_cnt = torch.cuda.device_count()
        gpu_ids = self.gpu_ids
        if gpu_cnt > 0 and gpu_ids:
            if gpu_ids == "all":
                gpu_ids = ",".join(list(map(str, range(gpu_cnt))))

            self.autocv_features["device"] = gpu_ids.split(",")

            if self.general_params["use_algos"] == "auto":
                self.general_params["use_algos"] = [["linear_l2", "cb"]]

        else:
            self.autocv_features["device"] = "cpu"

            if self.general_params["use_algos"] == "auto":
                self.general_params["use_algos"] = [["linear_l2", "lgb"]]

        # check all n_jobs params
        cpu_cnt = min(os.cpu_count(), self.cpu_limit)
        torch.set_num_threads(cpu_cnt)

        if "n_jobs" in self.autocv_features:
            self.autocv_features["n_jobs"] = min(self.autocv_features["n_jobs"], cpu_cnt)
        else:
            self.autocv_features["n_jobs"] = cpu_cnt

        if "n_jobs" in self.cv_simple_features:
            self.cv_simple_features["n_jobs"] = min(self.cv_simple_features["n_jobs"], cpu_cnt)
        else:
            self.cv_simple_features["n_jobs"] = cpu_cnt

        # other params as tabular
        super().infer_auto_params(train_data, multilevel_avail)

# --- from sb-ai-lab__LightAutoML::lightautoml/automl/presets/image_presets.py::TabularCVAutoML.infer_auto_params ---
def infer_auto_params(self, train_data: DataFrame, multilevel_avail: bool = False, target_col: str = None):
        """Infer automatic parameters."""
        # infer gpu params
        gpu_cnt = torch.cuda.device_count()
        gpu_ids = self.gpu_ids
        if gpu_cnt > 0 and gpu_ids:
            if gpu_ids == "all":
                gpu_ids = ",".join(list(map(str, range(gpu_cnt))))

            self.autocv_features["device"] = gpu_ids.split(",")

            if self.general_params["use_algos"] == "auto":
                self.general_params["use_algos"] = [["linear_l2", "cb"]]

        else:
            self.autocv_features["device"] = "cpu"

            if self.general_params["use_algos"] == "auto":
                self.general_params["use_algos"] = [["linear_l2", "lgb"]]

        # check all n_jobs params
        cpu_cnt = min(os.cpu_count(), self.cpu_limit)
        torch.set_num_threads(cpu_cnt)

        if "n_jobs" in self.autocv_features:
            self.autocv_features["n_jobs"] = min(self.autocv_features["n_jobs"], cpu_cnt)
        else:
            self.autocv_features["n_jobs"] = cpu_cnt

        if "n_jobs" in self.cv_simple_features:
            self.cv_simple_features["n_jobs"] = min(self.cv_simple_features["n_jobs"], cpu_cnt)
        else:
            self.cv_simple_features["n_jobs"] = cpu_cnt

        # other params as tabular
        super().infer_auto_params(train_data, multilevel_avail, target_col)

# --- from sberbank-ai-lab__LightAutoML::lightautoml/automl/presets/text_presets.py::TabularNLPAutoML.infer_auto_params ---
def infer_auto_params(self, train_data: DataFrame, multilevel_avail: bool = False):

        # infer gpu params
        gpu_cnt = torch.cuda.device_count()
        gpu_ids = self.gpu_ids
        if gpu_cnt > 0 and gpu_ids:
            if gpu_ids == "all":
                gpu_ids = ",".join(list(map(str, range(gpu_cnt))))

            self.nn_params["device"] = gpu_ids.split(",")
            self.text_params["device"] = gpu_ids.split(",")

            if self.general_params["use_algos"] == "auto":
                self.general_params["use_algos"] = [["linear_l2", "cb", "nn"]]

        else:
            self.nn_params["device"] = "cpu"
            self.text_params["device"] = "cpu"

            if self.general_params["use_algos"] == "auto":
                self.general_params["use_algos"] = [["linear_l2", "lgb"]]

        # check all n_jobs params
        cpu_cnt = min(os.cpu_count(), self.cpu_limit)
        torch.set_num_threads(cpu_cnt)

        self.nn_params["num_workers"] = min(self.nn_params["num_workers"], cpu_cnt)
        self.nn_params["lang"] = self.nn_params["lang"] or self.text_params["lang"]
        self.nn_params["bert_name"] = self.nn_params["bert_name"] or self.text_params["bert_model"]

        logger.info3("Model language mode: {}".format(self.nn_params["lang"]))

        if isinstance(self.autonlp_params["transformer_params"], dict):
            if "loader_params" in self.autonlp_params["transformer_params"]:
                self.autonlp_params["transformer_params"]["loader_params"]["num_workers"] = min(
                    self.autonlp_params["transformer_params"]["loader_params"]["num_workers"],
                    cpu_cnt,
                )
            else:
                self.autonlp_params["transformer_params"]["loader_params"] = {"num_workers": cpu_cnt}

        # other params as tabular
        super().infer_auto_params(train_data, multilevel_avail)

# --- from sb-ai-lab__LightAutoML::lightautoml/automl/presets/text_presets.py::TabularNLPAutoML.infer_auto_params ---
def infer_auto_params(self, train_data: DataFrame, multilevel_avail: bool = False, target_col: str = None):

        # infer gpu params
        gpu_cnt = torch.cuda.device_count()
        gpu_ids = self.gpu_ids
        if gpu_cnt > 0 and gpu_ids:
            if gpu_ids == "all":
                gpu_ids = ",".join(list(map(str, range(gpu_cnt))))

            self.nn_params["device"] = gpu_ids.split(",")
            self.text_params["device"] = gpu_ids.split(",")

            if self.general_params["use_algos"] == "auto":
                self.general_params["use_algos"] = [["linear_l2", "cb", "nn"]]

        else:
            self.nn_params["device"] = "cpu"
            self.text_params["device"] = "cpu"

            if self.general_params["use_algos"] == "auto":
                self.general_params["use_algos"] = [["linear_l2", "lgb"]]

        # check all n_jobs params
        cpu_cnt = min(os.cpu_count(), self.cpu_limit)
        torch.set_num_threads(cpu_cnt)

        self.nn_params["num_workers"] = min(self.nn_params["num_workers"], cpu_cnt)
        self.nn_params["lang"] = self.nn_params["lang"] or self.text_params["lang"]
        self.nn_params["bert_name"] = self.nn_params["bert_name"] or self.text_params["bert_model"]

        logger.info3(f"Model language mode: {self.nn_params['lang']}")

        if isinstance(self.autonlp_params["transformer_params"], dict):
            if "loader_params" in self.autonlp_params["transformer_params"]:
                self.autonlp_params["transformer_params"]["loader_params"]["num_workers"] = min(
                    self.autonlp_params["transformer_params"]["loader_params"]["num_workers"],
                    cpu_cnt,
                )
            else:
                self.autonlp_params["transformer_params"]["loader_params"] = {"num_workers": cpu_cnt}

        # other params as tabular
        super().infer_auto_params(train_data, multilevel_avail, target_col)
