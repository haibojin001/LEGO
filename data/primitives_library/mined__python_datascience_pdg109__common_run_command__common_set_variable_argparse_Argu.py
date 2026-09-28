# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg109::_common.run_command+_common.set_variable+argparse.ArgumentParser
# name: _common_argparse_primitive
# summary: Uses _common.run_command, _common.set_variable, argparse.ArgumentParser, argparse.ArgumentTypeError across 17 repos
# anchor_symbols: ['_common.run_command', '_common.set_variable', 'argparse.ArgumentParser', 'argparse.ArgumentTypeError', 'base64.b64decode', 'catboost.CatBoostClassifier']
# observed in 17 repos: ['DeepWisdom__AutoDL', 'InfuseAI__piperider', 'JosephLai241__URS', 'MatthewReid854__reliability', 'alteryx__evalml']...

# --- from microsoft__nni::examples/trials/mnist-tfv1/mnist_before.py::bias_variable ---
def bias_variable(shape):
    """bias_variable generates a bias variable of a given shape."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

# --- from microsoft__nni::examples/trials/mnist-tfv1/mnist.py::bias_variable ---
def bias_variable(shape):
    """bias_variable generates a bias variable of a given shape."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

# --- from underneathall__pinferencia::tests/api_tests/conftest.py::backend ---
def backend(backend_port):
    args = shlex.split(f"uvicorn --port {backend_port} tests.api_tests.app:service")
    p = Popen(args)
    for _ in range(60):
        try:
            requests.get(f"http://127.0.0.1:{backend_port}")
        except Exception:
            time.sleep(1)
    yield
    p.kill()

# --- from rasbt__mlxtend::mlxtend/externals/pyprind/prog_class.py::Prog._get_time ---
def _get_time(self, _time):
        if _time < 86400:
            return time.strftime("%H:%M:%S", time.gmtime(_time))
        else:
            s = (
                str(int(_time // 3600))
                + ":"
                + time.strftime("%M:%S", time.gmtime(_time))
            )
            return s

# --- from ploomber__ploomber::tests/sources/test_notebooksource.py::tmp_nbs_ipynb ---
def tmp_nbs_ipynb(tmp_nbs):
    """Modifies the nbs example to have one task with ipynb format"""
    # modify the spec so it has one ipynb task
    with open("pipeline.yaml") as f:
        spec = yaml.safe_load(f)

    spec["tasks"][0]["source"] = "load.ipynb"
    Path("pipeline.yaml").write_text(yaml.dump(spec))

    # generate notebook in ipynb format
    jupytext.write(jupytext.read("load.py"), "load.ipynb")

# --- from ploomber__ploomber::tests/spec/test_dagspec.py::test_import_tasks_from_non_list_yaml_file ---
def test_import_tasks_from_non_list_yaml_file(tmp_nbs):
    some_tasks = {"source": "extra_task.py", "product": "extra.ipynb"}
    Path("some_tasks.yaml").write_text(yaml.dump(some_tasks))

    spec_d = yaml.safe_load(Path("pipeline.yaml").read_text())
    spec_d["meta"]["import_tasks_from"] = "some_tasks.yaml"

    with pytest.raises(TypeError) as excinfo:
        DAGSpec(spec_d)
    assert "Expected list when loading YAML file" in str(excinfo.value)

# --- from InfuseAI__piperider::piperider_cli/recipes/utils.py::AbstractRecipeUtils.dryrun_ignored_execute_command_no_outputs ---
def dryrun_ignored_execute_command_no_outputs(command_line, env: Dict = None):
        cmd = shlex.split(command_line)
        proc = None

        try:
            proc = Popen(cmd, env=env or os.environ.copy(), cwd=FileSystem.WORKING_DIRECTORY)
            proc.communicate()
        except Exception:
            if proc:
                proc.kill()
                proc.communicate()
            else:
                return 1

        return proc.returncode

# --- from DeepWisdom__AutoDL::AutoDL_sample_code_submission/Auto_Tabular/model_lib/cb.py::CBModel.bayes_opt.objective ---
def objective(hyperparams):
            hyperparams = self.hyperparams.copy()
            hyperparams['iterations'] = 300
            model = CatBoostClassifier(**{**self.params, **hyperparams})
            model.fit(X_train, y_train)
            pred = model.predict_proba(X_eval)

            if self.is_multi_label:
                score = roc_auc_score(y_eval, pred[:, 1])
            else:
                score = roc_auc_score(y_eval, pred)

            return {'loss': -score, 'status': STATUS_OK}

# --- from ruc-datalab__DeepAnalyze::deepanalyze/ms-swift/swift/llm/infer/infer_engine/infer_engine.py::InferEngine.thread_run ---
def thread_run(target, args=(), kwargs=None):
        kwargs = kwargs or {}

        def func(target, queue, args, kwargs):
            try:
                queue.put(target(*args, **kwargs))
            except Exception as e:
                queue.put(e)

        queue = Queue()
        thread = Thread(target=func, args=(target, queue, args, kwargs))
        thread.start()
        thread.join()
        result = queue.get()
        if isinstance(result, Exception):
            raise result
        return result

# --- from microsoft__RD-Agent::rdagent/components/coder/finetune/unified_validator.py::LLMConfigValidator._inject_required_parameters ---
def _inject_required_parameters(self, config_yaml: str) -> str:
        """Inject required parameters for multi-task environments.

        Uses SYSTEM_MANAGED_PARAMS as the single source of truth.
        """
        config = yaml.safe_load(config_yaml)
        if not isinstance(config, dict):
            return config_yaml

        config.update(SYSTEM_MANAGED_PARAMS)

        logger.info(f"Injected required parameters: {SYSTEM_MANAGED_PARAMS}")
        return yaml.dump(config, default_flow_style=False, sort_keys=False)

# --- from InfuseAI__piperider::piperider_cli/recipes/github_action.py::run_external_command ---
def run_external_command(command_line, env: Dict = None):
    cmd = shlex.split(command_line)
    proc = None
    try:
        proc = Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env or os.environ.copy())
        outs, errs = proc.communicate()
    except BaseException as e:
        if proc:
            proc.kill()
            outs, errs = proc.communicate()
        else:
            return None, e, 1

    if outs is not None:
        outs = outs.decode().strip()
    if errs is not None:
        errs = errs.decode().strip()
    return outs, errs, proc.returncode

# --- from microsoft__RD-Agent::rdagent/scenarios/data_science/example/arf-12-hours-prediction-task/sample.py::copy_other_file ---
def copy_other_file(source: Path, target: Path):
    for item in source.iterdir():
        if item.name in {"train", "test"}:
            continue

        relative_path = item.relative_to(source)
        target_path = target / relative_path

        if item.is_dir():
            shutil.copytree(item, target_path, dirs_exist_ok=True)
            print(f"[COPY DIR] {item} -> {target_path}")
        elif item.is_file():
            target_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, target_path)
            print(f"[COPY FILE] {item} -> {target_path}")
