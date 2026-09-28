# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg499::importlib.util.module_from_spec+importlib.util.spec_from_file_location
# name: importlib_primitive
# summary: Uses importlib.util.module_from_spec, importlib.util.spec_from_file_location across 2 repos
# anchor_symbols: ['importlib.util.module_from_spec', 'importlib.util.spec_from_file_location']
# observed in 2 repos: ['HunterMcGushion__hyperparameter_hunter', 'Lightning-AI__torchmetrics']...

# --- from Lightning-AI__torchmetrics::setup.py::_load_py_module ---
def _load_py_module(fname: str, pkg: str = "torchmetrics"):
    spec = spec_from_file_location(os.path.join(pkg, fname), os.path.join(_PATH_SOURCE, pkg, fname))
    py = module_from_spec(spec)
    spec.loader.exec_module(py)
    return py

# --- from HunterMcGushion__hyperparameter_hunter::hyperparameter_hunter/compat/keras_optimization_helper.py::keras_prep_workflow ---
def keras_prep_workflow(model_initializer, build_fn, extra_params, source_script):
    """Conduct preparation steps necessary before hyperparameter optimization on a `Keras` model.
    Such steps include parsing and modifying `build_fn` to be of the form used by
    :class:`hyperparameter_hunter.optimization.protocol_core.BaseOptPro`, compiling a dummy model to
    identify universal locations of given hyperparameter choices, and creating a simplified
    characterization of the models to be built during optimization in order to enable similar
    Experiment collection

    Parameters
    ----------
    model_initializer: :class:`keras.wrappers.scikit_learn.<KerasClassifier; KerasRegressor>`
        A descendant of :class:`keras.wrappers.scikit_learn.BaseWrapper` used to build a Keras model
    build_fn: Callable
        The `build_fn` value provided to :meth:`keras.wrappers.scikit_learn.BaseWrapper.__init__`.
        Expected to return a compiled Keras model. May contain hyperparameter space choices
    extra_params: Dict
        The parameters expected to be passed to the extra methods of the compiled Keras model. Such
        methods include (but are not limited to) `fit`, `predict`, and `predict_proba`. Some of the
        common parameters given here include `epochs`, `batch_size`, and `callbacks`
    source_script: Str
        Absolute path to a Python file. Should end with one of following extensions: ".py", ".ipynb"

    Returns
    -------
    reusable_build_fn: Callable
        Modified `build_fn` in which hyperparameter space choices are replaced by dict lookups, and
        the signature is given a standard name, and additional input parameters necessary for reuse
    reusable_wrapper_params: Dict
        The parameters expected to be passed to the extra methods of the compiled Keras model. Such
        methods include (but are not limited to) `fit`, `predict`, and `predict_proba`. Some of the
        common parameters given here include `epochs`, `batch_size`, and `callbacks`
    dummy_layers: List
        The layers of a compiled dummy Keras model constructed according to the given
        hyperparameters, in which each layer is a dict containing at least the following: the name
        of the layer class, allowed and used args, and default and used kwargs
    dummy_compile_params: Dict
        The parameters used on the `compile` call for the dummy model. If a parameter is accepted
        by the `compile` method, but is not explicitly given, its default value is included in
        `dummy_compile_params`"""
    #################### Set Temporary Model-Builder Module Location ####################
    temp_module_name = f"__temp_model_builder_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S-%f')}"
    temp_module_dot_path = f"{TEMP_MODULES_DOT_PATH}.{temp_module_name}"
    temp_module_filepath = f"{TEMP_MODULES_DIR_PATH}/{temp_module_name}.py"

    #################### Prepare Model-Builder String ####################
    reusable_build_fn, expected_params = rewrite_model_builder(stringify_model_builder(build_fn))
    temp_module_str = build_temp_model_file(reusable_build_fn, source_script)

    #################### Save and Import Temporary Model Builder ####################
    write_python(temp_module_str, temp_module_filepath)

    if temp_module_name in sys.modules:
        del sys.modules[temp_module_name]

    temp_module_spec = spec_from_file_location(temp_module_dot_path, temp_module_filepath)
    temp_module = module_from_spec(temp_module_spec)
    temp_module_spec.loader.exec_module(temp_module)
    temp_build_fn = temp_module.build_fn

    #################### Translate Hyperparameter Names to Universal Paths ####################
    wrapper_params = dict(params={k: eval(v) for k, v in expected_params.items()}, **extra_params)
    # TODO: Intercept space choices that use callables (like `Categorical([glorot_normal(), orthogonal()])`)
    # TODO: Can't deal with them yet, due to imports unavailable in this context. Raise exception
    wrapper_params, dummified_params = check_dummy_params(wrapper_params)

    if ("optimizer_params" in dummified_params) and ("optimizer" in dummified_params):
        raise ValueError("Can't optimize `optimizer` with `optimizer_params`. Try them separately")

    compiled_dummy = initialize_dummy_model(model_initializer, temp_build_fn, wrapper_params)
    dummy_layers, dummy_compile_params = parameterize_compiled_keras_model(compiled_dummy)
    merged_compile_params = merge_compile_params(dummy_compile_params, dummified_params)
    # FLAG: Will need to deal with capitalization conflicts when comparing similar experiments: `optimizer`="Adam" vs "adam"

    consolidated_layers = consolidate_layers(dummy_layers, class_name_key=False, split_args=False)
    wrapper_params = deep_restricted_update(wrapper_params, dummified_params)

    return (temp_build_fn, wrapper_params, consolidated_layers, merged_compile_params)
