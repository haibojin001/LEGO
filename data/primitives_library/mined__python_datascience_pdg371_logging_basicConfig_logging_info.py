# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg371::logging.basicConfig+logging.info
# name: logging_primitive
# summary: Uses logging.basicConfig, logging.info across 2 repos
# anchor_symbols: ['logging.basicConfig', 'logging.info']
# observed in 2 repos: ['awslabs__gluonts', 'sematic-ai__sematic']...

# --- from sematic-ai__sematic::sematic/examples/retry/pipeline.py::raise_exception ---
def raise_exception() -> float:
    """
    A toy function to illustrate the retry mechanism
    """
    logging.basicConfig(level=logging.INFO)
    random_number = random.random()
    logging.info("Random number {}".format(random_number))
    if random_number < 0.1:
        return random_number

    logging.info("Raising exception")
    raise SomeException

# --- from awslabs__gluonts::src/gluonts/nursery/tsbench/src/evaluate.py::main ---
def main(
    dataset: str,
    model: str,
    seed: Optional[int],
    data_path: str,
    model_path: str,
    # Options
    validate: bool,
    use_tqdm: bool,
    # Common hyperparameters
    training_fraction: int,
    num_learning_rate_decays: int,
    learning_rate: float,
    context_length_multiple: int,
    # Model hyperparameters
    **kwargs: int,
) -> None:
    """
    Trains and evaluates a GluonTS model, logging all metrics and storing the
    generated forecasts on the test set (and, optionally, the validation set).
    """
    # Basic configuration
    env.use_tqdm = use_tqdm
    logging.basicConfig(level=logging.INFO)

    # Setup
    model_dir = Path(model_path)
    if seed is not None:
        np.random.seed(seed)
        mx.random.seed(seed)

    # Initialize data and model
    data = get_dataset_config(dataset, data_path)
    config = get_model_config(
        model,
        training_fraction=training_fraction,
        learning_rate=learning_rate,
        context_length_multiple=context_length_multiple,
        **{
            key[len(model) + 1 :]: value
            for key, value in kwargs.items()
            if key.startswith(model)
        },
    )
    logging.info("Using model configuration %s.", config)

    # Run training and evaluation
    logging.info("Fitting estimator...")
    fit_result = fit_estimator(
        config,
        data,
        num_learning_rate_decays=num_learning_rate_decays,
        validate=validate,
    )

    logging.info("Saving predictors...")
    fit_result.serialize_predictors(model_dir / "models")

    if validate and isinstance(config, TrainConfig):
        logging.info("Evaluating predictors on validation data...")
        fit_result.evaluate_predictors(
            data,
            data.data.val(),
            model_dir / "val_predictions",
            validation=True,
        )

    logging.info("Evaluating predictors on test data...")
    fit_result.evaluate_predictors(
        data, data.data.test(), model_dir / "predictions"
    )
