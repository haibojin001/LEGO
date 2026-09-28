# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg56::logging.FileHandler+logging.Formatter+logging.StreamHandler
# name: logging_primitive
# summary: Uses logging.FileHandler, logging.Formatter, logging.StreamHandler, logging.getLogger across 18 repos
# anchor_symbols: ['logging.FileHandler', 'logging.Formatter', 'logging.StreamHandler', 'logging.getLogger']
# observed in 18 repos: ['AgnostiqHQ__covalent', 'DeepWisdom__AutoDL', 'HDI-Project__ATM', 'HunterMcGushion__hyperparameter_hunter', 'InfuseAI__piperider']...

# --- from rpy2__rpy2::rpy2-rinterface/src/rpy2/situation/__init__.py::set_default_logging ---
def set_default_logging():
    logformatter = logging.Formatter('%(name)s: %(message)s')
    loghandler = logging.StreamHandler()
    loghandler.setFormatter(logformatter)
    logger.addHandler(loghandler)

# --- from ploomber__ploomber::src/ploomber/cli/parsers.py::_configure_logger ---
def _configure_logger(args):
    """Configure logger if user passed --log/--log-file args"""
    if hasattr(args, "log"):
        if args.log is not None:
            logging.basicConfig(level=args.log.upper())

    if hasattr(args, "log_file"):
        if args.log_file is not None:
            file_handler = logging.FileHandler(args.log_file)
            logging.getLogger().addHandler(file_handler)

# --- from AgnostiqHQ__covalent::tests/stress_tests/benchmarks/conftest.py::benchmark ---
def benchmark():
    logger = logging.getLogger("metricsLogger")
    logger.setLevel(logging.DEBUG)

    fileHandler = logging.FileHandler("metrics.log")
    fileHandler.setFormatter(logging.DEBUG)

    formatter = logging.Formatter("%(asctime)s-%(name)s-%(levelname)s-%(message)s")
    fileHandler.setFormatter(formatter)

    logger.addHandler(fileHandler)
    yield (
        run_benchmark,
        logger,
    )

# --- from DeepWisdom__AutoDL::AutoDL_sample_code_submission/Auto_Image/skeleton/projects/others.py::get_logger ---
def get_logger(name, stream=sys.stderr):
    formatter = logging.Formatter(fmt='[%(asctime)s %(levelname)s %(filename)s] %(message)s')

    handler = logging.StreamHandler(stream)
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    level = logging.INFO if os.environ.get('LOG_LEVEL', 'INFO') == 'INFO' else logging.DEBUG
    logger.setLevel(level)
    logger.addHandler(handler)
    return logger

# --- from ruc-datalab__DeepAnalyze::playground/TableQA/tests/utils/llm.py::setup_caller_logger ---
def setup_caller_logger():
    logger = logging.getLogger("model_caller")
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    return logger

# --- from microsoft__nni::examples/nas/legacy/cream/lib/utils/util.py::get_logger ---
def get_logger(file_path):
    """ Make python logger """
    log_format = '%(asctime)s | %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format=log_format, datefmt='%m/%d %I:%M:%S %p')
    logger = logging.getLogger('')

    formatter = logging.Formatter(log_format, datefmt='%m/%d %I:%M:%S %p')
    file_handler = logging.FileHandler(file_path)
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)

    return logger

# --- from DeepWisdom__AutoDL::AutoDL_sample_code_submission/Auto_Video/skeleton/projects/others.py::get_logger ---
def get_logger(name, stream=sys.stderr, file_path='debug.log'):
    formatter = logging.Formatter(fmt='[%(asctime)s %(levelname)s %(filename)s] %(message)s')

    handler = logging.StreamHandler(stream)
    # handler = logging.FileHandler(file_path)
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    level = logging.INFO if os.environ.get('LOG_LEVEL', 'INFO') == 'INFO' else logging.DEBUG
    logger.setLevel(level)
    logger.addHandler(handler)
    return logger

# --- from run-house__kubetorch::python_client/tests/utils.py::get_test_logger ---
def get_test_logger(name=None):
    """Use a generic logger for testing that doesn't require a kubetorch dependency."""
    logger = logging.getLogger(name or __name__)

    # Avoid adding handlers if they already exist
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

    return logger

# --- from run-house__kubetorch::python_client/tests/assets/app/summer_app.py::get_test_logger ---
def get_test_logger(name=None):
    """Use a generic logger for testing that doesn't require a kubetorch dependency."""
    logger = logging.getLogger(name or __name__)

    # Avoid adding handlers if they already exist
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

    return logger

# --- from microsoft__nni::examples/nas/legacy/cdarts/utils.py::get_logger ---
def get_logger(file_path):
    """ Make python logger """
    logger = logging.getLogger('cdarts')
    log_format = '%(asctime)s | %(message)s'
    formatter = logging.Formatter(log_format, datefmt='%m/%d %I:%M:%S %p')
    file_handler = logging.FileHandler(file_path)
    file_handler.setFormatter(formatter)
    # stream_handler = logging.StreamHandler()
    # stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    # logger.addHandler(stream_handler)
    logger.setLevel(logging.INFO)

    return logger

# --- from ruc-datalab__DeepAnalyze::playground/TableQA/tests/aitqa.py::setup_logger ---
def setup_logger(log_file):
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logger = logging.getLogger("aitqa_processor")
    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    console_handler = logging.StreamHandler()

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

# --- from alteryx__evalml::evalml/utils/logger.py::get_logger ---
def get_logger(name):
    """Get the logger with the associated name.

    Args:
        name (str): Name of the logger to get.

    Returns:
        The logger object with the associated name.
    """
    logger = logging.getLogger(name)
    if not len(logger.handlers):
        logger.setLevel(logging.DEBUG)
        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setLevel(logging.INFO)
        stdout_handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(stdout_handler)
    return logger
