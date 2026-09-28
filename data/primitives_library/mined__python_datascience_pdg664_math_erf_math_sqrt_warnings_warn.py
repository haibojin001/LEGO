# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg664::math.erf+math.sqrt+warnings.warn
# name: math_warnings_primitive
# summary: Uses math.erf, math.sqrt, warnings.warn across 2 repos
# anchor_symbols: ['math.erf', 'math.sqrt', 'warnings.warn']
# observed in 2 repos: ['OML-Team__open-metric-learning', 'devAmoghS__Machine-Learning-with-Python']...

# --- from devAmoghS__Machine-Learning-with-Python::helpers/probabilty.py::normal_cdf ---
def normal_cdf(x, mu=0, sigma=1.0):
    return (1 + math.erf((x - mu) / math.sqrt(2) / sigma)) / 2

# --- from OML-Team__open-metric-learning::oml/models/vit_dino/external/vision_transformer.py::_no_grad_trunc_normal_.norm_cdf ---
def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

# --- from OML-Team__open-metric-learning::oml/models/vit_unicom/external/vision_transformer.py::_trunc_normal_.norm_cdf ---
def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0
