# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg410::collections.OrderedDict+kale.embed.attention.PositionalEncoding+kale.embed.nn.FCNet
# name: collections_kale_primitive
# summary: Uses collections.OrderedDict, kale.embed.attention.PositionalEncoding, kale.embed.nn.FCNet, kale.embed.nn.RandomLayer across 4 repos
# anchor_symbols: ['collections.OrderedDict', 'kale.embed.attention.PositionalEncoding', 'kale.embed.nn.FCNet', 'kale.embed.nn.RandomLayer', 'kale.pipeline.drugban_trainer.DrugbanTrainer', 'kale.predict.class_domain_nets.DomainNetSmallImage']
# observed in 4 repos: ['DeepWisdom__AutoDL', 'bfortuner__ml-glossary', 'pykale__pykale', 'ruc-datalab__DeepAnalyze']...

# --- from pykale__pykale::tests/pipeline/test_multimodal_trainer.py::DummyMVAE.__init__ ---
def __init__(self):
        super().__init__()
        self.dummy_param = nn.Parameter(torch.zeros(1))

# --- from pykale__pykale::tests/helpers/boring_model.py::VideoBoringModel.__init__ ---
def __init__(self, in_channel):
        super().__init__()
        self.avg_pool3d = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(in_channel, 1024)

# --- from DeepWisdom__AutoDL::AutoDL_sample_code_submission/Auto_Video/architectures/mc3.py::BasicStem.__init__ ---
def __init__(self):
        super(BasicStem, self).__init__(
            nn.Conv3d(3, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2),
                      padding=(1, 3, 3), bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=False))

# --- from bfortuner__ml-glossary::code/cnn.py::linear_bn_relu_drop ---
def linear_bn_relu_drop(in_chans, out_chans, dropout=0.5, bias=False):
    layers = [
        nn.Linear(in_chans, out_chans, bias=bias),
        nn.BatchNorm1d(out_chans),
        nn.ReLU(inplace=True)
    ]
    if dropout > 0:
        layers.append(nn.Dropout(dropout))
    return layers

# --- from DeepWisdom__AutoDL::AutoDL_sample_code_submission/Auto_Tabular/model_lib/dnn.py::DnnModel.fc ---
def fc(self, x, out_dim, weight_decay):
        x = Dropout(0.2)(x)
        x = Dense(out_dim,
                  kernel_regularizer=keras.regularizers.l2(weight_decay),
                  use_bias=False,
                  )(x)
        x = BatchNormalization(axis=1)(x)
        x = Activation('relu')(x)
        return x

# --- from ruc-datalab__DeepAnalyze::deepanalyze/ms-swift/swift/tuners/scetuning/scetuning_components.py::choose_weight_type ---
def choose_weight_type(weight_type, dim):
    if weight_type == "gate":
        scaling = nn.Linear(dim, 1)
    elif weight_type == "scale":
        scaling = nn.Parameter(torch.Tensor(1))
        scaling.data.fill_(1)
    elif weight_type == "scale_channel":
        scaling = nn.Parameter(torch.Tensor(dim))
        scaling.data.fill_(1)
    elif weight_type and weight_type.startswith("scalar"):
        scaling = float(weight_type.split("_")[-1])
    else:
        scaling = None
    return scaling

# --- from ruc-datalab__DeepAnalyze::deepanalyze/ms-swift/swift/tuners/restuning_components.py::init_weight_type ---
def init_weight_type(dim, weight_type):
    if weight_type is None:
        scaling = None
    elif weight_type == "gate":
        scaling = nn.Linear(dim, 1)
    elif weight_type == "scale":
        scaling = nn.Parameter(torch.Tensor(1))
        scaling.data.fill_(1)
    elif weight_type == "scale_kv":
        scaling_k = nn.Parameter(torch.Tensor(1))
        scaling_k.data.fill_(1)
        scaling_v = nn.Parameter(torch.Tensor(1))
        scaling_v.data.fill_(1)
        scaling = (scaling_k, scaling_v)
    elif weight_type == "scale_channel":
        scaling = nn.Parameter(torch.Tensor(dim))
        scaling.data.fill_(1)
    elif weight_type == "scale_kv_channel":
        scaling_k = nn.Parameter(torch.Tensor(dim))
        scaling_k.data.fill_(1)
        scaling_v = nn.Parameter(torch.Tensor(dim))
        scaling_v.data.fill_(1)
        scaling = (scaling_k, scaling_v)
    elif weight_type and weight_type.startswith("scalar"):
        scaling = float(weight_type.split("_")[-1])
    else:
        scaling = None
    return scaling
