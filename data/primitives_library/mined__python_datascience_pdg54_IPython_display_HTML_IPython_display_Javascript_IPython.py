# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg54::IPython.display.HTML+IPython.display.Javascript+IPython.display.display
# name: IPython_Metal_primitive
# summary: Uses IPython.display.HTML, IPython.display.Javascript, IPython.display.display, Metal.MTLCreateSystemDefaultDevice across 22 repos
# anchor_symbols: ['IPython.display.HTML', 'IPython.display.Javascript', 'IPython.display.display', 'Metal.MTLCreateSystemDefaultDevice', 'Metal.MTLSize', 'annoy.AnnoyIndex']
# observed in 22 repos: ['BiomedSciAI__causallib', 'BlackHC__toma', 'HazyResearch__meerkat', 'HunterMcGushion__hyperparameter_hunter', 'capitalone__DataProfiler']...

# --- from vaexio__vaex::packages/vaex-core/vaex/test/dataset.py::TestDataset.test_covar.covar ---
def covar(x, y):
			mask = np.isfinite(x * y)
			#w = np.isfinite(x * y) * 1.0
			x = x[mask]
			y = y[mask]
			return np.cov([x, y], bias=1)[1,0]

# --- from vaexio__vaex::packages/vaex-core/vaex/ext/ipyvolume.py::PlotDefault.show ---
def show(self):
        container = p3.gcc()
        vbox = widgets.VBox([container, self.progress, widgets.VBox(self.tools), self.output])
        display(vbox)

# --- from sinaptik-ai__pandas-ai::extensions/sandbox/docker/tests/test_serializer.py::TestCustomEncoder.test_encode_numpy ---
def test_encode_numpy(self):
        data = {"int": np.int64(42), "float": np.float64(3.14)}
        encoded = json.dumps(data, cls=CustomEncoder)
        self.assertEqual(json.loads(encoded), {"int": 42, "float": 3.14})

# --- from HazyResearch__meerkat::tests/meerkat/ops/embed/test__init__.py::simple_text_transform ---
def simple_text_transform(text: str):
    return torch.tensor(
        [
            int.from_bytes(hashlib.sha256(token.encode("utf-8")).digest(), "big") % 100
            for token in text.split(" ")
        ]
    )[:1]

# --- from ploomber__sklearn-evaluation::src/sklearn_evaluation/report/serialize.py::figure2base64 ---
def figure2base64(fig):
    io = BytesIO()
    fig.savefig(io, format="png")
    try:
        fig_base64 = base64.encodebytes(io.getvalue())  # py3
    except Exception:
        fig_base64 = base64.encodestring(io.getvalue())  # py2
    return fig_base64

# --- from deepchecks__deepchecks::deepchecks/nlp/checks/data_integrity/under_annotated_segments.py::UnderAnnotatedSegments._get_box_boundaries ---
def _get_box_boundaries(feature_data: pd.Series, segment: Tuple[float, float]) -> Tuple[float, float]:
        lower = segment[0] if np.isfinite(segment[0]) else np.nanmin(feature_data)
        upper = segment[1] if np.isfinite(segment[1]) else np.nanmax(feature_data)
        return lower, upper

# --- from sfu-db__dataprep::dataprep/connector/generator/ui.py::_make_header ---
def _make_header(seq: int, title: str) -> HTML:
    """make a header"""

    return HTML(
        value=(
            """<h3 style="background-color:#E8E8E8; background-size: 100px; ">"""
            f"""<span style="color: #ff0000">{seq}.</span>{title}</h3>"""
        ),
        layout=Layout(width="100%"),
    )

# --- from HazyResearch__meerkat::meerkat/interactive/app/src/lib/shared/cell/basic/__init__.py::ScalarFormatter.encode ---
def encode(self, cell: Any):
        # check for native python nan
        if isinstance(cell, float) and math.isnan(cell):
            return "NaN"

        if isinstance(cell, np.generic):
            if pd.isna(cell):
                return "NaN"
            return cell.item()

        if hasattr(cell, "as_py"):
            return cell.as_py()
        return str(cell)

# --- from sfu-db__dataprep::dataprep/connector/generator/ui.py::ConfigGeneratorUI._make_generator_option ---
def _make_generator_option(self) -> VBox:

        self.table_path_box = Text(
            disabled=False, placeholder="table_path", layout={"width": "90%"}
        )

        return VBox(
            [
                _make_header(5, "Generator Options"),
                HBox([Label(value="Table Path:"), self.table_path_box], layout=BOX_LAYOUT),
            ],
            layout={"align_items": "center"},
        )

# --- from mwaskom__seaborn::tests/test_categorical.py::TestBarPlot.test_error_caps ---
def test_error_caps(self):

        x, y = ["a", "b", "c"] * 2, [1, 2, 3, 4, 5, 6]
        ax = barplot(x=x, y=y, capsize=.8, errorbar="pi")

        assert len(ax.patches) == len(ax.lines)
        for bar, error in zip(ax.patches, ax.lines):
            pos = error.get_xdata()
            assert len(pos) == 8
            assert np.nanmin(pos) == approx(bar.get_x())
            assert np.nanmax(pos) == approx(bar.get_x() + bar.get_width())

# --- from BlackHC__toma::toma/cpu_memory.py::set_cpu_memory_limit ---
def set_cpu_memory_limit(num_gigabytes):
    try:
        import resource

        num_bytes = int(num_gigabytes * 2 ** 30)
        _, hard_limit = resource.getrlimit(resource.RLIMIT_AS)
        if hard_limit != resource.RLIM_INFINITY:
            hard_limit = min(num_bytes, hard_limit)
        else:
            hard_limit = num_bytes
        resource.setrlimit(resource.RLIMIT_AS, (hard_limit, hard_limit))
    except ImportError:
        pass

# --- from mwaskom__seaborn::tests/test_categorical.py::TestBarPlot.test_error_caps_native_scale ---
def test_error_caps_native_scale(self):

        x, y = [2, 4, 20] * 2, [1, 2, 3, 4, 5, 6]
        ax = barplot(x=x, y=y, capsize=.8, native_scale=True, errorbar="pi")

        assert len(ax.patches) == len(ax.lines)
        for bar, error in zip(ax.patches, ax.lines):
            pos = error.get_xdata()
            assert len(pos) == 8
            assert np.nanmin(pos) == approx(bar.get_x())
            assert np.nanmax(pos) == approx(bar.get_x() + bar.get_width())
