# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg2::IPython.display.HTML+IPython.display.display+argparse.ArgumentParser
# name: IPython_argparse_primitive
# summary: Uses IPython.display.HTML, IPython.display.display, argparse.ArgumentParser, argparse.FileType across 23 repos
# anchor_symbols: ['IPython.display.HTML', 'IPython.display.display', 'argparse.ArgumentParser', 'argparse.FileType', 'base64.b64decode', 'boto3.client']
# observed in 23 repos: ['CamDavidsonPilon__lifelines', 'Data-Centric-AI-Community__fg-data-profiling', 'HDI-Project__ATM', 'HazyResearch__meerkat', 'JacksonWuxs__DaPy']...

# --- from piskvorky__gensim::gensim/utils.py::RepeatCorpus.__iter__ ---
def __iter__(self):
        return itertools.islice(itertools.cycle(self.corpus), self.reps)

# --- from alteryx__featuretools::featuretools/tests/testing_utils/generate_fake_dataframe.py::generate_fake_dataframe.randomize ---
def randomize(values_):
        random.seed(10)
        values = values_.copy()
        random.shuffle(values)
        return values

# --- from datafold__data-diff::tests/test_database_types.py::IntFaker.__iter__ ---
def __iter__(self) -> Iterator[int]:
        initial = -128
        step = 1
        return islice(chain(self.MANUAL_FAKES, accumulate(repeat(step), initial=initial)), self.max)

# --- from datafold__data-diff::tests/test_database_types.py::FloatFaker.__iter__ ---
def __iter__(self) -> Iterator[float]:
        initial = -10.0001
        step = 0.00571
        return islice(chain(self.MANUAL_FAKES, accumulate(repeat(step), initial=initial)), self.max)

# --- from Data-Centric-AI-Community__fg-data-profiling::src/data_profiling/utils/notebook.py::full_width ---
def full_width() -> None:
    """Resize the notebook to use it's full width"""
    from IPython.display import HTML, display

    display(HTML("<style>.container { width:100% !important; }</style>"))

# --- from HDI-Project__ATM::atm/utilities.py::base_64_to_object ---
def base_64_to_object(b64str):
    """
    Inverse of object_to_base_64.
    Decode base64-encoded string and then unpickle it.
    """
    decoded = base64.b64decode(b64str)
    return pickle.loads(decoded)

# --- from pgalko__BambooAI::bambooai/output_manager.py::OutputManager.send_html_content ---
def send_html_content(self, html_content, chain_id=None):
        """Display HTML content directly in the notebook"""
        if self.is_notebook:
            display(HTML(html_content))
        else:
            pass

# --- from approximatelabs__sketch::sketch/pandas_extension.py::SketchHelper.ask ---
def ask(self, question, call_display=True):
        result = call_prompt_on_dataframe(self._obj, ask_from_parts, question=question)
        if not call_display:
            return result
        display(HTML(f"""{result}"""))

# --- from explosion__spaCy::spacy/tests/matcher/test_matcher_api.py::test_matcher_empty_patterns_warns ---
def test_matcher_empty_patterns_warns(en_vocab):
    matcher = Matcher(en_vocab)
    assert len(matcher) == 0
    doc = Doc(en_vocab, words=["This", "is", "quite", "something"])
    with pytest.warns(UserWarning):
        matcher(doc)
    assert len(doc.ents) == 0

# --- from explosion__spaCy::spacy/tests/training/test_new_example.py::test_Example_from_dict_with_entities_invalid ---
def test_Example_from_dict_with_entities_invalid(annots):
    vocab = Vocab()
    predicted = Doc(vocab, words=annots["words"])
    with pytest.warns(UserWarning):
        example = Example.from_dict(predicted, annots)
    assert len(list(example.reference.ents)) == 0

# --- from pgalko__BambooAI::bambooai/output_manager.py::OutputManager.display_system_messages ---
def display_system_messages(self, message, chain_id=None):
        if self.is_notebook:
            display(HTML(f'<span style="color:{self.color_usr_input_rank};">-- info: \"{message}\"</span>'))
            time.sleep(1)
        else:
            cprint(f"\n--info: \"{message}\"", self.color_usr_input_rank)

# --- from HazyResearch__meerkat::meerkat/dataframe.py::DataFrame._ipython_display_ ---
def _ipython_display_(self):
        from IPython.display import HTML, display

        from meerkat.state import state

        if state.frontend_info is None:
            max_rows = meerkat.config.display.max_rows
            df, formatters = self._repr_pandas_(max_rows=max_rows)
            return display(
                HTML(df.to_html(formatters=formatters, max_rows=max_rows, escape=False))
            )

        return self.gui.table()._ipython_display_()
