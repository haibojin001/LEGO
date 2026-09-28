# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg218::IPython.display.display+ipywidgets.Output
# name: IPython_ipywidgets_primitive
# summary: Uses IPython.display.display, ipywidgets.Output across 2 repos
# anchor_symbols: ['IPython.display.display', 'ipywidgets.Output']
# observed in 2 repos: ['Data-Centric-AI-Community__fg-data-profiling', 'vaexio__vaex']...

# --- from Data-Centric-AI-Community__fg-data-profiling::src/data_profiling/report/presentation/flavours/widget/sample.py::WidgetSample.render ---
def render(self) -> widgets.VBox:
        out = Output()
        with out:
            display(self.content["sample"])

        name = widgets.HTML(f"<h4>{self.content['name']}</h4>")
        return widgets.VBox([name, out])

# --- from Data-Centric-AI-Community__fg-data-profiling::src/data_profiling/report/presentation/flavours/widget/duplicate.py::WidgetDuplicate.render ---
def render(self) -> widgets.VBox:
        out = Output()
        with out:
            display(self.content["duplicate"])

        name = widgets.HTML(f"<h4>{self.content['name']}</h4>")
        return widgets.VBox([name, out])

# --- from vaexio__vaex::packages/vaex-jupyter/vaex/jupyter/utils.py::interactive_selection.wrapped ---
def wrapped(f_interact):
        if not hasattr(f_interact, "widget"):
            output = widgets.Output()

            def _selection_changed(df, selection_name):
                with output:
                    clear_output(wait=True)
                    f_interact(df, selection_name)
            hook = df.signal_selection_changed.connect(_selection_changed)
            _selection_hooks.append((df, hook))
            _selection_changed(df, None)
            display(output)
            return functools.wraps(f_interact)
        else:
            def _selection_changed(df, selection_name):
                f_interact.widget.update(df, selection_name)
            hook = df.signal_selection_changed.connect(_selection_changed)
            _selection_hooks.append((df, hook))
            return functools.wraps(f_interact)

# --- from vaexio__vaex::packages/vaex-jupyter/vaex/jupyter/utils.py::interactive_selection ---
def interactive_selection(df):
    global _selection_hooks

    def wrapped(f_interact):
        if not hasattr(f_interact, "widget"):
            output = widgets.Output()

            def _selection_changed(df, selection_name):
                with output:
                    clear_output(wait=True)
                    f_interact(df, selection_name)
            hook = df.signal_selection_changed.connect(_selection_changed)
            _selection_hooks.append((df, hook))
            _selection_changed(df, None)
            display(output)
            return functools.wraps(f_interact)
        else:
            def _selection_changed(df, selection_name):
                f_interact.widget.update(df, selection_name)
            hook = df.signal_selection_changed.connect(_selection_changed)
            _selection_hooks.append((df, hook))
            return functools.wraps(f_interact)
    return wrapped
