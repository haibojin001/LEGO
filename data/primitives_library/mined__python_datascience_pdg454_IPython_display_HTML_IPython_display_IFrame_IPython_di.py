# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg454::IPython.display.HTML+IPython.display.IFrame+IPython.display.display
# name: IPython_primitive
# summary: Uses IPython.display.HTML, IPython.display.IFrame, IPython.display.display across 3 repos
# anchor_symbols: ['IPython.display.HTML', 'IPython.display.IFrame', 'IPython.display.display']
# observed in 3 repos: ['mljar__mljar-supervised', 'ploomber__ploomber', 'sfu-db__dataprep']...

# --- from sfu-db__dataprep::dataprep/clean/clean_df_gui.py::UserInterface.display ---
def display(self) -> None:
        """Display the GUI."""
        launch(self.df)

        path_to_local_server = "http://localhost:7680"
        display(IFrame(path_to_local_server, width=900, height=500))

# --- from mljar__mljar-supervised::supervised/base_automl.py::BaseAutoML._show_report ---
def _show_report(self, main_readme_html, width=900, height=1200):
        from IPython.display import HTML, IFrame

        if os.environ.get("KAGGLE_KERNEL_RUN_TYPE") is None:
            with open(main_readme_html) as fin:
                return HTML(fin.read())
        else:
            return IFrame(main_readme_html, width=width, height=height)

# --- from ploomber__ploomber::src/ploomber/dag/plot.py::embedded_html ---
def embedded_html(path):
    # set output cell to hold 100% of the embedded content
    display(HTML("<style>.output.output_scroll { height: 100% }</style>"))

    # create a copy of the file to embed in a local
    # directory inorder to load it with iFrame

    embedded_assets_dir = "embedded-assets"
    clear_embedded_assets_dir(embedded_assets_dir)

    original_file = Path(path)
    local_file_copy = os.path.join(embedded_assets_dir, original_file.name)

    with open(local_file_copy, "w+") as html:
        copy_of_html = Path(path).read_text()
        html.write(copy_of_html)

    iframe = IFrame(src=local_file_copy, width="100%", height=600)

    return iframe
