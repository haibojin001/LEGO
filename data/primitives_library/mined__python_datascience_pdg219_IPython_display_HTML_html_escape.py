# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg219::IPython.display.HTML+html.escape
# name: IPython_html_primitive
# summary: Uses IPython.display.HTML, html.escape across 2 repos
# anchor_symbols: ['IPython.display.HTML', 'html.escape']
# observed in 2 repos: ['Data-Centric-AI-Community__fg-data-profiling', 'fbdesignpro__sweetviz']...

# --- from Data-Centric-AI-Community__fg-data-profiling::src/data_profiling/report/presentation/flavours/widget/notebook.py::get_notebook_iframe_srcdoc ---
def get_notebook_iframe_srcdoc(config: Settings, profile: ProfileReport) -> "HTML":
    """Get the IPython HTML object with iframe with the srcdoc attribute

    Args:
        config: Settings
        profile: The profile report object

    Returns:
        IPython HTML object.
    """
    from IPython.display import HTML

    width = config.notebook.iframe.width
    height = config.notebook.iframe.height
    src = html.escape(profile.to_html())

    iframe = f'<iframe width="{width}" height="{height}" srcdoc="{src}" frameborder="0" allowfullscreen></iframe>'

    return HTML(iframe)

# --- from fbdesignpro__sweetviz::sweetviz/dataframe_report.py::DataframeReport.show_notebook ---
def show_notebook(self, w=None, h=None, scale=None, layout=None, filepath=None, file_layout=None, file_scale=None):
        w = self.use_config_if_none(w, "notebook_width")
        h = self.use_config_if_none(h, "notebook_height")
        scale = float(self.use_config_if_none(scale, "notebook_scale"))
        layout = self.use_config_if_none(layout, "notebook_layout")
        if layout not in ['widescreen', 'vertical']:
            raise ValueError(f"'layout' parameter must be either 'widescreen' or 'vertical'")

        sv_html.load_layout_globals_from_config()
        self.page_layout = layout
        self.scale = scale
        sv_html.set_summary_positions(self)
        sv_html.generate_html_detail(self)
        if self.associations_html_source:
            self.associations_html_source = sv_html.generate_html_associations(self, "source")
        if self.associations_html_compare:
            self.associations_html_compare = sv_html.generate_html_associations(self, "compare")
        self._page_html = sv_html.generate_html_dataframe_page(self)

        width=w
        height=h
        if str(height).lower() == "full":
            height = self.page_height

        # Output to iFrame
        import html
        self._page_html = html.escape(self._page_html)
        iframe = f' <iframe width="{width}" height="{height}" srcdoc="{self._page_html}" frameborder="0" allowfullscreen></iframe>'
        from IPython.display import display
        from IPython.display import HTML
        display(HTML(iframe))

        if filepath is not None:
            # We cannot just write out the same HTML as the notebook, as that one has been processed so as to
            # remove extraneous headings so it is nicely inserted into the notebook.
            # Instead, just do something similar to the "show_html()" code, but without its less-relevant printouts etc.
            # f = open(filepath, 'w', encoding="utf-8")
            # f.write(self._page_html)
            # f.close()
            scale = float(self.use_config_if_none(file_scale, "html_scale"))
            layout = self.use_config_if_none(file_layout, "html_layout")
            if layout not in ['widescreen', 'vertical']:
                raise ValueError(f"'layout' parameter for file output must be either 'widescreen' or 'vertical'")
            sv_html.load_layout_globals_from_config()
            self.page_layout = layout
            self.scale = scale
            sv_html.set_summary_positions(self)
            sv_html.generate_html_detail(self)
            if self.associations_html_source:
                self.associations_html_source = sv_html.generate_html_associations(self, "source")
            if self.associations_html_compare:
                self.associations_html_compare = sv_html.generate_html_associations(self, "compare")
            self._page_html = sv_html.generate_html_dataframe_page(self)

            f = open(filepath, 'w', encoding="utf-8")
            f.write(self._page_html)
            f.close()
            self.verbose_print(f"Report '{filepath}' was saved to storage.")

        if len(self.corr_warning):
            print("WARNING: one or more correlations had an edge-case/error and a 1.0 correlation was assigned\n"
                  "(likely due to only a single row containing non-NaN values for both correlated features)\n"
                  "Affected correlations:" + str(self.corr_warning))

        # Auto-log to comet_ml if desired & present
        self._comet_ml_logger = comet_ml_logger.CometLogger()
        if self._comet_ml_logger._logging:
            self.generate_comet_friendly_html()
            self._comet_ml_logger.log_html(self._page_html)
            self._comet_ml_logger.end()
