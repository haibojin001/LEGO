# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg686::streamlit.markdown+streamlit.warning
# name: streamlit_primitive
# summary: Uses streamlit.markdown, streamlit.warning across 2 repos
# anchor_symbols: ['streamlit.markdown', 'streamlit.warning']
# observed in 2 repos: ['WecoAI__aideml', 'underneathall__pinferencia']...

# --- from WecoAI__aideml::aide/webui/app.py::WebUI.load_css ---
def load_css():
        """
        Load custom CSS styles from 'style.css' file.
        """
        css_file = Path(__file__).parent / "style.css"
        if css_file.exists():
            with open(css_file) as f:
                st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
        else:
            st.warning(f"CSS file not found at: {css_file}")

# --- from underneathall__pinferencia::pinferencia/frontend/templates/base.py::BaseTemplate.render_header ---
def render_header(self):
        title = (
            self.metadata["display_name"]
            if self.metadata.get("display_name")
            else self.title
        )
        st.markdown(
            f'<h1 style="text-align: center;">{title}</h1>',
            unsafe_allow_html=True,
        )
        if self.metadata.get("description"):
            st.markdown(self.metadata["description"], unsafe_allow_html=True)
        if not self.metadata.get("input_type") or not self.metadata.get("output_type"):
            st.warning(
                "Request/response schema of the model service not properly defined."
                " Frontend may not work as expected. Refer to"
                " https://pinferencia.underneathall.app/how-to-guides/schema/ on"
                " how to define the schema of the request and response of the service."
            )
