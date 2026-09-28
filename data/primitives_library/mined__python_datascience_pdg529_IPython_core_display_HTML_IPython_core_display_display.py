# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg529::IPython.core.display.HTML+IPython.core.display.display
# name: IPython_primitive
# summary: Uses IPython.core.display.HTML, IPython.core.display.display across 4 repos
# anchor_symbols: ['IPython.core.display.HTML', 'IPython.core.display.display']
# observed in 4 repos: ['firmai__pandapy', 'hi-primus__optimus', 'polyaxon__traceml', 'sfu-db__dataprep']...

# --- from sfu-db__dataprep::dataprep/utils.py::display_html ---
def display_html(html_content: str) -> None:
    """Writes HTML content to a file and displays in browser."""
    if is_notebook():
        display(HTML(html_content))
    else:
        with NamedTemporaryFile(suffix=".html", mode="w", delete=False) as tmpf:
            tmpf.write(html_content)
            tmpf.flush()
            webbrowser.open_new_tab("file://" + tmpf.name)

# --- from hi-primus__optimus::optimus/helpers/output.py::print_html ---
def print_html(html):
    """
    Display() helper to print html code
    :param html: html code to be printed
    :return:
    """
    from IPython.core.display import display, HTML
    try:
        if "DATABRICKS_RUNTIME_VERSION" in os.environ:
            displayHTML(result)
        else:
            display(HTML(html))
        return True
    except NameError:
        return False

# --- from polyaxon__traceml::traceml/traceml/vendor/matplotlylib/mplexporter/tools.py::ipynb_vega_init ---
def ipynb_vega_init():
    """Initialize the IPython notebook display elements

    This function borrows heavily from the excellent vincent package:
    http://github.com/wrobstory/vincent
    """
    try:
        from IPython.core.display import display, HTML
    except ImportError:
        print("IPython Notebook could not be loaded.")

    require_js = """
    if (window['d3'] === undefined) {{
        require.config({{ paths: {{d3: "http://d3js.org/d3.v3.min"}} }});
        require(["d3"], function(d3) {{
          window.d3 = d3;
          {0}
        }});
    }};
    if (window['topojson'] === undefined) {{
        require.config(
            {{ paths: {{topojson: "http://d3js.org/topojson.v1.min"}} }}
            );
        require(["topojson"], function(topojson) {{
          window.topojson = topojson;
        }});
    }};
    """
    d3_geo_projection_js_url = "http://d3js.org/d3.geo.projection.v0.min.js"
    d3_layout_cloud_js_url = "http://wrobstory.github.io/d3-cloud/" "d3.layout.cloud.js"
    topojson_js_url = "http://d3js.org/topojson.v1.min.js"
    vega_js_url = "http://trifacta.github.com/vega/vega.js"

    dep_libs = """$.getScript("%s", function() {
        $.getScript("%s", function() {
            $.getScript("%s", function() {
                $.getScript("%s", function() {
                        $([IPython.events]).trigger("vega_loaded.vincent");
                })
            })
        })
    });""" % (
        d3_geo_projection_js_url,
        d3_layout_cloud_js_url,
        topojson_js_url,
        vega_js_url,
    )
    load_js = require_js.format(dep_libs)
    html = "<script>" + load_js + "</script>"
    display(HTML(html))

# --- from firmai__pandapy::pandapy/__init__.py::table ---
def table(array: np.ndarray, length: int = None, row_values: list = None, column_values: list = None, value_name: str = None) -> None:
    """
    An HTML table to be printed of structured numpy arrays. 

    :param array: np.ndarray, input array to be printed. 
    :param length: int, how many rows to print
    :param row_values: list, a list of the structured array's row names.
    :param column_values: list,  a list of the structured array's column names.
    :param value_name: str, the descriptor to appear on the top left of the table. 
    :return: None print
    """

    if not row_values:
      row_values = range(len(array))
    if not column_values:
      column_values=array.dtype.names
    if not value_name:
      value_name=""

    fields_original = array.dtype.fields
    
    is_unstructured = (array.dtype.names == None)

    if is_unstructured == False:
      array = np.array(array,dtype='object')

    """Numpy array HTML representation function."""
    # Fallbacks for cases where we cannot format HTML tables
    if array.size > 10_000:
        return f"Large numpy array {array.shape} of {array.dtype}"
    if (array.ndim != 2) and (is_unstructured) :
        return f"<pre>{escape(str(array))}</pre>"

    # Table format
    html = [f"<table><tr><th>{value_name}"]
    html += (f"<th>{j}" for j in column_values)

    if length != None:
      row_values = row_values[:length]

    for i, rv in enumerate(row_values):
        html.append(f"<tr><th>{rv}")
        for j, cv in enumerate(column_values):
          if is_unstructured:
            val = array[i,j]
            html.append("<td>")
            html.append(escape(f"{val:.2f}" if array.dtype == float else f"{val}"))
          else:
            val = array[i][j]
            html.append("<td>")
            html.append(escape(f"{val:.3f}" if fields_original[cv][0] == "float" else f"{val}"))
    html.append("</table>")
    display(HTML("".join(html)))
