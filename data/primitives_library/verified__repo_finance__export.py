from html import escape
from pathlib import Path
from xml.etree.ElementTree import Element, ElementTree, SubElement

import numpy as np
import pandas as pd


def export_table(table: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    safe = table.copy()

    def sanitize(value):
        if isinstance(value, pd.Timestamp) and value.tzinfo is not None:
            return value.isoformat()
        if isinstance(value, str) and value.startswith(
            ("=", "+", "-", "@", "\t", "\r")
        ):
            return "'" + value
        return value

    safe = safe.map(sanitize)
    safe.columns = [sanitize(column) for column in safe.columns]
    safe.index = safe.index.map(sanitize)

    if path.suffix.lower() == ".csv":
        safe.to_csv(path)
    elif path.suffix.lower() == ".xlsx":
        safe.to_excel(path, engine="openpyxl")
    else:
        raise ValueError("export path must end in .csv or .xlsx")

    return path


def network_gexf(
    correlation: pd.DataFrame, path: str | Path, threshold: float = 0.2
) -> Path:
    if (
        not correlation.index.equals(pd.Index(correlation.columns))
        or not np.isfinite(correlation).all().all()
    ):
        raise ValueError("finite square labeled matrix required")

    if not 0 <= threshold <= 1 or not np.allclose(correlation, correlation.T):
        raise ValueError("symmetric matrix and threshold in [0,1] required")

    root = Element(
        "gexf",
        xmlns="http://www.gexf.net/1.2draft",
        version="1.2",
    )
    graph = SubElement(
        root,
        "graph",
        mode="static",
        defaultedgetype="undirected",
    )
    nodes = SubElement(graph, "nodes")
    edges = SubElement(graph, "edges")

    for index, ticker in enumerate(correlation):
        SubElement(nodes, "node", id=str(index), label=str(ticker))

    edge_id = 0
    for row in range(len(correlation)):
        for column in range(row + 1, len(correlation)):
            weight = correlation.iloc[row, column]
            if abs(weight) >= threshold:
                SubElement(
                    edges,
                    "edge",
                    id=str(edge_id),
                    source=str(row),
                    target=str(column),
                    weight=str(weight),
                )
                edge_id += 1

    path = Path(path)
    ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)
    return path


def research_report(
    tables: dict[str, pd.DataFrame],
    path: str | Path,
    *,
    title: str = "Market research",
) -> Path:
    parts = [
        '<!doctype html><meta charset="utf-8">',
        f"<title>{escape(title)}</title>",
        "<style>body{font:16px system-ui;max-width:1200px;margin:40px auto;padding:20px}table{border-collapse:collapse}td,th{padding:8px;border-bottom:1px solid #ddd;text-align:right}h2{margin-top:40px}</style>",
        f"<h1>{escape(title)}</h1>",
    ]

    for name, table in tables.items():
        parts.extend([
            f"<h2>{escape(name)}</h2>",
            table.to_html(escape=True),
        ])
        metadata = {
            key: table.attrs[key]
            for key in ("source", "retrieved_at", "complete", "total_matches")
            if key in table.attrs
        }
        if metadata:
            parts.append(f"<p>{escape(str(metadata))}</p>")

    path = Path(path)
    path.write_text("\n".join(parts), encoding="utf-8")
    return path