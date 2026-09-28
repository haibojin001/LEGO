"""The flat legacy library keeps every safe source entry as an auditable proxy."""

import json

from lego.library.codeface import CodeFace
from lego.library.convert_legacy import convert


def test_convert_preserves_unparsed_and_no_export_entries(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    samples = {
        "valid.py": "def useful():\n    return 1\n",
        "imports.py": "import math\n",
        "broken.py": "def unfinished(\n",
    }
    for filename, text in samples.items():
        (source / filename).write_text(text)
    (source / "manifest.json").write_text(json.dumps([
        {"file": filename, "name": filename[:-3],
         "summary": f"{filename} capability", "kind": "mined"}
        for filename in samples
    ]))
    target = tmp_path / "library"
    report = convert(str(source), str(target))
    library = CodeFace(str(target), allow_unvalidated=True)
    assert len(CodeFace(str(target))) == 0

    assert report["manifest_entries"] == report["converted"] == 3
    assert report["validation_levels"] == {
        "syntax_only": 2, "raw_unparsed": 1}
    assert report["issues"] == {"no_exports": 1, "syntax_error": 1}
    assert len(library) == 3
    assert {p.name for p in library.prims} == {
        "valid", "imports", "broken"}
    assert all(p.meta["validated"] is False for p in library.prims)
    assert {p.meta["validation_level"] for p in library.prims} == {
        "syntax_only", "raw_unparsed"}
    assert any(p.name == "broken" for p in
               library.view().search("broken.py capability", top_m=3))
