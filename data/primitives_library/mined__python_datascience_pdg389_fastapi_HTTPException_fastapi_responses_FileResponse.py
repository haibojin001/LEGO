# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg389::fastapi.HTTPException+fastapi.responses.FileResponse
# name: fastapi_primitive
# summary: Uses fastapi.HTTPException, fastapi.responses.FileResponse across 2 repos
# anchor_symbols: ['fastapi.HTTPException', 'fastapi.responses.FileResponse']
# observed in 2 repos: ['ruc-datalab__DeepAnalyze', 'zama-ai__concrete-ml']...

# --- from zama-ai__concrete-ml::use_case_examples/deployment/server/server.py::get_client ---
def get_client():
        """Get client.

        Returns:
            FileResponse: client.zip

        Raises:
            HTTPException: if the file can't be find locally
        """
        path_to_client = (CLIENT_SERVER_PATH / "client.zip").resolve()
        if not path_to_client.exists():
            raise HTTPException(status_code=500, detail="Could not find client.")
        return FileResponse(path_to_client, media_type="application/zip")

# --- from zama-ai__concrete-ml::use_case_examples/hybrid_model/serve_model.py::get_client ---
def get_client(model_name: str = Form(), module_name: str = Form(), input_shape: str = Form()):
        """Get client.

        Returns:
            FileResponse: client.zip

        Raises:
            HTTPException: if the file can't be find locally
        """
        check_inputs(server, model_name, module_name, input_shape)
        return FileResponse(
            server.get_client(model_name, module_name, input_shape), media_type="application/zip"
        )

# --- from ruc-datalab__DeepAnalyze::demo/chat_v2/backend_app/services/workspace.py::get_workspace_file_response ---
def get_workspace_file_response(
    session_id: str,
    relative_path: str,
    *,
    download: bool = False,
) -> FileResponse:
    file_path = resolve_workspace_path(session_id, relative_path)
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail="File not found")

    response_kwargs = {
        "path": file_path,
        "content_disposition_type": "attachment" if download else "inline",
    }
    if download:
        response_kwargs["filename"] = file_path.name
    return FileResponse(**response_kwargs)
