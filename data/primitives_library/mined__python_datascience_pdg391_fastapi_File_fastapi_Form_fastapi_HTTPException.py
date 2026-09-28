# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg391::fastapi.File+fastapi.Form+fastapi.HTTPException
# name: fastapi_os_primitive
# summary: Uses fastapi.File, fastapi.Form, fastapi.HTTPException, os.makedirs across 2 repos
# anchor_symbols: ['fastapi.File', 'fastapi.Form', 'fastapi.HTTPException', 'os.makedirs', 'os.remove', 'os.urandom']
# observed in 2 repos: ['encord-team__encord-active', 'ruc-datalab__DeepAnalyze']...

# --- from ruc-datalab__DeepAnalyze::example/4c_competition/file_api.py::create_file ---
async def create_file(
    file: UploadFile = File(...),
    purpose: str = Form("file-extract")
):
    """Upload a file (OpenAI compatible)"""
    # Validate purpose
    if purpose not in VALID_FILE_PURPOSES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid purpose. Must be one of {VALID_FILE_PURPOSES}"
        )

    # Save file to a persistent location
    os.makedirs(FILE_STORAGE_DIR, exist_ok=True)
    file_id = f"file-{file.filename.replace('.', '-').replace('_', '-')[:8]}-{os.urandom(4).hex()}"
    file_path = os.path.join(FILE_STORAGE_DIR, file_id)

    try:
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)

        file_obj = storage.create_file(file.filename, file_path, purpose)
        return file_obj
    except Exception as e:
        # Clean up file if creation failed
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=str(e))

# --- from ruc-datalab__DeepAnalyze::API/file_api.py::create_file ---
async def create_file(
    file: UploadFile = File(...),
    purpose: str = Form("file-extract")
):
    """Upload a file (OpenAI compatible)"""
    # Validate purpose
    if purpose not in VALID_FILE_PURPOSES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid purpose. Must be one of {VALID_FILE_PURPOSES}"
        )

    # Save file to a persistent location
    os.makedirs(FILE_STORAGE_DIR, exist_ok=True)
    file_id = f"file-{file.filename.replace('.', '-').replace('_', '-')[:8]}-{os.urandom(4).hex()}"
    file_path = os.path.join(FILE_STORAGE_DIR, file_id)

    try:
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)

        file_obj = storage.create_file(file.filename, file_path, purpose)
        return file_obj
    except Exception as e:
        # Clean up file if creation failed
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=str(e))

# --- from encord-team__encord-active::src/encord_active/server/routers/project.py::search ---
def search(
    project: ProjectFileStructureDep,
    type: Annotated[SearchType, Form()],
    filters: Annotated[str, Form()] = "",
    query: Annotated[Optional[str], Form()] = None,
    image: Annotated[Optional[UploadFile], File()] = None,
    scope: Annotated[Optional[MetricScope], Form()] = None,
):
    if not (query or (image is not None)):
        raise HTTPException(status_code=422, detail="Invalid query. Either `query` or `image` should be specified")

    if filters:
        _filters = Filters.parse_raw(filters)
    else:
        _filters = Filters()

    querier = get_querier(project)

    merged_metrics = filtered_merged_metrics(project, _filters)

    def _search(ids: List[str]):
        snippet = None
        if type == SearchType.SEARCH:
            image_bytes = None
            if image is not None:
                image_bytes = image.file.read()
            _query = CLIPQuery(text=query, image=image_bytes, limit=-1, identifiers=ids)
            result = querier.search_semantics(_query)
        else:
            text_query = TextQuery(text=query, limit=-1, identifiers=ids)
            result = querier.search_with_code(text_query)
            if result:
                snippet = result.snippet

        if not result:
            raise HTTPException(status_code=422, detail="Invalid query")

        return [item.identifier for item in result.result_identifiers], snippet

    if scope == MetricScope.PREDICTION and _filters.prediction_filters is not None:
        _, _, predictions, _ = read_prediction_files(project, _filters.prediction_filters.type)
        if predictions is not None:
            ids, snippet = _search(get_ids(predictions["identifier"], scope))
            prediction_ids = predictions["identifier"].sort_values(
                key=lambda column: partial_column(column, 3).map(lambda id: ids.index(id))
            )
            return {"ids": prediction_ids.to_list(), "snippet": snippet}

    ids, snippet = _search(get_ids(merged_metrics.index, scope))
    return {"ids": ids, "snippet": snippet}
