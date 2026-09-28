# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg386::encord_active.lib.common.utils.partial_column+encord_active.lib.model_predictions.reader.get_model_predictions+encord_active.server.utils.filtered_merged_metrics
# name: encord_active_fastapi_primitive
# summary: Uses encord_active.lib.common.utils.partial_column, encord_active.lib.model_predictions.reader.get_model_predictions, encord_active.server.utils.filtered_merged_metrics, fastapi.Body across 3 repos
# anchor_symbols: ['encord_active.lib.common.utils.partial_column', 'encord_active.lib.model_predictions.reader.get_model_predictions', 'encord_active.server.utils.filtered_merged_metrics', 'fastapi.Body', 'fastapi.HTTPException', 'fastapi.responses.JSONResponse', 'fastapi.responses.ORJSONResponse']
# observed in 3 repos: ['encord-team__encord-active', 'ruc-datalab__DeepAnalyze', 'run-house__kubetorch']...

# --- from ruc-datalab__DeepAnalyze::demo/chat_v2/backend_app/routers/export.py::export_report ---
async def export_report(body: dict = Body(...)):
    try:
        return JSONResponse(export_report_from_body(body))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

# --- from encord-team__encord-active::src/encord_active/server/routers/project.py::read_item_ids ---
def read_item_ids(
    project: ProjectFileStructureDep,
    scope: Annotated[MetricScope, Body()],
    sort_by_metric: Annotated[str, Body()],
    filters: Filters = Filters(),
    ascending: Annotated[bool, Body()] = True,
    ids: Annotated[Optional[list[str]], Body()] = None,
):
    merged_metrics = filtered_merged_metrics(project, filters, scope)

    if scope == MetricScope.PREDICTION:
        if filters.prediction_filters is None:
            raise HTTPException(
                status_code=422, detail='Filters must contain "prediction_filters" when scope is "prediction"'
            )
        df, _ = get_model_predictions(project, filters.prediction_filters)
        df = apply_filters(df, filters, project, scope)

        if filters.prediction_filters.outcome == ObjectDetectionOutcomeType.FALSE_NEGATIVES:
            sort_by_metric = sort_by_metric.replace("(P)", "(O)")

        column = [col for col in df.columns if col.lower() == sort_by_metric.lower()][0]
        df = df[partial_column(df.index, 3).isin(partial_column(merged_metrics.index, 3).unique())]
        if filters.prediction_filters.outcome == ObjectDetectionOutcomeType.FALSE_NEGATIVES:
            df = df[df.index.isin(merged_metrics.index)]
    else:
        column = [col for col in merged_metrics.columns if col.lower() == sort_by_metric.lower()][0]
        df = merged_metrics

    res: pd.DataFrame = df[[column]].dropna().sort_values(by=[column], ascending=ascending)
    if ids:
        if filters.prediction_filters and filters.prediction_filters.type == MainPredictionType.OBJECT:
            res = res[partial_column(res.index, 3).isin(ids)]
        else:
            res = res[res.index.isin(ids)]
    res = res.reset_index().rename({"identifier": "id", column: "value"}, axis=1)

    return ORJSONResponse(res[["id", "value"]].to_dict("records"))

# --- from ruc-datalab__DeepAnalyze::demo/chat/backend.py::export_report ---
async def export_report(body: dict = Body(...)):
    """
    接收全部聊天历史（messages: [{role, content}...]），抽取 <Analyze>..</Analyze> ~ <Answer>..</Answer>
    仅生成 Markdown 文件并保存到 workspace；PDF 渲染留作 TODO。
    """
    try:
        messages = body.get("messages", [])
        title = (body.get("title") or "").strip()
        session_id = body.get("session_id", "default")
        workspace_dir = get_session_workspace(session_id)

        if not isinstance(messages, list):
            raise HTTPException(status_code=400, detail="messages must be a list")

        md_text = _extract_sections_from_messages(messages)
        if not md_text:
            md_text = (
                "(No <Analyze>/<Understand>/<Code>/<Execute>/<Answer> sections found.)"
            )

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_title = re.sub(r"[^\w\-_.]+", "_", title) if title else "Report"
        base_name = f"{safe_title}_{ts}" if title else f"Report_{ts}"

        # Save MD into generated/ folder under workspace
        export_dir = os.path.join(workspace_dir, "generated")
        os.makedirs(export_dir, exist_ok=True)

        md_path = _save_md(md_text, base_name, export_dir)

        # PDF 暂不生成（TODO）。
        pdf_path = _save_pdf(md_text, base_name, export_dir)

        result = {
            "message": "exported",
            "md": md_path.name,
            "pdf": pdf_path.name if pdf_path else None,
            "download_urls": {
                "md": build_download_url(f"{session_id}/generated/{md_path.name}"),
                "pdf": (
                    build_download_url(f"{session_id}/generated/{pdf_path.name}")
                    if pdf_path
                    else None
                ),
            },
        }
        return JSONResponse(result)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- from run-house__kubetorch::python_client/kubetorch/serving/http_server.py::test_reload ---
async def test_reload(request: Request, metadata: Dict = Body(...)):
    """Test endpoint to trigger reload with new metadata.

    This endpoint simulates the WebSocket push-based reload for testing purposes.
    It applies metadata, runs image setup, and recreates the supervisor.

    Example metadata:
    {
        "module": {
            "module_name": "my_module",
            "cls_or_fn_name": "my_function",
            "file_path": "/path/to/module",
            "init_args": null,
            "callable_type": "fn"
        },
        "runtime_config": {}
    }
    """
    global SUPERVISOR, _CACHED_CALLABLES

    try:
        # Apply the new metadata (sets env vars)
        _apply_metadata_from_dict(metadata)

        # Run image setup - use thread pool to avoid blocking event loop
        await asyncio.to_thread(run_image_setup)

        # Clear caches
        _CACHED_CALLABLES.clear()

        # Cleanup existing supervisor
        if SUPERVISOR:
            try:
                SUPERVISOR.cleanup()
            except Exception as e:
                logger.warning(f"Error during supervisor cleanup on test reload: {e}")
            SUPERVISOR = None

        # Recreate supervisor
        if os.environ.get("KT_CLS_OR_FN_NAME"):
            logger.info("Recreating supervisor during test reload")
            clear_cache()
            await asyncio.to_thread(load_callable)
            logger.info("Supervisor recreated successfully")

        # Set launch_id AFTER reload completes (same as _handle_reload)
        launch_id_val = metadata.get("launch_id") or metadata.get("launchId")
        if launch_id_val:
            os.environ["KT_LAUNCH_ID"] = launch_id_val

        return {"status": "ok", "message": "Reload completed successfully"}

    except Exception as e:
        logger.error(f"Error in test reload: {e}")
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})

# --- from encord-team__encord-active::src/encord_active/server/routers/project.py::get_2d_embeddings ---
def get_2d_embeddings(
    project: ProjectFileStructureDep, embedding_type: Annotated[EmbeddingType, Body()], filters: Filters
):
    embeddings_df = get_2d_embedding_data(project, embedding_type)

    if embeddings_df is None:
        raise HTTPException(
            status_code=404, detail=f'Embeddings of type "{embedding_type}" were not found for project "{project}"'
        )

    filtered = filtered_merged_metrics(project, filters)

    embeddings_df.set_index("identifier", inplace=True)
    embeddings_df = cast(DataFrame[Embedding2DSchema], embeddings_df[embeddings_df.index.isin(filtered.index)])

    if filters.prediction_filters is not None:
        embeddings_df["data_row_id"] = partial_column(embeddings_df.index, 3)
        predictions, labels = get_model_predictions(project, PredictionsFilters(type=filters.prediction_filters.type))

        if filters.prediction_filters.type == MainPredictionType.OBJECT:
            labels = labels[[LabelMatchSchema.is_false_negative]]
            labels = labels[labels[LabelMatchSchema.is_false_negative]].copy()
            labels["data_row_id"] = partial_column(labels.index, 3)
            labels["score"] = 0
            labels.drop(LabelMatchSchema.is_false_negative, axis=1, inplace=True)
            predictions = predictions[[PredictionMatchSchema.is_true_positive]].copy()
            predictions["data_row_id"] = partial_column(predictions.index, 3)
            predictions.rename(columns={PredictionMatchSchema.is_true_positive: "score"}, inplace=True)

            merged_score = pd.concat([labels, predictions], axis=0)
            grouped_score = (
                merged_score.groupby("data_row_id")[Embedding2DScoreSchema.score].mean().to_frame().reset_index()
            )
            embeddings_df = cast(
                DataFrame[Embedding2DSchema],
                embeddings_df.merge(grouped_score, on="data_row_id", how="outer")
                .fillna(0)
                .rename({"data_row_id": "identifier"}, axis=1),
            )
        else:
            predictions = predictions[[ClassificationPredictionMatchSchema.is_true_positive]]
            predictions["data_row_id"] = partial_column(predictions.index, 3)

            embeddings_df = cast(
                DataFrame[Embedding2DSchema],
                embeddings_df.merge(predictions, on="data_row_id", how="outer")
                .drop(columns=[Embedding2DSchema.label])
                .rename(
                    columns={
                        ClassificationPredictionMatchSchema.is_true_positive: Embedding2DSchema.label,
                        "data_row_id": "identifier",
                    }
                ),
            )

            embeddings_df["score"] = embeddings_df[Embedding2DSchema.label]
            embeddings_df[Embedding2DSchema.label] = embeddings_df[Embedding2DSchema.label].apply(
                lambda x: "Correct Classification" if x == 1.0 else "Misclassification"
            )

    return ORJSONResponse(embeddings_df.reset_index().rename({"identifier": "id"}, axis=1).to_dict("records"))
