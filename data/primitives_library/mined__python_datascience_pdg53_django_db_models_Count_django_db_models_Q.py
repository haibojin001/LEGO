# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg53::django.db.models.Count+django.db.models.Q
# name: django_primitive
# summary: Uses django.db.models.Count, django.db.models.Q across 3 repos
# anchor_symbols: ['django.db.models.Count', 'django.db.models.Q']
# observed in 3 repos: ['okfn-brasil__serenata-de-amor', 'polyaxon__haupt', 'sinaptik-ai__pandas-ai']...

# --- from sinaptik-ai__pandas-ai::extensions/ee/vectorstores/milvus/pandasai_milvus/milvus.py::Milvus._convert_ids ---
def _convert_ids(self, ids: Iterable[str]) -> List[str]:
        return [
            id
            if self._is_valid_uuid(id)
            else str(uuid.uuid5(uuid.UUID(UUID_NAMESPACE), id))
            for id in ids
        ]

# --- from sinaptik-ai__pandas-ai::extensions/ee/vectorstores/qdrant/pandasai_qdrant/qdrant.py::Qdrant._convert_ids ---
def _convert_ids(self, ids: Iterable[str]):
        return [
            (
                id
                if self._is_valid_uuid(id)
                else str(uuid.uuid5(uuid.UUID(UUID_NAMESPACE), id))
            )
            for id in ids
        ]

# --- from okfn-brasil__serenata-de-amor::jarbas/chamber_of_deputies/querysets.py::ReimbursementQuerySet.tuple_filter ---
def tuple_filter(self, **kwargs):
        filters = {_rename_key(k): v for k, v in _str_to_tuple(kwargs).items()}
        for key, values in filters.items():
            filter_ = reduce(lambda q, val: q | Q(**{key: val}), values, Q())
            self = self.filter(filter_)
        return self

# --- from polyaxon__haupt::haupt/haupt/db/managers/runs.py::get_stopping_pipelines_with_no_runs ---
def get_stopping_pipelines_with_no_runs(queryset):
    return (
        queryset.filter(
            kind__in=[V1RunKind.DAG, V1RunKind.MATRIX, V1RunKind.SCHEDULE],
            status=V1Statuses.STOPPING,
        )
        .annotate(
            unfinished=Count(
                "pipeline_runs",
                filter=~Q(
                    pipeline_runs__status__in=LifeCycle.DONE_VALUES
                    | LifeCycle.PENDING_VALUES
                ),
                distinct=True,
            )
        )
        .filter(unfinished=0)
    )

# --- from polyaxon__haupt::haupt/haupt/db/managers/artifacts.py::get_artifacts_by_keys ---
def get_artifacts_by_keys(
    run: BaseRun, namespace: uuid.UUID, artifacts: List[V1RunArtifact]
) -> Dict:
    results = {}
    for m in artifacts:
        state = m.state
        if not state:
            if m.is_input:
                state = m.get_state(namespace)
            else:
                state = run.uuid
        elif not isinstance(state, uuid.UUID):
            try:
                state = uuid.UUID(state)
            except (ValueError, KeyError):
                state = uuid.uuid5(namespace, state)
        results[(m.name, state)] = m

    return results
