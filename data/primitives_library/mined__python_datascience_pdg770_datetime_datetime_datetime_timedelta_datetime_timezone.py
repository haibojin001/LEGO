# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg770::datetime.datetime+datetime.timedelta+datetime.timezone
# name: datetime_primitive
# summary: Uses datetime.datetime, datetime.timedelta, datetime.timezone across 2 repos
# anchor_symbols: ['datetime.datetime', 'datetime.timedelta', 'datetime.timezone']
# observed in 2 repos: ['insitro__redun', 'sematic-ai__sematic']...

# --- from insitro__redun::redun/tests/test_utils.py::test_format_timestamp ---
def test_format_timestamp() -> None:
    """
    Ensure format_timestamp converts to local time before formatting.
    """
    timestamp = datetime(2024, 8, 27, 13, 1, 2, tzinfo=timezone.utc)
    assert format_timestamp(timestamp, timezone(timedelta(hours=2))) == "2024-08-27 15:01:02"
    assert format_timestamp(timestamp, timezone(timedelta(hours=-2))) == "2024-08-27 11:01:02"

# --- from sematic-ai__sematic::sematic/db/models/tests/test_factories.py::test_initialize_future_from_run ---
def test_initialize_future_from_run():
    created_at = datetime(year=2023, month=8, day=10, tzinfo=timezone(timedelta(hours=4)))
    run = Run(  # noqa: F811
        id="theid",
        original_run_id=None,
        future_state=FutureState.RAN,
        name="the name",
        function_path=f"{f2.__module__}.{f2.__name__}",
        parent_id="parentid",
        root_id="rootid",
        description="the description",
        tags=["foo", "bar"],
        nested_future_id="nestedid",
        container_image_uri="imageuri",
        created_at=created_at,
        updated_at=created_at + timedelta(hours=2),
        started_at=created_at + timedelta(hours=1),
        ended_at=created_at + timedelta(2),
        resolved_at=created_at + timedelta(2),
        failed_at=None,
        cache_key="cachekey",
    )
    requirements = ResourceRequirements(
        kubernetes=KubernetesResourceRequirements(
            node_selector={"foo": "bar"},
        )
    )
    run.resource_requirements = requirements

    kwargs = {"a": 1, "b": 2}
    future = initialize_future_from_run(run, kwargs=kwargs, use_same_id=True)

    assert future.id == run.id
    assert future.function is f2
    assert future.props.name == run.name
    assert future.props.tags == json.loads(run.tags)  # type: ignore
    assert future.props.state == FutureState[run.future_state]  # type: ignore
    assert future.props.resource_requirements == requirements
    assert future.props.scheduled_epoch_time == 1691614800
    assert future.kwargs == kwargs

    future2 = initialize_future_from_run(run, kwargs, use_same_id=False)
    assert future2.id != future.id

    run.function_path = "sematic.function._make_list"
    future3 = initialize_future_from_run(run, kwargs={"v0": 0, "v1": 1})
    assert future3.function.execute(**future3.kwargs) == [0, 1]
