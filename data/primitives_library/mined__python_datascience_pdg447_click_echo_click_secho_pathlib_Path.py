# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg447::click.echo+click.secho+pathlib.Path
# name: click_pathlib_primitive
# summary: Uses click.echo, click.secho, pathlib.Path across 3 repos
# anchor_symbols: ['click.echo', 'click.secho', 'pathlib.Path']
# observed in 3 repos: ['AgnostiqHQ__covalent', 'ploomber__ploomber', 'sematic-ai__sematic']...

# --- from AgnostiqHQ__covalent::covalent_dispatcher/_cli/groups/db_group.py::migrate ---
def migrate(ctx: click.Context) -> None:
    """
    Run database migrations
    """
    try:
        db = DataStore.factory()
        db.run_migrations()
        click.secho("Migrations are up to date.", fg="green")
    except Exception as migration_error:
        click.echo(str(migration_error))
        click.secho(MIGRATION_WARNING_MSG, fg="red")
        return ctx.exit(1)

# --- from ploomber__ploomber::tests/cli/test_custom.py::test_plot_uses_name_if_any ---
def test_plot_uses_name_if_any(tmp_nbs, monkeypatch):
    os.rename("pipeline.yaml", "pipeline.train.yaml")

    args_defaults = ["ploomber", "--entry-point", "pipeline.train.yaml"]
    monkeypatch.setattr(sys, "argv", args_defaults)
    mock = Mock()
    monkeypatch.setattr(dag_module.DAG, "plot", mock)
    plot.main(catch_exception=False)

    mock.assert_called_once_with(
        output="pipeline.train.png", include_products=False, backend=None
    )

# --- from ploomber__ploomber::tests/cli/test_examples.py::test_clones_if_outdated ---
def test_clones_if_outdated(clone_examples, monkeypatch, capsys):
    # mock metadata to make it look older
    metadata = _mock_metadata(
        timestamp=(datetime.now() - timedelta(days=1)).timestamp(),
        branch="another-branch",
    )
    monkeypatch.setattr(examples._ExamplesManager, "load_metadata", lambda _: metadata)

    mock_run = Mock()
    monkeypatch.setattr(examples.subprocess, "run", mock_run)

    # prevent actual deletion
    monkeypatch.setattr(examples.shutil, "rmtree", lambda _: None)

    examples.main(name=None, force=False)

    mock_run.assert_called_once()
    captured = capsys.readouterr()
    assert "Examples copy is more than 1 day old..." in captured.out

# --- from AgnostiqHQ__covalent::covalent_dispatcher/_cli/groups/db_group.py::alembic ---
def alembic(ctx: click.Context, alembic_args) -> None:
    """
    Alembic CLI
    """
    try:
        alembic_args = list(alembic_args)
        migrations_folder_path = Path(path.join(__file__, "./../../../../covalent_migrations/"))
        project_root_path = migrations_folder_path / Path("..")
        settings_file_path = migrations_folder_path / Path("alembic.ini")
        alembic_command = ["alembic", "-c", str(settings_file_path.resolve())] + alembic_args
        p = Popen(alembic_command, stdout=PIPE, stderr=PIPE, cwd=str(project_root_path.resolve()))
        output, error = p.communicate()
        if error:
            click.echo(error.decode("utf-8").strip())
        else:
            click.echo(output.decode("utf-8").strip())
    except Exception as migration_error:
        if migration_error:
            click.echo(f"{type(migration_error)}:{str(migration_error)}")
        click.secho(
            "There was an error forwarding arguments to alembic CLI please ensure that alembic is installed.",
            fg="red",
        )
        return ctx.exit(1)

# --- from sematic-ai__sematic::sematic/cli/logs.py::dump_log_storage ---
def dump_log_storage(storage_key: str):
    """Dumps all logs stored in a blob storage directory.

    The logs will be concatenated and dumped to stdout. Logs will not be
    "followed;" whatever logs are present at the time the dump is initiated
    will be dumped and the command will exit.

    This is useful primarily for infrastructure debugging purposes, such as
    to dump the full logs for a resolution
    (ex: logs/v2/run_id/{resolution_id}/driver/).
    It does not require API access, only storage access.
    """

    # Don't want Sematic logs interfering with the ones being pulled
    # from the remote
    logging.basicConfig(level=logging.ERROR)
    storage_key = storage_key if storage_key.endswith("/") else f"{storage_key}/"
    line_stream = line_stream_from_log_directory(
        storage_key,
        cursor_file=None,
        cursor_line_index=None,
        reverse=False,
    )
    found_lines = False

    for log_line in line_stream:
        found_lines = True
        print(log_line.line)

    if not found_lines:
        click.secho(f"No logs found in storage at '{storage_key}'", fg="red")
        sys.exit(1)

# --- from sematic-ai__sematic::sematic/cli/logs.py::logs ---
def logs(run_id: str, follow: bool):
    """Read the logs for the run directly from blob storage, and print to stdout.

    If the run is still producing logs, this will follow the logs "live" until
    no more log lines are being produced.
    """
    switch_env("user")  # only makes sense for this env

    # Don't want Sematic logs interfering with the ones being pulled
    # from the remote
    logging.basicConfig(level=logging.ERROR)

    try:
        api_client.get_run(run_id)
    except api_client.ResourceNotFoundError:
        click.secho(
            f"Could not find run with id '{run_id}' at {get_config().api_url}.",
            fg="red",
        )
        sys.exit(1)

    has_more = True
    cursor = None
    had_any = False

    while has_more:
        loaded = load_log_lines(
            run_id,
            forward_cursor_token=cursor,
            reverse_cursor_token=None,
            max_lines=DEFAULT_LOG_LOAD_MAX_SIZE,
            filter_strings=None,
            object_source=ObjectSource.API,
        )
        has_more = loaded.can_continue_forward

        if len(loaded.lines) == 0:
            if not (had_any or has_more):
                click.secho(loaded.log_info_message, fg="red")
                sys.exit(1)
            if not follow:
                break

            # give storage API a quick break while we wait for some logs to appear
            time.sleep(1)
            continue

        cursor = loaded.forward_cursor_token

        for line in loaded.lines:
            had_any = True
            click.echo(line)
