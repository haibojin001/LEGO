# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg257::sqlalchemy.MetaData+sqlalchemy.Table
# name: sqlalchemy_primitive
# summary: Uses sqlalchemy.MetaData, sqlalchemy.Table across 2 repos
# anchor_symbols: ['sqlalchemy.MetaData', 'sqlalchemy.Table']
# observed in 2 repos: ['InfuseAI__piperider', 'modin-project__modin']...

# --- from InfuseAI__piperider::piperider_cli/profiler/profiler.py::_run_in_executor ---
async def _run_in_executor(executor, func, *args):
    if executor:
        return await asyncio.get_running_loop().run_in_executor(executor, func, *args)
    else:
        return func(*args)

# --- from modin-project__modin::modin/experimental/core/io/sql/utils.py::get_table_metadata ---
def get_table_metadata(engine, table):
    """
    Extract all useful data from the given table.

    Parameters
    ----------
    engine : sqlalchemy.engine.base.Engine
        SQLAlchemy connection engine.
    table : str
        Table name.

    Returns
    -------
    sqlalchemy.sql.schema.Table
        Extracted metadata.
    """
    metadata = MetaData()
    metadata.reflect(bind=engine, only=[table])
    table_metadata = Table(table, metadata, autoload=True)
    return table_metadata

# --- from InfuseAI__piperider::piperider_cli/profiler/profiler.py::Profiler._fetch_metadata._fetch_table_task ---
def _fetch_table_task(subject: ProfileSubject):
                engine = self.data_source.get_engine_by_database(subject.database)
                schema = subject.schema.lower() if subject.schema is not None else None
                table = None
                try:
                    table = Table(subject.table, MetaData(), autoload_with=engine, schema=schema)
                except NoSuchTableError:
                    # ignore the table metadata fetch error
                    pass
                except Exception as e:
                    capture_exception(e)
                return subject, table
