# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg522::rich.console.Group+rich.progress.BarColumn+rich.progress.MofNCompleteColumn
# name: rich_primitive
# summary: Uses rich.console.Group, rich.progress.BarColumn, rich.progress.MofNCompleteColumn, rich.progress.Progress across 2 repos
# anchor_symbols: ['rich.console.Group', 'rich.progress.BarColumn', 'rich.progress.MofNCompleteColumn', 'rich.progress.Progress', 'rich.progress.TextColumn', 'rich.progress.TimeRemainingColumn']
# observed in 2 repos: ['JosephLai241__URS', 'refuel-ai__autolabel']...

# --- from refuel-ai__autolabel::src/autolabel/utils.py::_autolabel_progress ---
def _autolabel_progress(
    description: str = None,
    console: Optional[Console] = None,
    transient: bool = False,
    disable: bool = False,
) -> Progress:
    """Create a progress bar for autolabel."""
    columns: List[ProgressColumn] = (
        [TextColumn("[progress.description]{task.description}")] if description else []
    )
    columns.extend(
        (
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
        ),
    )
    return Progress(
        *columns,
        console=console,
        transient=transient,
        disable=disable,
    )

# --- from JosephLai241__URS::urs/praw_scrapers/static_scrapers/Comments.py::SortComments.sort_structured ---
def sort_structured(submission: Submission, url: str) -> List[Dict[str, Any]]:
        """
        Sort all comments in structured format.

        :param Submission submission: PRAW `Submission` object.
        :param str url: The submission's URL.

        :returns: A `list[dict[str, Any]]` containing `CommentNode`s in `dict`
            form.
        :rtype: `list[dict[str, Any]]`
        """

        renderable_column = RenderableColumn(renderable="|")
        spinner_column = SpinnerColumn(spinner_name="noise")
        text_column = TextColumn("Seeding Forest")

        progress_bar = Progress(
            spinner_column,
            text_column,
            BarColumn(),
            MofNCompleteColumn(),
            renderable_column,
            TimeRemainingColumn(),
        )

        forest = Forest(submission.id_from_url(url))

        with progress_bar:
            for comment in progress_bar.track(submission.comments.list()):
                comment_node = CommentNode(
                    json.dumps((Objectify().make_comment(comment, False)))
                )

                forest.seed_comment(comment_node)

        return forest.root.replies

# --- from refuel-ai__autolabel::src/autolabel/utils.py::gather_async_tasks_with_progress ---
async def gather_async_tasks_with_progress(
    tasks: Iterable,
    description: str = None,
    total: Optional[int] = None,
    advance: int = 1,
    transient: bool = False,
    console: Optional[Console] = None,
    disable: bool = False,
) -> Iterable:
    """
    Gather async tasks with progress bar

    Args:
        tasks (Iterable): A sequence of async tasks you wish to gather.
        description (str, optional): Description of task show next to progress bar. Defaults to `None`.
        total (int, optional): Total number of steps. Default is len(sequence).
        advance (int, optional): Number of steps to advance progress by. Defaults to 1. Total / advance must less than or equal to len(sequence) for progress to reach finished state.
        transient (bool, optional): Clear the progress on exit. Defaults to False.
        console (Console, optional): Console to write to. Default creates internal Console instance.
        disable (bool, optional): Disable display of progress.

    Returns:
        Iterable: Returns an iterable of the results of the async tasks.

    """
    progress = _autolabel_progress(
        description=description,
        transient=transient,
        console=console,
        disable=disable,
    )

    if total is None:
        total = len(tasks)

    group = Group(progress)
    live = LiveDisplay.get_instance(group, console=console).live

    async def _task_with_tracker(task, progress, progress_task, live):
        res = await task
        progress.advance(
            progress_task,
            advance=min(advance, total - progress.tasks[progress_task].completed),
        )
        live.refresh()
        return res

    with live:
        progress_task = progress.add_task(description, total=total)
        tasks = [
            _task_with_tracker(task, progress, progress_task, live) for task in tasks
        ]
        return await asyncio.gather(*tasks)
