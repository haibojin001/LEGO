# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg688::argparse.ArgumentParser+textwrap.dedent
# name: argparse_textwrap_primitive
# summary: Uses argparse.ArgumentParser, textwrap.dedent across 2 repos
# anchor_symbols: ['argparse.ArgumentParser', 'textwrap.dedent']
# observed in 2 repos: ['nok__sklearn-porter', 'violit-dev__violit']...

# --- from nok__sklearn-porter::sklearn_porter/cli/__main__.py::parse_args ---
def parse_args(args):
    header = 'sklearn-porter CLI v{}'.format(porter_version)
    footer = dedent(
        """
        Examples:
          `porter show`
          `porter port model --language js --template attached`
          `porter save model --directory /tmp --skip-warnings`
        
        Manuals:
          https://github.com/nok/sklearn-porter
          https://github.com/scikit-learn/scikit-learn
    """
    )
    parser = ArgumentParser(
        description=header,
        formatter_class=RawTextHelpFormatter,
        add_help=False,
        epilog=footer,
    )
    for group in parser._action_groups:
        group.title = str(group.title).capitalize()

    sp = parser.add_subparsers(
        metavar='command',
        dest='cmd',
    )
    sp.required = True

    show.config(sp)
    port.config(sp)
    save.config(sp)

    arg_version(parser)
    arg_help(parser)

    if len(sys.argv) == 1 and 'SKLEARN_PORTER_PYTEST' not in environ:
        parser.print_help(sys.stdout)
        sys.exit(1)

    return parser.parse_args(args)

# --- from violit-dev__violit::src/violit/cli.py::main ---
def main():
    parser = argparse.ArgumentParser(
        prog="violit",
        description="Violit CLI - Pure Python Web Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              violit create my_app
              violit run app.py
              violit run app.py --ws
              violit run app.py --reload --localhost
              violit run app.py --help
        """)
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # 1. violit run <script> [args...]
    run_parser = subparsers.add_parser(
        "run",
        help="Run a Violit script",
        description=textwrap.dedent("""\
            Run a Violit script.

            Any arguments after the script path are passed directly to app.run().
            That means the same runtime flags work in both styles:

              python app.py --reload --localhost
              violit run app.py --reload --localhost
        """),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Common examples:
              violit run app.py
                            violit run app.py --ws
              violit run app.py --reload
              violit run app.py --reload --localhost
              violit run app.py --port 8010
              violit run app.py --native
              violit run app.py --lite
              violit run app.py --make-migration "add_users"
              violit run app.py --help
        """)
    )
    run_parser.add_argument("script", help="Path to the python script (e.g. main.py)")
    run_parser.add_argument(
        "args",
        nargs=argparse.REMAINDER,
        metavar="APP_ARGS",
        help="Arguments passed directly to app.run() (for example: --reload --localhost --port 8010)"
    )
    
    # 2. violit create <project_name>
    create_parser = subparsers.add_parser("create", help="Create a new violit project")
    create_parser.add_argument("name", help="Name of the new project directory")

    args = parser.parse_args()

    if args.command == "run":
        run_app(args, args.args)
    elif args.command == "create":
        create_project(args)
