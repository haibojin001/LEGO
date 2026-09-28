import os
from time import sleep

from jinja2 import Environment, StrictUndefined

from cover_agent.lsp_logic.file_map.file_map import FileMap
from cover_agent.lsp_logic.multilspy import LanguageServer
from cover_agent.lsp_logic.multilspy.multilspy_config import MultilspyConfig
from cover_agent.lsp_logic.multilspy.multilspy_logger import MultilspyLogger
from cover_agent.settings.config_loader import get_settings
from cover_agent.utils import load_yaml


async def analyze_context(test_file, context_files, args, ai_caller):
    source_file = None
    context_files_include = context_files

    try:
        relative_test_path = os.path.relpath(test_file, args.project_root)
        context_file_names = ""

        for context_file in context_files:
            relative_context_path = os.path.relpath(
                context_file, args.project_root
            )
            context_file_names += f"`{relative_context_path}\n`"

        template_values = {
            "language": args.project_language,
            "test_file_name_rel": relative_test_path,
            "test_file_content": open(test_file, "r").read(),
            "context_files_names_rel": context_file_names,
        }

        jinja_environment = Environment(undefined=StrictUndefined)
        settings = get_settings()

        system_message = jinja_environment.from_string(
            settings.analyze_test_against_context.system
        ).render(template_values)
        user_message = jinja_environment.from_string(
            settings.analyze_test_against_context.user
        ).render(template_values)

        model_response, prompt_token_count, response_token_count = (
            ai_caller.call_model(
                prompt={"system": system_message, "user": user_message},
                stream=False,
            )
        )

        analysis = load_yaml(model_response)
        if int(analysis.get("is_this_a_unit_test", 0)) == 1:
            relative_source_path = analysis.get("main_file", "").strip().strip("`")
            source_file = os.path.join(args.project_root, relative_source_path)

            for context_file in context_files:
                if (
                    os.path.relpath(context_file, args.project_root)
                    == relative_source_path
                ):
                    context_files_include = [
                        item for item in context_files if item != context_file
                    ]

        if source_file:
            print(
                f"Test file: `{test_file}`,\n"
                f"is a unit test file for source file: `{source_file}`"
            )
        else:
            print(f"Test file: `{test_file}` is not a unit test file")
    except Exception as error:
        print(
            f"Error while analyzing test file {test_file} against context files: "
            f"{error}"
        )

    return source_file, context_files_include


async def find_test_file_context(args, lsp, test_file):
    try:
        relative_file = os.path.relpath(test_file, args.project_root)

        file_map = FileMap(
            test_file,
            parent_context=False,
            child_context=False,
            header_max=0,
            project_base_path=args.project_root,
        )
        query_results, captures = file_map.get_query_results()

        context_files, context_symbols = await lsp.get_direct_context(
            captures,
            args.project_language,
            args.project_root,
            relative_file,
        )

        non_empty_files = []
        for context_file in context_files:
            with open(context_file, "r") as file_handle:
                if file_handle.read().strip():
                    non_empty_files.append(context_file)

        context_files = non_empty_files
    except Exception as error:
        print(f"Error while getting context for test file {test_file}: {error}")
        context_files = []

    return context_files


async def initialize_language_server(args):
    logger = MultilspyLogger()
    configuration = MultilspyConfig.from_dict(
        {"code_language": args.project_language}
    )

    if args.project_language == "python":
        language_server = LanguageServer.create(
            configuration,
            logger,
            args.project_root,
        )
        sleep(0.1)
        return language_server

    raise NotImplementedError(
        "Unsupported language: {}".format(args.project_language)
    )