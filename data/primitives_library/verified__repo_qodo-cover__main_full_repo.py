import asyncio
import copy

from cover_agent.ai_caller import AICaller
from cover_agent.cover_agent import CoverAgent
from cover_agent.lsp_logic.ContextHelper import ContextHelper
from cover_agent.settings.config_loader import get_settings
from cover_agent.settings.config_schema import CoverAgentConfig
from cover_agent.utils import find_test_files, parse_args_full_repo


async def run():
    defaults = get_settings().get("default")
    arguments = parse_args_full_repo(defaults)

    if arguments.project_language != "python":
        raise NotImplementedError(
            "Unsupported language: {}".format(arguments.project_language)
        )

    context_helper = ContextHelper(arguments)
    test_files = find_test_files(arguments)

    print(
        "============\nTest files to be extended:\n"
        + "".join("{}\n============\n".format(path) for path in test_files)
    )

    async with context_helper.start_server():
        print("LSP server initialized.")

        caller = AICaller(
            model=arguments.model,
            api_base=getattr(arguments, "api_base", ""),
            generate_log_files=not arguments.suppress_log_files,
        )

        for test_file in test_files:
            context_files = await context_helper.find_test_file_context(test_file)
            print(
                "Context files for test file '{}':\n{}".format(
                    test_file,
                    "".join("{}\n".format(path) for path in context_files),
                )
            )

            print("\nAnalyzing test file against context files...")
            source_file, included_files = await context_helper.analyze_context(
                test_file,
                context_files,
                caller,
            )

            if not source_file:
                continue

            try:
                agent_args = copy.deepcopy(arguments)
                agent_args.source_file_path = source_file
                agent_args.test_command_dir = arguments.project_root
                agent_args.test_file_path = test_file
                agent_args.included_files = included_files

                configuration = CoverAgentConfig.from_cli_args_with_defaults(agent_args)
                CoverAgent(configuration).run()
            except Exception as error:
                print(
                    "Error running CoverAgent for test file '{}': {}".format(
                        test_file,
                        error,
                    )
                )


def main():
    asyncio.run(run())


if __name__ == "__main__":
    main()