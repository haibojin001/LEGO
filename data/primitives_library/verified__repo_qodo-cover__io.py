import base64
import os
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from prompt_toolkit.completion import Completer, Completion, ThreadedCompleter
from prompt_toolkit.cursor_shapes import ModalCursorShapeConfig
from prompt_toolkit.enums import EditingMode
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.lexers import PygmentsLexer
from prompt_toolkit.shortcuts import CompleteStyle, PromptSession
from prompt_toolkit.styles import Style
from pygments.lexers import MarkdownLexer, guess_lexer_for_filename
from pygments.token import Token
from rich.console import Console
from rich.markdown import Markdown
from rich.style import Style as RichStyle
from rich.text import Text


@dataclass
class ConfirmGroup:
    preference: str = None
    show_group: bool = True

    def __init__(self, items=None):
        if items is not None:
            self.show_group = len(items) > 1


class AutoCompleter(Completer):
    def __init__(
        self, root, rel_fnames, addable_rel_fnames, commands, encoding, abs_read_only_fnames=None
    ):
        self.addable_rel_fnames = addable_rel_fnames
        self.rel_fnames = rel_fnames
        self.encoding = encoding
        self.abs_read_only_fnames = abs_read_only_fnames or []

        basename_map = defaultdict(list)
        for name in addable_rel_fnames:
            base = os.path.basename(name)
            if base != name:
                basename_map[base].append(name)
        self.fname_to_rel_fnames = basename_map

        self.words = set()
        self.commands = commands
        self.command_completions = {}

        if commands:
            self.command_names = commands.get_commands()

        self.words.update(addable_rel_fnames)
        self.words.update(rel_fnames)

        files = [Path(root) / name for name in rel_fnames]
        if abs_read_only_fnames:
            files.extend(abs_read_only_fnames)
        self.all_fnames = files
        self.tokenized = False

    def tokenize(self):
        if self.tokenized:
            return

        self.tokenized = True
        for filename in self.all_fnames:
            try:
                with open(filename, "r", encoding=self.encoding) as handle:
                    data = handle.read()
            except (FileNotFoundError, UnicodeDecodeError, IsADirectoryError):
                continue

            try:
                lexer = guess_lexer_for_filename(filename, data)
            except Exception:
                continue

            names = []
            for token_type, value in lexer.get_tokens(data):
                if token_type in Token.Name:
                    names.append((value, "`%s`" % value))
            self.words.update(names)

    def get_command_completions(self, text, words):
        if len(words) == 1 and not text[-1].isspace():
            prefix = words[0].lower()
            return [name for name in self.command_names if name.startswith(prefix)]

        if len(words) <= 1 or text[-1].isspace():
            return []

        command = words[0]
        partial = words[-1].lower()
        matches, _, _ = self.commands.matching_commands(command)

        if len(matches) == 1:
            command = matches[0]
        elif command not in matches:
            return None

        if command not in self.command_completions:
            self.command_completions[command] = self.commands.get_completions(command)

        choices = self.command_completions[command]
        if choices is None:
            return None

        return [choice for choice in choices if partial in choice.lower()]

    def get_completions(self, document, complete_event):
        self.tokenize()

        text = document.text_before_cursor
        words = text.split()
        if not words:
            return

        if text and text[-1].isspace():
            return

        if text[0] == "/":
            options = self.get_command_completions(text, words)
            if options is not None:
                for option in sorted(options):
                    yield Completion(option, start_position=-len(words[-1]))
                return

        candidates = set(self.words)
        candidates.update(self.fname_to_rel_fnames)

        normalized = [
            item if type(item) is tuple else (item, item)
            for item in candidates
        ]

        needle = words[-1]
        found = []
        for match, insertion in normalized:
            if match.lower().startswith(needle.lower()):
                found.append((insertion, -len(needle), match))
                for full_name in self.fname_to_rel_fnames.get(match, []):
                    found.append((full_name, -len(needle), full_name))

        for insertion, position, display in sorted(found):
            yield Completion(insertion, start_position=position, display=display)


class InputOutput:
    num_error_outputs = 0
    num_user_asks = 0

    def __init__(
        self,
        pretty=True,
        yes=None,
        input_history_file=None,
        chat_history_file=None,
        input=None,
        output=None,
        user_input_color="blue",
        tool_output_color=None,
        tool_error_color="red",
        tool_warning_color="#FFA500",
        assistant_output_color="blue",
        completion_menu_color=None,
        completion_menu_bg_color=None,
        completion_menu_current_color=None,
        completion_menu_current_bg_color=None,
        code_theme="default",
        encoding="utf-8",
        dry_run=False,
        llm_history_file=None,
        editingmode=EditingMode.EMACS,
    ):
        self.editingmode = editingmode

        if os.environ.get("NO_COLOR") not in (None, ""):
            pretty = False

        self.user_input_color = user_input_color if pretty else None
        self.tool_output_color = tool_output_color if pretty else None
        self.tool_error_color = tool_error_color if pretty else None
        self.tool_warning_color = tool_warning_color if pretty else None
        self.assistant_output_color = assistant_output_color
        self.completion_menu_color = completion_menu_color if pretty else None
        self.completion_menu_bg_color = completion_menu_bg_color if pretty else None
        self.completion_menu_current_color = completion_menu_current_color if pretty else None
        self.completion_menu_current_bg_color = completion_menu_current_bg_color if pretty else None

        self.code_theme = code_theme
        self.input = input
        self.output = output
        self.pretty = bool(pretty)
        if output:
            self.pretty = False

        self.yes = yes
        self.input_history_file = input_history_file
        self.llm_history_file = llm_history_file
        self.chat_history_file = Path(chat_history_file) if chat_history_file is not None else None
        self.encoding = encoding
        self.dry_run = dry_run

        stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.append_chat_history("\n# aider chat started at %s\n\n" % stamp)

        self.prompt_session = None
        self.console = None

        if self.pretty:
            arguments = {
                "input": self.input,
                "output": self.output,
                "lexer": PygmentsLexer(MarkdownLexer),
                "editing_mode": self.editingmode,
                "cursor": ModalCursorShapeConfig(),
            }
            if self.input_history_file is not None:
                arguments["history"] = FileHistory(self.input_history_file)

            try:
                self.prompt_session = PromptSession(**arguments)
                self.console = Console()
            except Exception as err:
                self.console = Console(force_terminal=False, no_color=True)
                self.tool_error("Can't initialize prompt toolkit: %s" % err)

    def append_chat_history(self, text, linebreak=False):
        if self.chat_history_file is None:
            return

        try:
            parent = self.chat_history_file.parent
            if parent and not parent.exists():
                parent.mkdir(parents=True, exist_ok=True)
            with self.chat_history_file.open("a", encoding=self.encoding) as handle:
                handle.write(text)
                if linebreak and not text.endswith("\n"):
                    handle.write("\n")
        except OSError:
            pass

    def _write_plain(self, text):
        text = str(text)
        if self.output is not None and hasattr(self.output, "write") and not self.pretty:
            try:
                self.output.write(text + "\n")
                if hasattr(self.output, "flush"):
                    self.output.flush()
                return
            except Exception:
                pass
        print(text)

    def _message(self, messages):
        return " ".join(str(message) for message in messages)

    def tool_output(self, *messages, log_only=False):
        message = self._message(messages)
        self.append_chat_history(message + "\n")

        if log_only:
            return

        if self.pretty and self.console is not None:
            if self.tool_output_color:
                self.console.print(Text(message, style=RichStyle(color=self.tool_output_color)))
            else:
                self.console.print(message)
            return

        self._write_plain(message)

    def tool_error(self, *messages):
        self.num_error_outputs += 1
        message = self._message(messages)
        self.append_chat_history(message + "\n")

        if self.pretty and self.console is not None:
            if self.tool_error_color:
                self.console.print(Text(message, style=RichStyle(color=self.tool_error_color)))
            else:
                self.console.print(message)
            return

        self._write_plain(message)

    def tool_warning(self, *messages):
        message = self._message(messages)
        self.append_chat_history(message + "\n")

        if self.pretty and self.console is not None:
            if self.tool_warning_color:
                self.console.print(Text(message, style=RichStyle(color=self.tool_warning_color)))
            else:
                self.console.print(message)
            return

        self._write_plain(message)

    def assistant_output(self, message, pretty=True):
        message = str(message)
        self.append_chat_history("## Assistant\n\n%s\n\n" % message)

        if self.pretty and pretty and self.console is not None:
            if self.assistant_output_color:
                self.console.print(
                    Markdown(message, code_theme=self.code_theme),
                    style=RichStyle(color=self.assistant_output_color),
                )
            else:
                self.console.print(Markdown(message, code_theme=self.code_theme))
            return

        self._write_plain(message)

    def user_input(self, message):
        message = str(message)
        self.append_chat_history("## User\n\n%s\n\n" % message)

        if self.pretty and self.console is not None:
            if self.user_input_color:
                self.console.print(Text(message, style=RichStyle(color=self.user_input_color)))
            else:
                self.console.print(message)
            return

        self._write_plain(message)

    def _stream_input(self):
        if self.input is None:
            return None

        if hasattr(self.input, "readline"):
            value = self.input.readline()
        elif callable(self.input):
            value = self.input()
        elif isinstance(self.input, list):
            value = self.input.pop(0) if self.input else ""
        else:
            return None

        if value is None:
            return None
        value = str(value)
        if value == "":
            return None
        return value.rstrip("\r\n")

    def get_input(
        self, root, rel_fnames, addable_rel_fnames, commands, abs_read_only_fnames=None, edit_format=None
    ):
        supplied = self._stream_input()
        if supplied is not None:
            self.user_input(supplied)
            return supplied

        if self.input is not None:
            return None

        if self.prompt_session is not None:
            completer = AutoCompleter(
                root,
                rel_fnames,
                addable_rel_fnames,
                commands,
                self.encoding,
                abs_read_only_fnames,
            )

            palette = {}
            if self.completion_menu_color:
                palette["completion-menu"] = self.completion_menu_color
            if self.completion_menu_bg_color:
                palette["completion-menu"] = (
                    (palette.get("completion-menu", "") + " " + self.completion_menu_bg_color).strip()
                )
            if self.completion_menu_current_color:
                palette["completion-menu.completion.current"] = self.completion_menu_current_color
            if self.completion_menu_current_bg_color:
                palette["completion-menu.completion.current"] = (
                    (
                        palette.get("completion-menu.completion.current", "")
                        + " "
                        + self.completion_menu_current_bg_color
                    ).strip()
                )

            bindings = KeyBindings()

            @bindings.add("c-c")
            def _(event):
                event.app.exit(result=None)

            kwargs = {
                "completer": ThreadedCompleter(completer),
                "complete_style": CompleteStyle.MULTI_COLUMN,
                "key_bindings": bindings,
            }
            if palette:
                kwargs["style"] = Style.from_dict(palette)

            try:
                line = self.prompt_session.prompt("> ", **kwargs)
            except (EOFError, KeyboardInterrupt):
                return None
        else:
            try:
                line = input("> ")
            except (EOFError, KeyboardInterrupt):
                return None

        self.user_input(line)
        return line

    def prompt_ask(self, question, default=None):
        self.num_user_asks += 1

        if self.yes is True:
            return default

        prompt = str(question)
        if default is not None:
            prompt = "%s [%s] " % (prompt, default)
        elif not prompt.endswith(" "):
            prompt += " "

        supplied = self._stream_input()
        if supplied is not None:
            self.tool_output(prompt + supplied, log_only=True)
            return supplied if supplied else default

        if self.input is not None:
            return default

        try:
            if self.prompt_session is not None:
                answer = self.prompt_session.prompt(prompt)
            else:
                answer = input(prompt)
        except (EOFError, KeyboardInterrupt):
            return default

        return answer if answer else default

    def confirm_ask(
        self,
        question,
        default="y",
        subject=None,
        explicit_yes_required=False,
        group=None,
        allow_never=False,
    ):
        self.num_user_asks += 1

        if group is not None and group.preference is not None:
            return group.preference == "yes"

        if self.yes is True:
            return True

        if self.dry_run:
            return False

        yes_default = str(default).lower().startswith("y")
        choices = "y/N" if not yes_default else "Y/n"
        if allow_never:
            choices += "/never"
        if group is not None and group.show_group:
            choices += "/all"

        prompt = "%s (%s)" % (question, choices)
        if subject:
            prompt = "%s %s" % (prompt, subject)

        while True:
            answer = self.prompt_ask(prompt, None)
            if answer is None:
                return False

            response = str(answer).strip().lower()
            if not response:
                accepted = yes_default
            elif response in ("y", "yes"):
                accepted = True
            elif response in ("n", "no"):
                accepted = False
            elif allow_never and response == "never":
                if group is not None:
                    group.preference = "never"
                return False
            elif group is not None and group.show_group and response in ("a", "all"):
                group.preference = "yes"
                return True
            else:
                self.tool_warning("Please answer yes or no.")
                continue

            if explicit_yes_required and not response:
                accepted = False

            if group is not None:
                group.preference = "yes" if accepted else "no"
            return accepted

    def read_image(self, filename):
        try:
            with open(filename, "rb") as handle:
                return base64.b64encode(handle.read()).decode("utf-8")
        except OSError as err:
            self.tool_error("Unable to read image %s: %s" % (filename, err))
            return None