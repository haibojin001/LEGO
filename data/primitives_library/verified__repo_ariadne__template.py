import html
from enum import IntEnum
from os import path

TEMPLATE_DIR = path.join(path.dirname(path.abspath(__file__)), "templates")


def read_template(template: str) -> str:
    template_path = path.join(TEMPLATE_DIR, template)
    with open(template_path, encoding="utf-8") as file:
        return file.read()


class Token(IntEnum):
    STR = 0
    VAR = 1
    RAW = 2
    IF = 3
    IF_NOT = 4
    ELSE = 5
    ENDIF = 6


TokenBlock = tuple[Token, str | None]


def render_template(template: str, template_vars: dict | None = None) -> str:
    return parse_template(template).render(template_vars or {})


def parse_template(template: str):
    return build_template_ast(tokenize_template(template))


def tokenize_template(template: str) -> list[TokenBlock]:
    result: list[TokenBlock] = []
    position = 0
    length = len(template)

    while position <= length:
        openings = [
            index
            for index in (
                template.find("{{", position),
                template.find("{%", position),
            )
            if index >= 0
        ]

        if not openings:
            result.append((Token.STR, template[position:length]))
            break

        next_position = min(openings)
        if next_position > position:
            result.append((Token.STR, template[position:next_position]))
            position = next_position

        if template[position : position + 2] == "{{":
            token, position = tokenize_var(template, position)
        else:
            token, position = tokenize_block(template, position)

        result.append(token)

    return result


def tokenize_var(template: str, cursor: int) -> tuple[TokenBlock, int]:
    closing = template.find("}}", cursor)
    if closing == -1:
        raise ValueError(
            f"Unclosed variable tag at {cursor}: '{template[cursor : cursor + 20]}...'"
        )

    variable = template[cursor + 2 : closing].strip()
    if not variable:
        raise ValueError(
            f"Empty variable tag at {cursor}: '{template[cursor : cursor + 20]}...'"
        )

    return (Token.VAR, variable), closing + 2


def tokenize_block(template: str, cursor: int) -> tuple[TokenBlock, int]:
    closing = template.find("%}", cursor)
    if closing == -1:
        raise ValueError(
            f"Unclosed block tag at {cursor}: '{template[cursor : cursor + 20]}...'"
        )

    contents = template[cursor + 2 : closing].strip()
    if not contents:
        raise ValueError(
            f"Empty block tag at {cursor}: '{template[cursor : cursor + 20]}...'"
        )

    parts = [part.strip() for part in contents.split(" ")]
    name = parts[0]
    arguments = " ".join(parts[1:])
    block_name = name.lower()
    preview = f"'{template[cursor : cursor + 20]}...'"

    if block_name == "if":
        if not arguments:
            raise ValueError(f"'if' block without arguments at {cursor}: {preview}")
        token = (Token.IF, arguments)
    elif block_name == "ifnot":
        if not arguments:
            raise ValueError(f"'ifnot' block without arguments at {cursor}: {preview}")
        token = (Token.IF_NOT, arguments)
    elif block_name == "else":
        if arguments:
            raise ValueError(f"'else' block with arguments at {cursor}: {preview}")
        token = (Token.ELSE, None)
    elif block_name == "endif":
        if arguments:
            raise ValueError(f"'endif' block with arguments at {cursor}: {preview}")
        token = (Token.ENDIF, None)
    elif block_name == "raw":
        if not arguments:
            raise ValueError(f"'raw' block without arguments at {cursor}: {preview}")
        token = (Token.RAW, arguments)
    else:
        raise ValueError(f"Unknown block at {cursor}: {preview}")

    return token, closing + 2


def build_template_ast(tokens: list[TokenBlock]) -> "TemplateDocument":
    return TemplateDocument(ast_to_nodes(tokens))


def ast_to_nodes(tokens: list[TokenBlock]) -> list["TemplateNode"]:
    nodes: list[TemplateNode] = []
    index = 0
    count = len(tokens)

    while index < count:
        token_type, token_value = tokens[index]

        if token_type == Token.STR and token_value:
            nodes.append(TemplateText(token_value))
            index += 1
            continue

        if token_type == Token.VAR and token_value:
            nodes.append(TemplateVariable(token_value))
            index += 1
            continue

        if token_type == Token.RAW and token_value:
            nodes.append(TemplateVariable(token_value, escape=False))
            index += 1
            continue

        if token_type in (Token.IF, Token.IF_NOT) and token_value:
            if index + 1 == count:
                raise ValueError("Unclosed 'if' block found.")

            arguments = token_value.split(" ")
            nested = 0
            children: list[TokenBlock] = []
            inverted = token_type == Token.IF_NOT
            else_found = False

            for child in tokens[index + 1 :]:
                index += 1
                child_type = child[0]

                if child_type == Token.ENDIF:
                    if nested == 0:
                        index += 1
                        break
                    nested -= 1
                elif child_type == Token.ELSE:
                    if nested == 0:
                        if else_found:
                            raise ValueError("Multiple 'else' clauses found.")
                        nodes.append(
                            TemplateIfBlock(
                                arguments,
                                ast_to_nodes(children),
                                inverted,
                            )
                        )
                        children = []
                        inverted = not inverted
                        else_found = True
                        continue
                elif child_type in (Token.IF, Token.IF_NOT):
                    nested += 1

                children.append(child)
            else:
                raise ValueError("Unclosed 'if' block found.")

            nodes.append(
                TemplateIfBlock(
                    arguments,
                    ast_to_nodes(children),
                    inverted,
                )
            )
            continue

        if token_type == Token.ENDIF:
            raise ValueError("Extra 'endif' block found.")

    return nodes


class TemplateNode:
    def render(self, template_vars) -> str:
        raise NotImplementedError(
            "Subclasses of TemplateNode should define 'render' method."
        )


class TemplateDocument(TemplateNode):
    def __init__(self, nodes) -> None:
        self.nodes = nodes

    def render(self, template_vars) -> str:
        return "".join(node.render(template_vars) for node in self.nodes)


class TemplateText(TemplateNode):
    def __init__(self, value: str) -> None:
        self.value = value

    def render(self, template_vars) -> str:
        return self.value


class TemplateIfBlock(TemplateNode):
    def __init__(self, args: list[str], nodes, if_not: bool = False) -> None:
        self.args = args
        self.nodes = nodes
        self.if_not = if_not

    def render(self, template_vars) -> str:
        arguments_true = all(template_vars.get(argument) for argument in self.args)
        visible = not arguments_true if self.if_not else arguments_true
        if not visible:
            return ""
        return "".join(node.render(template_vars) for node in self.nodes)


class TemplateVariable(TemplateNode):
    def __init__(self, var_name: str, escape: bool = True) -> None:
        self.var_name = var_name
        self.escape = escape

    def render(self, template_vars) -> str:
        value = str(template_vars.get(self.var_name) or "")
        if self.escape:
            return html.escape(value)
        return value