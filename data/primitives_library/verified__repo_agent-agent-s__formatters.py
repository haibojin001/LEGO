from gui_agents.s3.utils.common_utils import (
    create_pyautogui_code,
    extract_agent_functions,
    parse_code_from_string,
    split_thinking_response,
)


def _has_single_action(response):
    code = parse_code_from_string(response)
    actions = extract_agent_functions(code)
    return len(actions) == 1


single_action_check = _has_single_action
single_action_error_msg = (
    "Incorrect code: There must be a single agent action in the code response."
)


def _format_single_action(response):
    return single_action_check(response), single_action_error_msg


SINGLE_ACTION_FORMATTER = _format_single_action


def _attempt_code_creation(agent, code, obs):
    """Attempts to create a pyautogui code snippet from the response code"""
    try:
        return create_pyautogui_code(agent, code, obs)
    except Exception:
        return None


def _is_valid_code(agent, obs, response):
    code = parse_code_from_string(response)
    return _attempt_code_creation(agent, code, obs) is not None


code_valid_check = _is_valid_code
code_valid_error_msg = (
    "Incorrect code: The agent action must be a valid function and use valid "
    "parameters from the docstring list."
)


def _format_valid_code(agent, obs, response):
    return code_valid_check(agent, obs, response), code_valid_error_msg


CODE_VALID_FORMATTER = _format_valid_code


def _has_thoughts_and_answer(response):
    return split_thinking_response(response)[1] != ""


thoughts_answer_tag_check = _has_thoughts_and_answer
thoughts_answer_tag_error_msg = (
    "Incorrect response: The response must contain both "
    "<thoughts>...</thoughts> and <answer>...</answer> tags."
)


def _format_thoughts_and_answer(response):
    return thoughts_answer_tag_check(response), thoughts_answer_tag_error_msg


THOUGHTS_ANSWER_TAG_FORMATTER = _format_thoughts_and_answer


def _has_integer_answer(response):
    return split_thinking_response(response)[0].strip().isdigit()


integer_answer_check = _has_integer_answer
integer_answer_error_msg = (
    "Incorrect response: The <answer>...</answer> tag must contain a single integer."
)


def _format_integer_answer(response):
    return integer_answer_check(response), integer_answer_error_msg


INTEGER_ANSWER_FORMATTER = _format_integer_answer