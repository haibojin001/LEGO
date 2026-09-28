import logging
from typing import Dict, List, Tuple, Optional

from gui_agents.s3.memory.procedural_memory import PROCEDURAL_MEMORY
from gui_agents.s3.utils.common_utils import call_llm_safe, split_thinking_response
from gui_agents.s3.core.mllm import LMMAgent

logger = logging.getLogger("desktopenv.agent")


def extract_code_block(action: str) -> Tuple[Optional[str], Optional[str]]:
    """Extract code and determine type from action string."""
    code_type = None
    code = None

    if "```python" in action:
        code_type = "python"
        code = action.split("```python")[1].split("```")[0].strip()
    elif "```bash" in action:
        code_type = "bash"
        code = action.split("```bash")[1].split("```")[0].strip()
    elif "```" in action:
        code = action.split("```")[1].split("```")[0].strip()

    logger.debug(
        f"Extracted code block: type={code_type}, length={len(code) if code else 0}"
    )
    return code_type, code


def execute_code(code_type: str, code: str, env_controller) -> Dict:
    """Execute code based on its type."""
    logger.info(f"CODING_AGENT_CODE_EXECUTION - Type: {code_type}\nCode:\n{code}")

    try:
        if code_type == "bash":
            return env_controller.run_bash_script(code, timeout=30)
        if code_type == "python":
            return env_controller.run_python_script(code)
        return {"status": "error", "error": f"Unknown code type: {code_type}"}
    except Exception as exc:
        logger.error(f"Error executing {code_type} code: {exc}")
        return {"status": "error", "error": str(exc)}


def format_result(result: Dict, step_count: int) -> str:
    """Format execution result into context string."""
    if not result:
        logger.warning(f"Step {step_count + 1}: No result returned from execution")
        return f"""
Step {step_count + 1} Error:
Error: No result returned from execution
"""

    status = result.get("status", "unknown")
    return_code = result.get("returncode", result.get("return_code", -1))

    if "returncode" in result:
        output = result.get("output", "")
        error = result.get("error", "")
    else:
        output = result.get("output", "")
        error = result.get("error", "")

    logger.debug(f"Step {step_count + 1}: Status={status}, Return Code={return_code}")

    text = f"Step {step_count + 1} Result:\n"
    text += f"Status: {status}\n"
    text += f"Return Code: {return_code}\n"

    if output:
        text += f"Output:\n{output}\n"
    if error:
        text += f"Error:\n{error}\n"

    return text


class CodeAgent:
    """A dedicated agent for executing code with a budget of steps."""

    def __init__(self, engine_params: Dict, budget: int = 20):
        """Initialize the CodeAgent."""
        if not engine_params:
            raise ValueError("engine_params cannot be None or empty")

        self.engine_params = engine_params
        self.budget = budget
        self.agent = None

        logger.info(f"CodeAgent initialized with budget={budget}")
        self.reset()

    def reset(self):
        """Reset the code agent state."""
        logger.debug("Resetting CodeAgent state")
        self.agent = LMMAgent(
            engine_params=self.engine_params,
            system_prompt=PROCEDURAL_MEMORY.CODE_AGENT_PROMPT,
        )

    def execute(self, task_instruction: str, screenshot: str, env_controller) -> Dict:
        """Execute code for the given task with a budget of steps."""
        if env_controller is None:
            raise ValueError("env_controller is required for code execution")

        print("\n🚀 STARTING CODE EXECUTION")
        print("=" * 60)
        print(f"Task: {task_instruction}")
        print(f"Budget: {self.budget} steps")
        print("=" * 60)

        logger.info(f"Starting code execution for task: {task_instruction}")
        logger.info(f"Budget: {self.budget} steps")

        self.reset()

        context = (
            f"Task: {task_instruction}\n\nCurrent screenshot is provided for context."
        )
        self.agent.add_message(context, image_content=screenshot, role="user")

        step_count = 0
        execution_history: List[Dict] = []
        completion_reason = "BUDGET_EXHAUSTED"

        while step_count < self.budget:
            logger.info(f"Step {step_count + 1}/{self.budget}")

            response = call_llm_safe(self.agent, temperature=1)

            print(f"\n🤖 CODING AGENT RESPONSE - Step {step_count + 1}/{self.budget}")
            print("=" * 60)
            print(response)
            print("=" * 60)

            logger.info(
                f"CODING_AGENT_LATEST_MESSAGE - Step {step_count + 1}:\n{response}"
            )

            if not response or response.strip() == "":
                error_msg = f"Step {step_count + 1}: LLM returned empty response"
                logger.error(error_msg)
                raise RuntimeError(error_msg)

            action, thoughts = split_thinking_response(response)
            execution_history.append(
                {
                    "step": step_count + 1,
                    "action": action,
                    "thoughts": thoughts,
                }
            )

            action_upper = action.upper().strip()
            if action_upper == "DONE":
                print(f"\n✅ TASK COMPLETED - Step {step_count + 1}")
                print("=" * 60)
                print("Agent signaled task completion")
                print("=" * 60)
                logger.info(f"Step {step_count + 1}: Task completed successfully")
                completion_reason = "DONE"
                break

            if action_upper == "FAIL":
                print(f"\n❌ TASK FAILED - Step {step_count + 1}")
                print("=" * 60)
                print("Agent signaled task failure")
                print("=" * 60)
                logger.info(f"Step {step_count + 1}: Task failed by agent request")
                completion_reason = "FAIL"
                break

            code_type, code = extract_code_block(action)

            if code:
                result = execute_code(code_type, code, env_controller)
                output = result.get("output", "")
                error = result.get("error", "")
                message = result.get("message", "")
                status = result.get("status", "")

                print(f"\n⚡ CODE EXECUTION RESULT - Step {step_count + 1}")
                print("-" * 50)
                print(f"Status: {status}")
                if output:
                    print(f"Output:\n{output}")
                if error:
                    print(f"Error:\n{error}")
                if message and not output and not error:
                    print(f"Message:\n{message}")
                print("-" * 50)

                log_lines = [
                    f"CODING_AGENT_EXECUTION_RESULT - Step {step_count + 1}:",
                    f"Status: {status}" if status else None,
                ]

                if output:
                    log_lines.append(
                        "Output:\n" + "-" * 40 + f"\n{output}\n" + "-" * 40
                    )
                if error:
                    log_lines.append(
                        "Error:\n" + "!" * 40 + f"\n{error}\n" + "!" * 40
                    )
                if message and not output and not error:
                    log_lines.append(
                        "Message:\n" + "-" * 40 + f"\n{message}\n" + "-" * 40
                    )

                logger.info("\n".join(line for line in log_lines if line))
            else:
                print(f"\n⚠️  NO CODE BLOCK FOUND - Step {step_count + 1}")
                print("-" * 50)
                print("Action did not contain executable code")
                print("-" * 50)

                logger.warning(f"Step {step_count + 1}: No code block found in action")
                result = {"status": "skipped", "message": "No code block found"}
                logger.info(
                    f"CODING_AGENT_EXECUTION_RESULT - Step {step_count + 1}:\n"
                    "Status: skipped\n"
                    "Message:\n"
                    "----------------------------------------\n"
                    "No code block found\n"
                    "----------------------------------------"
                )

            formatted_result = format_result(result, step_count)
            self.agent.add_message(formatted_result, role="user")
            step_count += 1

        if completion_reason == "BUDGET_EXHAUSTED":
            logger.info(f"Code execution budget exhausted after {self.budget} steps")
            print("\n⏱️  CODE EXECUTION BUDGET EXHAUSTED")
            print("=" * 60)
            print(f"Maximum of {self.budget} steps reached")
            print("=" * 60)

        return {
            "status": "success" if completion_reason == "DONE" else "failure",
            "completion_reason": completion_reason,
            "steps_taken": step_count + (1 if completion_reason in ("DONE", "FAIL") else 0),
            "execution_history": execution_history,
        }