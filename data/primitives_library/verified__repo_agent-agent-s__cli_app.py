import argparse
import datetime
import io
import logging
import os
import platform
import pyautogui
import signal
import sys
import time

from PIL import Image

from gui_agents.s3.agents.grounding import OSWorldACI
from gui_agents.s3.agents.agent_s import AgentS3
from gui_agents.s3.utils.local_env import LocalEnv

current_platform = platform.system().lower()

paused = False


def get_char():
    try:
        if platform.system() in ["Darwin", "Linux"]:
            import termios
            import tty

            fd = sys.stdin.fileno()
            old_settings = termios.tcgetattr(fd)
            try:
                tty.setraw(sys.stdin.fileno())
                ch = sys.stdin.read(1)
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
            return ch
        import msvcrt

        return msvcrt.getch().decode("utf-8", errors="ignore")
    except Exception:
        return input()


def signal_handler(signum, frame):
    global paused

    if not paused:
        print("\n\n🔸 Agent-S Workflow Paused 🔸")
        print("=" * 50)
        print("Options:")
        print("  • Press Ctrl+C again to quit")
        print("  • Press Esc to resume workflow")
        print("=" * 50)

        paused = True

        while paused:
            try:
                print("\n[PAUSED] Waiting for input... ", end="", flush=True)
                char = get_char()

                if ord(char) == 3:
                    print("\n\n🛑 Exiting Agent-S...")
                    sys.exit(0)
                elif ord(char) == 27:
                    print("\n\n▶️  Resuming Agent-S workflow...")
                    paused = False
                    break
                else:
                    print(f"\n   Unknown command: '{char}' (ord: {ord(char)})")
            except KeyboardInterrupt:
                print("\n\n🛑 Exiting Agent-S...")
                sys.exit(0)
    else:
        print("\n\n🛑 Exiting Agent-S...")
        sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

datetime_str: str = datetime.datetime.now().strftime("%Y%m%d@%H%M%S")

log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

file_handler = logging.FileHandler(
    os.path.join("logs", "normal-{:}.log".format(datetime_str)), encoding="utf-8"
)
debug_handler = logging.FileHandler(
    os.path.join("logs", "debug-{:}.log".format(datetime_str)), encoding="utf-8"
)
stdout_handler = logging.StreamHandler(sys.stdout)
sdebug_handler = logging.FileHandler(
    os.path.join("logs", "sdebug-{:}.log".format(datetime_str)), encoding="utf-8"
)

file_handler.setLevel(logging.INFO)
debug_handler.setLevel(logging.DEBUG)
stdout_handler.setLevel(logging.INFO)
sdebug_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter(
    fmt="\x1b[1;33m[%(asctime)s \x1b[31m%(levelname)s \x1b[32m%(module)s/%(lineno)d-%(processName)s\x1b[1;33m] \x1b[0m%(message)s"
)
file_handler.setFormatter(formatter)
debug_handler.setFormatter(formatter)
stdout_handler.setFormatter(formatter)
sdebug_handler.setFormatter(formatter)

stdout_handler.addFilter(logging.Filter("desktopenv"))
sdebug_handler.addFilter(logging.Filter("desktopenv"))

logger.addHandler(file_handler)
logger.addHandler(debug_handler)
logger.addHandler(stdout_handler)
logger.addHandler(sdebug_handler)

platform_os = platform.system()


def show_permission_dialog(code: str, action_description: str):
    if platform.system() == "Darwin":
        result = os.system(
            f'osascript -e \'display dialog "Do you want to execute this action?\n\n{code} which will try to {action_description}" with title "Action Permission" buttons {{"Cancel", "OK"}} default button "OK" cancel button "Cancel"\''
        )
        return result == 0
    elif platform.system() == "Linux":
        result = os.system(
            f'zenity --question --title="Action Permission" --text="Do you want to execute this action?\n\n{code}" --width=400 --height=200'
        )
        return result == 0
    return False


def scale_screen_dimensions(width: int, height: int, max_dim_size: int):
    scale_factor = min(max_dim_size / width, max_dim_size / height, 1)
    safe_width = int(width * scale_factor)
    safe_height = int(height * scale_factor)
    return safe_width, safe_height


def run_agent(agent, instruction: str, scaled_width: int, scaled_height: int):
    global paused

    obs = {}
    traj = "Task:\n" + instruction
    subtask_traj = ""

    for step in range(15):
        while paused:
            time.sleep(0.1)

        screenshot = pyautogui.screenshot()
        screenshot = screenshot.resize((scaled_width, scaled_height), Image.LANCZOS)

        buffered = io.BytesIO()
        screenshot.save(buffered, format="PNG")
        screenshot_bytes = buffered.getvalue()
        obs["screenshot"] = screenshot_bytes

        while paused:
            time.sleep(0.1)

        print(f"\n🔄 Step {step + 1}/15: Getting next action from agent...")

        info, code = agent.predict(instruction=instruction, observation=obs)

        if "done" in code[0].lower() or "fail" in code[0].lower():
            if platform.system() == "Darwin":
                os.system(
                    'osascript -e \'display dialog "Task Completed" with title "OpenACI Agent" buttons "OK" default button "OK"\''
                )
            elif platform.system() == "Linux":
                os.system(
                    'zenity --info --title="OpenACI Agent" --text="Task Completed" --width=200 --height=100'
                )
            break

        if "next" in code[0].lower():
            continue

        if "wait" in code[0].lower():
            print("⏳ Agent requested wait...")
            time.sleep(5)
            continue

        time.sleep(1.0)
        print("EXECUTING CODE:", code[0])

        while paused:
            time.sleep(0.1)

        exec(code[0])
        time.sleep(1.0)

        if "reflection" in info and "executor_plan" in info:
            traj += (
                "\n\nReflection:\n"
                + str(info["reflection"])
                + "\n\n----------------------\n\nPlan:\n"
                + info["executor_plan"]
            )


def main():
    parser = argparse.ArgumentParser(description="Run AgentS3 with specified model.")
    parser.add_argument(
        "--provider",
        type=str,
        default="openai",
        help="Specify the provider to use (e.g., openai, anthropic, etc)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-5-2025-08-07",
        help="Specify the model to use (e.g., gpt-5-2025-08-07)",
    )
    parser.add_argument(
        "--model_url",
        type=str,
        default="",
        help="The URL of the main generation model API.",
    )
    parser.add_argument(
        "--model_api_key",
        type=str,
        default="",
        help="The API key of the main generation model.",
    )
    parser.add_argument(
        "--model_temperature",
        type=float,
        default=None,
        help="Temperature to fix the generation model at (e.g. o3 can only be run with 1.0)",
    )
    parser.add_argument(
        "--ground_provider",
        type=str,
        required=True,
        help="The provider for the grounding model",
    )
    parser.add_argument(
        "--ground_url",
        type=str,
        required=True,
        help="The URL of the grounding model",
    )
    parser.add_argument(
        "--ground_api_key",
        type=str,
        default="",
        help="The API key of the grounding model.",
    )
    parser.add_argument(
        "--ground_model",
        type=str,
        required=True,
        help="The model name for the grounding model",
    )
    parser.add_argument(
        "--grounding_width",
        type=int,
        required=True,
        help="Width of the grounding model input",
    )
    parser.add_argument(
        "--grounding_height",
        type=int,
        required=True,
        help="Height of the grounding model input",
    )
    parser.add_argument(
        "--instruction",
        type=str,
        default="",
        help="Task instruction for Agent-S.",
    )

    args = parser.parse_args()

    engine_params_for_generation = {
        "engine_type": args.provider,
        "model": args.model,
        "base_url": args.model_url,
        "api_key": args.model_api_key,
    }
    if args.model_temperature is not None:
        engine_params_for_generation["temperature"] = args.model_temperature

    grounding_engine_params = {
        "engine_type": args.ground_provider,
        "model": args.ground_model,
        "base_url": args.ground_url,
        "api_key": args.ground_api_key,
    }

    grounding_agent = OSWorldACI(
        platform=platform_os,
        engine_params_for_generation=grounding_engine_params,
        width=args.grounding_width,
        height=args.grounding_height,
    )

    agent = AgentS3(
        engine_params_for_generation=engine_params_for_generation,
        grounding_agent=grounding_agent,
        platform=platform_os,
    )

    LocalEnv(platform=platform_os)

    instruction = args.instruction
    if not instruction:
        instruction = input("Enter your instruction: ")

    screen_width, screen_height = pyautogui.size()
    scaled_width, scaled_height = scale_screen_dimensions(
        screen_width, screen_height, args.grounding_width
    )
    run_agent(agent, instruction, scaled_width, scaled_height)


if __name__ == "__main__":
    main()