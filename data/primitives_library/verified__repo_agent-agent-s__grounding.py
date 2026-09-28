import base64
import logging
import re
from collections import defaultdict
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

import pytesseract
from PIL import Image
from pytesseract import Output

from gui_agents.s3.memory.procedural_memory import PROCEDURAL_MEMORY
from gui_agents.s3.core.mllm import LMMAgent
from gui_agents.s3.utils.common_utils import call_llm_safe
from gui_agents.s3.agents.code_agent import CodeAgent


logger = logging.getLogger("desktopenv.agent")


class ACI:
    def __init__(self):
        self.notes: List[str] = []


def agent_action(func):
    func.is_agent_action = True
    return func


UBUNTU_APP_SETUP = """import difflib
import subprocess
import time
import pyautogui

pyautogui.press("escape")
time.sleep(0.5)
windows = subprocess.check_output(["wmctrl", "-lx"]).decode("utf-8").splitlines()
titles = [line.split(None, 4)[2] for line in windows if len(line.split(None, 4)) >= 3]
matches = difflib.get_close_matches("APP_NAME", titles, n=1, cutoff=0.1)
if matches:
    chosen = matches[0]
    window_id = None
    for line in windows:
        if chosen in line:
            window_id = line.split()[0]
            break
    if window_id:
        subprocess.run(["wmctrl", "-ia", window_id])
        subprocess.run(
            ["wmctrl", "-ir", window_id, "-b", "add,maximized_vert,maximized_horz"]
        )
"""


SET_CELL_VALUES_CMD = r"""import json
import subprocess
import unicodedata
import uno

def _normalise(value):
    if value is None:
        return None
    if not isinstance(value, str):
        return value
    if "\\u" in value or "\\U" in value or "\\x" in value:
        try:
            value = json.loads('"' + value.replace('"', '\\"') + '"')
        except Exception:
            try:
                value = value.encode("utf-8").decode("unicode_escape")
            except Exception:
                pass
    return unicodedata.normalize("NFC", value)

def _cell_position(reference):
    letters = "".join(character for character in reference if character.isalpha())
    digits = "".join(character for character in reference if character.isdigit())
    if not letters or not digits:
        raise ValueError("Invalid cell reference: " + str(reference))
    column = 0
    for character in letters.upper():
        column = column * 26 + ord(character) - ord("A") + 1
    return column - 1, int(digits) - 1

def _document_kind(component):
    if component.supportsService("com.sun.star.sheet.SpreadsheetDocument"):
        return "Calc"
    if component.supportsService("com.sun.star.text.TextDocument"):
        return "Writer"
    if component.supportsService("com.sun.star.presentation.PresentationDocument"):
        return "Impress"
    return None

def set_cell_values(values, app_name="Untitled 1", sheet_name="Sheet1"):
    app_name = _normalise(app_name)
    sheet_name = _normalise(sheet_name)
    indexed = {{}}
    for address, value in values.items():
        indexed[_cell_position(address)] = value

    subprocess.run(
        'echo "osworld-public-evaluation" | sudo -S ss --kill --tcp state TIME-WAIT sport = :2002',
        shell=True,
        capture_output=True,
        text=True,
    )
    subprocess.Popen(
        ["soffice", "--accept=socket,host=localhost,port=2002;urp;StarOffice.Service"]
    )

    local_context = uno.getComponentContext()
    resolver = local_context.ServiceManager.createInstanceWithContext(
        "com.sun.star.bridge.UnoUrlResolver", local_context
    )
    context = resolver.resolve(
        "uno:socket,host=localhost,port=2002;urp;StarOffice.ComponentContext"
    )
    desktop = context.ServiceManager.createInstanceWithContext(
        "com.sun.star.frame.Desktop", context
    )

    candidates = []
    for component in desktop.Components:
        if _document_kind(component) == "Calc":
            candidates.append(component)
    if not candidates:
        raise ValueError("Could not find a LibreOffice Calc document.")

    selected = next(
        (component for component in candidates if component.Title == app_name),
        candidates[0],
    )
    try:
        sheet = selected.Sheets.getByName(sheet_name)
    except Exception as error:
        raise ValueError(
            "Could not find sheet " + str(sheet_name) + " in " + str(app_name)
        ) from error

    for (column, row), value in indexed.items():
        cell = sheet.getCellByPosition(column, row)
        if isinstance(value, bool):
            cell.Value = 1 if value else 0
        elif isinstance(value, (int, float)):
            cell.Value = value
        elif isinstance(value, str):
            if value.startswith("="):
                cell.Formula = value
            else:
                cell.String = value
        elif value is None:
            cell.clearContents(0)
        else:
            raise ValueError("Unsupported cell value type: " + str(type(value)))

set_cell_values(new_cell_values, app_name="{app_name}", sheet_name="{sheet_name}")
"""


class OSWorldACI(ACI):
    def __init__(
        self,
        env,
        platform: str,
        engine_params_for_generation: Dict,
        engine_params_for_grounding: Dict,
        width: int = 1920,
        height: int = 1080,
        code_agent_budget: int = 20,
        code_agent_engine_params: Dict = None,
    ):
        super().__init__()
        self.env = env
        self.platform = platform
        self.width = width
        self.height = height
        self.notes = []
        self.obs = None

        self.grounding_model = LMMAgent(engine_params_for_grounding)
        self.engine_params_for_grounding = engine_params_for_grounding
        self.text_span_agent = LMMAgent(
            engine_params=engine_params_for_generation,
            system_prompt=PROCEDURAL_MEMORY.PHRASE_TO_WORD_COORDS_PROMPT,
        )

        if code_agent_engine_params is None:
            code_agent_engine_params = engine_params_for_generation
        self.code_agent = CodeAgent(code_agent_engine_params, code_agent_budget)
        self.current_task_instruction = None
        self.last_code_agent_result = None

    def generate_coords(self, ref_expr: str, obs: Dict) -> List[int]:
        self.grounding_model.reset()
        prompt = (
            f"Query:{ref_expr}\n"
            "Output only the coordinate of one point in your response.\n"
        )
        self.grounding_model.add_message(
            text_content=prompt,
            image_content=obs["screenshot"],
            put_text_last=True,
        )
        response = call_llm_safe(self.grounding_model)
        print("RAW GROUNDING MODEL RESPONSE:", response)
        values = re.findall(r"\d+", response)
        assert len(values) >= 2
        return [int(values[0]), int(values[1])]

    @staticmethod
    def _image_from_data(image_data: Any) -> Image.Image:
        if isinstance(image_data, Image.Image):
            return image_data
        if isinstance(image_data, str):
            try:
                image_data = base64.b64decode(image_data)
            except Exception:
                image_data = image_data.encode()
        return Image.open(BytesIO(image_data))

    def get_ocr_elements(self, b64_image_data: str) -> Tuple[str, List]:
        image = self._image_from_data(b64_image_data)
        image_data = pytesseract.image_to_data(image, output_type=Output.DICT)

        for index, word in enumerate(image_data.get("text", [])):
            image_data["text"][index] = re.sub(
                r"^[^a-zA-Z\s.,!?;:\-\+]+|[^a-zA-Z\s.,!?;:\-\+]+$",
                "",
                word or "",
            )

        elements: List[Dict[str, Any]] = []
        table_lines = ["Text Table:", "Word id\tText"]
        groups = defaultdict(list)

        texts = image_data.get("text", [])
        for index, text in enumerate(texts):
            text = (text or "").strip()
            if not text:
                continue

            try:
                confidence = float(image_data.get("conf", ["0"])[index])
            except (TypeError, ValueError, IndexError):
                confidence = 0.0
            if confidence < 0:
                continue

            block = image_data.get("block_num", [0])[index]
            paragraph = image_data.get("par_num", [0])[index]
            line = image_data.get("line_num", [0])[index]
            group_key = (block, paragraph, line)
            left = int(image_data.get("left", [0])[index])
            top = int(image_data.get("top", [0])[index])
            width = int(image_data.get("width", [0])[index])
            height = int(image_data.get("height", [0])[index])

            element = {
                "id": len(elements),
                "text": text,
                "group_num": group_key,
                "word_num": image_data.get("word_num", [0])[index],
                "left": left,
                "top": top,
                "width": width,
                "height": height,
                "x": left,
                "y": top,
            }
            elements.append(element)
            groups[group_key].append(element)
            table_lines.append(f"{element['id']}\t{text}")

        for group_index, group in enumerate(groups.values()):
            for word_index, element in enumerate(group):
                element["group_num"] = group_index
                element["word_num"] = word_index

        return "\n".join(table_lines), elements

    def generate_text_coords(self, ref_expr: str, obs: Dict) -> List[int]:
        screenshot = obs["screenshot"]
        table, elements = self.get_ocr_elements(screenshot)
        if not elements:
            return self.generate_coords(ref_expr, obs)

        self.text_span_agent.reset()
        prompt = (
            f"{table}\n\n"
            f"Find the word ids corresponding to this phrase: {ref_expr}\n"
            "Return only the relevant word ids, separated by commas."
        )
        self.text_span_agent.add_message(text_content=prompt, put_text_last=True)
        response = call_llm_safe(self.text_span_agent)
        ids = [int(value) for value in re.findall(r"\d+", response)]

        selected = [elements[index] for index in ids if 0 <= index < len(elements)]
        if not selected:
            return self.generate_coords(ref_expr, obs)

        left = min(item["left"] for item in selected)
        top = min(item["top"] for item in selected)
        right = max(item["left"] + item["width"] for item in selected)
        bottom = max(item["top"] + item["height"] for item in selected)
        return [(left + right) // 2, (top + bottom) // 2]

    def ground_text(self, ref_expr: str, obs: Optional[Dict] = None) -> List[int]:
        return self.generate_text_coords(ref_expr, obs or self.obs)

    def _coords(self, description: str) -> List[int]:
        if self.obs is None:
            raise ValueError("No screenshot observation is available for grounding.")
        return self.generate_coords(description, self.obs)

    @agent_action
    def click(
        self,
        element_description: str,
        num_clicks: int = 1,
        button: str = "left",
        hold_keys: List[str] = [],
    ) -> str:
        x, y = self._coords(element_description)
        prefix = ""
        suffix = ""
        if hold_keys:
            prefix = "\n".join(f"pyautogui.keyDown({key!r})" for key in hold_keys) + "\n"
            suffix = "\n" + "\n".join(
                f"pyautogui.keyUp({key!r})" for key in reversed(hold_keys)
            )
        return (
            f"{prefix}pyautogui.click({x}, {y}, clicks={num_clicks}, "
            f"button={button!r}){suffix}"
        )

    @agent_action
    def double_click(self, element_description: str, button: str = "left") -> str:
        return self.click(element_description, num_clicks=2, button=button)

    @agent_action
    def right_click(self, element_description: str) -> str:
        return self.click(element_description, button="right")

    @agent_action
    def type(self, text: str) -> str:
        return f"pyperclip.copy({text!r})\npyautogui.hotkey('ctrl', 'v')"

    @agent_action
    def press(self, key: str, presses: int = 1, interval: float = 0.1) -> str:
        return f"pyautogui.press({key!r}, presses={presses}, interval={interval})"

    @agent_action
    def hotkey(self, keys: List[str]) -> str:
        return f"pyautogui.hotkey(*{list(keys)!r})"

    @agent_action
    def scroll(self, clicks: int, element_description: Optional[str] = None) -> str:
        if element_description:
            x, y = self._coords(element_description)
            return f"pyautogui.moveTo({x}, {y})\npyautogui.scroll({clicks})"
        return f"pyautogui.scroll({clicks})"

    @agent_action
    def drag_and_drop(
        self,
        source_description: str,
        target_description: str,
        duration: float = 1.0,
    ) -> str:
        source_x, source_y = self._coords(source_description)
        target_x, target_y = self._coords(target_description)
        return (
            f"pyautogui.moveTo({source_x}, {source_y})\n"
            f"pyautogui.dragTo({target_x}, {target_y}, duration={duration})"
        )

    @agent_action
    def switch_applications(self, app_name: str) -> str:
        if self.platform.lower() in {"ubuntu", "linux"}:
            return UBUNTU_APP_SETUP.replace("APP_NAME", app_name)
        if self.platform.lower() in {"windows", "win"}:
            return "pyautogui.hotkey('alt', 'tab')"
        if self.platform.lower() in {"darwin", "macos", "mac"}:
            return "pyautogui.hotkey('command', 'tab')"
        return "pyautogui.hotkey('alt', 'tab')"

    @agent_action
    def wait(self, seconds: float = 1.0) -> str:
        return f"time.sleep({seconds})"

    @agent_action
    def save_to_knowledge(self, text: str) -> str:
        self.notes.append(text)
        return ""

    @agent_action
    def set_cell_values(
        self,
        cell_values: Dict[str, Any],
        app_name: str = "Untitled 1",
        sheet_name: str = "Sheet1",
    ) -> str:
        command = SET_CELL_VALUES_CMD.replace(
            "new_cell_values", repr(cell_values), 1
        )
        return command.format(app_name=app_name, sheet_name=sheet_name)