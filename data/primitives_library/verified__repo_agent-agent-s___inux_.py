import base64
import logging
import os
import time
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import requests

from gui_agents.s1.aci.ACI import ACI
from gui_agents.s1.utils.common_utils import box_iou

import platform

if platform.system() == "Linux":
    try:
        import pyatspi
        from pyatspi import Accessible, StateType, STATE_SHOWING
        from pyatspi import Action as ATAction
        from pyatspi import Component
        from pyatspi import Text as ATText
        from pyatspi import Value as ATValue
        from lxml.etree import _Element
        import lxml.etree
        import concurrent.futures
    except ImportError:
        pyatspi = None
        Accessible = Any
        StateType = Any
        STATE_SHOWING = Any
        ATAction = Any
        Component = Any
        ATText = Any
        ATValue = Any
        _Element = Any


_accessibility_ns_map_ubuntu = {
    "st": "https://accessibility.ubuntu.example.org/ns/state",
    "attr": "https://accessibility.ubuntu.example.org/ns/attributes",
    "cp": "https://accessibility.ubuntu.example.org/ns/component",
    "doc": "https://accessibility.ubuntu.example.org/ns/document",
    "docattr": "https://accessibility.ubuntu.example.org/ns/document/attributes",
    "txt": "https://accessibility.ubuntu.example.org/ns/text",
    "val": "https://accessibility.ubuntu.example.org/ns/value",
    "act": "https://accessibility.ubuntu.example.org/ns/action",
}

MAX_DEPTH = 50
MAX_WIDTH = 1024

logger = logging.getLogger("desktopenv.agent")

state_ns = "https://accessibility.ubuntu.example.org/ns/state"
component_ns = "https://accessibility.ubuntu.example.org/ns/component"
attributes_ns = "https://accessibility.ubuntu.example.org/ns/attributes"
value_ns = "https://accessibility.ubuntu.example.org/ns/value"


def agent_action(func):
    func.is_agent_action = True
    return func


class LinuxACI(ACI):
    def __init__(self, top_app=None, vm_version="new", top_app_only=True, ocr=True):
        self.active_apps = set()
        self.top_app = top_app
        self.top_app_only = top_app_only
        self.ocr = ocr
        self.index_out_of_range_flag = False
        self.top_active_app = None
        self.notes = []
        self.clipboard = ""
        self.preserved_nodes = []
        self.ocr_elements = []
        self.app_setup_code = """import subprocess;
import difflib;
import pyautogui;
pyautogui.press('escape');
time.sleep(0.5);
output = subprocess.check_output(['wmctrl', '-lx']);
output = output.decode('utf-8').splitlines();
window_titles = [line.split(None, 4)[2] for line in output];
closest_matches = difflib.get_close_matches('APP_NAME', window_titles, n=1, cutoff=0.1);
if closest_matches:
    closest_match = closest_matches[0];
    for line in output:
        if closest_match in line:
            window_id = line.split()[0]
            break;
subprocess.run(['wmctrl', '-ia', window_id])
subprocess.run(['wmctrl', '-ir', window_id, '-b', 'add,maximized_vert,maximized_horz'])
"""

        global state_ns, component_ns, attributes_ns, value_ns
        if vm_version == "old":
            state_ns = "uri:deskat:state.at-spi.gnome.org"
            component_ns = "uri:deskat:component.at-spi.gnome.org"
        else:
            attributes_ns = "https://accessibility.windows.example.org/ns/attributes"
            state_ns = "https://accessibility.ubuntu.example.org/ns/state"
            component_ns = "https://accessibility.ubuntu.example.org/ns/component"
            value_ns = "https://accessibility.ubuntu.example.org/ns/value"

    def get_active_apps(self, obs: Dict) -> List[str]:
        tree = ET.ElementTree(ET.fromstring(obs["accessibility_tree"]))
        apps = []
        excluded = {"gjs", "gnome-shell"}
        for node in tree.iter():
            if (
                node.tag.endswith("application")
                and list(node)
                and node.attrib.get("name", "") not in excluded
            ):
                apps.append(node.attrib.get("name", "").replace("\\", ""))
        return apps

    def check_new_apps(self, old_apps, new_apps):
        return new_apps - old_apps

    def get_top_app(self, obs):
        return self.top_app

    def find_active_applications(self, tree):
        to_keep = ["gnome-shell"]
        apps_with_active_tag = []
        for application in list(tree.getroot()):
            app_name = application.attrib.get("name")
            for frame in application:
                active = frame.attrib.get("{{{:}}}active".format(state_ns), "false")
                if active == "true":
                    apps_with_active_tag.append(app_name)
        if apps_with_active_tag:
            to_keep.append(apps_with_active_tag[-1])
        return to_keep

    def filter_active_app(self, tree):
        for application in list(tree.getroot()):
            app_name = application.attrib.get("name")
            for frame in application:
                active = frame.attrib.get("{{{:}}}active".format(state_ns), "false")
                if active == "true":
                    return app_name
        return None

    @staticmethod
    def _coordinates(node) -> Tuple[int, int]:
        raw = node.get("{{{:}}}screencoord".format(component_ns), "(-1, -1)")
        try:
            value = eval(raw, {"__builtins__": {}}, {})
            return int(value[0]), int(value[1])
        except Exception:
            return -1, -1

    @staticmethod
    def _size(node) -> Tuple[int, int]:
        raw = node.get("{{{:}}}size".format(component_ns), "(0, 0)")
        try:
            value = eval(raw, {"__builtins__": {}}, {})
            return int(value[0]), int(value[1])
        except Exception:
            return 0, 0

    def filter_nodes(self, tree, show_all=False):
        preserved_nodes = []
        excluded = {"panel", "window", "filler", "frame", "separator", "scroll-bar"}

        for node in tree.iter():
            if node.tag in excluded:
                continue
            state = "visible" if show_all else "showing"
            if node.attrib.get("{{{:}}}{}".format(state_ns, state)) != "true":
                continue
            x, y = self._coordinates(node)
            if x >= 0 and y >= 0:
                preserved_nodes.append(node)
        return preserved_nodes

    def linearize_tree(self, preserved_nodes):
        result = ["id\ttag\tname\ttext"]
        for idx, node in enumerate(preserved_nodes):
            if node.text:
                text = node.text
                if '"' in text:
                    text = '"{}"'.format(text.replace('"', '""'))
            else:
                text = '""'
            result.append(
                "{}\t{}\t{}\t{}".format(
                    idx,
                    node.tag,
                    node.get("name", ""),
                    text,
                )
            )
        return result

    def extract_elements_from_screenshot(self, screenshot) -> Dict:
        def send_image_to_ocr(image):
            url = os.environ.get("OCR_SERVER_ADDRESS", "")
            if not url:
                raise Exception("OCR SERVER ADDRESS NOT SET")
            encoded = base64.b64encode(image).decode("utf-8")
            print("Getting OCR response")
            started = time.time()
            response = requests.post(url, json={"img_bytes": encoded})
            print("Got OCR response in", time.time() - started)
            if response.status_code == 200:
                return response.json()
            return {
                "error": "Request failed with status code {}".format(
                    response.status_code
                ),
                "results": [],
            }

        return send_image_to_ocr(screenshot)["results"]

    @staticmethod
    def _ocr_box(element):
        if isinstance(element, dict):
            for key in ("bbox", "box", "bounding_box", "coordinates"):
                if key in element:
                    value = element[key]
                    break
            else:
                value = None
        else:
            value = None

        if value is None:
            return None
        try:
            arr = np.asarray(value, dtype=float)
            if arr.shape == (4, 2):
                xs, ys = arr[:, 0], arr[:, 1]
                return [xs.min(), ys.min(), xs.max(), ys.max()]
            arr = arr.reshape(-1)
            if len(arr) == 4:
                x1, y1, a, b = arr
                if a > x1 and b > y1:
                    return [x1, y1, a, b]
                return [x1, y1, x1 + a, y1 + b]
        except Exception:
            return None
        return None

    @staticmethod
    def _ocr_text(element):
        if isinstance(element, dict):
            for key in ("text", "label", "content", "word"):
                if key in element:
                    return str(element[key])
        if isinstance(element, (list, tuple)) and element:
            return str(element[-1])
        return ""

    def add_ocr_elements(
        self, screenshot, linearized_accessibility_tree, preserved_nodes
    ):
        boxes = []
        for node in preserved_nodes:
            x, y = self._coordinates(node)
            width, height = self._size(node)
            boxes.append([x, y, x + width, y + height])

        try:
            ocr_results = self.extract_elements_from_screenshot(screenshot)
        except Exception as exc:
            logger.warning("Unable to obtain OCR results: %s", exc)
            return linearized_accessibility_tree, preserved_nodes

        self.ocr_elements = []
        tree_boxes = np.asarray(boxes, dtype=float) if boxes else np.empty((0, 4))
        for result in ocr_results or []:
            bbox = self._ocr_box(result)
            text = self._ocr_text(result)
            if bbox is None or not text:
                continue

            should_add = True
            if len(tree_boxes):
                try:
                    overlaps = box_iou(
                        np.asarray([bbox], dtype=float),
                        tree_boxes,
                    )
                    should_add = float(np.max(overlaps)) < 0.1
                except Exception:
                    should_add = True

            if should_add:
                idx = len(preserved_nodes)
                x1, y1, x2, y2 = bbox
                width, height = x2 - x1, y2 - y1
                ocr_node = {
                    "name": text,
                    "text": text,
                    "bbox": (x1, y1, width, height),
                    "tag": "ocr",
                }
                preserved_nodes.append(ocr_node)
                self.ocr_elements.append(ocr_node)
                escaped = text if '"' not in text else '"{}"'.format(text.replace('"', '""'))
                linearized_accessibility_tree.append(
                    "{}\tocr\t{}\t{}".format(idx, text, escaped)
                )

        return linearized_accessibility_tree, preserved_nodes

    def get_accessibility_tree(self, obs, show_all=False):
        tree = ET.ElementTree(ET.fromstring(obs["accessibility_tree"]))

        if self.top_app_only:
            active_apps = self.find_active_applications(tree)
            self.top_active_app = self.filter_active_app(tree)
            for application in list(tree.getroot()):
                if application.attrib.get("name") not in active_apps:
                    tree.getroot().remove(application)

        self.preserved_nodes = self.filter_nodes(tree, show_all)
        linearized = self.linearize_tree(self.preserved_nodes)

        if self.ocr and obs.get("screenshot") is not None:
            linearized, self.preserved_nodes = self.add_ocr_elements(
                obs["screenshot"], linearized, self.preserved_nodes
            )

        return "\n".join(linearized)

    def linearize_accessibility_tree(self, obs, show_all=False):
        return self.get_accessibility_tree(obs, show_all)

    def _element_center(self, element_id):
        self.index_out_of_range_flag = False
        try:
            element_id = int(element_id)
            node = self.preserved_nodes[element_id]
        except (ValueError, TypeError, IndexError):
            self.index_out_of_range_flag = True
            return None

        if isinstance(node, dict):
            x, y, width, height = node["bbox"]
        else:
            x, y = self._coordinates(node)
            width, height = self._size(node)

        if x < 0 or y < 0:
            self.index_out_of_range_flag = True
            return None
        return x + width // 2, y + height // 2

    def _element_code(self, element_id):
        center = self._element_center(element_id)
        if center is None:
            return None
        return center[0], center[1]

    @agent_action
    def click(self, element_id, num_clicks=1, button="left", hold_keys=[]):
        point = self._element_code(element_id)
        if point is None:
            return ""
        keys = list(hold_keys or [])
        prefix = "".join("pyautogui.keyDown({});".format(repr(key)) for key in keys)
        suffix = "".join(
            "pyautogui.keyUp({});".format(repr(key)) for key in reversed(keys)
        )
        return "{}pyautogui.click({}, {}, clicks={}, button={});{}".format(
            prefix, point[0], point[1], int(num_clicks), repr(button), suffix
        )

    @agent_action
    def type(self, text, interval=0.05):
        return "pyperclip.copy({});pyautogui.hotkey('ctrl', 'v')".format(repr(str(text)))

    @agent_action
    def keypress(self, keys):
        if isinstance(keys, str):
            keys = [keys]
        keys = list(keys)
        if len(keys) == 1:
            return "pyautogui.press({})".format(repr(keys[0]))
        return "pyautogui.hotkey({})".format(", ".join(repr(key) for key in keys))

    @agent_action
    def hotkey(self, keys):
        return self.keypress(keys)

    @agent_action
    def scroll(self, element_id, direction, num_clicks=1):
        point = self._element_code(element_id)
        if point is None:
            return ""
        amount = abs(int(num_clicks))
        direction = str(direction).lower()
        if direction in {"down", "right"}:
            amount = -amount
        return "pyautogui.moveTo({}, {});pyautogui.scroll({})".format(
            point[0], point[1], amount
        )

    @agent_action
    def hover(self, element_id):
        point = self._element_code(element_id)
        if point is None:
            return ""
        return "pyautogui.moveTo({}, {})".format(point[0], point[1])

    @agent_action
    def drag_and_drop(self, source_id, target_id, hold_keys=[]):
        source = self._element_code(source_id)
        target = self._element_code(target_id)
        if source is None or target is None:
            return ""
        keys = list(hold_keys or [])
        prefix = "".join("pyautogui.keyDown({});".format(repr(key)) for key in keys)
        suffix = "".join(
            "pyautogui.keyUp({});".format(repr(key)) for key in reversed(keys)
        )
        return (
            "{}pyautogui.moveTo({}, {});pyautogui.dragTo({}, {}, duration=1);{}"
        ).format(prefix, source[0], source[1], target[0], target[1], suffix)

    @agent_action
    def wait(self, seconds=1):
        return "time.sleep({})".format(float(seconds))

    @agent_action
    def focus(self, element_id):
        return self.click(element_id)

    @agent_action
    def set_value(self, element_id, value):
        click_code = self.click(element_id)
        if not click_code:
            return ""
        return "{};pyautogui.hotkey('ctrl', 'a');pyperclip.copy({});pyautogui.hotkey('ctrl', 'v')".format(
            click_code, repr(str(value))
        )

    @agent_action
    def select(self, element_id):
        return self.click(element_id)

    @agent_action
    def copy(self):
        return "pyautogui.hotkey('ctrl', 'c')"

    @agent_action
    def paste(self):
        return "pyautogui.hotkey('ctrl', 'v')"