import base64
import logging
import os
import time
import xml.etree.ElementTree as ET
from typing import Dict, List, Tuple

import numpy as np
import requests

from gui_agents.s1.utils.common_utils import box_iou

logger = logging.getLogger("desktopenv.agent")

state_ns = "uri:deskat:state.at-spi.gnome.org"
component_ns = "uri:deskat:component.at-spi.gnome.org"


def agent_action(func):
    func.is_agent_action = True
    return func


class GroundingAgent:
    def __init__(self, vm_version: str, top_app=None, top_app_only=True, ocr=True):
        self.active_apps = set()
        self.top_app = top_app
        self.top_app_only = top_app_only
        self.ocr = ocr
        self.index_out_of_range_flag = False
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
        self.top_active_app = None
        self.notes = []
        self.clipboard = ""
        self.preserved_nodes = []
        self.linearized_accessibility_tree = []
        self.vm_version = vm_version

    def get_current_applications(self, obs):
        tree = ET.ElementTree(ET.fromstring(obs["accessibility_tree"]))
        applications = []
        for item in tree.getroot():
            applications.append(item.get("name", "").replace("\\", ""))
        return applications

    def check_new_apps(self, old_apps, new_apps):
        return new_apps - old_apps

    def find_active_applications(self, tree):
        retained = ["Program Manager"]
        active = []

        for application in list(tree.getroot()):
            application_name = application.get("name")
            for frame in application:
                active_value = frame.attrib.get(
                    "{{{:}}}active".format(state_ns), "false"
                )
                if active_value == "true":
                    active.append(application_name)

        print(active)

        if active:
            retained.append(active[-1])

        return retained

    def filter_active_app(self, tree):
        for application in list(tree.getroot()):
            application_name = application.attrib.get("name")
            for frame in application:
                active_value = frame.attrib.get(
                    "{{{:}}}active".format(state_ns), "false"
                )
                if active_value == "true":
                    return application_name
        return None

    def filter_nodes(self, tree, show_all=False):
        kept = []
        excluded = {
            "panel",
            "window",
            "filler",
            "frame",
            "separator",
            "scroll-bar",
        }

        state_name = f"{{{state_ns}}}enabled" if show_all else f"{{{state_ns}}}visible"

        for node in tree.iter():
            if node.tag in excluded:
                continue

            if node.attrib.get(state_name) != "true":
                continue

            coordinates = eval(
                node.get("{{{:}}}screencoord".format(component_ns), "(-1, -1)")
            )

            if coordinates[0] >= 0 and coordinates[1] >= 0:
                kept.append(node)

        return kept

    def linearize_tree(self, preserved_nodes):
        rows = ["id\ttag\tname\ttext"]

        for index, node in enumerate(preserved_nodes):
            if node.text:
                text = (
                    node.text
                    if '"' not in node.text
                    else '"{:}"'.format(node.text.replace('"', '""'))
                )
            else:
                text = '""'

            rows.append(
                "{:}\t{:}\t{:}\t{:}".format(
                    index,
                    node.tag,
                    node.get("name", ""),
                    text,
                )
            )

        return rows

    def extract_elements_from_screenshot(self, screenshot) -> Dict:
        def request_ocr(image):
            url = "http://127.0.0.1:8083/ocr/"
            if url == "":
                raise Exception("OCR SERVER ADDRESS NOT SET")

            encoded = base64.b64encode(image).decode("utf-8")
            response = requests.post(url, json={"img_bytes": encoded})

            if response.status_code == 200:
                return response.json()

            return {
                "error": "Request failed with status code {}".format(
                    response.status_code
                ),
                "results": [],
            }

        return request_ocr(screenshot)["results"]

    @staticmethod
    def _ocr_item_to_box_and_text(item):
        box = None
        text = ""

        if isinstance(item, dict):
            box = (
                item.get("bbox")
                or item.get("box")
                or item.get("bounding_box")
                or item.get("coordinates")
            )
            text = item.get("text") or item.get("content") or item.get("label") or ""
        elif isinstance(item, (list, tuple)):
            if len(item) >= 2:
                box = item[0]
                text = item[1]
            elif len(item) == 1:
                box = item[0]

        if box is None:
            return None, str(text)

        try:
            values = np.asarray(box, dtype=float)
            if values.shape == (4,):
                x1, y1, x2, y2 = values.tolist()
            else:
                values = values.reshape(-1, 2)
                x1 = float(values[:, 0].min())
                y1 = float(values[:, 1].min())
                x2 = float(values[:, 0].max())
                y2 = float(values[:, 1].max())
            return [x1, y1, x2, y2], str(text)
        except Exception:
            return None, str(text)

    def add_ocr_elements(
        self, screenshot, linearized_accessibility_tree, preserved_nodes
    ):
        tree_bboxes = []

        for node in preserved_nodes:
            coordinates = eval(
                node.get("{{{:}}}screencoord".format(component_ns), "(-1, -1)")
            )
            size = eval(node.get("{{{:}}}size".format(component_ns), "(-1, -1)"))
            tree_bboxes.append(
                [
                    coordinates[0],
                    coordinates[1],
                    coordinates[0] + size[0],
                    coordinates[1] + size[1],
                ]
            )

        try:
            ocr_results = self.extract_elements_from_screenshot(screenshot)
        except Exception as exc:
            print(f"Error: {exc}")
            ocr_results = []

        if not ocr_results:
            return linearized_accessibility_tree

        added_index = len(preserved_nodes)

        for item in ocr_results:
            ocr_box, content = self._ocr_item_to_box_and_text(item)
            if ocr_box is None:
                continue

            overlaps = False
            if tree_bboxes:
                try:
                    ious = box_iou(
                        np.asarray(tree_bboxes, dtype=float),
                        np.asarray([ocr_box], dtype=float),
                    )
                    if hasattr(ious, "detach"):
                        ious = ious.detach().cpu().numpy()
                    ious = np.asarray(ious)
                    overlaps = bool(ious.size and np.max(ious) > 0.1)
                except Exception:
                    for existing in tree_bboxes:
                        left = max(existing[0], ocr_box[0])
                        top = max(existing[1], ocr_box[1])
                        right = min(existing[2], ocr_box[2])
                        bottom = min(existing[3], ocr_box[3])
                        intersection = max(0, right - left) * max(0, bottom - top)
                        existing_area = max(0, existing[2] - existing[0]) * max(
                            0, existing[3] - existing[1]
                        )
                        ocr_area = max(0, ocr_box[2] - ocr_box[0]) * max(
                            0, ocr_box[3] - ocr_box[1]
                        )
                        union = existing_area + ocr_area - intersection
                        if union and intersection / union > 0.1:
                            overlaps = True
                            break

            if overlaps:
                continue

            escaped_content = content.replace('"', '""')
            linearized_accessibility_tree.append(
                "{}\tocr\t{}\t\"{}\"".format(
                    added_index,
                    content,
                    escaped_content,
                )
            )
            tree_bboxes.append(ocr_box)
            added_index += 1

        return linearized_accessibility_tree

    def get_linearized_accessibility_tree(self, obs, show_all=False):
        tree = ET.ElementTree(ET.fromstring(obs["accessibility_tree"]))
        self.active_apps = set(self.get_current_applications(obs))
        self.top_active_app = self.filter_active_app(tree)

        if self.top_app_only:
            applications_to_keep = self.find_active_applications(tree)

            if self.top_app is not None:
                applications_to_keep.append(self.top_app)

            for application in list(tree.getroot()):
                if application.get("name") not in applications_to_keep:
                    tree.getroot().remove(application)

        self.preserved_nodes = self.filter_nodes(tree, show_all=show_all)
        self.linearized_accessibility_tree = self.linearize_tree(self.preserved_nodes)

        if self.ocr and obs.get("screenshot") is not None:
            self.linearized_accessibility_tree = self.add_ocr_elements(
                obs["screenshot"],
                self.linearized_accessibility_tree,
                self.preserved_nodes,
            )

        return "\n".join(self.linearized_accessibility_tree)

    def linearize_and_annotate_tree(self, obs, show_all=False):
        return self.get_linearized_accessibility_tree(obs, show_all=show_all)

    def get_obs(self, obs, show_all=False):
        return self.get_linearized_accessibility_tree(obs, show_all=show_all)

    def _node_center(self, element_id):
        self.index_out_of_range_flag = False

        try:
            element_id = int(element_id)
            node = self.preserved_nodes[element_id]
        except (ValueError, TypeError, IndexError):
            self.index_out_of_range_flag = True
            return None

        coordinates = eval(
            node.get("{{{:}}}screencoord".format(component_ns), "(-1, -1)")
        )
        size = eval(node.get("{{{:}}}size".format(component_ns), "(-1, -1)"))

        return (
            int(coordinates[0] + size[0] / 2),
            int(coordinates[1] + size[1] / 2),
        )

    @staticmethod
    def _quote(value):
        return repr(str(value))

    @agent_action
    def click(self, element_id, button="left", clicks=1):
        point = self._node_center(element_id)
        if point is None:
            return ""

        return "pyautogui.click({}, {}, clicks={}, button={})".format(
            point[0],
            point[1],
            int(clicks),
            self._quote(button),
        )

    @agent_action
    def double_click(self, element_id, button="left"):
        return self.click(element_id, button=button, clicks=2)

    @agent_action
    def right_click(self, element_id):
        return self.click(element_id, button="right", clicks=1)

    @agent_action
    def type(self, text):
        return "pyautogui.write({}, interval=0.01)".format(self._quote(text))

    @agent_action
    def scroll(self, element_id, clicks):
        point = self._node_center(element_id)
        if point is None:
            return ""

        return "pyautogui.moveTo({}, {}); pyautogui.scroll({})".format(
            point[0],
            point[1],
            int(clicks),
        )

    @agent_action
    def drag_and_drop(self, drag_from, drop_to, duration=1):
        source = self._node_center(drag_from)
        if source is None:
            return ""

        destination = self._node_center(drop_to)
        if destination is None:
            return ""

        return (
            "pyautogui.moveTo({}, {}); pyautogui.dragTo({}, {}, duration={}, button='left')"
        ).format(
            source[0],
            source[1],
            destination[0],
            destination[1],
            duration,
        )

    @agent_action
    def hotkey(self, keys):
        if isinstance(keys, str):
            keys = [keys]
        rendered = ", ".join(self._quote(key) for key in keys)
        return "pyautogui.hotkey({})".format(rendered)

    @agent_action
    def hold_and_press(self, keys, presses):
        if isinstance(keys, str):
            keys = [keys]
        if isinstance(presses, str):
            presses = [presses]

        held = ", ".join(self._quote(key) for key in keys)
        pressed = ", ".join(self._quote(key) for key in presses)

        return (
            "pyautogui.keyDown({}); pyautogui.press([{}]); pyautogui.keyUp({})"
        ).format(held, pressed, held)

    @agent_action
    def press(self, key, presses=1, interval=0.0):
        return "pyautogui.press({}, presses={}, interval={})".format(
            self._quote(key),
            int(presses),
            interval,
        )

    @agent_action
    def wait(self, seconds=1):
        return "time.sleep({})".format(seconds)

    @agent_action
    def copy(self):
        return "pyautogui.hotkey('ctrl', 'c')"

    @agent_action
    def paste(self):
        return "pyautogui.hotkey('ctrl', 'v')"

    @agent_action
    def select_all(self):
        return "pyautogui.hotkey('ctrl', 'a')"

    @agent_action
    def note(self, text):
        self.notes.append(text)
        return ""

    @agent_action
    def set_clipboard(self, text):
        self.clipboard = text
        return "import pyperclip; pyperclip.copy({})".format(self._quote(text))