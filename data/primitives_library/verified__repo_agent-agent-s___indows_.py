import base64
import os
import platform
from typing import Any, Dict, List, Tuple

import numpy as np
import psutil
import requests

from gui_agents.s1.aci.ACI import ACI, agent_action
from gui_agents.s1.utils.common_utils import box_iou

if platform.system() == "Windows":
    import pywinauto
    from pywinauto import Desktop
    import win32gui
    import win32process


def _normalize_key(key: str) -> str:
    return "ctrl" if key == "control" else key


def list_apps_in_directories():
    directories_to_search = [
        os.environ.get("PROGRAMFILES", "C:\\Program Files"),
        os.environ.get("PROGRAMFILES(X86)", "C:\\Program Files (x86)"),
    ]
    apps = []
    for directory in directories_to_search:
        if os.path.exists(directory):
            for root, dirs, files in os.walk(directory):
                for file in files:
                    if file.endswith(".exe"):
                        apps.append(file)
    return apps


class UIElement:
    def __init__(self, element):
        self.element = element

    def title(self):
        try:
            return self.element.window_text()
        except Exception:
            try:
                return self.element.element_info.name
            except Exception:
                return ""

    def text(self):
        try:
            return self.element.element_info.name
        except Exception:
            try:
                return self.element.window_text()
            except Exception:
                return ""

    def role(self):
        try:
            return self.element.element_info.control_type
        except Exception:
            try:
                return self.element.friendly_class_name()
            except Exception:
                return "Unknown"

    def position(self):
        try:
            rectangle = self.element.rectangle()
            return rectangle.left, rectangle.top
        except Exception:
            return None

    def size(self):
        try:
            rectangle = self.element.rectangle()
            return rectangle.width(), rectangle.height()
        except Exception:
            return None

    def children(self):
        try:
            return [UIElement(child) for child in self.element.children()]
        except Exception:
            return []

    @staticmethod
    def get_current_applications(obs):
        applications = []

        try:
            desktop = Desktop(backend="uia")
            for window in desktop.windows():
                try:
                    title = window.window_text()
                    if title:
                        applications.append(title)
                except Exception:
                    continue
            if applications:
                return applications
        except Exception:
            pass

        for key in ("active_apps", "applications", "apps"):
            value = obs.get(key) if isinstance(obs, dict) else None
            if value is not None:
                if isinstance(value, dict):
                    return list(value.keys())
                if isinstance(value, (list, tuple, set)):
                    return list(value)
                return [value]

        return applications

    @staticmethod
    def get_top_app(obs):
        try:
            handle = win32gui.GetForegroundWindow()
            if handle:
                return win32gui.GetWindowText(handle)
        except Exception:
            pass

        if isinstance(obs, dict):
            for key in ("window_title", "top_app", "active_app"):
                value = obs.get(key)
                if value is not None:
                    return value

            apps = UIElement.get_current_applications(obs)
            if apps:
                return apps[0]

        return ""


class WindowsACI(ACI):
    def __init__(self, top_app_only: bool = True, ocr: bool = False):
        super().__init__(top_app_only=top_app_only, ocr=ocr)
        self.nodes = []
        self.all_apps = list_apps_in_directories()

    def get_active_apps(self, obs: Dict) -> List[str]:
        return UIElement.get_current_applications(obs)

    def get_top_app(self, obs: Dict) -> str:
        return UIElement.get_top_app(obs)

    def preserve_nodes(self, tree, exclude_roles=None):
        if exclude_roles is None:
            exclude_roles = set()

        preserved_nodes = []

        def traverse_and_preserve(element):
            role = element.role()

            if role not in exclude_roles:
                position = element.position()
                size = element.size()
                if position and size:
                    x, y = position
                    w, h = size

                    if x >= 0 and y >= 0 and w > 0 and h > 0:
                        preserved_nodes.append(
                            {
                                "position": (x, y),
                                "size": (w, h),
                                "title": element.title(),
                                "text": element.text(),
                                "role": role,
                            }
                        )

            children = element.children()
            if children:
                for child_element in children:
                    traverse_and_preserve(child_element)

        traverse_and_preserve(tree)
        return preserved_nodes

    def extract_elements_from_screenshot(self, screenshot: bytes) -> Dict[str, Any]:
        url = os.environ.get("OCR_SERVER_ADDRESS")
        if not url:
            raise EnvironmentError("OCR SERVER ADDRESS NOT SET")

        encoded_screenshot = base64.b64encode(screenshot).decode("utf-8")
        response = requests.post(url, json={"img_bytes": encoded_screenshot})

        if response.status_code != 200:
            return {
                "error": f"Request failed with status code {response.status_code}",
                "results": [],
            }
        return response.json()

    def add_ocr_elements(
        self,
        screenshot,
        linearized_accessibility_tree: List[str],
        preserved_nodes: List[Dict],
    ) -> Tuple[List[str], List[Dict]]:
        if preserved_nodes:
            tree_bboxes = np.array(
                [
                    [
                        node["position"][0],
                        node["position"][1],
                        node["position"][0] + node["size"][0],
                        node["position"][1] + node["size"][1],
                    ]
                    for node in preserved_nodes
                ],
                dtype=np.float32,
            )
        else:
            tree_bboxes = np.empty((0, 4), dtype=np.float32)

        try:
            ocr_bboxes = self.extract_elements_from_screenshot(screenshot)
        except Exception as e:
            print(f"Error: {e}")
            ocr_bboxes = []
        else:
            if ocr_bboxes:
                preserved_nodes_index = len(preserved_nodes)

                ocr_boxes_array = np.array(
                    [
                        [
                            int(box.get("left", 0)),
                            int(box.get("top", 0)),
                            int(box.get("right", 0)),
                            int(box.get("bottom", 0)),
                        ]
                        for _, _, box in ocr_bboxes["results"]
                    ],
                    dtype=np.float32,
                )

                if len(tree_bboxes) > 0:
                    max_ious = box_iou(tree_bboxes, ocr_boxes_array).max(axis=0)
                else:
                    max_ious = np.zeros(len(ocr_boxes_array))

                for idx, ((_, content, box), max_iou) in enumerate(
                    zip(ocr_bboxes["results"], max_ious)
                ):
                    if max_iou < 0.1:
                        x1 = int(box.get("left", 0))
                        y1 = int(box.get("top", 0))
                        x2 = int(box.get("right", 0))
                        y2 = int(box.get("bottom", 0))

                        linearized_accessibility_tree.append(
                            f"{preserved_nodes_index}\tButton\t\t{content}\t\t"
                        )

                        preserved_nodes.append(
                            {
                                "position": (x1, y1),
                                "size": (x2 - x1, y2 - y1),
                                "title": "",
                                "text": content,
                                "role": "Button",
                            }
                        )
                        preserved_nodes_index += 1

        return linearized_accessibility_tree, preserved_nodes

    def linearize_and_annotate_tree(
        self, obs: Dict, show_all_elements: bool = False
    ) -> str:
        desktop = Desktop(backend="uia")
        try:
            tree = desktop.window(
                handle=win32gui.GetForegroundWindow()
            ).wrapper_object()
        except Exception as e:
            print(f"Error accessing foreground window: {e}")
            self.nodes = []
            return ""

        exclude_roles = ["Pane", "Group", "Unknown"]
        preserved_nodes = self.preserve_nodes(UIElement(tree), exclude_roles).copy()

        if not preserved_nodes and show_all_elements:
            preserved_nodes = self.preserve_nodes(
                UIElement(tree), exclude_roles=[]
            ).copy()

        tree_elements = ["id\trole\ttitle\ttext"]
        for idx, node in enumerate(preserved_nodes):
            tree_elements.append(
                f"{idx}\t{node['role']}\t{node['title']}\t{node['text']}"
            )

        if self.ocr:
            screenshot = obs.get("screenshot", None)
            if screenshot is not None:
                tree_elements, preserved_nodes = self.add_ocr_elements(
                    screenshot, tree_elements, preserved_nodes
                )

        self.nodes = preserved_nodes
        return "\n".join(tree_elements)

    def find_element(self, element_id: int) -> Dict:
        if not self.nodes:
            print("No elements found in the accessibility tree.")
            raise IndexError("No elements to select.")
        try:
            return self.nodes[element_id]
        except IndexError:
            print("The index of the selected element was out of range.")
            self.index_out_of_range_flag = True
            return self.nodes[0]

    @staticmethod
    def _element_center(element: Dict) -> Tuple[int, int]:
        x, y = element["position"]
        width, height = element["size"]
        return x + width // 2, y + height // 2

    @agent_action
    def open(self, app_or_file_name: str):
        command = (
            "import pyautogui; import time; "
            "pyautogui.hotkey('win', 'r', interval=0.5); "
            f"pyautogui.typewrite({repr(app_or_file_name)}); "
            "pyautogui.press('enter'); time.sleep(1.0)"
        )
        return command

    @agent_action
    def switch_applications(self, app_or_file_name):
        command = (
            "import pyautogui; import time; "
            "pyautogui.hotkey('win', 'd', interval=0.5); "
            f"pyautogui.typewrite({repr(app_or_file_name)}); "
            "pyautogui.press('enter'); time.sleep(1.0)"
        )
        return command

    @agent_action
    def click(self, element_id: int, num_clicks: int = 1, button: str = "left"):
        element = self.find_element(element_id)
        x, y = self._element_center(element)
        return (
            "import pyautogui; "
            f"pyautogui.click({x}, {y}, clicks={num_clicks}, "
            f"interval=0.1, button={repr(button)})"
        )

    @agent_action
    def type(self, text: str):
        return (
            "import pyautogui; import pyperclip; "
            f"pyperclip.copy({repr(text)}); "
            "pyautogui.hotkey('ctrl', 'v')"
        )

    @agent_action
    def press(self, key: str):
        return f"import pyautogui; pyautogui.press({_normalize_key(key)!r})"

    @agent_action
    def hotkey(self, keys: List[str]):
        normalized_keys = [_normalize_key(key) for key in keys]
        arguments = ", ".join(repr(key) for key in normalized_keys)
        return f"import pyautogui; pyautogui.hotkey({arguments})"

    @agent_action
    def scroll(self, element_id: int, clicks: int):
        element = self.find_element(element_id)
        x, y = self._element_center(element)
        return (
            "import pyautogui; "
            f"pyautogui.moveTo({x}, {y}); pyautogui.scroll({clicks})"
        )

    @agent_action
    def drag_and_drop(self, drag_from: int, drop_to: int):
        source = self.find_element(drag_from)
        target = self.find_element(drop_to)
        source_x, source_y = self._element_center(source)
        target_x, target_y = self._element_center(target)
        return (
            "import pyautogui; "
            f"pyautogui.moveTo({source_x}, {source_y}); "
            "pyautogui.mouseDown(); "
            f"pyautogui.moveTo({target_x}, {target_y}, duration=0.5); "
            "pyautogui.mouseUp()"
        )

    @agent_action
    def wait(self, seconds: float = 1.0):
        return f"import time; time.sleep({seconds})"