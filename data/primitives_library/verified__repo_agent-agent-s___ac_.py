import base64
import os
import platform
from typing import Any, Dict, List, Tuple

import numpy as np
import requests

from gui_agents.s1.aci.ACI import ACI, agent_action
from gui_agents.s1.utils.common_utils import box_iou

if platform.system() == "Darwin":
    from AppKit import NSWorkspace
    from ApplicationServices import (
        AXUIElementCopyAttributeNames,
        AXUIElementCopyAttributeValue,
        AXUIElementCreateSystemWide,
    )


def _normalize_key(key: str) -> str:
    return "command" if key == "cmd" else key


def list_apps_in_directories(directories):
    apps = []
    for directory in directories:
        if os.path.exists(directory):
            apps.extend(
                app for app in os.listdir(directory) if app.endswith(".app")
            )
    return apps


class UIElement:
    def __init__(self, ref):
        self.ref = ref

    def attribute(self, name):
        try:
            result = AXUIElementCopyAttributeValue(self.ref, name, None)
        except Exception:
            return None

        if isinstance(result, tuple):
            if len(result) > 1:
                return result[1]
            return None
        return result

    def attributes(self):
        try:
            result = AXUIElementCopyAttributeNames(self.ref, None)
        except Exception:
            return []

        if isinstance(result, tuple):
            return result[1] if len(result) > 1 and result[1] is not None else []
        return result or []

    def children(self):
        return self.attribute("AXChildren")

    @staticmethod
    def get_current_applications(obs: Dict) -> List[str]:
        root = obs.get("accessibility_tree")
        if root is None:
            return []

        root_element = UIElement(root)
        applications = []
        children = root_element.children() or []

        for child in children:
            element = UIElement(child)
            role = element.attribute("AXRole")
            if str(role) == "AXApplication":
                title = element.attribute("AXTitle")
                if title is not None:
                    applications.append(str(title))

        if not applications:
            focused = root_element.attribute("AXFocusedApplication")
            if focused is not None:
                title = UIElement(focused).attribute("AXTitle")
                if title is not None:
                    applications.append(str(title))

        return applications

    @staticmethod
    def get_top_app(obs: Dict) -> str:
        root = obs.get("accessibility_tree")
        if root is None:
            return ""

        focused = UIElement(root).attribute("AXFocusedApplication")
        if focused is None:
            return ""

        title = UIElement(focused).attribute("AXTitle")
        return "" if title is None else str(title)


class MacOSACI(ACI):
    def __init__(self, top_app_only: bool = True, ocr: bool = False):
        super().__init__(top_app_only=top_app_only, ocr=ocr)
        directories_to_search = ["/System/Applications", "/Applications"]
        self.all_apps = list_apps_in_directories(directories_to_search)

    def get_active_apps(self, obs: Dict) -> List[str]:
        return UIElement.get_current_applications(obs)

    def get_top_app(self, obs: Dict) -> str:
        return UIElement.get_top_app(obs)

    def preserve_nodes(self, tree, exclude_roles=None):
        if exclude_roles is None:
            exclude_roles = set()

        preserved_nodes = []

        def traverse_and_preserve(element):
            role = element.attribute("AXRole")

            if role not in exclude_roles:
                position = element.attribute("AXPosition")
                size = element.attribute("AXSize")

                if position and size:
                    pos_parts = position.__repr__().split().copy()
                    x_part = next(
                        part for part in pos_parts if part.startswith("x:")
                    )
                    y_part = next(
                        part for part in pos_parts if part.startswith("y:")
                    )

                    x = float(x_part.split(":")[1])
                    y = float(y_part.split(":")[1])

                    size_parts = size.__repr__().split().copy()
                    width_part = next(
                        part for part in size_parts if part.startswith("w:")
                    )
                    height_part = next(
                        part for part in size_parts if part.startswith("h:")
                    )

                    w = float(width_part.split(":")[1])
                    h = float(height_part.split(":")[1])

                    if x >= 0 and y >= 0 and w > 0 and h > 0:
                        preserved_nodes.append(
                            {
                                "position": (x, y),
                                "size": (w, h),
                                "title": str(element.attribute("AXTitle")),
                                "text": str(element.attribute("AXDescription"))
                                or str(element.attribute("AXValue")),
                                "role": str(element.attribute("AXRole")),
                            }
                        )

            children = element.children()
            if children:
                for child_ref in children:
                    traverse_and_preserve(UIElement(child_ref))

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
                        for _, _, box in ocr_bboxes
                    ],
                    dtype=np.float32,
                )

                if len(tree_bboxes) > 0:
                    max_ious = box_iou(tree_bboxes, ocr_boxes_array).max(axis=0)
                else:
                    max_ious = np.zeros(len(ocr_boxes_array))

                for (_, content, box), max_iou in zip(ocr_bboxes, max_ious):
                    if max_iou < 0.1:
                        x1 = int(box.get("left", 0))
                        y1 = int(box.get("top", 0))
                        x2 = int(box.get("right", 0))
                        y2 = int(box.get("bottom", 0))

                        linearized_accessibility_tree.append(
                            f"{preserved_nodes_index}\tAXButton\t\t{content}\t\t"
                        )

                        preserved_nodes.append(
                            {
                                "position": (x1, y1),
                                "size": (x2 - x1, y2 - y1),
                                "title": "",
                                "text": content,
                                "role": "AXButton",
                            }
                        )
                        preserved_nodes_index += 1

        return linearized_accessibility_tree, preserved_nodes

    def linearize_and_annotate_tree(
        self, obs: Dict, show_all_elements: bool = False
    ) -> str:
        accessibility_tree = obs["accessibility_tree"]
        screenshot = obs["screenshot"]

        self.top_app = (
            NSWorkspace.sharedWorkspace().frontmostApplication().localizedName()
        )

        tree = UIElement(accessibility_tree.attribute("AXFocusedApplication"))
        exclude_roles = ["AXGroup", "AXLayoutArea", "AXLayoutItem", "AXUnknown"]
        preserved_nodes = self.preserve_nodes(tree, exclude_roles).copy()

        tree_elements = ["id\trole\ttitle\ttext"]
        for idx, node in enumerate(preserved_nodes):
            tree_elements.append(
                f"{idx}\t{node['role']}\t{node['title']}\t{node['text']}"
            )

        if self.ocr:
            tree_elements, preserved_nodes = self.add_ocr_elements(
                screenshot, tree_elements, preserved_nodes, "AXButton"
            )

        self.nodes = preserved_nodes
        return "\n".join(tree_elements)

    def find_element(self, element_id: int) -> Dict:
        try:
            return self.nodes[element_id]
        except IndexError:
            print("The index of the selected element was out of range.")
            self.index_out_of_range = True
            return None