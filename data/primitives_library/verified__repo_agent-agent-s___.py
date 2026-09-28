import logging
import xml.etree.ElementTree as ET
from typing import Any, Dict, List

logger = logging.getLogger("desktopenv.agent")


def agent_action(func):
    func.is_agent_action = True
    return func


class ACI:
    def __init__(self, top_app_only: bool = True, ocr: bool = False):
        self.top_app_only = top_app_only
        self.ocr = ocr
        self.index_out_of_range_flag = False
        self.notes: List[str] = []
        self.clipboard = ""
        self.nodes: List[Any] = []
        self._active_apps: List[str] = []
        self._top_app = None

    @staticmethod
    def _tag_name(tag: Any) -> str:
        if not isinstance(tag, str):
            return ""
        if "}" in tag:
            tag = tag.rsplit("}", 1)[-1]
        return tag.replace("_", " ").strip().lower()

    @staticmethod
    def _truthy(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
            "active",
            "focused",
            "showing",
            "visible",
        }

    def _tree_from_observation(self, obs: Dict) -> Any:
        if not obs:
            return None

        tree = obs.get("accessibility_tree", obs.get("tree"))
        if tree is None:
            return None

        if isinstance(tree, bytes):
            try:
                tree = tree.decode("utf-8")
            except Exception:
                return None

        if isinstance(tree, str):
            try:
                return ET.fromstring(tree)
            except (ET.ParseError, ValueError, TypeError):
                return None

        return tree

    def _node_role(self, node: Any) -> str:
        if isinstance(node, ET.Element):
            return self._tag_name(node.tag)

        if isinstance(node, dict):
            role = (
                node.get("role")
                or node.get("tag")
                or node.get("type")
                or node.get("control_type")
                or ""
            )
            return self._tag_name(role)

        role = getattr(node, "role", None) or getattr(node, "tag", None) or ""
        return self._tag_name(role)

    def _node_name(self, node: Any) -> str:
        if isinstance(node, ET.Element):
            attributes = node.attrib
            name = (
                attributes.get("name")
                or attributes.get("label")
                or attributes.get("title")
                or attributes.get("value")
                or attributes.get("text")
                or (node.text or "")
            )
            return str(name).strip()

        if isinstance(node, dict):
            name = (
                node.get("name")
                or node.get("label")
                or node.get("title")
                or node.get("value")
                or node.get("text")
                or node.get("description")
                or ""
            )
            return str(name).strip()

        name = (
            getattr(node, "name", None)
            or getattr(node, "label", None)
            or getattr(node, "title", None)
            or getattr(node, "text", None)
            or ""
        )
        return str(name).strip()

    def _node_dict(self, node: Any) -> Dict:
        if isinstance(node, ET.Element):
            result = dict(node.attrib)
            result["role"] = self._node_role(node)
            result["name"] = self._node_name(node)
            result["_element"] = node
            return result

        if isinstance(node, dict):
            result = dict(node)
            result["role"] = self._node_role(node)
            result["name"] = self._node_name(node)
            return result

        result = {}
        for key in (
            "name",
            "role",
            "tag",
            "description",
            "value",
            "states",
            "state",
            "bounds",
            "position",
            "x",
            "y",
            "width",
            "height",
        ):
            if hasattr(node, key):
                result[key] = getattr(node, key)
        result["role"] = self._node_role(node)
        result["name"] = self._node_name(node)
        result["_element"] = node
        return result

    def _children(self, node: Any) -> List[Any]:
        if isinstance(node, ET.Element):
            return list(node)

        if isinstance(node, dict):
            children = (
                node.get("children")
                or node.get("child")
                or node.get("nodes")
                or node.get("elements")
                or []
            )
            return children if isinstance(children, list) else list(children)

        children = getattr(node, "children", None)
        if children is None:
            return []
        try:
            return list(children)
        except TypeError:
            return []

    def _walk(self, tree: Any) -> List[Any]:
        if tree is None:
            return []

        result = []
        stack = [tree]
        while stack:
            node = stack.pop()
            result.append(node)
            children = self._children(node)
            stack.extend(reversed(children))
        return result

    def get_active_apps(self, obs: Dict) -> List[str]:
        if not obs:
            self._active_apps = []
            self._top_app = None
            return []

        supplied_apps = obs.get("active_apps")
        if supplied_apps is None:
            supplied_apps = obs.get("apps")

        if supplied_apps is not None and not isinstance(supplied_apps, (str, bytes)):
            names = []
            for app in supplied_apps:
                if isinstance(app, str):
                    name = app
                elif isinstance(app, dict):
                    name = app.get("name") or app.get("title") or app.get("app_name")
                else:
                    name = getattr(app, "name", None) or getattr(app, "title", None)
                if name:
                    names.append(str(name))
            if names:
                self._active_apps = names
                self._top_app = names[0]
                return names

        tree = self._tree_from_observation(obs)
        applications = []
        marked_active = []

        for node in self._walk(tree):
            role = self._node_role(node)
            if role not in {"application", "app"}:
                continue

            name = self._node_name(node)
            if not name:
                continue

            applications.append(name)
            node_dict = self._node_dict(node)
            state = str(
                node_dict.get("states", node_dict.get("state", ""))
            ).lower()
            if (
                "active" in state
                or "focused" in state
                or self._truthy(node_dict.get("active", False))
                or self._truthy(node_dict.get("focused", False))
            ):
                marked_active.append(name)

        self._active_apps = marked_active or applications
        self._top_app = self._active_apps[0] if self._active_apps else None
        return list(self._active_apps)

    def get_top_app(self):
        if self._top_app:
            return self._top_app

        if self._active_apps:
            return self._active_apps[0]

        for node in self.nodes:
            role = self._node_role(node)
            if role in {"application", "app"}:
                name = self._node_name(node)
                if name:
                    self._top_app = name
                    return name

        return None

    def preserve_nodes(self, tree: Any, exclude_roles: set = None) -> List[Dict]:
        excluded = {
            self._tag_name(role)
            for role in (exclude_roles or set())
        }
        preserved = []

        for node in self._walk(tree):
            role = self._node_role(node)
            if role in excluded:
                continue

            node_dict = self._node_dict(node)
            if not node_dict["role"]:
                continue

            preserved.append(node_dict)

        return preserved

    def linearize_and_annotate_tree(
        self, obs: Dict, show_all_elements: bool = False
    ) -> str:
        self.index_out_of_range_flag = False
        tree = self._tree_from_observation(obs)

        self.get_active_apps(obs)

        excluded_roles = set()
        if not show_all_elements:
            excluded_roles = {
                "desktop",
                "root",
                "panel",
                "filler",
                "separator",
                "scroll bar",
                "scrollbar",
                "status bar",
                "statusbar",
                "tool bar",
                "toolbar",
                "menu bar",
                "menubar",
            }

        all_nodes = self.preserve_nodes(tree, excluded_roles)

        if self.top_app_only and self._top_app:
            filtered_nodes = []
            active_app_seen = False
            for node in all_nodes:
                role = self._node_role(node)
                if role in {"application", "app"}:
                    active_app_seen = self._node_name(node) == self._top_app
                    if active_app_seen:
                        filtered_nodes.append(node)
                    continue
                if active_app_seen:
                    filtered_nodes.append(node)
            if filtered_nodes:
                all_nodes = filtered_nodes

        self.nodes = all_nodes

        lines = []
        for index, node in enumerate(self.nodes):
            node["id"] = index
            role = str(node.get("role", "")).strip()
            name = str(node.get("name", "")).strip()
            lines.append(f"{index}\t{role}\t{name}")

        return "\n".join(lines)

    def find_element(self, element_id: int) -> Dict:
        self.index_out_of_range_flag = False

        try:
            index = int(element_id)
        except (TypeError, ValueError):
            self.index_out_of_range_flag = True
            return {}

        if index < 0 or index >= len(self.nodes):
            self.index_out_of_range_flag = True
            return {}

        node = self.nodes[index]
        if isinstance(node, dict):
            return node
        return self._node_dict(node)