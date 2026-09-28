import base64
import io
import json
import os
import pickle
import re
import tempfile
import time
import xml.etree.ElementTree as ET
from io import BytesIO
from typing import Dict, List, Tuple, Union
from xml.etree.ElementTree import Element

import numpy as np
import tiktoken
from PIL import Image, ImageDraw, ImageFont
from pydantic import BaseModel, ValidationError


def find_leaf_nodes(xlm_file_str):
    if not xlm_file_str:
        return []

    root = ET.fromstring(xlm_file_str)
    leaf_nodes = []

    def collect_leaf_nodes(node, result):
        if not list(node):
            result.append(node)
        for child in node:
            collect_leaf_nodes(child, result)

    collect_leaf_nodes(root, leaf_nodes)
    return leaf_nodes


state_ns = "uri:deskat:state.at-spi.gnome.org"
component_ns = "uri:deskat:component.at-spi.gnome.org"


class Node(BaseModel):
    name: str
    info: str


class Dag(BaseModel):
    nodes: List[Node]
    edges: List[List[Node]]


NUM_IMAGE_TOKEN = 1105


def call_llm_safe(agent) -> Union[str, Dag]:
    max_retries = 3
    attempt = 0
    response = ""

    while attempt < max_retries:
        try:
            response = agent.get_response()
            break
        except Exception as e:
            attempt += 1
            print(f"Attempt {attempt} failed: {e}")
            if attempt == max_retries:
                print("Max retries reached. Handling failure.")
        time.sleep(1.0)

    return response


def get_input_token_length(input_string):
    encoding = tiktoken.encoding_for_model("gpt-4")
    return len(encoding.encode(input_string))


def calculate_tokens(messages, num_image_token=NUM_IMAGE_TOKEN) -> Tuple[int, int]:
    num_input_images = 0
    output_message = messages[-1]
    input_message = messages[:-1]

    input_string = ""
    for message in input_message:
        input_string += message["content"][0]["text"] + "\n"
        if len(message["content"]) > 1:
            num_input_images += 1

    input_text_tokens = get_input_token_length(input_string)
    input_image_tokens = num_image_token * num_input_images
    output_tokens = get_input_token_length(output_message["content"][0]["text"])

    return input_text_tokens + input_image_tokens, output_tokens


def judge_node(node: Element, platform="ubuntu", check_image=False) -> bool:
    keeps = (
        node.tag.startswith("document")
        or node.tag.endswith("item")
        or node.tag.endswith("button")
        or node.tag.endswith("heading")
        or node.tag.endswith("label")
        or node.tag.endswith("scrollbar")
        or node.tag.endswith("searchbox")
        or node.tag.endswith("textbox")
        or node.tag.endswith("link")
        or node.tag.endswith("tabelement")
        or node.tag.endswith("textfield")
        or node.tag.endswith("textarea")
        or node.tag.endswith("menu")
        or node.tag.endswith("menu-item")
        or node.tag
        in {
            "alert",
            "canvas",
            "check-box",
            "combo-box",
            "entry",
            "icon",
            "image",
            "paragraph",
            "scroll-bar",
            "section",
            "slider",
            "static",
            "table-cell",
            "terminal",
            "text",
            "netuiribbontab",
            "start",
            "trayclockwclass",
            "traydummysearchcontrol",
            "uiimage",
            "uiproperty",
            "uiribboncommandbar",
        }
    )

    keeps = (
        keeps
        and (
            (
                platform == "ubuntu"
                and node.get("{{{:}}}showing".format(state_ns), "false") == "true"
                and node.get("{{{:}}}visible".format(state_ns), "false") == "true"
            )
            or (
                platform == "windows"
                and node.get("{{{:}}}visible".format(state_ns), "false") == "true"
            )
        )
        and (
            node.get("name", "") != ""
            or (node.text is not None and len(node.text) > 0)
            or (check_image and node.get("image", "false") == "true")
        )
    )

    coordinates = eval(
        node.get("{{{:}}}screencoord".format(component_ns), "(-1, -1)")
    )
    sizes = eval(node.get("{{{:}}}size".format(component_ns), "(-1, -1)"))

    keeps = (
        keeps
        and coordinates[0] >= 0
        and coordinates[1] >= 0
        and sizes[0] > 0
        and sizes[1] > 0
    )
    return keeps


def filter_nodes(root: Element, platform="ubuntu", check_image=False):
    filtered_nodes = []
    all_nodes = []

    for node in root.iter():
        all_nodes.append(node)

    for node in root.iter():
        if judge_node(node, platform, check_image):
            filtered_nodes.append(node)

    return filtered_nodes


def draw_bounding_boxes(nodes, image_file_content, down_sampling_ratio=1.0):
    image_stream = io.BytesIO(image_file_content)
    image = Image.open(image_stream)

    if float(down_sampling_ratio) != 1.0:
        image = image.resize(
            (
                int(image.size[0] * down_sampling_ratio),
                int(image.size[1] * down_sampling_ratio),
            )
        )

    draw = ImageDraw.Draw(image)
    marks = []
    drew_nodes = []
    text_informations: List[str] = ["index\ttag\tname\ttext"]

    try:
        font = ImageFont.truetype("arial.ttf", 15)
    except IOError:
        font = ImageFont.load_default()

    index = 1

    for _node in nodes:
        coords_str = _node.attrib.get(
            "{uri:deskat:component.at-spi.gnome.org}screencoord"
        )
        size_str = _node.attrib.get("{uri:deskat:component.at-spi.gnome.org}size")

        if coords_str and size_str:
            try:
                coords = tuple(map(int, coords_str.strip("()").split(", ")))
                size = tuple(map(int, size_str.strip("()").split(", ")))

                original_coords = tuple(coords)
                original_size = tuple(size)

                if float(down_sampling_ratio) != 1.0:
                    coords = tuple(int(coord * down_sampling_ratio) for coord in coords)
                    size = tuple(int(value * down_sampling_ratio) for value in size)

                if size[0] <= 0 or size[1] <= 0:
                    raise ValueError(f"Size must be positive, got: {size}")

                bottom_right = (coords[0] + size[0], coords[1] + size[1])

                if bottom_right[0] < coords[0] or bottom_right[1] < coords[1]:
                    raise ValueError(
                        f"Invalid coordinates or size, coords: {coords}, size: {size}"
                    )

                cropped_image = image.crop((*coords, *bottom_right))
                if len(set(list(cropped_image.getdata()))) == 1:
                    continue

                draw.rectangle([coords, bottom_right], outline="red", width=1)

                text_position = (coords[0], bottom_right[1])
                text_bbox = draw.textbbox(
                    text_position, str(index), font=font, anchor="lb"
                )
                draw.rectangle(text_bbox, fill="black")
                draw.text(
                    text_position,
                    str(index),
                    fill="white",
                    font=font,
                    anchor="lb",
                )

                marks.append([index, original_coords, original_size])
                text_informations.append(
                    "{}\t{}\t{}\t{}".format(
                        index,
                        _node.tag,
                        _node.get("name", ""),
                        _node.text if _node.text is not None else "",
                    )
                )
                drew_nodes.append(_node)
                index += 1
            except ValueError as e:
                print(f"Error processing node: {e}")

    output_image_stream = BytesIO()
    image.save(output_image_stream, format="PNG")

    return marks, drew_nodes, output_image_stream.getvalue(), "\n".join(text_informations)


def encode_image(image_file_content):
    return base64.b64encode(image_file_content).decode("utf-8")


def parse_dag(response):
    if isinstance(response, Dag):
        return response

    if not isinstance(response, str):
        return None

    json_match = re.search(r"```(?:json)?\s*(.*?)\s*```", response, re.DOTALL)
    dag_string = json_match.group(1) if json_match else response

    try:
        data = json.loads(dag_string)
        return Dag(**data)
    except (json.JSONDecodeError, ValidationError) as e:
        print(f"Failed to parse DAG: {e}")
        return None


def save_to_pickle(data, file_path=None):
    if file_path is None:
        fd, file_path = tempfile.mkstemp(suffix=".pkl")
        os.close(fd)

    with open(file_path, "wb") as file:
        pickle.dump(data, file)

    return file_path


def load_from_pickle(file_path):
    with open(file_path, "rb") as file:
        return pickle.load(file)


def image_to_numpy(image_file_content):
    image = Image.open(BytesIO(image_file_content))
    return np.array(image)