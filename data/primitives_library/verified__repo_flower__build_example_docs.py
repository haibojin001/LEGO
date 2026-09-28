import os
import re
import shutil
import subprocess
from pathlib import Path

CategoryContent = dict[str, str]

ROOT = Path(__file__).resolve().parents[2]
INDEX = ROOT / "examples" / "docs" / "source" / "index.rst"

INITIAL_TEXT = """
Flower Examples Documentation
-----------------------------

Welcome to Flower Examples' documentation. `Flower <https://flower.ai>`_ is
a friendly federated AI framework.

Join the Flower Community
-------------------------

The Flower Community is growing quickly - we're a friendly group of researchers,
engineers, students, professionals, academics, and other enthusiasts.

.. button-link:: https://flower.ai/join-slack
    :color: primary
    :shadow:

    Join us on Slack

Quickstart Examples
-------------------

Flower Quickstart Examples are a collection of demo projects that show how you
can use Flower in combination with other existing frameworks or technologies.

"""

TABLE_HEADERS = (
    "\n.. list-table::\n   :widths: 50 15 15 15\n   "
    ":header-rows: 1\n\n   * - Title\n     - Framework\n     - Dataset\n     - Tags\n\n"
)

CATEGORIES: dict[str, CategoryContent] = {
    "quickstart": {"table": TABLE_HEADERS, "list": ""},
    "advanced": {"table": TABLE_HEADERS, "list": ""},
    "other": {"table": TABLE_HEADERS, "list": ""},
}

URLS: dict[str, str] = {
    "Android": "https://www.android.com/",
    "catboost": "https://catboost.ai/docs/en/",
    "C++": "https://isocpp.org/",
    "Docker": "https://www.docker.com/",
    "JAX": "https://jax.readthedocs.io/en/latest/",
    "Java": "https://www.java.com/",
    "Keras": "https://keras.io/",
    "Kotlin": "https://kotlinlang.org/",
    "MLX": "https://ml-explore.github.io/mlx/build/html/index.html",
    "MONAI": "https://monai.io/",
    "PEFT": "https://huggingface.co/docs/peft/index",
    "Swift": "https://www.swift.org/",
    "TensorFlowLite": "https://www.tensorflow.org/lite",
    "fastai": "https://fast.ai/",
    "lifelines": "https://lifelines.readthedocs.io/en/latest/index.html",
    "lightning": "https://lightning.ai/docs/pytorch/stable/",
    "numpy": "https://numpy.org/",
    "opacus": "https://opacus.ai/",
    "pandas": "https://pandas.pydata.org/",
    "scikit-learn": "https://scikit-learn.org/",
    "tensorboard": "https://www.tensorflow.org/tensorboard",
    "tensorflow": "https://www.tensorflow.org/",
    "torch": "https://pytorch.org/",
    "torchvision": "https://pytorch.org/vision/stable/index.html",
    "transformers": "https://huggingface.co/docs/transformers/index",
    "wandb": "https://wandb.ai/home",
    "whisper": "https://huggingface.co/openai/whisper-tiny",
    "xgboost": "https://xgboost.readthedocs.io/en/stable/",
    "Adult Census Income": "https://www.kaggle.com/datasets/uciml/adult-census-income/data",
    "Alpaca-GPT4": "https://huggingface.co/datasets/vicgalle/alpaca-gpt4",
    "CIFAR-10": "https://huggingface.co/datasets/uoft-cs/cifar10",
    "HIGGS": "https://archive.ics.uci.edu/dataset/280/higgs",
    "IMDB": "https://huggingface.co/datasets/stanfordnlp/imdb",
    "Iris": "https://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html",
    "MNIST": "https://huggingface.co/datasets/ylecun/mnist",
    "MedNIST": "https://medmnist.com/",
    "Oxford Flower-102": "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/",
    "SpeechCommands": "https://huggingface.co/datasets/google/speech_commands",
    "Titanic": "https://www.kaggle.com/competitions/titanic",
    "Waltons": (
        "https://lifelines.readthedocs.io/en/latest/"
        "lifelines.datasets.html#lifelines.datasets.load_waltons"
    ),
}


def _convert_to_link(search_result: str) -> str:
    if "," in search_result:
        converted = ""
        for item in search_result.split(","):
            converted += f"{_convert_to_link(item)}, "
        return converted[:-2]

    value = search_result.strip()
    target = URLS.get(value)
    if target:
        return f"`{value.strip()} <{target.strip()}>`_"
    return value


def _read_metadata(example: str) -> tuple[str, str, str, str]:
    with open(os.path.join(example, "README.md"), encoding="utf-8") as readme:
        contents = readme.read()

    block = re.search(r"^---(.*?)^---", contents, re.DOTALL | re.MULTILINE)
    if block is None:
        raise ValueError("Metadata block not found")
    metadata = block.group(1)

    title_result = re.search(r"^# (.+)$", contents, re.MULTILINE)
    if title_result is None:
        raise ValueError("Title not found in metadata")
    title = title_result.group(1).strip()

    tags_result = re.search(r"^tags:\s*\[(.+?)\]$", metadata, re.MULTILINE)
    if tags_result is None:
        raise ValueError("Tags not found in metadata")
    tags = tags_result.group(1).strip()

    dataset_result = re.search(
        r"^dataset:\s*\[(.*?)\]$", metadata, re.DOTALL | re.MULTILINE
    )
    if dataset_result is None:
        raise ValueError("Dataset not found in metadata")
    dataset = dataset_result.group(1).strip()

    framework_result = re.search(
        r"^framework:\s*\[(.*?|)\]$", metadata, re.DOTALL | re.MULTILINE
    )
    if framework_result is None:
        raise ValueError("Framework not found in metadata")
    framework = framework_result.group(1).strip()

    dataset = _convert_to_link(re.sub(r"\s+", " ", dataset).strip())
    framework = _convert_to_link(re.sub(r"\s+", " ", framework).strip())
    return title, tags, dataset, framework


def _add_table_entry(example: str, tag: str, table_var: str) -> bool:
    title, tags, dataset, framework = _read_metadata(example)
    name = Path(example).stem
    entry = (
        f"   * - `{title} <{name}.html>`_ \n     "
        f"- {framework} \n     - {dataset} \n     - {tags}\n\n"
    )
    if tag in tags:
        CATEGORIES[table_var]["table"] += entry
        CATEGORIES[table_var]["list"] += f"  {name}\n"
        return True
    return False


def _copy_markdown_files(example: str) -> None:
    for filename in os.listdir(example):
        if filename.endswith(".md"):
            source = os.path.join(example, filename)
            destination = os.path.join(
                str(ROOT),
                "examples",
                "docs",
                "source",
                os.path.basename(example) + ".md",
            )
            shutil.copyfile(source, destination)


def _add_gh_button(example: str) -> None:
    button = (
        '[<img src="_static/view-gh.png" alt="View on GitHub" width="200"/>]'
        f"(https://github.com/flwrlabs/flower/blob/main/examples/{example})"
    )
    filename = os.path.join(
        str(ROOT), "examples", "docs", "source", example + ".md"
    )
    with open(filename, "r+", encoding="utf-8") as readme:
        contents = readme.read()
        if button not in contents:
            contents = re.sub(
                r"(^# .+$)",
                rf"\1\n\n{button}",
                contents,
                count=1,
                flags=re.MULTILINE,
            )
            readme.seek(0)
            readme.write(contents)
            readme.truncate()


def _copy_images(example: str) -> None:
    image_dir = os.path.join(example, "_static")
    output_dir = os.path.join(str(ROOT), "examples", "docs", "source", "_static")
    if os.path.isdir(image_dir):
        for filename in os.listdir(image_dir):
            if filename.endswith((".jpg", ".png", ".jpeg")):
                shutil.copyfile(
                    os.path.join(image_dir, filename),
                    os.path.join(output_dir, filename),
                )


def _add_all_entries() -> None:
    examples_dir = os.path.join(ROOT, "examples")
    for example in sorted(os.listdir(examples_dir)):
        example_path = os.path.join(examples_dir, example)
        if os.path.isdir(example_path) and example != "docs":
            _copy_markdown_files(example_path)
            _add_gh_button(example)
            _copy_images(example_path)


def _main() -> None:
    if INDEX.exists():
        INDEX.unlink()

    with INDEX.open("w", encoding="utf-8") as index_file:
        index_file.write(INITIAL_TEXT)

    examples_dir = os.path.join(ROOT, "examples")
    for example in sorted(os.listdir(examples_dir)):
        example_path = os.path.join(examples_dir, example)
        if os.path.isdir(example_path) and example != "docs":
            _copy_markdown_files(example_path)
            _add_gh_button(example)
            _copy_images(example_path)

            if _add_table_entry(example_path, "quickstart", "quickstart"):
                continue
            if _add_table_entry(example_path, "advanced", "advanced"):
                continue
            _add_table_entry(example_path, "other", "other")

    with INDEX.open("a", encoding="utf-8") as index_file:
        index_file.write(CATEGORIES["quickstart"]["table"])
        index_file.write("\n.. toctree::\n   :maxdepth: 1\n   :hidden:\n\n")
        index_file.write(CATEGORIES["quickstart"]["list"])

        index_file.write("\nAdvanced Examples\n-----------------\n")
        index_file.write(CATEGORIES["advanced"]["table"])
        index_file.write("\n.. toctree::\n   :maxdepth: 1\n   :hidden:\n\n")
        index_file.write(CATEGORIES["advanced"]["list"])

        index_file.write("\nOther Examples\n--------------\n")
        index_file.write(CATEGORIES["other"]["table"])
        index_file.write("\n.. toctree::\n   :maxdepth: 1\n   :hidden:\n\n")
        index_file.write(CATEGORIES["other"]["list"])

    subprocess.run(["make", "html"], cwd=ROOT / "examples" / "docs", check=False)


if __name__ == "__main__":
    _main()