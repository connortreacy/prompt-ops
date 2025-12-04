"""
Template files for project scaffolding.
"""

import json
import os

import yaml

# Directory containing the template files
TEMPLATE_DIR = os.path.dirname(os.path.abspath(__file__))


def get_template_path(filename):
    """Get the absolute path to a template file."""
    return os.path.join(TEMPLATE_DIR, filename)


def get_template_content(filename):
    """Get the content of a template file."""
    with open(get_template_path(filename), "r") as f:
        return f.read()


def get_sample_dataset():
    """Get the sample dataset as a Python object."""
    with open(get_template_path("sample_dataset.json"), "r") as f:
        return json.load(f)


def get_config_template(model: str) -> dict:
    """
    Load the sample config template and substitute the model name.

    Args:
        model: The model identifier (e.g., "openrouter/meta-llama/llama-3.3-70b-instruct")

    Returns:
        dict: The config dictionary with model name substituted
    """
    template_path = get_template_path("sample_config.yaml")
    with open(template_path, "r") as f:
        content = f.read()

    # Replace placeholder with actual model name
    content = content.replace("${MODEL}", model)

    return yaml.safe_load(content)
