"""Tests specific to the Python base image.

These tests verify Python-specific labels and configurations
that differ from the CUDA image.
"""

import os

import pytest

# --- Python-Specific Label Tests ---


def test_accelerator_label_cpu(python_container):
    """Verify accelerator label is 'cpu' for Python image."""
    labels = python_container.get_labels()
    accelerator = labels.get("com.opendatahub.accelerator")
    assert accelerator == "cpu", f"Expected accelerator='cpu', got: {accelerator}"


def test_python_version_label(python_container):
    """Verify Python version label matches expected version.

    The expected version is controlled by the PYTHON_VERSION environment variable.
    If not set, version validation is skipped (only checks label exists).
    """
    labels = python_container.get_labels()
    python_version = labels.get("com.opendatahub.python", "")
    assert python_version, "Python version label should be set"

    expected_version = os.environ.get("PYTHON_VERSION")
    if expected_version is None:
        pytest.skip(
            "PYTHON_VERSION not set - skipping version validation. "
            "Set PYTHON_VERSION env var to validate specific version."
        )
    assert expected_version in python_version, (
        f"Expected Python version label to contain {expected_version}, got: {python_version}"
    )
