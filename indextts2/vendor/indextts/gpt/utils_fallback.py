# coding=utf-8
# Copyright 2020 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Local fallbacks for names removed from ``transformers.utils`` in v5.

``transformers`` 5.0.0 removed several constants and helpers from
``transformers.utils`` that the vendored ``PreTrainedModel`` still references:
``FLAX_WEIGHTS_NAME``, ``TF2_WEIGHTS_NAME``, ``TF_WEIGHTS_NAME``,
``is_offline_mode``, ``is_remote_url``, ``download_url`` and
``is_safetensors_available``. This module vendors the original v4
implementations so the vendored code keeps working on both ``transformers`` 4.x
and 5.x.
"""

import os
import tempfile
import warnings
from urllib.parse import urlparse

import huggingface_hub
from huggingface_hub.file_download import http_get

# Removed weight-name constants (v4 values).
TF2_WEIGHTS_NAME = "tf_model.h5"
TF_WEIGHTS_NAME = "model.ckpt"
FLAX_WEIGHTS_NAME = "flax_model.msgpack"


def is_offline_mode():
    return huggingface_hub.constants.HF_HUB_OFFLINE


def is_remote_url(url_or_filename):
    parsed = urlparse(url_or_filename)
    return parsed.scheme in ("http", "https")


def download_url(url, proxies=None):
    """
    Downloads a given url in a temporary file. This function is not safe to use in multiple processes. Its only use is
    for deprecated behavior allowing to download config/models with a single url instead of using the Hub.
    """
    warnings.warn(
        f"Using `from_pretrained` with the url of a file (here {url}) is deprecated and won't be possible anymore in"
        " v5 of Transformers. You should host your file on the Hub (hf.co) instead and use the repository ID. Note"
        " that this is not compatible with the caching system (your file will be downloaded at each execution) or"
        " multiple processes (each process will download the file in a different temporary file).",
        FutureWarning,
    )
    tmp_fd, tmp_file = tempfile.mkstemp()
    with os.fdopen(tmp_fd, "wb") as f:
        http_get(url, f, proxies=proxies)
    return tmp_file


def is_safetensors_available():
    try:
        import safetensors  # noqa: F401

        return True
    except ImportError:
        return False
