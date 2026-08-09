"""
Chapter 2, Step 0: Download the model from ModelScope
====================================

Before running any experiments, download Qwen2.5-0.5B-Instruct to the local machine.
Later scripts will preferentially load the model from the local copy, avoiding a
network download every time.

Usage:
    pip install modelscope
    python 0-download_model.py
"""

import os
from modelscope import snapshot_download

# Directory where the model is saved
LOCAL_MODEL_DIR = "./Qwen2.5-0.5B-Instruct"

# Model ID on ModelScope
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"


def download_model():
    if os.path.exists(LOCAL_MODEL_DIR) and os.path.exists(
        os.path.join(LOCAL_MODEL_DIR, "config.json")
    ):
        print(f"Model already exists at {LOCAL_MODEL_DIR}, skipping download.")
        print(f"To re-download, delete the {LOCAL_MODEL_DIR} directory and try again.")
        return LOCAL_MODEL_DIR

    print(f"Downloading model {MODEL_ID} from ModelScope ...")
    print("The model is about 1GB, please be patient.")
    # Use local_dir instead of cache_dir: cache_dir would nest an extra
    # level under the repo path (./Qwen2.5-0.5B-Instruct/Qwen/Qwen2___5-0___5B-Instruct/),
    # which would make config.json unfindable when later scripts load from LOCAL_MODEL_DIR.
    model_dir = snapshot_download(
        MODEL_ID,
        local_dir=LOCAL_MODEL_DIR,
    )
    print(f"Model download complete, saved to: {model_dir}")
    return model_dir


if __name__ == "__main__":
    download_model()
