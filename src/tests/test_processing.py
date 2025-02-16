# TODO do not monkeypatch env variables, load the .env file instead
import os
from dotenv import load_dotenv
import shutil
import tarfile
import pytest
import tempfile
import pandas as pd
from pathlib import Path
from processing.processing import processing


load_dotenv()


@pytest.fixture(scope="function", autouse=False)
def directory():
    directory = tempfile.mkdtemp()
    data_directory = Path(directory) / "data"
    data_directory.mkdir(parents=True, exist_ok=True)

    import subprocess
    s3_folder_dir = 's3://birdsdetection/dataset/images/009.Brewer_Blackbird/*'
    destination = data_directory
    command = ['s5cmd', 'cp', s3_folder_dir, destination]
    subprocess.run(command, check=False, capture_output=True, text=True)

    directory = Path(directory)

    processing(directory, unittesting=True)

    yield directory

    shutil.rmtree(directory)


def test_preprocess_generates_transformed_data(directory):
    print("test")