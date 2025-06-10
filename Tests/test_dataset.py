import os;
# import pytest;

file_path = os.path.join("Data", "ds_salaries.csv") # Fix path

def test_file_exists():
    """Check if the dataset file exists"""
    assert os.path.exists(file_path), f"File not found: {file_path}"

def test_file_size():
    """Check if the dataset file is between 200k b and 250kb"""
    assert os.path.getsize(file_path) > 200000, "Dataset file is more than 200 kb"
    assert os.path.getsize(file_path) < 250000, "Dataset file is less than 250 kb"