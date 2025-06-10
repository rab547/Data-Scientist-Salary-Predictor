import os;

file_path = 'Data/ds_salaries.csv'

def test_exists():
    assert os.path.exists(file_path)

def test_size():
    assert os.path.getsize(file_path) > 200000 # more than 200 kb
    assert os.path.getsize(file_path) < 250000 # less than 250 kb