# Create and set up a virtual environment
setup:
	uv venv env
	env/bin/python -m ensurepip --default-pip
	env/bin/python -m pip install --upgrade pip
	env/bin/python -m pip install -r requirements.txt

#need to manually run $source env/bin/activate after $make setup

# Run tests
test:
	. env/bin/activate && pytest Tests/test_dataset.py && pytest Tests/test_data.py
