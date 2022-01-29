# Install VSCODE python plugins 
IntelliSense (Pylance), Linting, Debugging (multi-threaded, remote), Jupyter Notebooks, code formatting, refactoring, unit tests

# upgrade pip version
python -m pip install --upgrade pip

# Create a Virtual Environment with local version
python -m venv rrsenv

# select python interpreter on the editior from virtual environment folder
ctl + shift + P
search : Python Interpreter
select rrsenv as interpreter environment

# Enter into virtual environment
.\rrsenv\Scripts\activate

# Following Commands has been executed to run the project successfully
# Inside the porject directory
pip3 install -r requirements.txt

# to Exit from virualenv
deactivate

# To delete virtual environment, delete env directory
rm -rf rrsenv

# Before final deployment
pip3 freeze > requirements.txt

# To run application
python main.py