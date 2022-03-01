# /src

Source code for the implementation will be stored here.

## Setup
Install Python3, pip, Jupyter Notebook. 

### Virtual Environment (optional; recommended)
I recommend using a virtual environment with [venv](https://docs.python.org/3/library/venv.html) to avoid polluting your global pip environment. 
In this `/src` directory:
```sh
python3 -m venv envbandit
```
Activating the environment can be done like so:
```sh
source envbandit/bin/activate
```
and deactivating like this:
```sh
deactivate
```

While your virtual env is active, installing Python packages with pip will add them to the environment.

### Install Packages
Install the required packages with 
```
pip install -r requirements.txt
```
