# Web-B-Gone

Workspace for the group Web-B-Gone of the 'Big Data and Language Technologies' course SoSe22.

## Setup
You can use [Docker](https://www.docker.com/101-tutorial) to setup this project. If you are not familiar with Docker, 
please visit the linked tutorial.

Clone this repository and create a docker image with the ``Dockerfile``. This image contains the entrypoint to the ``startup.py``. 
The program needs three directories to work correctly:
 - an input directory where the data is located (default: ``./data``)
 - a working directory where the index and other stuff is saved for multiple use (default: ``./working``)
 - an output directory where the results are saved (default: ``./out``)
 
It's possible to set the directories in the config.json, if so the ``config.json - PATH`` has to be the
parameter after ``-cfg``.

## Usage
In the main method of startup.py some example calls of the main functionalities of this project, like model training, 
evaluation, etc. are given. These are for illustration purposes and can be adjusted as desired.

## Dataset
The dataset can be downloaded [here](https://academictorrents.com/details/411576c7e80787e4b40452360f5f24acba9b5159). 
In order to use the dataset for this project, it first needs to be refactored. Please use the following methods in the 
same order from classification.preprocessing.swde_setup:
 - setup_swde_dataset(zip_path: Path)
 - swde_setup.restructure_swde()

## Models 
All trained and evaluated models are stored in the GIT as working.zip. To use them, they can be extracted and moved to 
the /working directory.

## Paper
The entire GIT project is based on the paper "Information-Extraction from websites with a NER-Approach", which can also 
be found in the GIT.
