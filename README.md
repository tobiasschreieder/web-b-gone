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
