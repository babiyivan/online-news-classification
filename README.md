# TUW-NLP2024

This project uses python ^3.12 and poetry for dependecy management
You can create a python virtual enviroment with pyenv
~~~
pyenv install 3.12 # install fitting python version if needed
pyenv local 3.12 # select fitting python version to use in this folder
~~~

To install poetry do 
~~~
pip install pipx
python -m pipx ensurepath
 
pipx install poetry
~~~

To add new dependencies to poetry you can run 
~~~
poetry add <library>
~~~
TO install all the needed dependencies run:
~~~
poetry env use 3.12
poetry install --no-root
~~~

To start the poetry virtual enviroment you can type
~~~
poetry shell # Now you are inside the virtual env
~~~

