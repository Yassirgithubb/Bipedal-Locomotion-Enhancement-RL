You can check your python version with

    python3 --version

install python virtualenv

    pip3 install virtualenv
    mkdir ~/.virtualenvs
    pip3 install virtualenvwrapper

add the following to your ~/.bashrc

    export WORKON_HOME=~/.virtualenvs
    export VIRTUALENVWRAPPER_PYTHON=/usr/bin/python3
    source ~/.local/bin/virtualenvwrapper.sh

Open a new terminal tab and create your virtual environment:

    source ~/.profile
    mkvirtualenv --system-site-packages leggedgym

To activate your python virtual environment everytime you are working with it, use

    workon leggedgym

Make sure that your virtual env is always used when installing packages