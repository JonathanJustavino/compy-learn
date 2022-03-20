# target system
FROM ubuntu:20.04 as compy-learn

# copy repo data required for setup
COPY ./ /usr/share/compy-learn

# repo is working dir
WORKDIR /usr/share/compy-learn

# set common environment variables
ENV COMPY_LEARN=/usr/share/compy-learn \
    COMPY_LEARN_BIN=/usr/share/compy-learn-bin \
    VENV=/usr/share/venv/compy-learn

# allow installation usint apt
RUN apt-get update -y

# upgrade to install fixes
RUN apt-get upgrade -y

# install common packages missing on ubuntu docker image (required to start setup script)
RUN apt-get install -y lsb-release sudo

# install tzdata (not installed on ubuntu docker image)
RUN DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get -y install dialog tzdata

# install python and virtual environment module
RUN apt-get install -y python3 python-is-python3 python3-venv

# initialize virtual python environment
RUN python -m venv $VENV

# install all dependencies
RUN ./.docker-support/run-with-venv.sh "./install_deps.sh cpu"

# running all tests (running longer time)
#RUN ./.docker-support/run-with-venv.sh "python setup.py test"

# final installation setup
RUN ./.docker-support/run-with-venv.sh "python setup.py install"

# save built binaries to restore it on mounted folders
RUN ./.docker-support/save-binaries.sh

# do not stop container by an exiting process
CMD ["tail", "-f", "/dev/null"]
