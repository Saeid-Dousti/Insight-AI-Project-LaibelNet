# Dockerfile for building streamline app

# pull miniconda image
FROM continuumio/miniconda3


# copy local files into container
COPY . /tmp/
#COPY requirements.txt /tmp/
#COPY setup.py /tmp/
#COPY st_functions.py /tmp/

#COPY data /tmp/data
#COPY LaibelNet /tmp/LaibelNet
#COPY config /tmp/config
#COPY pickledir /tmp/pickledir

# .streamlit to make enableCORS=False
#COPY .streamlit /tmp/.streamlit

# install python 3.7
RUN conda install python=3.7

EXPOSE 8080

# change directory
WORKDIR /tmp

# install dependencies
RUN apt-get update && apt-get install -y vim g++
RUN pip install -r requirements.txt

# run commands
CMD ["streamlit", "run", "app.py"]