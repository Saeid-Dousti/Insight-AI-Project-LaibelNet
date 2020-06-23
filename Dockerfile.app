# Dockerfile for building streamline app

# pull miniconda image
FROM continuumio/miniconda3


# copy local files into container
COPY app.py /tmp/
COPY requirements.txt /tmp/
COPY cloudwine /tmp/cloudwine
#COPY data /tmp/data
# .streamlit for something to do with making enableCORS=False
COPY .streamlit /tmp/.streamlit
COPY build /tmp/build

# install python 3.8.3
RUN conda install python=3.8.3
# RUN conda install faiss-cpu=1.5.1 -c pytorch -y

ENV PORT 8080
ENV GOOGLE_APPLICATION_CREDENTIALS=/tmp/build/storage-read-only-service-account.json

# change directory
WORKDIR /tmp

# install dependencies
RUN apt-get update && apt-get install -y vim g++
RUN pip install -r requirements.txt

# run commands
CMD ["streamlit", "run", "app.py"]