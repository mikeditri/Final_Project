FROM ubuntu:16.04
MAINTAINER <Mike D'Itri >

RUN apt-get update
RUN apt-get install -y python3-pip
RUN pip3 install --upgrade pip
RUN pip3 install Numpy
RUN pip3 install pandas 
RUN pip3 install matplotlib
RUN pip3 install seaborn
RUN pip3 install tqdm
RUN pip3 install sklearn
WORKDIR /app
CMD ["python3", "/app/Final_script.py"]