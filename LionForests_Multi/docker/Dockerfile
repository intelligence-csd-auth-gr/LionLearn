FROM ubuntu:18.04
FROM python:3.7.8

RUN apt-get update
RUN apt-get dist-upgrade -y
RUN apt-get install unzip

RUN DEBIAN_FRONTEND=noninteractive apt-get -y dist-upgrade
RUN DEBIAN_FRONTEND=noninteractive apt-get -y install software-properties-common

WORKDIR /usr/src/app

COPY lionforests.zip .

RUN unzip lionforests.zip 
RUN rm /usr/src/app/lionforests.zip

RUN pip install --upgrade setuptools

RUN pip install --no-cache-dir -r requirements.txt
RUN rm /usr/src/app/requirements.txt

RUN python3 -m pip install --upgrade notebook

CMD ["jupyter", "notebook", "--port=8888", "--no-browser", "--ip=0.0.0.0", "--allow-root"]