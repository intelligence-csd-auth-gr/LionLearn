FROM ubuntu:18.04
FROM python:3.7.8

RUN apt-get update
RUN apt-get dist-upgrade -y
RUN apt-get install unzip

RUN DEBIAN_FRONTEND=noninteractive apt-get -y dist-upgrade
RUN DEBIAN_FRONTEND=noninteractive apt-get -y install software-properties-common
RUN DEBIAN_FRONTEND=noninteractive apt-add-repository ppa:swi-prolog/stable
RUN DEBIAN_FRONTEND=noninteractive apt-get -y install swi-prolog

WORKDIR /usr/src/app

COPY lionets.zip .

RUN unzip lionets.zip
RUN rm /usr/src/app/lionets.zip

RUN pip install --upgrade setuptools

RUN pip install --no-cache-dir -r /usr/src/app/lionets/requirements.txt
RUN rm /usr/src/app/lionets/requirements.txt

RUN cp /usr/src/app/lionets/utilities/checks.py /usr/local/lib/python3.7/site-packages/innvestigate/utils/keras
RUN rm /usr/src/app/lionets/utilities/checks.py

RUN cp /usr/src/app/lionets/utilities/relevance_analyzer.py /usr/local/lib/python3.7/site-packages/innvestigate/analyzer/relevance_based
RUN rm /usr/src/app/lionets/utilities/relevance_analyzer.py
RUN python3 -m pip install --upgrade notebook

CMD ["jupyter", "notebook", "--port=8888", "--no-browser", "--ip=0.0.0.0", "--allow-root"]