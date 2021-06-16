FROM python:3.8
MAINTAINER Boiko Pavlo 'pboyko172839465@gmail.com'

WORKDIR /usr/src/app
COPY ./app/requirements.txt ./
RUN pip3 install --no-cache-dir -r requirements.txt 
COPY ./app .

ENTRYPOINT  [ "python","-m", "flask_app"]

