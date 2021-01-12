FROM ubuntu:18.04

MAINTANER FirstName LastName "abcd@domain.com"

RUN apt-get update -y && \
    apt-get install -y python-pip python-dev

# copy the requirements.txt first to leverage Docker cache
COPY ./requirements.txt /app/requirements.txt

WORKDIR /app

RUN pip install -r requirements.txt

COPY . /app

ENTRYPOINT [ "python" ]

CMD [ "main.py" ]