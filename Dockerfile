FROM python:3.8-slim-buster

RUN apt update -y
RUN apt install -y build-essential

COPY . .
RUN pip install -r requirements.txt
ENTRYPOINT ["gunicorn", "--certfile=server.crt", "--keyfile=server.key", "--timeout", "600" ,"application:app", "-b 0.0.0.0:443"]

EXPOSE 443
