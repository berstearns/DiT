FROM  python:3.9.16-slim-buster
RUN apt-get update -y
RUN apt-get install\
	    ffmpeg\
	    libsm6\
	    libxext6\
	    git\
	    gcc\
	    g++ -y
WORKDIR /app
COPY ./requirements.txt /app/requirements.txt
RUN  --mount=type=cache,target=/root/.cache \
	pip install -r requirements.txt
RUN  --mount=type=cache,target=/root/.cache \
	pip install 'git+https://github.com/facebookresearch/detectron2.git'
COPY . /app
# CMD ["uvicorn", "api:app", "--port=8080", "--reload", "--host=0.0.0.0"]
# CMD ["gunicorn", "flask_api:app", "-c", "gunicorn_config.py"]
CMD ["python"]
