FROM tiangolo/uvicorn-gunicorn-fastapi:python3.11

EXPOSE 8000
COPY ./deployment/backend/requirements.txt /deployment/backend/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /deployment/backend/requirements.txt

COPY ./deployment/backend /backend
COPY ./deployment/model /model

WORKDIR /backend

CMD ["python3", "-m", "src"]        
