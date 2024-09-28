FROM python:3.11    

EXPOSE 8501
COPY ./deployment/frontend/requirements.txt /frontend/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /frontend/requirements.txt

COPY ./deployment/frontend /frontend
WORKDIR /frontend

CMD ["streamlit", "run", "src/app.py"]  