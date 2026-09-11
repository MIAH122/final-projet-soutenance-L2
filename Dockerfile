FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .

RUN pip install --default-timeout=1000 --retries=10 numpy==2.3.5
RUN pip install --default-timeout=1000 --retries=10 pandas==2.2.3
RUN pip install --default-timeout=1000 --retries=10 scipy==1.16.1
RUN pip install --default-timeout=1000 --retries=10 scikit-learn==1.8.0
RUN pip install --default-timeout=1000 --retries=10 streamlit==1.32.0
RUN pip install --default-timeout=1000 --retries=10 joblib==1.5.2
RUN pip install --default-timeout=1000 --retries=10 python-dotenv==1.1.1

# Étape de sécurité : force la bonne version de numpy en dernier,
# au cas où une dépendance l'aurait changée silencieusement
RUN pip install --force-reinstall --no-deps numpy==2.3.5

COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py"]