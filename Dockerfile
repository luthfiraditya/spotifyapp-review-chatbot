FROM python:3.10

WORKDIR /app

COPY requirements.txt .

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8123

CMD ["streamlit", "run", "app.py", "--server.port=8123", "--server.address=0.0.0.0"]
