FROM python:3.11-slim

WORKDIR /app

# Instala dependencias
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia el resto del proyecto
COPY . .

# Asegura que las carpetas existan
RUN mkdir -p /app/models /app/data

# Ejecuta el pipeline
CMD ["python", "main.py"]
