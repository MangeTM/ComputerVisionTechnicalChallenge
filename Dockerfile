# Usar la imagen oficial de Python como base
FROM python:latest

# Establecer el directorio de trabajo dentro del contenedor
WORKDIR /app

# Copiar el script de Python y los requisitos
COPY main.py .
COPY requirements.txt .

# Instalar dependencias
RUN pip install -r requirements.txt
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 -y

# Comando por defecto para ejecutar tu script
CMD ["python", "main.py"]
