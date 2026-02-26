# Usa un'immagine Python ufficiale e leggera
FROM python:3.10-slim

# Variabili d'ambiente per ottimizzare Python nel container
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
# Aggiunge la root al path di Python così funzionano gli import "from src..."
ENV PYTHONPATH=/app

# Imposta la cartella di lavoro interna al container
WORKDIR /app

# Installa i compilatori C++ necessari per gRPC e le librerie matematiche
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copia PRIMA il file dei requisiti e installa le librerie
# (Questo trucco sfrutta la cache di Docker: se cambi il codice ma non le librerie, ci mette un secondo a buildare)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia TUTTO il resto del tuo codice nel container
COPY . .

# Espone la porta standard di gRPC (se usi una porta diversa, cambiala qui)
EXPOSE 50051

# Comando di default: apriamo una shell bash vuota.
# Sovrascriveremo questo comando quando facciamo "docker run"
CMD ["/bin/bash"]