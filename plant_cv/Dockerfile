# Build from the plant_cv/ folder:  docker build -t maxik-server .
FROM python:3.12-slim

# libgomp1: OpenMP runtime that onnxruntime needs on slim images
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements-server.txt .
RUN pip install --no-cache-dir -r requirements-server.txt

COPY maxik_cv ./maxik_cv
COPY webapp.py dashboard_ui.py bed_ui.py ./
COPY samples ./samples
COPY models_synthetic/plant_gate_trained.onnx models_synthetic/plant_health.onnx models_synthetic/plant_health.json ./models_synthetic/
COPY models_tomato/tomato_health.onnx models_tomato/tomato_health.json ./models_tomato/
COPY models_tomato_field/tomato_health.onnx models_tomato_field/tomato_health.json ./models_tomato_field/
COPY models_leaf/leaf_detector.onnx models_leaf/leaf_detector.json ./models_leaf/
COPY models_species/species.onnx models_species/species.json ./models_species/
COPY models_bed/bed_detector.onnx models_bed/bed_detector.json ./models_bed/

# Set MAXIK_API_TOKEN in the service's environment settings (never bake it into the image).
ENV PORT=8080
EXPOSE 8080
CMD waitress-serve --host=0.0.0.0 --port=${PORT} --threads=4 webapp:app
