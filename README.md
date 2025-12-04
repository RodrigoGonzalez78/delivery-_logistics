
# 🚚 Predicción de Tiempos de Entrega en Logística
## 📂 Estructura del Proyecto

```text
delivery_project/
├── config/
│   └── config.yaml           # Hiperparámetros y rutas de datos
├── data/
│   ├── raw/                  # Dataset original (CSV)
│   └── processed/            # Datos procesados (opcional)
├── models/                   # Artefactos: Modelos (.keras) y Preprocesadores (.pkl)
├── src/
│   ├── api/
│   │   ├── static/           # Archivos del Frontend (HTML/CSS)
│   │   └── main.py           # Servidor FastAPI
│   ├── features/
│   │   └── build_features.py # Pipelines de transformación de datos
│   └── models/
│       ├── model_arch.py     # Definición de la arquitectura de la Red Neuronal
│       └── train_model.py    # Script de entrenamiento y validación
├── Dockerfile                # Configuración para Docker
├── requirements.txt          # Dependencias de Python
└── README.md                 # Documentación

```

### Crear Entorno

```Bash
python -m venv venv
source venv/bin/activate  
```
### Instalar dependencias

```Bash
pip install -r requirements.txt
```

### Entrenar el Modelo
Este script carga los datos, ejecuta el preprocesamiento, entrena la red neuronal y guarda los artefactos en la carpeta models/.

```Bash
python src/models/train_model.py
```

### Ejecución con Docker

```Bash

# Construir la imagen
docker build -t delivery-api .

# Correr el contenedor
docker run -p 8000:8000 delivery-api

```


