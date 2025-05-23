from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional
from contextlib import asynccontextmanager
import pickle
import os
from datetime import datetime
import uvicorn
from pathlib import Path # ¡Nueva importación!

# Modelos Pydantic para request/response
class PredictionRequest(BaseModel):
    municipio: str
    months_ahead: Optional[int] = 12

class PredictionData(BaseModel):
    fecha: str
    prediccion: float
    limite_inferior: float
    limite_superior: float

class PredictionResponse(BaseModel):
    municipio: str
    predicciones: List[PredictionData]
    total_meses: int
    fecha_consulta: str

class MunicipioResponse(BaseModel):
    municipios: List[str]
    total: int

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    models_loaded: int

# Variable global para almacenar modelos cargados
models_data = {}
municipios_list = []

class ModelLoader:
    @staticmethod
    def load_all_models(directory='trained_models'):
        """Cargar todos los modelos al iniciar la API"""
        global models_data, municipios_list
        
        # Ajusta el directorio de modelos para que sea relativo a main.py si es necesario
        # Esto asume que 'trained_models' está en el mismo directorio que main.py (api/)
        model_dir_path = Path(__file__).parent / directory
        
        if not model_dir_path.exists():
            raise FileNotFoundError(f"Directorio {model_dir_path} no encontrado")
        
        # Cargar lista de municipios
        municipios_file = model_dir_path / "municipios.pkl"
        if not municipios_file.exists():
            raise FileNotFoundError(f"Archivo de municipios no encontrado en {municipios_file}")
            
        with open(municipios_file, 'rb') as f:
            municipios_list = pickle.load(f)
        
        # Cargar cada modelo
        loaded_count = 0
        for municipio in municipios_list:
            try:
                filename = model_dir_path / f"model_{municipio.lower().replace(' ', '_')}.pkl"
                if filename.exists():
                    with open(filename, 'rb') as f:
                        models_data[municipio] = pickle.load(f)
                        loaded_count += 1
            except Exception as e:
                print(f"Error cargando modelo para {municipio}: {str(e)}")
        
        print(f"Modelos cargados exitosamente: {loaded_count}/{len(municipios_list)}")
        return loaded_count

    @staticmethod
    def predict_municipio(municipio: str, months_ahead: int = 12):
        """Realizar predicción para un municipio específico"""
        if municipio not in models_data:
            # Buscar municipio con coincidencia parcial (case insensitive)
            matching_municipios = [m for m in municipios_list 
                                   if municipio.lower() in m.lower() or m.lower() in municipio.lower()]
            if matching_municipios:
                municipio = matching_municipios[0]
            else:
                raise ValueError(f"Municipio '{municipio}' no encontrado")
        
        model_info = models_data[municipio]
        model = model_info['model']
        
        # Crear dataframe futuro
        future = model.make_future_dataframe(periods=months_ahead, freq='MS')
        
        # Realizar predicción
        forecast = model.predict(future)
        
        # Extraer predicciones futuras
        future_forecast = forecast.tail(months_ahead)
        
        predictions = []
        for _, row in future_forecast.iterrows():
            predictions.append(PredictionData(
                fecha=row['ds'].strftime('%Y-%m-%d'),
                prediccion=round(row['yhat'], 2),
                limite_inferior=round(row['yhat_lower'], 2),
                limite_superior=round(row['yhat_upper'], 2)
            ))
        
        return predictions

# --- INICIO DE CORRECCIONES DE RUTA ---
# Definir la ruta base del proyecto (un nivel arriba de 'api/')
BASE_DIR = Path(__file__).resolve().parent.parent
# Definir la ruta de la carpeta frontend
FRONTEND_DIR = BASE_DIR / "frontend"
# --- FIN DE CORRECCIONES DE RUTA ---

# Definir lifespan ANTES de usar en FastAPI
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gestionar el ciclo de vida de la aplicación"""
    # Startup
    try:
        ModelLoader.load_all_models()
        print("API iniciada correctamente")
    except Exception as e:
        print(f"Error al cargar modelos: {str(e)}")
        print("Nota: Asegúrate de que 'trained_models' esté en el mismo directorio que main.py")
        print("Y ejecuta 'python train_model.py' primero para generar los modelos si no existen.")
        # En lugar de fallar, continuamos pero sin modelos
    
    yield
    
    # Shutdown (si necesario)
    print("API cerrándose...")

# Inicializar FastAPI DESPUÉS de definir lifespan
app = FastAPI(
    title="TarifaPredict API",
    description="API para predicción de tarifas municipales usando Prophet",
    version="1.0.0",
    lifespan=lifespan
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción, especificar dominios específicos
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# *** NUEVAS LÍNEAS PARA SERVIR EL FRONTEND (MODIFICADAS PARA USAR FRONTEND_DIR) ***
# Montar archivos estáticos del frontend
if FRONTEND_DIR.is_dir(): # Usamos .is_dir() para verificar que la carpeta existe
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")
    
    @app.get("/")
    async def serve_frontend():
        """Servir la aplicación frontend"""
        # Usamos FRONTEND_DIR para especificar la ruta exacta del index.html
        return FileResponse(FRONTEND_DIR / 'index.html')

# *** MODIFICAR EL ENDPOINT ROOT (MODIFICADO PARA USAR FRONTEND_DIR) ***
@app.get("/")
async def root():
    """Endpoint de salud de la API con opción de ver frontend"""
    # Usamos .is_file() para verificar que el archivo index.html existe dentro de la carpeta frontend
    if (FRONTEND_DIR / "index.html").is_file():
        return {
            "status": "OK",
            "timestamp": datetime.now().isoformat(),
            "models_loaded": len(models_data),
            "message": "API funcionando correctamente",
            "frontend_available": True,
            "frontend_url": "/app"
        }
    else:
        return HealthResponse(
            status="OK",
            timestamp=datetime.now().isoformat(),
            models_loaded=len(models_data)
        )

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Verificar estado de la API"""
    return HealthResponse(
        status="healthy" if len(models_data) > 0 else "no_models",
        timestamp=datetime.now().isoformat(),
        models_loaded=len(models_data)
    )

@app.get("/municipios", response_model=MunicipioResponse)
async def get_municipios():
    """Obtener lista de municipios disponibles"""
    return MunicipioResponse(
        municipios=sorted(municipios_list),
        total=len(municipios_list)
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict_tarifa(request: PredictionRequest):
    """Realizar predicción de tarifas para un municipio"""
    try:
        # Validar parámetros
        if request.months_ahead < 1 or request.months_ahead > 24:
            raise HTTPException(
                status_code=400, 
                detail="months_ahead debe estar entre 1 y 24"
            )
        
        # Realizar predicción
        predictions = ModelLoader.predict_municipio(
            request.municipio, 
            request.months_ahead
        )
        
        return PredictionResponse(
            municipio=request.municipio,
            predicciones=predictions,
            total_meses=len(predictions),
            fecha_consulta=datetime.now().isoformat()
        )
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")

@app.get("/predict/{municipio}", response_model=PredictionResponse)
async def predict_tarifa_get(municipio: str, months_ahead: int = 12):
    """Realizar predicción usando GET (alternativo)"""
    request = PredictionRequest(municipio=municipio, months_ahead=months_ahead)
    return await predict_tarifa(request)

@app.get("/municipios/search/{query}")
async def search_municipios(query: str):
    """Buscar municipios por nombre parcial"""
    matching = [m for m in municipios_list 
                if query.lower() in m.lower()]
    return {
        "query": query,
        "matches": matching,
        "total": len(matching)
    }

# Endpoint para obtener información del modelo de un municipio
@app.get("/model-info/{municipio}")
async def get_model_info(municipio: str):
    """Obtener información sobre el modelo de un municipio"""
    if municipio not in models_data:
        raise HTTPException(status_code=404, detail="Municipio no encontrado")
    
    model_info = models_data[municipio]
    
    return {
        "municipio": municipio,
        "ultima_fecha_entrenamiento": model_info['last_date'].isoformat(),
        "valor_promedio_historico": round(model_info['mean_value'], 2),
        "total_datos_entrenamiento": len(model_info['training_data'])
    }

# Función para ejecutar el servidor
def run_server():
    """Ejecutar servidor de desarrollo"""
    # Al ejecutar desde 'api/' con 'uvicorn main:app', Uvicorn ya sabe dónde buscar 'main'.
    # Si ejecutas desde la raíz del proyecto ('TRABAJO FINAL ELECTIVA/'),
    # tendrías que usar 'uvicorn api.main:app'
    uvicorn.run(
        "main:app", 
        host="127.0.0.1", 
        port=3000, 
        reload=True,
        log_level="info"
    )

if __name__ == "__main__":
    run_server()