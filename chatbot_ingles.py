from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
import re
import os
import uvicorn
from typing import List

# Configuración para producción
app = FastAPI(
    title="Chatbot Educativo de Inglés Básico",
    description="API para identificación de errores y retroalimentación pedagógica en inglés básico (A1-A2)",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Variables globales
data = pd.DataFrame()
model_pipeline = None

def load_and_prepare_data():
    """Carga y prepara el dataset de manera robusta"""
    try:
        # Intentar cargar desde diferentes ubicaciones
        possible_paths = [
            "dataset_chatbot_ingles.csv",
            "./dataset_chatbot_ingles.csv",
            os.path.join(os.path.dirname(__file__), "dataset_chatbot_ingles.csv")
        ]
        
        data = None
        for path in possible_paths:
            try:
                if os.path.exists(path):
                    data = pd.read_csv(path)
                    print(f"✅ Dataset cargado desde: {path} - {len(data)} registros")
                    break
            except Exception as e:
                print(f"Error intentando cargar {path}: {e}")
                continue
        
        if data is None or data.empty:
            print("⚠️ No se encontró dataset, creando datos de ejemplo...")
            # Crear datos de ejemplo si no se encuentra el archivo
            data = pd.DataFrame({
                'pregunta': [
                    "Fill in the blank: 'They ___ at home'",
                    "Complete: 'She ___ to school every day'",
                    "Fill in: 'I am ___ the park'",
                    "Choose: 'He ___ a book yesterday'",
                    "Complete: 'We ___ happy'"
                ],
                'respuesta_estudiante': [
                    "They is at home",
                    "She go to school every day", 
                    "I am on the park",
                    "He readed a book yesterday",
                    "We is happy"
                ],
                'respuesta_correcta': [
                    "They are at home",
                    "She goes to school every day",
                    "I am in the park", 
                    "He read a book yesterday",
                    "We are happy"
                ],
                'tipo_error': [
                    "error_concordancia",
                    "error_tercera_persona",
                    "error_preposicion",
                    "error_tiempo_verbal",
                    "error_concordancia"
                ],
                'retroalimentacion': [
                    "Con 'they' se usa 'are': They are at home.",
                    "En tercera persona singular se agrega 's': She goes to school.",
                    "Se usa 'in' para lugares cerrados: I am in the park.",
                    "El pasado de 'read' es 'read': He read a book yesterday.",
                    "Con 'we' se usa 'are': We are happy."
                ]
            })
        
        # Limpiar datos
        for column in ['pregunta', 'respuesta_estudiante', 'respuesta_correcta', 'tipo_error', 'retroalimentacion']:
            if column in data.columns:
                data[column] = data[column].astype(str).str.strip()
        
        return data
        
    except Exception as e:
        print(f"❌ Error cargando dataset: {e}")
        return pd.DataFrame()

def train_model(data):
    """Entrena el modelo de clasificación"""
    if data.empty:
        print("❌ No se puede entrenar el modelo: dataset vacío")
        return None
    
    try:
        # Crear características para el modelo
        X = data['pregunta'] + ' ' + data['respuesta_estudiante']
        y = data['tipo_error']
        
        print(f"📊 Entrenando con {len(X)} ejemplos, {y.nunique()} clases")
        
        # Crear pipeline con TF-IDF y LinearSVC
        model_pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(
                lowercase=True,
                stop_words='english',
                ngram_range=(1, 2),
                max_features=1000,
                min_df=1,
                max_df=0.95
            )),
            ('classifier', LinearSVC(
                random_state=42, 
                max_iter=2000,
                C=1.0,
                class_weight='balanced'
            ))
        ])
        
        # Entrenar el modelo
        model_pipeline.fit(X, y)
        print(f"✅ Modelo entrenado exitosamente")
        
        return model_pipeline
        
    except Exception as e:
        print(f"❌ Error entrenando modelo: {e}")
        return None

# Inicializar al arrancar la aplicación
@app.on_event("startup")
async def startup_event():
    global data, model_pipeline
    print("🚀 Iniciando aplicación...")
    data = load_and_prepare_data()
    model_pipeline = train_model(data)
    print("✅ Aplicación lista")

# Esquemas
class RespuestaEntrada(BaseModel):
    pregunta: str
    respuesta_estudiante: str

class RespuestaAPI(BaseModel):
    es_correcta: bool
    tipo_error: str
    retroalimentacion: str
    respuesta_correcta: str = None

# Funciones auxiliares
def normalizar_texto(texto: str) -> str:
    """Normaliza el texto para comparación"""
    texto = texto.lower().strip()
    texto = re.sub(r'[^\w\s]', '', texto)
    texto = re.sub(r'\s+', ' ', texto)
    return texto

def son_respuestas_equivalentes(respuesta1: str, respuesta2: str) -> bool:
    """Compara si dos respuestas son equivalentes"""
    norm1 = normalizar_texto(respuesta1)
    norm2 = normalizar_texto(respuesta2)
    return norm1 == norm2

def obtener_retroalimentacion_especifica(pregunta: str, tipo_error: str, respuesta_correcta: str) -> str:
    """Obtiene retroalimentación específica"""
    global data
    
    # Buscar retroalimentación específica
    filtro_especifico = data[
        (data['pregunta'].str.lower() == pregunta.lower()) & 
        (data['tipo_error'].str.lower() == tipo_error.lower())
    ]
    
    if not filtro_especifico.empty:
        return filtro_especifico['retroalimentacion'].iloc[0]
    
    # Buscar retroalimentación general
    filtro_general = data[data['tipo_error'].str.lower() == tipo_error.lower()]
    if not filtro_general.empty:
        return filtro_general['retroalimentacion'].iloc[0]
    
    # Retroalimentación por defecto
    retroalimentacion_default = {
        'error_concordancia': f"Revisa la concordancia entre sujeto y verbo. La respuesta correcta es: {respuesta_correcta}",
        'error_preposicion': f"Verifica el uso correcto de las preposiciones. La respuesta correcta es: {respuesta_correcta}",
        'error_tercera_persona': f"Recuerda agregar 's' en tercera persona singular. La respuesta correcta es: {respuesta_correcta}",
        'error_tiempo_verbal': f"Revisa el tiempo verbal utilizado. La respuesta correcta es: {respuesta_correcta}",
        'error_articulo': f"Verifica el uso correcto de los artículos (a, an, the). La respuesta correcta es: {respuesta_correcta}",
        'error_vocabulario': f"Revisa la palabra utilizada. La respuesta correcta es: {respuesta_correcta}",
        'error_orden_palabras': f"Verifica el orden correcto de las palabras. La respuesta correcta es: {respuesta_correcta}"
    }
    
    return retroalimentacion_default.get(tipo_error.lower(), 
                                       f"Revisa tu respuesta. La respuesta correcta es: {respuesta_correcta}")

# Endpoints
@app.get("/")
async def root():
    """Endpoint de bienvenida"""
    return {
        "mensaje": "Chatbot Educativo de Inglés Básico",
        "descripción": "API para identificación de errores y retroalimentación pedagógica",
        "version": "1.0.0",
        "status": "online",
        "endpoints": {
            "clasificar": "/clasificar/ (POST)",
            "salud": "/health (GET)",
            "estadisticas": "/stats (GET)",
            "documentacion": "/docs (GET)"
        }
    }

@app.get("/health")
async def health_check():
    """Verificar el estado de la API"""
    global data, model_pipeline
    return {
        "status": "healthy",
        "dataset_loaded": not data.empty if data is not None else False,
        "model_trained": model_pipeline is not None,
        "total_records": len(data) if data is not None and not data.empty else 0,
        "python_version": "3.11.7",
        "framework": "FastAPI"
    }

@app.get("/stats")
async def get_statistics():
    """Obtener estadísticas del dataset"""
    global data
    
    if data is None or data.empty:
        return {
            "error": "Dataset no disponible",
            "usando_datos_ejemplo": True,
            "total_registros": 0
        }
    
    try:
        return {
            "total_registros": len(data),
            "tipos_error": data['tipo_error'].value_counts().to_dict(),
            "preguntas_unicas": data['pregunta'].nunique(),
            "ejemplos_por_tipo": {
                error_type: {
                    "cantidad": count,
                    "ejemplo": {
                        "pregunta": data[data['tipo_error'] == error_type]['pregunta'].iloc[0],
                        "respuesta_incorrecta": data[data['tipo_error'] == error_type]['respuesta_estudiante'].iloc[0],
                        "respuesta_correcta": data[data['tipo_error'] == error_type]['respuesta_correcta'].iloc[0]
                    }
                }
                for error_type, count in data['tipo_error'].value_counts().head(5).items()
            }
        }
    except Exception as e:
        return {"error": f"Error obteniendo estadísticas: {str(e)}"}

@app.post("/clasificar/", response_model=RespuestaAPI)
async def clasificar_respuesta(entrada: RespuestaEntrada):
    """Clasifica la respuesta del estudiante y proporciona retroalimentación"""
    global data, model_pipeline
    
    if data is None or data.empty or model_pipeline is None:
        raise HTTPException(status_code=500, detail="Servicio no disponible: modelo no entrenado")
    
    try:
        pregunta_entrada = entrada.pregunta.strip()
        respuesta_estudiante = entrada.respuesta_estudiante.strip()
        
        # Buscar la pregunta en el dataset
        opciones = data[data['pregunta'].str.lower() == pregunta_entrada.lower()]
        
        if opciones.empty:
            # Si no encontramos la pregunta exacta, usar el modelo para predecir
            texto_completo = f"{pregunta_entrada} {respuesta_estudiante}"
            tipo_error_predicho = model_pipeline.predict([texto_completo])[0]
            
            return RespuestaAPI(
                es_correcta=False,
                tipo_error=tipo_error_predicho,
                retroalimentacion=f"Tipo de error identificado: {tipo_error_predicho}. Revisa tu respuesta y verifica la gramática utilizada.",
                respuesta_correcta="No disponible para esta pregunta"
            )
        
        # Obtener la respuesta correcta
        respuesta_correcta = opciones['respuesta_correcta'].iloc[0].strip()
        
        # Verificar si la respuesta es correcta
        if son_respuestas_equivalentes(respuesta_estudiante, respuesta_correcta):
            mensajes_felicitacion = [
                "¡Excelente! Tu respuesta es correcta.",
                "¡Muy bien! Has respondido correctamente.",
                "¡Perfecto! Tu respuesta está bien.",
                "¡Correcto! Sigue así.",
                "¡Fantástico! Tu respuesta es exacta."
            ]
            import random
            return RespuestaAPI(
                es_correcta=True,
                tipo_error="ninguno",
                retroalimentacion=random.choice(mensajes_felicitacion),
                respuesta_correcta=respuesta_correcta
            )
        
        # Si la respuesta es incorrecta, predecir el tipo de error
        texto_completo = f"{pregunta_entrada} {respuesta_estudiante}"
        tipo_error_predicho = model_pipeline.predict([texto_completo])[0]
        
        # Obtener retroalimentación específica
        retroalimentacion = obtener_retroalimentacion_especifica(
            pregunta_entrada, tipo_error_predicho, respuesta_correcta
        )
        
        return RespuestaAPI(
            es_correcta=False,
            tipo_error=tipo_error_predicho,
            retroalimentacion=retroalimentacion,
            respuesta_correcta=respuesta_correcta
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error procesando la solicitud: {str(e)}")

@app.post("/evaluar_multiple/")
async def evaluar_respuestas_multiples(respuestas: List[RespuestaEntrada]):
    """Evalúa múltiples respuestas de una vez"""
    if len(respuestas) > 10:
        raise HTTPException(status_code=400, detail="Máximo 10 respuestas por solicitud")
    
    resultados = []
    for i, respuesta in enumerate(respuestas):
        try:
            resultado = await clasificar_respuesta(respuesta)
            resultados.append({
                "indice": i,
                "pregunta": respuesta.pregunta,
                "resultado": resultado
            })
        except Exception as e:
            resultados.append({
                "indice": i,
                "pregunta": respuesta.pregunta,
                "error": str(e)
            })
    
    return {"resultados": resultados}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
