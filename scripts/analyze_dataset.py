import pandas as pd
import requests

# Descargar y analizar el dataset
url = "https://hebbkx1anhila5yf.public.blob.vercel-storage.com/dataset_chatbot_ingles-pK0xx1LsFZTGnPSnC8VBjoOe3vUNl7.csv"
response = requests.get(url)

# Guardar el archivo CSV
with open("dataset_chatbot_ingles.csv", "wb") as f:
    f.write(response.content)

# Cargar y analizar el dataset
df = pd.read_csv("dataset_chatbot_ingles.csv")

print("=== ANÁLISIS DEL DATASET ===")
print(f"Número de filas: {len(df)}")
print(f"Columnas: {list(df.columns)}")
print("\n=== PRIMERAS 5 FILAS ===")
print(df.head())

print("\n=== TIPOS DE ERRORES ÚNICOS ===")
print(df['tipo_error'].value_counts())

print("\n=== EJEMPLOS POR TIPO DE ERROR ===")
for error_type in df['tipo_error'].unique():
    print(f"\n--- {error_type} ---")
    example = df[df['tipo_error'] == error_type].iloc[0]
    print(f"Pregunta: {example['pregunta']}")
    print(f"Respuesta estudiante: {example['respuesta_estudiante']}")
    print(f"Respuesta correcta: {example['respuesta_correcta']}")
    print(f"Retroalimentación: {example['retroalimentacion']}")
