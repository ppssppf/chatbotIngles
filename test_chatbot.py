import requests
import json

# URL base de la API (cambiar según el entorno)
BASE_URL = "http://localhost:8000"

def test_api():
    """Función para probar la API del chatbot"""
    
    print("=== PROBANDO CHATBOT EDUCATIVO DE INGLÉS ===\n")
    
    # Test 1: Verificar estado de la API
    print("1. Verificando estado de la API...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"Estado: {response.json()}")
    except Exception as e:
        print(f"Error: {e}")
    
    print("\n" + "="*50 + "\n")
    
    # Test 2: Obtener estadísticas
    print("2. Obteniendo estadísticas del dataset...")
    try:
        response = requests.get(f"{BASE_URL}/stats")
        stats = response.json()
        print(f"Total de registros: {stats['total_registros']}")
        print(f"Tipos de error: {list(stats['tipos_error'].keys())}")
    except Exception as e:
        print(f"Error: {e}")
    
    print("\n" + "="*50 + "\n")
    
    # Test 3: Probar respuestas correctas e incorrectas
    test_cases = [
        {
            "pregunta": "Fill in the blank: 'They ___ at home'",
            "respuesta_estudiante": "They are at home",
            "esperado": "correcta"
        },
        {
            "pregunta": "Fill in the blank: 'They ___ at home'",
            "respuesta_estudiante": "They is at home",
            "esperado": "error_concordancia"
        },
        {
            "pregunta": "Complete: 'She ___ to school every day'",
            "respuesta_estudiante": "She go to school every day",
            "esperado": "error_tercera_persona"
        },
        {
            "pregunta": "Fill in: 'I am ___ the park'",
            "respuesta_estudiante": "I am on the park",
            "esperado": "error_preposicion"
        }
    ]
    
    print("3. Probando casos de prueba...")
    for i, test_case in enumerate(test_cases, 1):
        print(f"\nCaso {i}:")
        print(f"Pregunta: {test_case['pregunta']}")
        print(f"Respuesta estudiante: {test_case['respuesta_estudiante']}")
        
        try:
            response = requests.post(
                f"{BASE_URL}/clasificar/",
                json={
                    "pregunta": test_case["pregunta"],
                    "respuesta_estudiante": test_case["respuesta_estudiante"]
                }
            )
            
            if response.status_code == 200:
                resultado = response.json()
                print(f"Es correcta: {resultado['es_correcta']}")
                print(f"Tipo de error: {resultado['tipo_error']}")
                print(f"Retroalimentación: {resultado['retroalimentacion']}")
                if resultado.get('respuesta_correcta'):
                    print(f"Respuesta correcta: {resultado['respuesta_correcta']}")
            else:
                print(f"Error HTTP {response.status_code}: {response.text}")
                
        except Exception as e:
            print(f"Error: {e}")
        
        print("-" * 30)

if __name__ == "__main__":
    test_api()
