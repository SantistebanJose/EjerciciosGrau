from flask import Flask, request, jsonify
from flask_cors import CORS
import random
import json
import numpy as np
from fractions import Fraction
import joblib
import os
import math
from datetime import datetime

# Importar los generadores de ejercicios
from algoritmoGenerarEjercicios import (
    BalanceExerciseGenerator,
    ArithmeticProgressionGenerator,
    LinearEquationGenerator,
    LinearFunctionGenerator,
    ProportionalityGenerator,
    InequalityGenerator,
    ProportionalityConstantGenerator
)

app = Flask(__name__)
CORS(app)  # Habilitar CORS para todas las rutas

# Configuración para la base de datos (simulado por ahora con un dict)
student_history = {}

# Inicializar generadores de ejercicios
exercise_types = {
    'balanza': BalanceExerciseGenerator(),
    'progresion_aritmetica': ArithmeticProgressionGenerator(),
    'ecuacion_lineal': LinearEquationGenerator(),
    'funcion_lineal': LinearFunctionGenerator(),
    'proporcionalidad': ProportionalityGenerator(),
    'desigualdades': InequalityGenerator(),
    'constante_proporcionalidad': ProportionalityConstantGenerator()
}

# Mapa de conceptos con dependencias (para recomendaciones)
concept_map = {
    'balanza': ['ecuacion_lineal'],
    'progresion_aritmetica': ['ecuacion_lineal', 'funcion_lineal'],
    'ecuacion_lineal': ['funcion_lineal', 'desigualdades'],
    'funcion_lineal': ['proporcionalidad'],
    'proporcionalidad': ['constante_proporcionalidad'],
    'desigualdades': ['proporcionalidad'],
    'constante_proporcionalidad': []
}

# Intentar cargar el modelo de recomendación si existe
def load_recommendation_model():
    model_path = 'tutor_recommendation_model.joblib'
    try:
        if os.path.exists(model_path):
            return joblib.load(model_path)
    except Exception as e:
        print(f"Error cargando modelo: {e}")
    
    # Crear un modelo inicial si no existe
    from sklearn.ensemble import RandomForestClassifier
    return RandomForestClassifier(n_estimators=50)

recommendation_model = load_recommendation_model()

# Función para guardar el modelo
def save_model():
    try:
        joblib.dump(recommendation_model, 'tutor_recommendation_model.joblib')
        return True
    except Exception as e:
        print(f"Error guardando modelo: {e}")
        return False

# Función para entrenar el modelo con nuevos datos
def train_recommendation_model(student_id):
    if student_id not in student_history or len(student_history[student_id]) < 5:
        return False  # No hay suficientes datos para entrenar
    
    try:
        import pandas as pd
        # Preparar datos para entrenamiento
        df = pd.DataFrame(student_history[student_id])
        
        # Características para el modelo
        X = df[['difficulty', 'steps_completed', 'mistakes', 'help_requested', 'time_taken']].values
        
        # Etiquetas: 1 si completó con éxito, 0 si no
        y = df['completed'].astype(int).values
        
        # Entrenar modelo
        recommendation_model.fit(X, y)
        save_model()
        return True
    except Exception as e:
        print(f"Error entrenando modelo: {e}")
        return False

# Función para generar recomendaciones
def get_recommendation(student_id, current_performance):
    # Si no hay suficientes datos, usar recomendación basada en reglas
    if student_id not in student_history or len(student_history[student_id]) < 5:
        return rule_based_recommendation(current_performance)
    
    try:
        # Convertir el rendimiento actual a un formato para el modelo
        features = np.array([[
            current_performance['difficulty'],
            current_performance['steps_completed'],
            current_performance['mistakes'],
            current_performance['help_requested'],
            current_performance['time_taken']
        ]])
        
        # Predecir probabilidad de éxito con diversos tipos de ejercicios
        success_probs = {}
        current_difficulty = current_performance['difficulty']
        
        for exercise_type in exercise_types:
            # Simular rendimiento en diferentes niveles de dificultad
            for diff_level in range(1, 4):
                features[0][0] = diff_level
                prob = recommendation_model.predict_proba(features)[0][1]
                success_probs[(exercise_type, diff_level)] = prob
        
        # Encontrar el mejor ejercicio recomendado (desafiante pero alcanzable)
        best_recommendation = max(success_probs.items(), key=lambda x: x[1] if x[1] >= 0.6 else 0)
        
        return {
            'exercise_type': best_recommendation[0][0],
            'difficulty': best_recommendation[0][1],
            'success_probability': float(best_recommendation[1]),
            'rationale': "Basado en tu rendimiento histórico, este ejercicio debería ser desafiante pero alcanzable."
        }
    except Exception as e:
        print(f"Error generando recomendación: {e}")
        return rule_based_recommendation(current_performance)

# Recomendación basada en reglas cuando no hay suficientes datos
def rule_based_recommendation(performance):
    current_type = performance['exercise_type']
    current_diff = performance['difficulty']
    mistakes = performance['mistakes']
    
    # Si hay muchos errores, mantener mismo tipo pero bajar dificultad
    if mistakes > 3 and current_diff > 1:
        return {
            'exercise_type': current_type,
            'difficulty': current_diff - 1,
            'success_probability': 0.8,
            'rationale': "Practicar el mismo concepto con menor dificultad te ayudará a reforzar lo básico."
        }
    # Si hay pocos errores, subir dificultad
    elif mistakes < 2 and current_diff < 3:
        return {
            'exercise_type': current_type,
            'difficulty': current_diff + 1,
            'success_probability': 0.7,
            'rationale': "¡Buen trabajo! Estás listo para un desafío mayor en este tema."
        }
    # Si domina el tema, recomendar un tema relacionado
    elif mistakes == 0 and current_diff == 3:
        # Buscar un tema que dependa de este
        for next_topic, dependencies in concept_map.items():
            if current_type in dependencies:
                return {
                    'exercise_type': next_topic,
                    'difficulty': 1,
                    'success_probability': 0.75,
                    'rationale': f"¡Has dominado este tema! Es un buen momento para explorar {next_topic}, que se basa en lo que ya aprendiste."
                }
        
        # Si no hay dependencia clara, recomendar un tema al azar
        next_topic = random.choice(list(exercise_types.keys()))
        return {
            'exercise_type': next_topic,
            'difficulty': 1,
            'success_probability': 0.65,
            'rationale': "Es un buen momento para explorar un nuevo tema."
        }
    
    # Mantener mismo tipo y dificultad
    return {
        'exercise_type': current_type,
        'difficulty': current_diff,
        'success_probability': 0.75,
        'rationale': "Sigue practicando para reforzar este concepto."
    }

# Función para registrar la interacción del estudiante
def track_interaction(student_id, exercise_type, difficulty, steps_completed, mistakes, help_requested, time_taken, completed):
    # Inicializar historial del estudiante si no existe
    if student_id not in student_history:
        student_history[student_id] = []
    
    interaction = {
        'timestamp': datetime.now().isoformat(),
        'exercise_type': exercise_type,
        'difficulty': difficulty,
        'steps_completed': steps_completed,
        'mistakes': mistakes,
        'help_requested': help_requested,
        'time_taken': time_taken,
        'completed': completed
    }
    
    student_history[student_id].append(interaction)
    
    # Reentrenar el modelo cada cierto número de interacciones
    if len(student_history[student_id]) % 5 == 0:
        train_recommendation_model(student_id)

# Rutas de la API
@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'ok', 'message': 'API del Sistema Tutor Inteligente funcionando correctamente'})

@app.route('/api/exercise-types', methods=['GET'])
def get_exercise_types():
    types = {
        'balanza': 'Relaciones de equivalencia (ejercicios de balanzas)',
        'progresion_aritmetica': 'Progresiones aritméticas',
        'ecuacion_lineal': 'Ecuaciones lineales',
        'funcion_lineal': 'Funciones lineales',
        'proporcionalidad': 'Proporcionalidad directa e inversa',
        'desigualdades': 'Desigualdades',
        'constante_proporcionalidad': 'Constante de proporcionalidad'
    }
    
    difficulty_levels = {
        1: 'Principiante',
        2: 'Intermedio',
        3: 'Avanzado'
    }
    
    return jsonify({
        'exercise_types': types,
        'difficulty_levels': difficulty_levels
    })

@app.route('/api/exercise/generate', methods=['POST'])
def generate_exercise():
    data = request.json
    
    # Validar datos de entrada
    if not data or 'exercise_type' not in data or 'difficulty_level' not in data:
        return jsonify({'error': 'Datos incompletos. Se requiere tipo de ejercicio y nivel de dificultad'}), 400
    
    exercise_type = data['exercise_type']
    difficulty_level = int(data['difficulty_level'])
    
    # Validar tipo de ejercicio
    if exercise_type not in exercise_types:
        return jsonify({'error': f'Tipo de ejercicio no válido. Opciones: {", ".join(exercise_types.keys())}'}), 400
    
    # Validar nivel de dificultad
    if difficulty_level not in [1, 2, 3]:
        return jsonify({'error': 'Nivel de dificultad no válido. Opciones: 1 (Principiante), 2 (Intermedio), 3 (Avanzado)'}), 400
    
    # Generar ejercicio
    difficulty_names = {1: 'principiante', 2: 'intermedio', 3: 'avanzado'}
    generator = exercise_types[exercise_type]
    
    try:
        # Generar el ejercicio
        exercise = generator.generate_exercise(difficulty_names[difficulty_level])
        
        # Convertir objeto exercise a un formato serializable para JSON
        exercise_json = json.loads(json.dumps(exercise, default=lambda o: str(o) if isinstance(o, (Fraction, complex)) else o.__dict__ if hasattr(o, '__dict__') else repr(o)))
        
        # Incluir metadatos adicionales
        response = {
            'exercise_id': random.randint(10000, 99999),  # En una implementación real, este sería un ID único generado por la base de datos
            'exercise_type': exercise_type,
            'difficulty_level': difficulty_level,
            'difficulty_name': difficulty_names[difficulty_level],
            'exercise_data': exercise_json,
            'exercise_text': get_exercise_text(exercise_type, exercise)
        }
        
        return jsonify(response)
    except Exception as e:
        return jsonify({'error': f'Error generando ejercicio: {str(e)}'}), 500

def get_exercise_text(exercise_type, exercise):
    """Obtener texto descriptivo del ejercicio para mostrar en la web"""
    # Esta función simplificada devuelve un texto básico según el tipo de ejercicio
    # En una implementación real, esto dependería del formato específico de cada tipo de ejercicio
    
    exercise_texts = {
        'balanza': f"Encuentra el valor de la caja misteriosa en la balanza en equilibrio.",
        'progresion_aritmetica': f"Encuentra el término faltante en la progresión aritmética.",
        'ecuacion_lineal': f"Resuelve la ecuación lineal.",
        'funcion_lineal': f"Encuentra la función lineal que pasa por los puntos dados.",
        'proporcionalidad': f"Resuelve el problema de proporcionalidad {'directa' if exercise.get('is_direct', True) else 'inversa'}.",
        'desigualdades': f"Resuelve la desigualdad y encuentra un valor entero que la satisfaga.",
        'constante_proporcionalidad': f"Calcula la constante de proporcionalidad y escribe la fórmula."
    }
    
    return exercise_texts.get(exercise_type, "Resuelve el ejercicio.")

@app.route('/api/exercise/submit', methods=['POST'])
def submit_exercise():
    data = request.json
    
    # Validar datos de entrada
    required_fields = ['student_id', 'exercise_id', 'exercise_type', 'difficulty_level', 
                      'user_answer', 'time_taken', 'steps_completed', 'mistakes', 'help_requested']
    
    for field in required_fields:
        if field not in data:
            return jsonify({'error': f'Campo requerido faltante: {field}'}), 400
    
    # Procesar la respuesta
    # En una implementación real, esto verificaría la respuesta contra el ejercicio guardado
    # Por ahora, simularemos una evaluación
    
    exercise_type = data['exercise_type']
    difficulty_level = int(data['difficulty_level'])
    user_answer = data['user_answer']
    
    # Simular evaluación (en implementación real, esto verificaría contra la solución real)
    success_probability = random.random()
    is_correct = success_probability > 0.3  # 70% de probabilidad de ser correcto para la simulación
    
    # Registrar la interacción
    track_interaction(
        student_id=data['student_id'],
        exercise_type=exercise_type,
        difficulty=difficulty_level,
        steps_completed=data['steps_completed'],
        mistakes=data['mistakes'],
        help_requested=data['help_requested'],
        time_taken=data['time_taken'],
        completed=is_correct
    )
    
    # Preparar retroalimentación
    feedback = {
        'is_correct': is_correct,
        'correct_answer': generate_correct_answer(exercise_type),  # Simulado
        'explanation': generate_explanation(exercise_type, is_correct),  # Simulado
        'next_steps': suggest_next_steps(exercise_type, is_correct, difficulty_level)  # Simulado
    }
    
    return jsonify({
        'result': 'success' if is_correct else 'incorrect',
        'feedback': feedback
    })

def generate_correct_answer(exercise_type):
    """Genera una respuesta correcta simulada basada en el tipo de ejercicio"""
    # En una implementación real, esto retornaría la solución real del ejercicio
    answers = {
        'balanza': "El valor de la caja misteriosa es 7",
        'progresion_aritmetica': "El término faltante es 15",
        'ecuacion_lineal': "x = 3",
        'funcion_lineal': "f(x) = 2x + 5",
        'proporcionalidad': "y = 4x",
        'desigualdades': "x > 2",
        'constante_proporcionalidad': "k = 2.5"
    }
    return answers.get(exercise_type, "Respuesta correcta")

def generate_explanation(exercise_type, is_correct):
    """Genera una explicación simulada basada en el tipo de ejercicio y si es correcto"""
    if is_correct:
        explanations = {
            'balanza': "Correcto. Calculaste el valor de la caja despejando correctamente la ecuación de equilibrio.",
            'progresion_aritmetica': "Correcto. Identificaste la diferencia común y calculaste el término faltante.",
            'ecuacion_lineal': "Correcto. Despejaste correctamente la variable y encontraste el valor de x.",
            'funcion_lineal': "Correcto. Identificaste la pendiente y el corte con el eje y de la función.",
            'proporcionalidad': "Correcto. Aplicaste correctamente el concepto de proporcionalidad.",
            'desigualdades': "Correcto. Resolviste la desigualdad y encontraste un valor válido.",
            'constante_proporcionalidad': "Correcto. Calculaste la constante de proporcionalidad y escribiste la fórmula."
        }
    else:
        explanations = {
            'balanza': "Incorrecto. Recuerda que en una balanza en equilibrio, ambos lados tienen el mismo peso.",
            'progresion_aritmetica': "Incorrecto. Verifica que estás calculando la diferencia común correctamente.",
            'ecuacion_lineal': "Incorrecto. Revisa los pasos para despejar la variable.",
            'funcion_lineal': "Incorrecto. Verifica el cálculo de la pendiente usando los puntos dados.",
            'proporcionalidad': "Incorrecto. Verifica si se trata de proporcionalidad directa o inversa.",
            'desigualdades': "Incorrecto. Recuerda que al multiplicar por un número negativo, el sentido de la desigualdad cambia.",
            'constante_proporcionalidad': "Incorrecto. Revisa la fórmula para calcular la constante de proporcionalidad."
        }
    
    return explanations.get(exercise_type, "Explicación detallada.")

def suggest_next_steps(exercise_type, is_correct, difficulty_level):
    """Sugiere pasos siguientes basados en el resultado"""
    if is_correct:
        if difficulty_level < 3:
            return f"¡Buen trabajo! Puedes intentar un ejercicio de nivel {difficulty_level + 1} para desafiarte más."
        else:
            next_topics = concept_map.get(exercise_type, [])
            if next_topics:
                return f"¡Excelente! Has dominado este tema. Puedes avanzar a un tema relacionado como {', '.join(next_topics)}."
            else:
                return "¡Excelente! Has dominado este tema. Puedes elegir otro tema para seguir avanzando."
    else:
        return "Revisa la explicación y practica más con ejercicios similares para reforzar este concepto."

@app.route('/api/recommendation', methods=['POST'])
def get_student_recommendation():
    data = request.json
    
    # Validar datos de entrada
    if not data or 'student_id' not in data:
        return jsonify({'error': 'ID de estudiante requerido'}), 400
    
    student_id = data['student_id']
    
    # Si hay datos adicionales de rendimiento actual, usarlos
    current_performance = data.get('current_performance', None)
    
    # Si no hay datos de rendimiento actual, usar el último ejercicio del historial
    if not current_performance:
        if student_id not in student_history or not student_history[student_id]:
            # Si no hay historial, recomendar ejercicio básico
            recommendation = {
                'exercise_type': 'balanza',
                'difficulty': 1,
                'success_probability': 0.9,
                'rationale': "Recomendamos comenzar con ejercicios básicos de balanza para familiarizarte con el sistema."
            }
        else:
            # Usar el último ejercicio del historial
            last_exercise = student_history[student_id][-1]
            recommendation = get_recommendation(student_id, last_exercise)
    else:
        # Usar los datos de rendimiento proporcionados
        recommendation = get_recommendation(student_id, current_performance)
    
    return jsonify(recommendation)

@app.route('/api/student/history', methods=['GET'])
def get_student_history():
    student_id = request.args.get('student_id')
    
    if not student_id:
        return jsonify({'error': 'ID de estudiante requerido'}), 400
    
    if student_id not in student_history:
        return jsonify({'student_id': student_id, 'history': []})
    
    # Calcular estadísticas
    history = student_history[student_id]
    
    total_exercises = len(history)
    completed_exercises = sum(1 for item in history if item['completed'])
    success_rate = (completed_exercises / total_exercises) * 100 if total_exercises > 0 else 0
    
    # Analizar progreso por tipo de ejercicio
    progress_by_type = {}
    for exercise in history:
        ex_type = exercise['exercise_type']
        if ex_type not in progress_by_type:
            progress_by_type[ex_type] = {'total': 0, 'completed': 0}
        
        progress_by_type[ex_type]['total'] += 1
        if exercise['completed']:
            progress_by_type[ex_type]['completed'] += 1
    
    # Calcular tasa de éxito por tipo
    for ex_type in progress_by_type:
        total = progress_by_type[ex_type]['total']
        completed = progress_by_type[ex_type]['completed']
        progress_by_type[ex_type]['success_rate'] = (completed / total) * 100 if total > 0 else 0
    
    return jsonify({
        'student_id': student_id,
        'total_exercises': total_exercises,
        'completed_exercises': completed_exercises,
        'success_rate': success_rate,
        'progress_by_type': progress_by_type,
        'history': history  # Historial completo (podría ser paginado en una implementación real)
    })

@app.route('/api/student/progress', methods=['GET'])
def get_student_progress():
    student_id = request.args.get('student_id')
    
    if not student_id:
        return jsonify({'error': 'ID de estudiante requerido'}), 400
    
    if student_id not in student_history:
        return jsonify({'student_id': student_id, 'message': 'No hay datos de progreso disponibles'})
    
    # Analizar progreso por tipo de ejercicio y nivel
    progress = {}
    for ex_type in exercise_types:
        progress[ex_type] = {
            'name': get_exercise_display_name(ex_type),
            'levels': {
                1: {'total': 0, 'completed': 0, 'success_rate': 0},
                2: {'total': 0, 'completed': 0, 'success_rate': 0},
                3: {'total': 0, 'completed': 0, 'success_rate': 0}
            },
            'total': 0,
            'completed': 0,
            'success_rate': 0,
            'mastery_level': 0  # 0: No iniciado, 1-3: Nivel alcanzado
        }
    
    # Procesar historial
    for exercise in student_history[student_id]:
        ex_type = exercise['exercise_type']
        difficulty = exercise['difficulty']
        
        if ex_type in progress:
            # Incrementar contadores por nivel
            progress[ex_type]['levels'][difficulty]['total'] += 1
            if exercise['completed']:
                progress[ex_type]['levels'][difficulty]['completed'] += 1
            
            # Incrementar contadores totales
            progress[ex_type]['total'] += 1
            if exercise['completed']:
                progress[ex_type]['completed'] += 1
    
    # Calcular tasas de éxito y nivel de dominio
    for ex_type in progress:
        # Calcular tasas de éxito por nivel
        for level in progress[ex_type]['levels']:
            level_data = progress[ex_type]['levels'][level]
            level_data['success_rate'] = (level_data['completed'] / level_data['total'] * 100) if level_data['total'] > 0 else 0
        
        # Calcular tasa de éxito total
        total = progress[ex_type]['total']
        completed = progress[ex_type]['completed']
        progress[ex_type]['success_rate'] = (completed / total * 100) if total > 0 else 0
        
        # Determinar nivel de dominio (básico: nivel con al menos 3 ejercicios completados y 70% de éxito)
        mastery_level = 0
        for level in [3, 2, 1]:  # Empezar desde el nivel más alto
            level_data = progress[ex_type]['levels'][level]
            if level_data['completed'] >= 3 and level_data['success_rate'] >= 70:
                mastery_level = level
                break
        
        progress[ex_type]['mastery_level'] = mastery_level
    
    return jsonify({
        'student_id': student_id,
        'progress': progress
    })

def get_exercise_display_name(exercise_type):
    """Obtener nombre de visualización para cada tipo de ejercicio"""
    display_names = {
        'balanza': 'Relaciones de equivalencia',
        'progresion_aritmetica': 'Progresiones aritméticas',
        'ecuacion_lineal': 'Ecuaciones lineales',
        'funcion_lineal': 'Funciones lineales',
        'proporcionalidad': 'Proporcionalidad',
        'desigualdades': 'Desigualdades',
        'constante_proporcionalidad': 'Constante de proporcionalidad'
    }
    return display_names.get(exercise_type, exercise_type)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)