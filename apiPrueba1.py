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



app = Flask(__name__)

# Definimos los temas disponibles
temas = [
    "Relaciones de equivalencia (ejercicios de balanzas)",
    "Progresiones aritméticas",
    "Ecuaciones lineales",
    "Funciones lineales",
    "Proporcionalidad directa e inversa",
    "Desigualdades",
    "Constante de proporcionalidad",
    "Recomendación personalizada",
]

def _run_exercise_session(self, topic, difficulty_level):
    """Ejecuta una sesión de ejercicios del tema y dificultad seleccionados"""
    difficulty_names = {1: 'principiante', 2: 'intermedio', 3: 'avanzado'}
    difficulty = difficulty_names[difficulty_level]

    # Iniciar temporizador para medir tiempo de resolución
    start_time = time.time()

    # Generar ejercicio
    generator = self.exercise_types[topic]
    exercise = generator.generate_exercise(difficulty)

    # Variables para seguimiento del progreso
    steps_completed = 0
    mistakes = 0
    help_requested = 0
    completed = False

    # Presentar y resolver el ejercicio interactivamente
    generator.format_exercise(exercise)
    result = generator.solve_interactive(exercise, track_progress=True)

    # Actualizar variables de seguimiento
    steps_completed = result['steps_completed']
    mistakes = result['mistakes']
    help_requested = result['help_requested']
    completed = result['completed']

    # Calcular tiempo total
    time_taken = time.time() - start_time

    # Registrar interacción
    self.track_interaction(
        exercise_type=topic,
        difficulty=difficulty_level,
        steps_completed=steps_completed,
        mistakes=mistakes,
        help_requested=help_requested,
        time_taken=time_taken,
        completed=completed
    )

    # Mostrar feedback y recomendación
    self._show_feedback(topic, difficulty_level, steps_completed, mistakes, help_requested, time_taken, completed)

    # Preguntar si quiere otro ejercicio del mismo tipo
    print("\n¿Quieres hacer otro ejercicio del mismo tipo? (s/n)")
    if input("> ").lower() == 's':
        self._run_exercise_session(topic, difficulty_level)

@app.route('/temas', methods=['GET'])
def obtener_temas():
    """Retorna la lista de temas disponibles"""
    return jsonify({
        'temas': temas
    })

@app.route('/elegir_tema', methods=['POST'])
def elegir_tema():
    """Permite al usuario elegir un tema basado en el número del tema"""
    data = request.get_json()
    numero_tema = data.get('tema')
    
    if numero_tema and 1 <= numero_tema <= 8:
        # Respondemos con el tema correspondiente
        return jsonify({
            'mensaje': f"Has elegido el tema: {temas[numero_tema - 1]}"
        })
    else:
        # Si el número está fuera de rango
        return jsonify({
            'error': 'Por favor, elige un número válido entre 1 y 8.'
        }), 400

@app.route('/salir', methods=['GET'])
def salir():
    """Mensaje de salida de la API"""
    return jsonify({
        'mensaje': '¡Hasta luego! Gracias por usar la API.'
    })


if __name__ == '__main__':
    app.run(debug=True)
