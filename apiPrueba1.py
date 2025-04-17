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
