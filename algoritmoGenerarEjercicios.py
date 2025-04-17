import random
import time
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import joblib
import os
from datetime import datetime
import matplotlib.pyplot as plt
from fractions import Fraction




class IntelligentTutorSystem:
    def __init__(self):
        # Tipos de ejercicios disponibles
        self.exercise_types = {
            'balanza': BalanceExerciseGenerator(),
            'progresion_aritmetica': ArithmeticProgressionGenerator(),
            'ecuacion_lineal': LinearEquationGenerator(),
            'funcion_lineal': LinearFunctionGenerator(),
            'proporcionalidad': ProportionalityGenerator(),
            'desigualdades': InequalityGenerator(),
            'constante_proporcionalidad': ProportionalityConstantGenerator()
        }

        # Modelo de ML para recomendaciones
        self.recommendation_model = self._initialize_recommendation_model()

        # Registro de interacciones del estudiante
        self.student_history = []

        # Mapa de conceptos con dependencias
        self.concept_map = {
            'balanza': ['ecuacion_lineal'],
            'progresion_aritmetica': ['ecuacion_lineal', 'funcion_lineal'],
            'ecuacion_lineal': ['funcion_lineal', 'desigualdades'],
            'funcion_lineal': ['proporcionalidad'],
            'proporcionalidad': ['constante_proporcionalidad'],
            'desigualdades': ['proporcionalidad'],
            'constante_proporcionalidad': []
        }

    def _initialize_recommendation_model(self):
        """Inicializa o carga el modelo de recomendación"""
        model_path = 'tutor_recommendation_model.joblib'

        if os.path.exists(model_path):
            try:
                return joblib.load(model_path)
            except:
                pass

        # Crear un modelo inicial
        return RandomForestClassifier(n_estimators=50)

    def save_model(self):
        """Guarda el modelo de recomendación entrenado"""
        if len(self.student_history) > 10:  # Solo guardar si hay suficientes datos
            try:
                joblib.dump(self.recommendation_model, 'tutor_recommendation_model.joblib')
                print("Modelo de recomendación guardado correctamente.")
            except Exception as e:
                print(f"Error al guardar el modelo: {e}")

    def track_interaction(self, exercise_type, difficulty, steps_completed, mistakes, help_requested, time_taken, completed):
        """Registra la interacción del estudiante para entrenar el modelo"""
        interaction = {
            'timestamp': datetime.now(),
            'exercise_type': exercise_type,
            'difficulty': difficulty,
            'steps_completed': steps_completed,
            'mistakes': mistakes,
            'help_requested': help_requested,
            'time_taken': time_taken,
            'completed': completed
        }

        self.student_history.append(interaction)

        # Reentrenar el modelo cada cierto número de interacciones
        if len(self.student_history) % 5 == 0:
            self._train_recommendation_model()

    def _train_recommendation_model(self):
        """Entrena el modelo de recomendación con el historial del estudiante"""
        if len(self.student_history) < 5:
            return  # No hay suficientes datos para entrenar

        try:
            # Preparar datos para entrenamiento
            df = pd.DataFrame(self.student_history)

            # Características para el modelo
            X = df[['difficulty', 'steps_completed', 'mistakes', 'help_requested', 'time_taken']].values

            # Etiquetas: 1 si completó con éxito, 0 si no
            y = df['completed'].astype(int).values

            # Entrenar modelo
            self.recommendation_model.fit(X, y)
            print("Modelo de recomendación actualizado.")
        except Exception as e:
            print(f"Error al entrenar el modelo: {e}")

    def get_recommendation(self, student_performance):
        """Genera una recomendación basada en el rendimiento del estudiante"""
        if len(self.student_history) < 5:
            # Si no hay suficientes datos, usar recomendación basada en reglas
            return self._rule_based_recommendation(student_performance)

        try:
            # Convertir el rendimiento actual a un formato para el modelo
            features = np.array([[
                student_performance['difficulty'],
                student_performance['steps_completed'],
                student_performance['mistakes'],
                student_performance['help_requested'],
                student_performance['time_taken']
            ]])

            # Predecir probabilidad de éxito con diversos tipos de ejercicios
            success_probs = {}
            current_difficulty = student_performance['difficulty']

            for exercise_type in self.exercise_types:
                # Simular rendimiento en diferentes niveles de dificultad
                for diff_level in range(1, 4):
                    features[0][0] = diff_level
                    prob = self.recommendation_model.predict_proba(features)[0][1]
                    success_probs[(exercise_type, diff_level)] = prob

            # Encontrar el mejor ejercicio recomendado (desafiante pero alcanzable)
            best_recommendation = max(success_probs.items(), key=lambda x:
                                     x[1] if x[1] >= 0.6 else 0)

            return {
                'exercise_type': best_recommendation[0][0],
                'difficulty': best_recommendation[0][1],
                'success_probability': best_recommendation[1],
                'rationale': "Basado en tu rendimiento histórico, este ejercicio debería ser desafiante pero alcanzable."
            }
        except Exception as e:
            print(f"Error al generar recomendación: {e}")
            return self._rule_based_recommendation(student_performance)

    def _rule_based_recommendation(self, performance):
        """Genera recomendación basada en reglas cuando no hay suficientes datos"""
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
            for next_topic, dependencies in self.concept_map.items():
                if current_type in dependencies:
                    return {
                        'exercise_type': next_topic,
                        'difficulty': 1,
                        'success_probability': 0.75,
                        'rationale': f"¡Has dominado este tema! Es un buen momento para explorar {next_topic}, que se basa en lo que ya aprendiste."
                    }

            # Si no hay dependencia clara, recomendar un tema al azar
            next_topic = random.choice(list(self.exercise_types.keys()))
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

    def start_session(self):
        """Inicia una sesión interactiva con el estudiante"""
        print("\n" + "=" * 60)
        print("BIENVENIDO AL SISTEMA TUTOR INTELIGENTE DE MATEMATICAS".center(60))
        print("=" * 60)
        print("\nEste sistema te guiara en tu aprendizaje de matematicas,")
        print("adaptandose a tu nivel y recomendando ejercicios personalizados.")

        # Si hay historial, dar la bienvenida personalizada
        if self.student_history:
            print(f"\n¡Bienvenido de nuevo! Has completado {len(self.student_history)} ejercicios.")

            # Calcular estadísticas
            completed = sum(1 for item in self.student_history if item['completed'])
            success_rate = completed / len(self.student_history) * 100
            print(f"Tu tasa de éxito es del {success_rate:.2f}%")

        while True:
            # Mostrar opciones de temas
            print("\n TEMAS DISPONIBLES")
            print("1. Relaciones de equivalencia (ejercicios de balanzas)")
            print("2. Progresiones aritmeticas")
            print("3. Ecuaciones lineales")
            print("4. Funciones lineales")
            print("5. Proporcionalidad directa e inversa")
            print("6. Desigualdades")
            print("7. Constante de proporcionalidad")
            print("8. Recomendacion personalizada")
            print("9. Salir")

            choice = input("\n> Selecciona una opcion: ")

            if choice == '9':
                self.save_model()
                print("\n¡Gracias por usar el Sistema Tutor Inteligente! ¡Hasta pronto! 👋")
                break

            topic_map = {
                '1': 'balanza',
                '2': 'progresion_aritmetica',
                '3': 'ecuacion_lineal',
                '4': 'funcion_lineal',
                '5': 'proporcionalidad',
                '6': 'desigualdades',
                '7': 'constante_proporcionalidad'
            }

            # Si se solicita recomendación personalizada
            if choice == '8':
                if not self.student_history:
                    print("\n⚠️ Aún no tenemos suficientes datos para darte una recomendación personalizada.")
                    print("Te recomendamos comenzar con ejercicios de balanza en nivel principiante.")
                    choice = '1'
                    difficulty_level = 1
                else:
                    # Usar último rendimiento para recomendar
                    last_performance = self.student_history[-1]
                    recommendation = self.get_recommendation(last_performance)

                    print("\n🔮 RECOMENDACIÓN PERSONALIZADA 🔮")
                    print(f"Te recomendamos practicar: {recommendation['exercise_type'].replace('_', ' ').title()}")
                    print(f"Nivel de dificultad: {recommendation['difficulty']}")
                    print(f"Probabilidad de éxito estimada: {recommendation['success_probability']*100:.2f}%")
                    print(f"Razón: {recommendation['rationale']}")

                    print("\n¿Deseas seguir esta recomendación? (s/n)")
                    if input("> ").lower() == 's':
                        topic = recommendation['exercise_type']
                        difficulty_level = recommendation['difficulty']
                    else:
                        continue
            else:
                # Si eligió un tema específico
                if choice not in topic_map:
                    print("\n⚠️ Opción no válida. Por favor, intenta de nuevo.")
                    continue

                topic = topic_map[choice]

                # Seleccionar nivel de dificultad
                print("\nSelecciona el nivel de dificultad:")
                print("1. Principiante")
                print("2. Intermedio")
                print("3. Avanzado")

                try:
                    difficulty_level = int(input("> "))
                    if difficulty_level < 1 or difficulty_level > 3:
                        raise ValueError()
                except ValueError:
                    print("\n⚠️ Nivel no válido. Estableciendo nivel principiante.")
                    difficulty_level = 1

            # Iniciar ejercicio del tema seleccionado
            self._run_exercise_session(topic, difficulty_level)

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

    def _show_feedback(self, topic, difficulty, steps, mistakes, help_requests, time_taken, completed):
        """Muestra feedback personalizado al estudiante"""
        difficulty_names = {1: 'principiante', 2: 'intermedio', 3: 'avanzado'}

        print("\n" + "=" * 60)
        print("📊 ANÁLISIS DE TU DESEMPEÑO 📊".center(60))
        print("=" * 60)

        # Calificación simbólica
        if mistakes == 0 and help_requests == 0:
            rating = "⭐⭐⭐⭐⭐"
        elif mistakes <= 1 and help_requests <= 1:
            rating = "⭐⭐⭐⭐"
        elif mistakes <= 2 and help_requests <= 2:
            rating = "⭐⭐⭐"
        elif completed:
            rating = "⭐⭐"
        else:
            rating = "⭐"

        print(f"\nCalificación: {rating}")
        print(f"Tiempo empleado: {time_taken:.1f} segundos")
        print(f"Pasos completados: {steps}")
        print(f"Errores cometidos: {mistakes}")
        print(f"Ayudas solicitadas: {help_requests}")

        # Comentarios personalizados
        if completed and mistakes == 0:
            print("\n👏 ¡Excelente trabajo! Dominas este concepto.")
        elif completed and mistakes <= 2:
            print("\n👍 Buen trabajo. Estás comprendiendo el concepto, pero aún hay espacio para mejorar.")
        elif completed:
            print("\n🙂 Has completado el ejercicio, pero necesitas más práctica para consolidar este concepto.")
        else:
            print("\n🤔 Este concepto parece desafiante para ti. Sigamos practicando con más ejercicios.")

        # Mostrar recomendación
        performance = {
            'exercise_type': topic,
            'difficulty': difficulty,
            'steps_completed': steps,
            'mistakes': mistakes,
            'help_requested': help_requests,
            'time_taken': time_taken,
            'completed': completed
        }

        recommendation = self.get_recommendation(performance)

        print("\n📝 RECOMENDACIÓN PARA TU PRÓXIMO PASO:")
        print(f"• {recommendation['rationale']}")

        # Ofrecer contenido adicional relacionado
        print("\n📚 CONTENIDO RECOMENDADO:")
        if mistakes > 2:
            print("• Revisa los conceptos básicos sobre este tema")
            print(f"• Practica más ejercicios de nivel {difficulty_names[difficulty]}")
        else:
            print("• Explora ejemplos más complejos de este tema")
            if topic in self.concept_map and self.concept_map[topic]:
                next_topics = ", ".join(self.concept_map[topic])
                print(f"• Prepárate para avanzar hacia: {next_topics}")


class ExerciseGenerator:
    """Clase base para todos los generadores de ejercicios"""
    def __init__(self):
        self.difficulty_levels = {
            'principiante': {'max_value': 10, 'operations': ['suma', 'resta']},
            'intermedio': {'max_value': 20, 'operations': ['suma', 'resta', 'multiplicacion']},
            'avanzado': {'max_value': 50, 'operations': ['suma', 'resta', 'multiplicacion', 'division']}
        }

    def generate_exercise(self, level):
        """Método que debe ser implementado por las subclases"""
        raise NotImplementedError

    def format_exercise(self, exercise):
        """Método que debe ser implementado por las subclases"""
        raise NotImplementedError

    def solve_interactive(self, exercise, track_progress=False):
        """Método que debe ser implementado por las subclases"""
        raise NotImplementedError


class BalanceExerciseGenerator(ExerciseGenerator):
    def generate_exercise(self, level='principiante'):
        """Genera un ejercicio de balanza basado en el nivel de dificultad."""
        config = self.difficulty_levels.get(level, self.difficulty_levels['principiante'])

        # Generar valores para el ejercicio
        if level == 'principiante':
            x = random.randint(1, config['max_value'])
            weight1 = random.randint(1, 5)
            operation = random.choice(config['operations'])

            if operation == 'suma':
                weight2 = x + weight1
            else:  # resta
                weight2 = x + weight1
                x, weight1 = weight1, x  # Intercambiar para que la resta tenga sentido

        elif level == 'intermedio':
            x = random.randint(1, config['max_value'])
            weight1 = random.randint(1, 10)
            operation = random.choice(config['operations'])

            if operation == 'suma':
                weight2 = x + weight1
            elif operation == 'resta':
                weight2 = x + weight1
                x, weight1 = weight1, x  # Intercambiar para que la resta tenga sentido
            else:  # multiplicacion
                factor = random.randint(2, 4)
                weight2 = x * factor
                weight1 = (factor - 1) * x

        else:  # avanzado
            x = random.randint(1, config['max_value'])
            operation = random.choice(config['operations'])

            if operation == 'suma':
                weight1 = random.randint(1, 15)
                weight2 = x + weight1
            elif operation == 'resta':
                weight1 = random.randint(1, 15)
                weight2 = x + weight1
                x, weight1 = weight1, x  # Intercambiar para que la resta tenga sentido
            elif operation == 'multiplicacion':
                factor = random.randint(2, 5)
                weight2 = x * factor
                weight1 = (factor - 1) * x
            else:  # division
                divisor = random.randint(2, 5)
                x = random.randint(1, 10) * divisor  # Asegurar que x sea divisible
                weight2 = x // divisor + random.randint(1, 5)
                weight1 = weight2 * divisor - x

        # Crear el ejercicio con formato
        exercise = {
            'level': level,
            'mystery_box_weight': x,
            'left_side': {'mystery_box': True, 'weight1': weight1},
            'right_side': {'weight': weight2},
            'operation': operation
        }

        return exercise

    def format_exercise(self, exercise):
        """Formatea el ejercicio para mostrarlo al usuario."""
        level = exercise['level']
        x = exercise['mystery_box_weight']
        weight1 = exercise['left_side']['weight1']
        weight2 = exercise['right_side']['weight']
        operation = exercise['operation']

        print("\n" + "=" * 60)
        print(f"🧪 EJERCICIO DE BALANZA - NIVEL: {level.upper()} 🧪".center(60))
        print("=" * 60)
        print("\n📝 Ejercicio: Descubriendo el valor de una caja misteriosa 🎁")
        print("\nTienes una balanza en equilibrio:")
        print(f"• A un lado hay una caja misteriosa (X) y un bloque de {weight1} kg.")
        print(f"• Al otro lado hay un bloque de {weight2} kg.")

        # Mostrar una representación visual simple
        print("\n" + "-" * 40)
        print("   [X]   ", end="")
        print(f"[{weight1}]", end="")
        print("      =      ", end="")
        print(f"[{weight2}]")
        print("-" * 40)

        print("\n💎 ¿Cuánto pesa la caja misteriosa?")

        return exercise

    def solve_interactive(self, exercise, track_progress=False):
        """Permite al usuario resolver el ejercicio interactivamente con seguimiento."""
        # Variables para tracking
        steps_completed = 0
        mistakes = 0
        help_requested = 0
        completed = False

        x = exercise['mystery_box_weight']
        weight1 = exercise['left_side']['weight1']
        weight2 = exercise['right_side']['weight']
        level = exercise['level']

        # Paso 1: Identificar la operación
        print("\n🔍 PASO 1: ¿Qué operación debes hacer para encontrar el valor de X?")

        options = []
        if level == 'principiante':
            options = [
                f"A) Sumar {weight1} a {weight2}",
                f"B) Restar {weight1} de {weight2}",
            ]
        elif level == 'intermedio':
            options = [
                f"A) Sumar {weight1} a {weight2}",
                f"B) Restar {weight1} de {weight2}",
                f"C) Multiplicar {weight1} por {weight2 // weight1} (si es posible)"
            ]
        else:  # avanzado
            options = [
                f"A) Sumar {weight1} a {weight2}",
                f"B) Restar {weight1} de {weight2}",
                f"C) Multiplicar {weight1} por un factor",
                f"D) Dividir {weight2} entre un factor"
            ]

        print("\n✅ Opciones:")
        for option in options:
            print(option)

        # Determinar la respuesta correcta
        if exercise['operation'] == 'suma':
            correct_option = "B"  # Restar weight1 de weight2
        elif exercise['operation'] == 'resta':
            correct_option = "B"  # Restar weight1 de weight2
        elif exercise['operation'] == 'multiplicacion':
            correct_option = "C"
        else:  # division
            correct_option = "D"

        # Pedir respuesta al usuario
        print("\n¿Cuál es tu respuesta? (A, B, C, D o 'ayuda' para una pista)")
        user_answer = input("> ").upper()

        if user_answer == 'AYUDA':
            help_requested += 1
            print("\n🔔 PISTA: Recuerda que en una balanza en equilibrio, el peso de ambos lados es igual.")
            print("Por lo tanto, si X + weight1 = weight2, entonces X = weight2 - weight1.")
            print("\n¿Cuál es tu respuesta ahora?")
            user_answer = input("> ").upper()

        if user_answer == correct_option:
            print("\n✅ ¡Correcto! Esa es la operación adecuada.")
            steps_completed += 1
        else:
            print(f"\n❌ No es correcto. La respuesta correcta es la opción {correct_option}.")
            mistakes += 1

            # Explicación según la operación
            if exercise['operation'] in ['suma', 'resta']:
                print(f"\nExplicación: Si X + {weight1} = {weight2}, entonces X = {weight2} - {weight1}")
            elif exercise['operation'] == 'multiplicacion':
                factor = weight2 // x
                print(f"\nExplicación: Aquí hay una relación multiplicativa. X × {factor} = {weight2} + {weight1}")
            else:  # division
                divisor = (weight2 + weight1) // x
                print(f"\nExplicación: Aquí hay una relación de división. X = ({weight2} + {weight1}) ÷ {divisor}")

        # Paso 2: Resolver la ecuación
        print("\n🔍 PASO 2: Resuelve la ecuación para encontrar el valor de X")

        if exercise['operation'] in ['suma', 'resta']:
            print(f"\nSi X + {weight1} = {weight2}, entonces X = {weight2} - {weight1}")
        elif exercise['operation'] == 'multiplicacion':
            factor = weight2 // x
            print(f"\nSi X × {factor} = {weight2}, entonces X = {weight2} ÷ {factor}")
        else:  # division
            divisor = (weight2 + weight1) // x
            print(f"\nSi X ÷ {divisor} = {weight2}, entonces X = {weight2} × {divisor}")

        print("\n¿Cuál es el valor de X? (o escribe 'ayuda' para una pista)")
        user_answer = input("> ").lower()

        if user_answer == 'ayuda':
            help_requested += 1
            print("\n🔔 PISTA: Sustituye los valores en la ecuación y resuelve paso a paso.")

            if exercise['operation'] in ['suma', 'resta']:
                print(f"X = {weight2} - {weight1} = {weight2 - weight1}")
            elif exercise['operation'] == 'multiplicacion':
                factor = weight2 // x
                print(f"X = {weight2} ÷ {factor} = {weight2 // factor}")
            else:  # division
                divisor = (weight2 + weight1) // x
                print(f"X = {weight2} × {divisor} = {weight2 * divisor}")

            print("\n¿Cuál es tu respuesta ahora?")
            user_answer = input("> ")

        try:
            user_value = float(user_answer)

            # Verificar respuesta con margen de error para decimales
            if abs(user_value - x) < 0.001:
                print("\n✅ ¡Correcto! El valor de X es " + str(x))
                steps_completed += 1
                completed = True
            else:
                print(f"\n❌ No es correcto. El valor de X es {x}")
                mistakes += 1

                # Mostrar solución paso a paso
                print("\nVeamos la solución paso a paso:")

                if exercise['operation'] in ['suma', 'resta']:
                    print(f"1. Tenemos la ecuación: X + {weight1} = {weight2}")
                    print(f"2. Despejamos X: X = {weight2} - {weight1}")
                    print(f"3. Calculamos: X = {weight2} - {weight1} = {x}")
                elif exercise['operation'] == 'multiplicacion':
                    factor = weight2 // x
                    print(f"1. Tenemos una relación multiplicativa: X × {factor} = {weight2}")
                    print(f"2. Despejamos X: X = {weight2} ÷ {factor}")
                    print(f"3. Calculamos: X = {weight2} ÷ {factor} = {x}")
                else:  # division
                    divisor = (weight2 + weight1) // x
                    print(f"1. Tenemos una relación de división: X ÷ {divisor} = {weight2}")
                    print(f"2. Despejamos X: X = {weight2} × {divisor}")
                    print(f"3. Calculamos: X = {weight2} × {divisor} = {x}")

                # Dar otra oportunidad
                print("\n¿Quieres intentarlo una vez más? (s/n)")
                if input("> ").lower() == 's':
                    print("\n¿Cuál es el valor de X?")
                    second_try = float(input("> "))

                    if abs(second_try - x) < 0.001:
                        print("\n✅ ¡Correcto en el segundo intento! El valor de X es " + str(x))
                        steps_completed += 1
                        completed = True
                    else:
                        print(f"\n❌ No es correcto. El valor de X es {x}")
        except ValueError:
            print("\n⚠️ Por favor, ingresa un número válido.")
            mistakes += 1

        # Paso 3: Verificación (opcional para niveles intermedio y avanzado)
        if level in ['intermedio', 'avanzado'] and completed:
            print("\n🔍 PASO 3 (Opcional): Verifica tu respuesta")
            print("\n¿Quieres comprobar tu respuesta? (s/n)")
            if input("> ").lower() == 's':
                print("\n📊 VERIFICACIÓN:")
                print(f"1. Si ponemos X = {x} en la balanza:")
                print(f"2. Lado izquierdo: X + {weight1} = {x} + {weight1} = {x + weight1}")
                print(f"3. Lado derecho: {weight2}")
                print(f"4. Como {x + weight1} = {weight2}, la balanza está equilibrada. ✓")
                steps_completed += 1

        # Resultados del ejercicio
        print("\n" + "=" * 60)
        print("🏆 RESULTADO DEL EJERCICIO 🏆".center(60))
        print("=" * 60)

        if completed:
            print("\n¡Felicidades! Has completado el ejercicio correctamente.")
        else:
            print("\nSigue practicando. Recuerda que cada error es una oportunidad para aprender.")

        # Devolver estadísticas si se solicita seguimiento
        if track_progress:
            return {
                'steps_completed': steps_completed,
                'mistakes': mistakes,
                'help_requested': help_requested,
                'completed': completed
            }

        return None


class ArithmeticProgressionGenerator(ExerciseGenerator):
    def generate_exercise(self, level='principiante'):
        """Genera un ejercicio de progresión aritmética basado en el nivel de dificultad."""
        config = self.difficulty_levels.get(level, self.difficulty_levels['principiante'])

        # Generar valores para el ejercicio
        if level == 'principiante':
            # Progresión simple con diferencia común entre 1 y 3
            common_diff = random.randint(1, 3)
            first_term = random.randint(1, 5)
            terms_count = 5
            missing_position = random.randint(2, 4)  # No el primero ni el último

        elif level == 'intermedio':
            # Progresión con diferencia común mayor y más términos
            common_diff = random.randint(2, 5)
            first_term = random.randint(1, 10)
            terms_count = 6
            missing_position = random.randint(2, 5)

        else:  # avanzado
            # Progresión más compleja, posiblemente con diferencia común negativa
            common_diff = random.randint(-5, 10)
            if common_diff == 0:  # Evitar diferencia cero
                common_diff = 1
            first_term = random.randint(-10, 20)
            terms_count = 7
            missing_position = random.randint(2, 6)

        # Generar la progresión completa
        progression = [first_term + i * common_diff for i in range(terms_count)]
        missing_value = progression[missing_position - 1]  # El valor que falta

        # Crear una copia con el valor faltante como None
        display_progression = progression.copy()
        display_progression[missing_position - 1] = None

        # Crear el ejercicio
        exercise = {
            'level': level,
            'progression': display_progression,
            'complete_progression': progression,
            'common_diff': common_diff,
            'missing_position': missing_position,
            'missing_value': missing_value
        }

        return exercise

    def format_exercise(self, exercise):
        """Formatea el ejercicio para mostrarlo al usuario."""
        level = exercise['level']
        progression = exercise['progression']

        print("\n" + "=" * 60)
        print(f"🧪 EJERCICIO DE PROGRESIÓN ARITMÉTICA - NIVEL: {level.upper()} 🧪".center(60))
        print("=" * 60)
        print("\n📝 Ejercicio: Encuentra el término faltante en la progresión aritmética")

        # Mostrar la progresión con el término faltante como '?'
        progression_display = [str(term) if term is not None else '?' for term in progression]
        print("\nProgresión: " + ", ".join(progression_display))

        print("\n💎 ¿Cuál es el valor del término faltante?")

        return exercise

    def solve_interactive(self, exercise, track_progress=False):
        """Permite al usuario resolver el ejercicio interactivamente con seguimiento."""
        # Variables para tracking
        steps_completed = 0
        mistakes = 0
        help_requested = 0
        completed = False

        progression = exercise['progression']
        complete_progression = exercise['complete_progression']
        common_diff = exercise['common_diff']
        missing_position = exercise['missing_position']
        missing_value = exercise['missing_value']
        level = exercise['level']

        # Paso 1: Identificar la diferencia común
        print("\n🔍 PASO 1: Encuentra la diferencia común de la progresión aritmética")

        print("\n¿Cuál es la diferencia común? (o escribe 'ayuda' para una pista)")
        user_answer = input("> ").lower()

        if user_answer == 'ayuda':
            help_requested += 1
            print("\n🔔 PISTA: La diferencia común es el valor que se suma a cada término para obtener el siguiente.")
            print("Puedes calcularla restando dos términos consecutivos: an+1 - an")

            # Encontrar dos términos consecutivos conocidos
            known_terms = [(i+1, term) for i, term in enumerate(progression) if term is not None]
            if len(known_terms) >= 2:
                for i in range(len(known_terms) - 1):
                    if known_terms[i+1][0] - known_terms[i][0] == 1:  # Son consecutivos
                        pos1, val1 = known_terms[i]
                        pos2, val2 = known_terms[i+1]
                        print(f"Por ejemplo: {val2} - {val1} = {val2 - val1}")
                        break

            print("\n¿Cuál es tu respuesta ahora?")
            user_answer = input("> ")

        try:
            user_diff = int(user_answer)

            if user_diff == common_diff:
                print("\n✅ ¡Correcto! La diferencia común es " + str(common_diff))
                steps_completed += 1
            else:
                print(f"\n❌ No es correcto. La diferencia común es {common_diff}")
                mistakes += 1

                # Mostrar cálculo de la diferencia común
                print("\nVeamos cómo calcular la diferencia común:")

                # Buscar pares de términos consecutivos para mostrar
                for i in range(len(complete_progression) - 1):
                    term1 = complete_progression[i]
                    term2 = complete_progression[i + 1]
                    print(f"Término {i+2} - Término {i+1} = {term2} - {term1} = {term2 - term1}")

                # Dar otra oportunidad
                print("\n¿Quieres intentarlo una vez más? (s/n)")
                if input("> ").lower() == 's':
                    print("\n¿Cuál es la diferencia común?")
                    second_try = int(input("> "))

                    if second_try == common_diff:
                        print("\n✅ ¡Correcto en el segundo intento! La diferencia común es " + str(common_diff))
                        steps_completed += 1
                    else:
                        print(f"\n❌ No es correcto. La diferencia común es {common_diff}")
        except ValueError:
            print("\n⚠️ Por favor, ingresa un número entero válido.")
            mistakes += 1

        # Paso 2: Calcular el valor faltante
        print("\n🔍 PASO 2: Calcula el término faltante")

        # Mostrar la fórmula general según el nivel
        if level == 'principiante':
            print("\nEn una progresión aritmética, puedes encontrar cualquier término usando:")
            print("- Los términos vecinos y la diferencia común")
            print("- La posición del término y la diferencia común")
        else:
            print("\nEn una progresión aritmética, cada término sigue la fórmula:")
            print("an = a1 + (n - 1) × d")
            print("Donde:")
            print("- an es el término en la posición n")
            print("- a1 es el primer término")
            print("- d es la diferencia común")

        print("\n¿Cuál es el valor del término faltante? (o escribe 'ayuda' para una pista)")
        user_answer = input("> ").lower()

        if user_answer == 'ayuda':
            help_requested += 1
            print("\n🔔 PISTA: Puedes usar la fórmula general o analizar los términos vecinos.")

            # Determinar el método más sencillo según los datos disponibles
            if progression[0] is not None:  # Si conocemos el primer término
                a1 = progression[0]
                print(f"Como conocemos el primer término ({a1}) y la diferencia común ({common_diff}),")
                print(f"podemos usar la fórmula: a{missing_position} = {a1} + ({missing_position} - 1) × {common_diff}")
            else:
                # Buscar un término conocido cercano
                known_pos, known_val = next((i+1, val) for i, val in enumerate(progression) if val is not None)
                print(f"Conocemos el término a{known_pos} = {known_val}")
                print(f"Para encontrar a{missing_position}, podemos usar:")
                print(f"a{missing_position} = {known_val} + ({missing_position - known_pos}) × {common_diff}")

            print("\n¿Cuál es tu respuesta ahora?")
            user_answer = input("> ")

        try:
            user_value = int(user_answer)

            if user_value == missing_value:
                print("\n✅ ¡Correcto! El término faltante es " + str(missing_value))
                steps_completed += 1
                completed = True
            else:
                print(f"\n❌ No es correcto. El término faltante es {missing_value}")
                mistakes += 1

                # Mostrar solución paso a paso
                print("\nVeamos la solución paso a paso:")

                # Si conocemos el primer término
                if progression[0] is not None:
                    a1 = progression[0]
                    print(f"1. Usamos la fórmula: an = a1 + (n - 1) × d")
                    print(f"2. Sustituimos: a{missing_position} = {a1} + ({missing_position} - 1) × {common_diff}")
                    print(f"3. Calculamos: a{missing_position} = {a1} + {missing_position - 1} × {common_diff} = {a1 + (missing_position - 1) * common_diff}")
                else:
                    # Usar un término conocido como referencia
                    known_pos, known_val = next((i+1, val) for i, val in enumerate(progression) if val is not None)
                    print(f"1. Conocemos el término a{known_pos} = {known_val}")
                    print(f"2. La diferencia de posiciones es: {missing_position} - {known_pos} = {missing_position - known_pos}")
                    print(f"3. El cambio de valor es: ({missing_position - known_pos}) × {common_diff} = {(missing_position - known_pos) * common_diff}")
                    print(f"4. Por lo tanto: a{missing_position} = {known_val} + {(missing_position - known_pos) * common_diff} = {known_val + (missing_position - known_pos) * common_diff}")

                # Dar otra oportunidad
                print("\n¿Quieres intentarlo una vez más? (s/n)")
                if input("> ").lower() == 's':
                    print("\n¿Cuál es el valor del término faltante?")
                    second_try = int(input("> "))

                    if second_try == missing_value:
                        print("\n✅ ¡Correcto en el segundo intento! El término faltante es " + str(missing_value))
                        steps_completed += 1
                        completed = True
                    else:
                        print(f"\n❌ No es correcto. El término faltante es {missing_value}")
        except ValueError:
            print("\n⚠️ Por favor, ingresa un número entero válido.")
            mistakes += 1

        # Paso 3: Verificación (opcional para niveles intermedio y avanzado)
        if level in ['intermedio', 'avanzado'] and completed:
            print("\n🔍 PASO 3 (Opcional): Verifica tu respuesta")
            print("\n¿Quieres comprobar tu respuesta? (s/n)")
            if input("> ").lower() == 's':
                print("\n📊 VERIFICACIÓN:")

                # Mostrar la progresión completa
                progression_str = ", ".join(str(term) for term in complete_progression)
                print(f"Progresión completa: {progression_str}")

                # Verificar la diferencia común
                print("\nVerificando la diferencia común:")
                for i in range(len(complete_progression) - 1):
                    print(f"{complete_progression[i+1]} - {complete_progression[i]} = {complete_progression[i+1] - complete_progression[i]}")

                print(f"\nComo vemos, la diferencia común es {common_diff} en todos los casos. ✓")
                steps_completed += 1

        # Resultados del ejercicio
        print("\n" + "=" * 60)
        print("🏆 RESULTADO DEL EJERCICIO 🏆".center(60))
        print("=" * 60)

        if completed:
            print("\n¡Felicidades! Has completado el ejercicio correctamente.")
        else:
            print("\nSigue practicando. Recuerda que cada error es una oportunidad para aprender.")

        # Devolver estadísticas si se solicita seguimiento
        if track_progress:
            return {
                'steps_completed': steps_completed,
                'mistakes': mistakes,
                'help_requested': help_requested,
                'completed': completed
            }

        return None


class LinearEquationGenerator(ExerciseGenerator):
    def generate_exercise(self, level='principiante'):
        """Genera un ejercicio de ecuación lineal basado en el nivel de dificultad."""
        config = self.difficulty_levels.get(level, self.difficulty_levels['principiante'])

        # Generar valores para el ejercicio
        if level == 'principiante':
            # Ecuación simple: ax + b = c
            a = random.randint(1, 5)
            x = random.randint(1, 10)
            b = random.randint(1, 10)
            c = a * x + b
            equation_type = 'simple'

        elif level == 'intermedio':
            # Ecuación con variables en ambos lados: ax + b = cx + d
            a = random.randint(2, 8)
            c = random.randint(1, a - 1)  # Para que a > c y la solución sea positiva
            x = random.randint(1, 15)
            b = random.randint(1, 20)
            d = a * x + b - c * x  # Para que se cumpla la igualdad
            equation_type = 'two_sides'

        else:  # avanzado
            # Tipo de ecuación aleatoria
            equation_types = ['fractions', 'parentheses', 'complex']
            equation_type = random.choice(equation_types)

            if equation_type == 'fractions':
                # Ecuación con fracciones: (ax + b)/c = d/e
                a = random.randint(1, 10)
                b = random.randint(1, 15)
                c = random.randint(2, 5)  # Denominador
                e = random.randint(2, 5)  # Denominador
                x = Fraction(random.randint(1, 10), 1)  # La solución será un entero para simplificar

                # Calculamos d para que se cumpla la igualdad
                left = Fraction(a * x + b, c)
                d = left * e

            elif equation_type == 'parentheses':
                # Ecuación con paréntesis: a(x + b) = c(x + d)
                a = random.randint(2, 8)
                b = random.randint(1, 10)
                c = random.randint(1, a - 1)  # Para que a > c y la solución sea positiva
                x = random.randint(1, 15)

                # Calculamos d para que se cumpla la igualdad
                # a(x + b) = c(x + d)
                # ax + ab = cx + cd
                # ax - cx = cd - ab
                # x(a - c) = cd - ab
                # d = (x(a - c) + ab) / c
                d = (x * (a - c) + a * b) / c

                # Si d no es entero, ajustamos x para que d sea entero
                if d != int(d):
                    # Calculamos x tal que (x(a - c) + ab) es divisible por c
                    for test_x in range(1, 21):
                        test_d = (test_x * (a - c) + a * b) / c
                        if test_d == int(test_d):
                            x = test_x
                            d = int(test_d)
                            break
                else:
                    d = int(d)

            else:  # complex
                # Ecuación más compleja con múltiples términos
                a = random.randint(2, 6)
                b = random.randint(1, 10)
                c = random.randint(1, 5)
                d = random.randint(1, 10)
                e = random.randint(1, 4)
                f = random.randint(1, 8)

                # La ecuación será: ax + b - cx = d + ex - f
                # Reorganizando: ax - cx - ex = d - f - b
                # x(a - c - e) = d - f - b
                x = Fraction(d - f - b, a - c - e)

                # Si x no es entero o tiene un denominador grande, ajustamos los coeficientes
                if x.denominator > 5:
                    # Intentamos con otros valores
                    for new_a in range(2, 10):
                        for new_e in range(1, 5):
                            new_x = Fraction(d - f - b, new_a - c - new_e)
                            if new_x.denominator <= 2 and new_x.numerator > 0:
                                a = new_a
                                e = new_e
                                x = new_x
                                break
                        if x.denominator <= 2:
                            break

        # Crear el ejercicio con formato
        exercise = {
            'level': level,
            'type': equation_type,
            'x_value': x,
        }

        # Añadir coeficientes según el tipo de ecuación
        if equation_type == 'simple':
            exercise.update({
                'a': a,
                'b': b,
                'c': c
            })
        elif equation_type == 'two_sides':
            exercise.update({
                'a': a,
                'b': b,
                'c': c,
                'd': d
            })
        elif equation_type == 'fractions':
            exercise.update({
                'a': a,
                'b': b,
                'c': c,
                'd': d,
                'e': e
            })
        elif equation_type == 'parentheses':
            exercise.update({
                'a': a,
                'b': b,
                'c': c,
                'd': d
            })
        else:  # complex
            exercise.update({
                'a': a,
                'b': b,
                'c': c,
                'd': d,
                'e': e,
                'f': f
            })

        return exercise

    def format_exercise(self, exercise):
        """Formatea el ejercicio para mostrarlo al usuario."""
        level = exercise['level']
        equation_type = exercise['type']

        print("\n" + "=" * 60)
        print(f"🧪 EJERCICIO DE ECUACIÓN LINEAL - NIVEL: {level.upper()} 🧪".center(60))
        print("=" * 60)
        print("\n📝 Ejercicio: Resuelve la siguiente ecuación lineal para x")

        # Construir la ecuación según el tipo
        if equation_type == 'simple':
            a, b, c = exercise['a'], exercise['b'], exercise['c']
            equation = f"{a}x + {b} = {c}"

        elif equation_type == 'two_sides':
            a, b, c, d = exercise['a'], exercise['b'], exercise['c'], exercise['d']
            equation = f"{a}x + {b} = {c}x + {d}"

        elif equation_type == 'fractions':
            a, b, c, d, e = exercise['a'], exercise['b'], exercise['c'], exercise['d'], exercise['e']
            equation = f"({a}x + {b})/{c} = {d}/{e}"

        elif equation_type == 'parentheses':
            a, b, c, d = exercise['a'], exercise['b'], exercise['c'], exercise['d']
            equation = f"{a}(x + {b}) = {c}(x + {d})"

        else:  # complex
            a, b, c, d, e, f = exercise['a'], exercise['b'], exercise['c'], exercise['d'], exercise['e'], exercise['f']
            equation = f"{a}x + {b} - {c}x = {d} + {e}x - {f}"

        print(f"\nEcuación: {equation}")
        print("\n💎 ¿Cuál es el valor de x?")

        return exercise

    def solve_interactive(self, exercise, track_progress=False):
        """Permite al usuario resolver el ejercicio interactivamente con seguimiento."""
        # Variables para tracking
        steps_completed = 0
        mistakes = 0
        help_requested = 0
        completed = False

        x_value = exercise['x_value']
        level = exercise['level']
        equation_type = exercise['type']

        # Paso 1: Identificar los pasos para resolver la ecuación
        print("\n🔍 PASO 1: ¿Cuál es la estrategia para resolver esta ecuación?")

        options = []

        if equation_type == 'simple':
            options = [
                "A) Despejar x directamente",
                "B) Agrupar términos similares y luego despejar x",
                "C) Multiplicar ambos lados para eliminar fracciones"
            ]
            correct_option = "A"

        elif equation_type == 'two_sides':
            options = [
                "A) Despejar x directamente",
                "B) Agrupar términos con x en un lado y constantes en otro",
                "C) Multiplicar ambos lados para eliminar fracciones"
            ]
            correct_option = "B"

        elif equation_type == 'fractions':
            options = [
                "A) Despejar x directamente",
                "B) Agrupar términos con x en un lado",
                "C) Multiplicar ambos lados por los denominadores para eliminar fracciones"
            ]
            correct_option = "C"

        elif equation_type == 'parentheses':
            options = [
                "A) Despejar x directamente",
                "B) Distribuir los coeficientes, agrupar términos similares y despejar x",
                "C) Multiplicar ambos lados para eliminar fracciones"
            ]
            correct_option = "B"

        else:  # complex
            options = [
                "A) Despejar x directamente",
                "B) Agrupar términos con x en un lado y constantes en otro, luego despejar x",
                "C) Convertir a una ecuación cuadrática"
            ]
            correct_option = "B"

        print("\n✅ Opciones de estrategia:")
        for option in options:
            print(option)

        print("\n¿Cuál estrategia usarías? (A, B, C o 'ayuda' para una pista)")
        user_answer = input("> ").upper()

        if user_answer == 'AYUDA':
            help_requested += 1

            if equation_type == 'simple':
                print("\n🔔 PISTA: En una ecuación de la forma ax + b = c, puedes despejar x directamente restando b de ambos lados y luego dividiendo por a.")
            elif equation_type == 'two_sides':
                print("\n🔔 PISTA: Cuando tienes variables en ambos lados, lo mejor es agrupar todos los términos con x en un lado y las constantes en el otro.")
            elif equation_type == 'fractions':
                print("\n🔔 PISTA: Para eliminar las fracciones, multiplica ambos lados de la ecuación por el mínimo común múltiplo de los denominadores.")
            elif equation_type == 'parentheses':
                print("\n🔔 PISTA: Primero debes distribuir los coeficientes dentro de los paréntesis, luego agrupar términos similares.")
            else:  # complex
                print("\n🔔 PISTA: Con múltiples términos, lo mejor es agrupar los términos con x en un lado y las constantes en el otro.")

            print("\n¿Cuál es tu respuesta ahora?")
            user_answer = input("> ").upper()

        if user_answer == correct_option:
            print("\n✅ ¡Correcto! Esa es la estrategia adecuada.")
            steps_completed += 1
        else:
            print(f"\n❌ No es correcto. La estrategia adecuada es la opción {correct_option}.")
            mistakes += 1

        # Paso 2: Resolver paso a paso
        print("\n🔍 PASO 2: Vamos a resolver la ecuación paso a paso.")

        # Guiar la resolución según el tipo de ecuación
        if equation_type == 'simple':
            a, b, c = exercise['a'], exercise['b'], exercise['c']

            print(f"\nTenemos la ecuación: {a}x + {b} = {c}")
            print(f"Paso 1: Restamos {b} de ambos lados")
            print(f"{a}x + {b} - {b} = {c} - {b}")
            print(f"{a}x = {c - b}")

            print(f"\nPaso 2: Dividimos ambos lados por {a}")
            print(f"{a}x ÷ {a} = {c - b} ÷ {a}")

            result = (c - b) / a
            if result == int(result):
                result = int(result)
            print(f"x = {result}")

            print("\n¿Cuál es el valor de x? (o escribe 'ayuda' para una pista)")
            user_answer = input("> ").lower()

        elif equation_type == 'two_sides':
            a, b, c, d = exercise['a'], exercise['b'], exercise['c'], exercise['d']

            print(f"\nTenemos la ecuación: {a}x + {b} = {c}x + {d}")
            print(f"Paso 1: Restamos {c}x de ambos lados para agrupar términos con x")
            print(f"{a}x + {b} - {c}x = {c}x + {d} - {c}x")
            print(f"{a-c}x + {b} = {d}")

            print(f"\nPaso 2: Restamos {b} de ambos lados")
            print(f"{a-c}x + {b} - {b} = {d} - {b}")
            print(f"{a-c}x = {d - b}")

            print(f"\nPaso 3: Dividimos ambos lados por {a-c}")
            print(f"{a-c}x ÷ {a-c} = {d - b} ÷ {a-c}")

            result = (d - b) / (a - c)
            if result == int(result):
                result = int(result)
            print(f"x = {result}")

            print("\n¿Cuál es el valor de x? (o escribe 'ayuda' para una pista)")
            user_answer = input("> ").lower()

        elif equation_type == 'fractions':
            a, b, c, d, e = exercise['a'], exercise['b'], exercise['c'], exercise['d'], exercise['e']

            print(f"\nTenemos la ecuación: ({a}x + {b})/{c} = {d}/{e}")
            print(f"Paso 1: Multiplicamos ambos lados por {c*e} para eliminar fracciones")
            print(f"{c*e} × ({a}x + {b})/{c} = {c*e} × {d}/{e}")
            print(f"{e} × ({a}x + {b}) = {c} × {d}")
            print(f"{e*a}x + {e*b} = {c*d}")

            print(f"\nPaso 2: Restamos {e*b} de ambos lados")
            print(f"{e*a}x + {e*b} - {e*b} = {c*d} - {e*b}")
            print(f"{e*a}x = {c*d - e*b}")

            print(f"\nPaso 3: Dividimos ambos lados por {e*a}")
            print(f"{e*a}x ÷ {e*a} = {c*d - e*b} ÷ {e*a}")

            result = Fraction(c*d - e*b, e*a)
            print(f"x = {result}")

            print("\n¿Cuál es el valor de x? (Escribe la fracción como 'a/b' o el decimal. O escribe 'ayuda' para una pista)")
            user_answer = input("> ").lower()

        elif equation_type == 'parentheses':
            a, b, c, d = exercise['a'], exercise['b'], exercise['c'], exercise['d']

            print(f"\nTenemos la ecuación: {a}(x + {b}) = {c}(x + {d})")
            print(f"Paso 1: Distribuimos los coeficientes")
            print(f"{a}x + {a*b} = {c}x + {c*d}")

            print(f"\nPaso 2: Restamos {c}x de ambos lados para agrupar términos con x")
            print(f"{a}x + {a*b} - {c}x = {c}x + {c*d} - {c}x")
            print(f"{a-c}x + {a*b} = {c*d}")

            print(f"\nPaso 3: Restamos {a*b} de ambos lados")
            print(f"{a-c}x + {a*b} - {a*b} = {c*d} - {a*b}")
            print(f"{a-c}x = {c*d - a*b}")

            print(f"\nPaso 4: Dividimos ambos lados por {a-c}")
            print(f"{a-c}x ÷ {a-c} = {c*d - a*b} ÷ {a-c}")

            result = (c*d - a*b) / (a - c)
            if result == int(result):
                result = int(result)
            print(f"x = {result}")

            print("\n¿Cuál es el valor de x? (o escribe 'ayuda' para una pista)")
            user_answer = input("> ").lower()

        else:  # complex
            a, b, c, d, e, f = exercise['a'], exercise['b'], exercise['c'], exercise['d'], exercise['e'], exercise['f']

            print(f"\nTenemos la ecuación: {a}x + {b} - {c}x = {d} + {e}x - {f}")
            print(f"Paso 1: Agrupamos términos con x en el lado izquierdo")
            print(f"{a}x - {c}x - {e}x = {d} - {f} - {b}")
            print(f"{a-c-e}x = {d - f - b}")

            print(f"\nPaso 2: Dividimos ambos lados por {a-c-e}")
            print(f"{a-c-e}x ÷ {a-c-e} = {d - f - b} ÷ {a-c-e}")

            result = Fraction(d - f - b, a - c - e)
            print(f"x = {result}")

            print("\n¿Cuál es el valor de x? (Escribe la fracción como 'a/b' o el decimal. O escribe 'ayuda' para una pista)")
            user_answer = input("> ").lower()

        if user_answer == 'ayuda':
            help_requested += 1

            if isinstance(x_value, Fraction) and x_value.denominator > 1:
                print(f"\n🔔 PISTA: La respuesta es una fracción. Puedes escribirla como '{x_value.numerator}/{x_value.denominator}' o como decimal.")
            else:
                print(f"\n🔔 PISTA: Asegúrate de seguir los pasos correctamente. Verifica tus cálculos.")

            print("\n¿Cuál es tu respuesta ahora?")
            user_answer = input("> ").lower()

        try:
            # Evaluar la respuesta del usuario
            if '/' in user_answer:
                # Convertir fracción
                num, denom = map(int, user_answer.split('/'))
                user_value = Fraction(num, denom)
            else:
                # Convertir decimal/entero
                user_value = float(user_answer)
                # Si es un entero, convertirlo a int
                if user_value == int(user_value):
                    user_value = int(user_value)

            # Comparar con la respuesta correcta
            if isinstance(x_value, Fraction):
                correct = abs(float(user_value) - float(x_value)) < 0.0001
            else:
                correct = abs(user_value - x_value) < 0.0001

            if correct:
                print(f"\n✅ ¡Correcto! x = {x_value}")
                steps_completed += 1
                completed = True
            else:
                print(f"\n❌ No es correcto. El valor de x es {x_value}")
                mistakes += 1

                # Dar otra oportunidad
                print("\n¿Quieres intentarlo una vez más? (s/n)")
                if input("> ").lower() == 's':
                    print("\n¿Cuál es el valor de x?")
                    second_try = input("> ").lower()

                    # Evaluar segundo intento
                    if '/' in second_try:
                        num, denom = map(int, second_try.split('/'))
                        second_value = Fraction(num, denom)
                    else:
                        second_value = float(second_try)
                        if second_value == int(second_value):
                            second_value = int(second_value)

                    if isinstance(x_value, Fraction):
                        second_correct = abs(float(second_value) - float(x_value)) < 0.0001
                    else:
                        second_correct = abs(second_value - x_value) < 0.0001

                    if second_correct:
                        print(f"\n✅ ¡Correcto en el segundo intento! x = {x_value}")
                        steps_completed += 1
                        completed = True
                    else:
                        print(f"\n❌ No es correcto. El valor de x es {x_value}")
        except ValueError:
            print("\n⚠️ Por favor, ingresa un número válido o una fracción como 'a/b'.")
            mistakes += 1

        # Paso 3: Verificación (opcional)
        if completed and level in ['intermedio', 'avanzado']:
            print("\n🔍 PASO 3 (Opcional): Verifica tu solución")
            print("\n¿Quieres comprobar tu respuesta? (s/n)")
            if input("> ").lower() == 's':
                print("\n📊 VERIFICACIÓN:")

                # Según el tipo de ecuación
                if equation_type == 'simple':
                    a, b, c = exercise['a'], exercise['b'], exercise['c']
                    left = a * x_value + b
                    right = c
                    print(f"Sustituimos x = {x_value} en la ecuación original {a}x + {b} = {c}")
                    print(f"Lado izquierdo: {a} × {x_value} + {b} = {left}")
                    print(f"Lado derecho: {c}")

                elif equation_type == 'two_sides':
                    a, b, c, d = exercise['a'], exercise['b'], exercise['c'], exercise['d']
                    left = a * x_value + b
                    right = c * x_value + d
                    print(f"Sustituimos x = {x_value} en la ecuación original {a}x + {b} = {c}x + {d}")
                    print(f"Lado izquierdo: {a} × {x_value} + {b} = {left}")
                    print(f"Lado derecho: {c} × {x_value} + {d} = {right}")

                elif equation_type == 'fractions':
                    a, b, c, d, e = exercise['a'], exercise['b'], exercise['c'], exercise['d'], exercise['e']
                    left = Fraction(a * x_value + b, c)
                    right = Fraction(d, e)
                    print(f"Sustituimos x = {x_value} en la ecuación original ({a}x + {b})/{c} = {d}/{e}")
                    print(f"Lado izquierdo: ({a} × {x_value} + {b})/{c} = {left}")
                    print(f"Lado derecho: {d}/{e} = {right}")

                elif equation_type == 'parentheses':
                    a, b, c, d = exercise['a'], exercise['b'], exercise['c'], exercise['d']
                    left = a * (x_value + b)
                    right = c * (x_value + d)
                    print(f"Sustituimos x = {x_value} en la ecuación original {a}(x + {b}) = {c}(x + {d})")
                    print(f"Lado izquierdo: {a} × ({x_value} + {b}) = {a} × {x_value + b} = {left}")
                    print(f"Lado derecho: {c} × ({x_value} + {d}) = {c} × {x_value + d} = {right}")

                else:  # complex
                    a, b, c, d, e, f = exercise['a'], exercise['b'], exercise['c'], exercise['d'], exercise['e'], exercise['f']
                    left = a * x_value + b - c * x_value
                    right = d + e * x_value - f
                    print(f"Sustituimos x = {x_value} en la ecuación original {a}x + {b} - {c}x = {d} + {e}x - {f}")
                    print(f"Lado izquierdo: {a} × {x_value} + {b} - {c} × {x_value} = {left}")
                    print(f"Lado derecho: {d} + {e} × {x_value} - {f} = {right}")

                # Comprobar igualdad
                if abs(float(left) - float(right)) < 0.0001:
                    print("\n✅ La solución es correcta. Se cumple la igualdad.")
                else:
                    print("\n❌ Hay un error en la verificación. Revisa los cálculos.")

                steps_completed += 1

        # Resultados del ejercicio
        print("\n" + "=" * 60)
        print("🏆 RESULTADO DEL EJERCICIO 🏆".center(60))
        print("=" * 60)

        if completed:
            print("\n¡Felicidades! Has completado el ejercicio correctamente.")
        else:
            print("\nSigue practicando. Recuerda que cada error es una oportunidad para aprender.")

        # Devolver estadísticas si se solicita seguimiento
        if track_progress:
            return {
                'steps_completed': steps_completed,
                'mistakes': mistakes,
                'help_requested': help_requested,
                'completed': completed
            }

        return None


class LinearFunctionGenerator(ExerciseGenerator):
    """Generador de ejercicios de funciones lineales"""

    def generate_exercise(self, level='principiante'):
        """Genera un ejercicio de función lineal basado en el nivel de dificultad."""
        config = self.difficulty_levels.get(level, self.difficulty_levels['principiante'])

        # Generar valores para el ejercicio según el nivel
        if level == 'principiante':
            # Para principiantes: forma y = mx + b con m, b enteros positivos pequeños
            m = random.randint(1, 3)
            b = random.randint(0, 5)
            x_values = [random.randint(-5, 5) for _ in range(3)]

        elif level == 'intermedio':
            # Para intermedios: incluir pendientes negativas y mayores valores
            m = random.randint(-5, 5)
            while m == 0:  # Evitar pendiente cero
                m = random.randint(-5, 5)
            b = random.randint(-10, 10)
            x_values = [random.randint(-10, 10) for _ in range(3)]

        else:  # avanzado
            # Para avanzados: incluir fracciones y números más complejos
            numerator = random.randint(-5, 5)
            while numerator == 0:
                numerator = random.randint(-5, 5)

            denominator = random.randint(1, 4)
            m = Fraction(numerator, denominator)
            b = random.randint(-15, 15)
            x_values = [random.randint(-15, 15) for _ in range(4)]

        # Calcular los valores de y correspondientes
        y_values = []
        for x in x_values:
            if isinstance(m, Fraction):
                y = m * x + b
                y_values.append(y)
            else:
                y_values.append(m * x + b)

        # Crear el ejercicio
        exercise = {
            'level': level,
            'slope': m,
            'y_intercept': b,
            'x_values': x_values,
            'y_values': y_values,
            'equation': f'y = {m}x + {b}' if b >= 0 else f'y = {m}x - {abs(b)}'
        }

        return exercise

    def format_exercise(self, exercise):
        """Formatea el ejercicio para mostrarlo al usuario."""
        level = exercise['level']
        slope = exercise['slope']
        y_intercept = exercise['y_intercept']
        x_values = exercise['x_values']
        y_values = exercise['y_values']

        print("\n" + "=" * 60)
        print(f"📈 EJERCICIO DE FUNCIÓN LINEAL - NIVEL: {level.upper()} 📈".center(60))
        print("=" * 60)

        if level == 'principiante':
            print("\n📝 Ejercicio: Identifica y utiliza una función lineal")
            print("\nObserva los siguientes puntos de una función lineal:")
            for i in range(len(x_values)):
                print(f"• Punto {i+1}: ({x_values[i]}, {y_values[i]})")

            print("\n💎 Encuentra la función lineal que pasa por estos puntos.")
            print("Luego, utiliza la función para calcular valores de 'y' para nuevos valores de 'x'.")

        elif level == 'intermedio':
            print("\n📝 Ejercicio: Encuentra la función lineal y analiza sus características")
            print("\nObserva los siguientes puntos de una función lineal:")
            for i in range(len(x_values)):
                print(f"• Punto {i+1}: ({x_values[i]}, {y_values[i]})")

            print("\n💎 Encuentra la ecuación de la función lineal en la forma y = mx + b.")
            print("Analiza la pendiente y el corte con el eje y de la función.")

        else:  # avanzado
            print("\n📝 Ejercicio: Función lineal y sus aplicaciones")
            print("\nObserva los siguientes puntos de una función lineal:")
            for i in range(min(3, len(x_values))):
                print(f"• Punto {i+1}: ({x_values[i]}, {y_values[i]})")

            print("\n💎 Encuentra la ecuación de la función lineal y analiza:")
            print("• La pendiente y su significado")
            print("• El corte con el eje y")
            print("• Los cortes con ambos ejes")
            print("• El crecimiento o decrecimiento de la función")

        return exercise

    def solve_interactive(self, exercise, track_progress=False):
        """Permite al usuario resolver el ejercicio de función lineal interactivamente."""
        # Variables para tracking
        steps_completed = 0
        mistakes = 0
        help_requested = 0
        completed = False

        level = exercise['level']
        slope = exercise['slope']
        y_intercept = exercise['y_intercept']
        x_values = exercise['x_values']
        y_values = exercise['y_values']

        # Paso 1: Identificación de la pendiente
        print("\n🔍 PASO 1: Identifica la pendiente (m) de la función lineal")
        print("\n¿Cuál es el valor de la pendiente? (o escribe 'ayuda' para una pista)")

        user_answer = input("> ").lower()

        if user_answer == 'ayuda':
            help_requested += 1
            print("\n🔔 PISTA: La pendiente se puede calcular usando la fórmula:")
            print("m = (y₂ - y₁) / (x₂ - x₁) usando dos puntos cualesquiera.")
            print(f"Por ejemplo, con los puntos ({x_values[0]}, {y_values[0]}) y ({x_values[1]}, {y_values[1]})")

            # Calcular la pendiente para el ejemplo
            if x_values[1] - x_values[0] != 0:
                m_example = (y_values[1] - y_values[0]) / (x_values[1] - x_values[0])
                print(f"m = ({y_values[1]} - {y_values[0]}) / ({x_values[1]} - {x_values[0]}) = {m_example}")

            print("\n¿Cuál es tu respuesta ahora?")
            user_answer = input("> ")

        # Verificar la respuesta de la pendiente
        try:
            # Convertir la respuesta a un número o fracción
            if '/' in user_answer:
                parts = user_answer.split('/')
                user_slope = Fraction(int(parts[0]), int(parts[1]))
            else:
                user_slope = float(user_answer)

            # Verificar si la respuesta es correcta
            correct_slope = False
            if isinstance(slope, Fraction):
                correct_slope = abs(user_slope - float(slope)) < 0.001
            else:
                correct_slope = abs(user_slope - slope) < 0.001

            if correct_slope:
                print("\n✅ ¡Correcto! La pendiente es " + str(slope))
                steps_completed += 1
            else:
                print(f"\n❌ No es correcto. La pendiente es {slope}")
                mistakes += 1

                # Mostrar cálculo paso a paso
                print("\nCálculo paso a paso:")
                x1, y1 = x_values[0], y_values[0]
                x2, y2 = x_values[1], y_values[1]
                print(f"Usando los puntos ({x1}, {y1}) y ({x2}, {y2}):")
                print(f"m = (y₂ - y₁) / (x₂ - x₁)")
                print(f"m = ({y2} - {y1}) / ({x2} - {x1})")
                print(f"m = {y2 - y1} / {x2 - x1}")
                print(f"m = {slope}")
        except:
            print("\n⚠️ Por favor, ingresa un número válido o una fracción (n/d).")
            mistakes += 1

        # Paso 2: Identificación del corte con el eje y (b)
        print("\n🔍 PASO 2: Identifica el corte con el eje y (b) de la función")
        print("\n¿Cuál es el valor de b? (o escribe 'ayuda' para una pista)")

        user_answer = input("> ").lower()

        if user_answer == 'ayuda':
            help_requested += 1
            print("\n🔔 PISTA: El corte con el eje y (b) se puede calcular usando la ecuación:")
            print("y = mx + b")
            print("Despejando b: b = y - mx")
            print(f"Usando el punto ({x_values[0]}, {y_values[0]}) y m = {slope}:")
            b_calculation = y_values[0] - slope * x_values[0]
            print(f"b = {y_values[0]} - {slope} × {x_values[0]} = {b_calculation}")

            print("\n¿Cuál es tu respuesta ahora?")
            user_answer = input("> ")

        # Verificar la respuesta del corte con el eje y
        try:
            user_intercept = float(user_answer)

            if abs(user_intercept - y_intercept) < 0.001:
                print("\n✅ ¡Correcto! El corte con el eje y es " + str(y_intercept))
                steps_completed += 1
            else:
                print(f"\n❌ No es correcto. El corte con el eje y es {y_intercept}")
                mistakes += 1

                # Mostrar cálculo paso a paso
                print("\nCálculo paso a paso:")
                x1, y1 = x_values[0], y_values[0]
                print(f"Usando el punto ({x1}, {y1}) y m = {slope}:")
                print(f"b = y - mx")
                print(f"b = {y1} - {slope} × {x1}")
                print(f"b = {y1} - {slope * x1}")
                print(f"b = {y_intercept}")
        except:
            print("\n⚠️ Por favor, ingresa un número válido.")
            mistakes += 1

        # Paso 3: Escribir la ecuación completa
        print("\n🔍 PASO 3: Escribe la ecuación completa de la función lineal")
        print("\n¿Cuál es la ecuación de la función lineal en la forma y = mx + b?")
        print("(escribe 'ayuda' para una pista)")

        user_answer = input("> ").lower()

        if user_answer == 'ayuda':
            help_requested += 1
            print("\n🔔 PISTA: Sustituye los valores de m y b en la ecuación y = mx + b.")
            if y_intercept >= 0:
                print(f"Con m = {slope} y b = {y_intercept}, la ecuación es y = {slope}x + {y_intercept}")
            else:
                print(f"Con m = {slope} y b = {y_intercept}, la ecuación es y = {slope}x - {abs(y_intercept)}")

            print("\n¿Cuál es tu respuesta ahora?")
            user_answer = input("> ")

        # Verificar ecuación
        # Aceptar diferentes formatos de respuesta
        equation = exercise['equation'].replace(" ", "").lower()
        user_eq = user_answer.replace(" ", "").lower()
        user_eq = user_eq.replace("y=", "")
        equation = equation.replace("y=", "")

        if user_eq == equation or (slope == 1 and user_eq.replace("1x", "x") == equation.replace("1x", "x")):
            print("\n✅ ¡Correcto! La ecuación es " + exercise['equation'])
            steps_completed += 1
        else:
            print(f"\n❌ No es correcto. La ecuación es {exercise['equation']}")
            mistakes += 1

        # Paso 4 (opcional para intermedio y avanzado): Análisis adicional
        if level in ['intermedio', 'avanzado']:
            completed_analysis = False
            print("\n🔍 PASO 4: Analiza las características de la función")

            # a) Interpretar el significado de la pendiente
            print("\n¿La función es creciente, decreciente o constante? (o escribe 'ayuda')")
            user_answer = input("> ").lower()

            if user_answer == 'ayuda':
                help_requested += 1
                print("\n🔔 PISTA: Una función es creciente si m > 0, decreciente si m < 0, y constante si m = 0.")
                print(f"En este caso, m = {slope}")
                print("\n¿Cuál es tu respuesta ahora?")
                user_answer = input("> ").lower()

            correct_trend = False
            if (slope > 0 and 'creciente' in user_answer) or \
               (slope < 0 and 'decreciente' in user_answer) or \
               (slope == 0 and 'constante' in user_answer):
                print("\n✅ ¡Correcto!")
                correct_trend = True
                steps_completed += 1
            else:
                print("\n❌ No es correcto.")
                if slope > 0:
                    print("La función es creciente porque m > 0.")
                elif slope < 0:
                    print("La función es decreciente porque m < 0.")
                else:
                    print("La función es constante porque m = 0.")
                mistakes += 1

            # b) Calcular corte con el eje x (si aplica y es nivel avanzado)
            if level == 'avanzado' and slope != 0:
                print("\n¿En qué punto la función corta al eje x? (o escribe 'ayuda')")
                user_answer = input("> ").lower()

                if user_answer == 'ayuda':
                    help_requested += 1
                    print("\n🔔 PISTA: El corte con el eje x ocurre cuando y = 0.")
                    print(f"Resuelve: 0 = {slope}x + {y_intercept}")
                    print(f"{-y_intercept} = {slope}x")
                    x_intercept = -y_intercept / slope
                    print(f"x = {x_intercept}")
                    print("\n¿Cuál es tu respuesta ahora?")
                    user_answer = input("> ")

                try:
                    if 'no' in user_answer and y_intercept == 0 and slope != 0:
                        print("\n✅ ¡Correcto! La función solo corta el eje x en el origen (0,0)")
                        steps_completed += 1
                    else:
                        x_intercept = -y_intercept / slope

                        # Verificar formato de respuesta (aceptar punto o valor de x)
                        if ',' in user_answer:  # formato (x,y)
                            parts = user_answer.strip('()').split(',')
                            user_x = float(parts[0])
                            user_y = float(parts[1]) if len(parts) > 1 else 0

                            if abs(user_x - x_intercept) < 0.001 and abs(user_y) < 0.001:
                                print(f"\n✅ ¡Correcto! La función corta el eje x en ({x_intercept}, 0)")
                                steps_completed += 1
                            else:
                                print(f"\n❌ No es correcto. La función corta el eje x en ({x_intercept}, 0)")
                                mistakes += 1
                        else:  # solo valor de x
                            user_x = float(user_answer)
                            if abs(user_x - x_intercept) < 0.001:
                                print(f"\n✅ ¡Correcto! La función corta el eje x cuando x = {x_intercept}")
                                steps_completed += 1
                            else:
                                print(f"\n❌ No es correcto. La función corta el eje x cuando x = {x_intercept}")
                                mistakes += 1
                except:
                    print("\n⚠️ Por favor, ingresa un valor numérico o un punto (x,y).")
                    mistakes += 1

            # Completado análisis adicional
            if level == 'intermedio' and correct_trend:
                completed_analysis = True
            elif level == 'avanzado' and steps_completed >= 4:
                completed_analysis = True

            if completed_analysis:
                completed = True
        else:
            # Para nivel principiante, verificar con un valor adicional
            print("\n🔍 PASO 4: Utiliza la función para calcular un nuevo valor")

            # Generar un nuevo valor de x
            new_x = random.randint(-10, 10)
            new_y = slope * new_x + y_intercept

            print(f"\nUtiliza la función para calcular y cuando x = {new_x}")
            print("(escribe 'ayuda' para una pista)")

            user_answer = input("> ").lower()

            if user_answer == 'ayuda':
                help_requested += 1
                print("\n🔔 PISTA: Sustituye el valor de x en la ecuación.")
                print(f"y = {slope}x + {y_intercept}")
                print(f"y = {slope} × {new_x} + {y_intercept}")
                print(f"y = {slope * new_x} + {y_intercept}")
                print(f"y = {new_y}")

                print("\n¿Cuál es tu respuesta ahora?")
                user_answer = input("> ")

            try:
                user_y = float(user_answer)

                if abs(user_y - new_y) < 0.001:
                    print("\n✅ ¡Correcto! y = " + str(new_y))
                    steps_completed += 1
                    completed = True
                else:
                    print(f"\n❌ No es correcto. y = {new_y}")
                    mistakes += 1

                    # Mostrar cálculo paso a paso
                    print("\nCálculo paso a paso:")
                    print(f"y = {slope}x + {y_intercept}")
                    print(f"y = {slope} × {new_x} + {y_intercept}")
                    print(f"y = {slope * new_x} + {y_intercept}")
                    print(f"y = {new_y}")

                    # Dar otra oportunidad
                    print("\n¿Quieres intentarlo una vez más? (s/n)")
                    if input("> ").lower() == 's':
                        print(f"\n¿Cuál es el valor de y cuando x = {new_x}?")
                        second_try = float(input("> "))

                        if abs(second_try - new_y) < 0.001:
                            print("\n✅ ¡Correcto en el segundo intento! y = " + str(new_y))
                            steps_completed += 1
                            completed = True
                        else:
                            print(f"\n❌ No es correcto. y = {new_y}")
            except:
                print("\n⚠️ Por favor, ingresa un número válido.")
                mistakes += 1

        # Resultados del ejercicio
        print("\n" + "=" * 60)
        print("🏆 RESULTADO DEL EJERCICIO 🏆".center(60))
        print("=" * 60)

        if (level == 'principiante' and steps_completed >= 3) or \
           (level == 'intermedio' and steps_completed >= 3) or \
           (level == 'avanzado' and steps_completed >= 4):
            completed = True
            print("\n¡Felicidades! Has completado el ejercicio correctamente.")
        else:
            print("\nSigue practicando. Recuerda que cada error es una oportunidad para aprender.")

        # Devolver estadísticas si se solicita seguimiento
        if track_progress:
            return {
                'steps_completed': steps_completed,
                'mistakes': mistakes,
                'help_requested': help_requested,
                'completed': completed
            }

        return None

class ProportionalityGenerator(ExerciseGenerator):
    def generate_exercise(self, level='principiante'):
        """Genera un ejercicio de proporcionalidad directa o inversa según el nivel."""
        config = self.difficulty_levels.get(level, self.difficulty_levels['principiante'])

        # Determinar si es proporcionalidad directa o inversa
        is_direct = random.choice([True, False])
        prop_type = "directa" if is_direct else "inversa"

        if level == 'principiante':
            # Proporcionalidad con números enteros pequeños
            if is_direct:
                # Para proporcionalidad directa: y = kx
                k = random.randint(2, 5)
                x1 = random.randint(1, 6)
                y1 = k * x1
                x2 = random.randint(7, 10)
                y2 = k * x2
            else:
                # Para proporcionalidad inversa: xy = k
                k = random.randint(12, 24)
                x1 = random.randint(2, 4)
                y1 = k // x1
                x2 = random.randint(6, 8)
                y2 = k // x2

        elif level == 'intermedio':
            # Proporcionalidad con números más grandes
            if is_direct:
                k = random.randint(3, 7)
                x1 = random.randint(4, 10)
                y1 = k * x1
                x2 = random.randint(11, 20)
                y2 = k * x2
            else:
                k = random.randint(24, 60)
                x1 = random.randint(3, 6)
                y1 = k // x1
                x2 = random.randint(10, 15)
                y2 = k // x2

        else:  # avanzado
            # Proporcionalidad con fracciones o decimales
            if is_direct:
                k_num = random.randint(1, 5)
                k_den = random.randint(2, 5)
                k = k_num / k_den
                x1 = random.randint(4, 10)
                y1 = k * x1
                x2 = random.randint(12, 25)
                y2 = k * x2
            else:
                k = random.randint(30, 100)
                x1 = random.randint(5, 10)
                y1 = k / x1
                x2 = random.randint(15, 25)
                y2 = k / x2

        # Crear el ejercicio
        exercise = {
            'level': level,
            'prop_type': prop_type,
            'is_direct': is_direct,
            'constant': k,
            'pair1': {'x': x1, 'y': y1},
            'pair2': {'x': x2, 'y': y2},
            # Crear un valor desconocido para que el estudiante lo calcule
            'unknown': random.choice(['pair1_y', 'pair2_y', 'pair2_x'])
        }

        # Asegurarse de que los valores sean enteros cuando sea posible
        if level in ['principiante', 'intermedio']:
            exercise['pair1']['y'] = int(exercise['pair1']['y'])
            exercise['pair2']['y'] = int(exercise['pair2']['y'])
        else:
            # Redondear a 2 decimales para el nivel avanzado
            exercise['pair1']['y'] = round(exercise['pair1']['y'], 2)
            exercise['pair2']['y'] = round(exercise['pair2']['y'], 2)

        # Ocultar el valor que el estudiante debe calcular
        if exercise['unknown'] == 'pair1_y':
            exercise['pair1']['y_hidden'] = exercise['pair1']['y']
            exercise['pair1']['y'] = '?'
        elif exercise['unknown'] == 'pair2_y':
            exercise['pair2']['y_hidden'] = exercise['pair2']['y']
            exercise['pair2']['y'] = '?'
        else:  # pair2_x
            exercise['pair2']['x_hidden'] = exercise['pair2']['x']
            exercise['pair2']['x'] = '?'

        return exercise

    def format_exercise(self, exercise):
        """Formatea el ejercicio de proporcionalidad para mostrarlo al usuario."""
        level = exercise['level']
        prop_type = exercise['prop_type']
        x1 = exercise['pair1']['x']
        y1 = exercise['pair1']['y']
        x2 = exercise['pair2']['x']
        y2 = exercise['pair2']['y']

        print("\n" + "=" * 60)
        print(f"🧮 EJERCICIO DE PROPORCIONALIDAD {prop_type.upper()} - NIVEL: {level.upper()} 🧮".center(60))
        print("=" * 60)

        print(f"\n📝 Ejercicio: Resuelve el problema de proporcionalidad {prop_type}")

        print("\nObserva la siguiente relación entre magnitudes:")
        print(f"• Para x = {x1}, y = {y1}")
        print(f"• Para x = {x2}, y = {y2}")

        if exercise['is_direct']:
            print("\n💡 Recuerda: En la proporcionalidad directa, cuando una magnitud aumenta, la otra también aumenta.")
            print("   La fórmula es: y = k·x, donde k es la constante de proporcionalidad.")
        else:
            print("\n💡 Recuerda: En la proporcionalidad inversa, cuando una magnitud aumenta, la otra disminuye.")
            print("   La fórmula es: x·y = k, donde k es la constante de proporcionalidad.")

        # Representación visual simple
        print("\n" + "-" * 40)
        print("       x       |       y       ")
        print("-" * 40)
        print(f"      {x1}       |      {y1}      ")
        print(f"      {x2}       |      {y2}      ")
        print("-" * 40)

        print(f"\n💎 Calcula el valor que falta, marcado con '?'")

        return exercise

    def solve_interactive(self, exercise, track_progress=False):
        """Permite al usuario resolver el ejercicio de proporcionalidad interactivamente."""
        steps_completed = 0
        mistakes = 0
        help_requested = 0
        completed = False

        level = exercise['level']
        is_direct = exercise['is_direct']
        prop_type = exercise['prop_type']
        unknown = exercise['unknown']

        # Paso 1: Identificar el tipo de proporcionalidad
        print("\n🔍 PASO 1: Identifica el tipo de proporcionalidad")
        print("\n¿Qué tipo de proporcionalidad observas en este problema?")
        print("A) Proporcionalidad directa")
        print("B) Proporcionalidad inversa")
        print("C) No hay proporcionalidad")
        print("\n¿Cuál es tu respuesta? (A, B, C o 'ayuda' para una pista)")

        user_answer = input("> ").upper()

        if user_answer == 'AYUDA':
            help_requested += 1
            print("\n🔔 PISTA:")
            print("• En proporcionalidad directa, si una magnitud aumenta, la otra también aumenta.")
            print("• En proporcionalidad inversa, si una magnitud aumenta, la otra disminuye.")
            print("• Compara los valores dados para ver si hay algún patrón.")

            print("\n¿Cuál es tu respuesta ahora?")
            user_answer = input("> ").upper()

        correct_option = "A" if is_direct else "B"

        if user_answer == correct_option:
            print(f"\n✅ ¡Correcto! Es una proporcionalidad {prop_type}.")
            steps_completed += 1
        else:
            print(f"\n❌ No es correcto. Es una proporcionalidad {prop_type}.")
            mistakes += 1

            # Explicación
            if is_direct:
                print("\nExplicación: Cuando x aumenta, y también aumenta. Esto es proporcionalidad directa.")
            else:
                print("\nExplicación: Cuando x aumenta, y disminuye. Esto es proporcionalidad inversa.")

        # Paso 2: Encontrar la constante de proporcionalidad
        print("\n🔍 PASO 2: Encuentra la constante de proporcionalidad (k)")

        # Obtener los valores no ocultos
        x1 = exercise['pair1']['x']
        y1 = exercise['pair1']['y']
        x2 = exercise['pair2']['x']
        y2 = exercise['pair2']['y']

        # Determinar qué valor usar para calcular k
        if unknown == 'pair1_y':
            use_pair = 'pair2'
            x_value = x2
            y_value = exercise['pair2']['y']
        else:
            use_pair = 'pair1'
            x_value = x1
            y_value = exercise['pair1']['y']

        # Para permitir el cálculo si el valor que necesitamos es el oculto
        if y_value == '?':
            use_pair = 'pair2'
            x_value = x2
            y_value = exercise['pair2']['y']
            if y_value == '?':  # Si aún es '?', usar el valor oculto
                y_value = exercise['pair2']['y_hidden']

        print(f"\nUsando los valores x = {x_value} e y = {y_value}, calcula k:")

        if is_direct:
            print(f"• Para proporcionalidad directa: k = y/x = {y_value}/{x_value}")
        else:
            print(f"• Para proporcionalidad inversa: k = x·y = {x_value}·{y_value}")

        print("\n¿Cuál es el valor de k? (o escribe 'ayuda' para una pista)")

        user_answer = input("> ").lower()

        # Calcular el valor correcto de k
        if is_direct and y_value != '?':
            k_correct = y_value / x_value
        elif not is_direct and y_value != '?':
            k_correct = x_value * y_value
        else:
            k_correct = exercise['constant']

        # Redondear para comparar
        k_correct_rounded = round(k_correct, 2)

        if user_answer == 'ayuda':
            help_requested += 1
            print("\n🔔 PISTA:")
            if is_direct:
                print(f"• Para proporcionalidad directa: k = y/x = {y_value}/{x_value} = {k_correct_rounded}")
            else:
                print(f"• Para proporcionalidad inversa: k = x·y = {x_value}·{y_value} = {k_correct_rounded}")

            print("\n¿Cuál es tu respuesta ahora?")
            user_answer = input("> ")

        try:
            user_k = float(user_answer)

            # Verificar con margen de error
            if abs(user_k - k_correct) < 0.1:
                print(f"\n✅ ¡Correcto! La constante de proporcionalidad k = {k_correct_rounded}")
                steps_completed += 1
            else:
                print(f"\n❌ No es correcto. La constante de proporcionalidad k = {k_correct_rounded}")
                mistakes += 1

                # Mostrar cálculo paso a paso
                if is_direct:
                    print(f"\nCálculo: k = y/x = {y_value}/{x_value} = {k_correct_rounded}")
                else:
                    print(f"\nCálculo: k = x·y = {x_value}·{y_value} = {k_correct_rounded}")
        except ValueError:
            print("\n⚠️ Por favor, ingresa un número válido.")
            mistakes += 1
            # Mostrar el valor correcto
            print(f"La constante de proporcionalidad k = {k_correct_rounded}")

        # Paso 3: Calcular el valor desconocido
        print("\n🔍 PASO 3: Calcula el valor desconocido usando la constante de proporcionalidad")

        # Determinar qué valor calcular
        if unknown == 'pair1_y':
            print(f"\nAhora, calcula y cuando x = {x1}:")
            if is_direct:
                print(f"• Usando la fórmula y = k·x: y = {k_correct_rounded}·{x1}")
            else:
                print(f"• Usando la fórmula x·y = k: {x1}·y = {k_correct_rounded}")
            correct_value = exercise['pair1']['y_hidden']
        elif unknown == 'pair2_y':
            print(f"\nAhora, calcula y cuando x = {x2}:")
            if is_direct:
                print(f"• Usando la fórmula y = k·x: y = {k_correct_rounded}·{x2}")
            else:
                print(f"• Usando la fórmula x·y = k: {x2}·y = {k_correct_rounded}")
            correct_value = exercise['pair2']['y_hidden']
        else:  # pair2_x
            print(f"\nAhora, calcula x cuando y = {y2}:")
            if is_direct:
                print(f"• Usando la fórmula y = k·x: {y2} = {k_correct_rounded}·x")
            else:
                print(f"• Usando la fórmula x·y = k: x·{y2} = {k_correct_rounded}")
            correct_value = exercise['pair2']['x_hidden']

        print("\n¿Cuál es el valor desconocido? (o escribe 'ayuda' para una pista)")
        user_answer = input("> ").lower()

        if user_answer == 'ayuda':
            help_requested += 1
            print("\n🔔 PISTA:")
            if unknown in ['pair1_y', 'pair2_y']:
                if is_direct:
                    x_val = x1 if unknown == 'pair1_y' else x2
                    print(f"• Para calcular y: y = k·x = {k_correct_rounded}·{x_val} = {correct_value}")
                else:
                    x_val = x1 if unknown == 'pair1_y' else x2
                    print(f"• Para calcular y: y = k/x = {k_correct_rounded}/{x_val} = {correct_value}")
            else:  # pair2_x
                if is_direct:
                    print(f"• Para calcular x: x = y/k = {y2}/{k_correct_rounded} = {correct_value}")
                else:
                    print(f"• Para calcular x: x = k/y = {k_correct_rounded}/{y2} = {correct_value}")

            print("\n¿Cuál es tu respuesta ahora?")
            user_answer = input("> ")

        try:
            user_value = float(user_answer)

            # Redondear el valor correcto para comparar
            correct_value_rounded = round(correct_value, 2)

            # Verificar con margen de error
            if abs(user_value - correct_value) < 0.1:
                print(f"\n✅ ¡Correcto! El valor desconocido es {correct_value_rounded}")
                steps_completed += 1
                completed = True
            else:
                print(f"\n❌ No es correcto. El valor desconocido es {correct_value_rounded}")
                mistakes += 1

                # Mostrar cálculo paso a paso
                if unknown in ['pair1_y', 'pair2_y']:
                    x_val = x1 if unknown == 'pair1_y' else x2
                    if is_direct:
                        print(f"\nCálculo: y = k·x = {k_correct_rounded}·{x_val} = {correct_value_rounded}")
                    else:
                        print(f"\nCálculo: y = k/x = {k_correct_rounded}/{x_val} = {correct_value_rounded}")
                else:  # pair2_x
                    if is_direct:
                        print(f"\nCálculo: x = y/k = {y2}/{k_correct_rounded} = {correct_value_rounded}")
                    else:
                        print(f"\nCálculo: x = k/y = {k_correct_rounded}/{y2} = {correct_value_rounded}")

                # Dar otra oportunidad
                print("\n¿Quieres intentarlo una vez más? (s/n)")
                if input("> ").lower() == 's':
                    print("\n¿Cuál es el valor desconocido?")
                    second_try = float(input("> "))

                    if abs(second_try - correct_value) < 0.1:
                        print(f"\n✅ ¡Correcto en el segundo intento! El valor desconocido es {correct_value_rounded}")
                        steps_completed += 1
                        completed = True
                    else:
                        print(f"\n❌ No es correcto. El valor desconocido es {correct_value_rounded}")
        except ValueError:
            print("\n⚠️ Por favor, ingresa un número válido.")
            mistakes += 1
            # Mostrar el valor correcto
            print(f"El valor desconocido es {round(correct_value, 2)}")

        # Paso 4: Verificación (opcional para niveles intermedio y avanzado)
        if level in ['intermedio', 'avanzado'] and completed:
            print("\n🔍 PASO 4 (Opcional): Verifica tu respuesta")
            print("\n¿Quieres comprobar tu respuesta? (s/n)")
            if input("> ").lower() == 's':
                print("\n📊 VERIFICACIÓN:")

                if unknown == 'pair1_y':
                    if is_direct:
                        print(f"1. Usamos la fórmula y = k·x")
                        print(f"2. Sustituimos: y = {k_correct_rounded}·{x1} = {correct_value_rounded}")
                        print(f"3. Comprobamos: {x1}/{correct_value_rounded} = {x2}/{y2} ✓")
                    else:
                        print(f"1. Usamos la fórmula x·y = k")
                        print(f"2. Sustituimos: {x1}·y = {k_correct_rounded}")
                        print(f"3. Despejamos: y = {k_correct_rounded}/{x1} = {correct_value_rounded}")
                        print(f"4. Comprobamos: {x1}·{correct_value_rounded} = {x2}·{y2} = {k_correct_rounded} ✓")

                elif unknown == 'pair2_y':
                    if is_direct:
                        print(f"1. Usamos la fórmula y = k·x")
                        print(f"2. Sustituimos: y = {k_correct_rounded}·{x2} = {correct_value_rounded}")
                        print(f"3. Comprobamos: {x1}/{y1} = {x2}/{correct_value_rounded} ✓")
                    else:
                        print(f"1. Usamos la fórmula x·y = k")
                        print(f"2. Sustituimos: {x2}·y = {k_correct_rounded}")
                        print(f"3. Despejamos: y = {k_correct_rounded}/{x2} = {correct_value_rounded}")
                        print(f"4. Comprobamos: {x1}·{y1} = {x2}·{correct_value_rounded} = {k_correct_rounded} ✓")

                else:  # pair2_x
                    if is_direct:
                        print(f"1. Usamos la fórmula y = k·x")
                        print(f"2. Sustituimos: {y2} = {k_correct_rounded}·x")
                        print(f"3. Despejamos: x = {y2}/{k_correct_rounded} = {correct_value_rounded}")
                        print(f"4. Comprobamos: {y1}/{x1} = {y2}/{correct_value_rounded} = {k_correct_rounded} ✓")
                    else:
                        print(f"1. Usamos la fórmula x·y = k")
                        print(f"2. Sustituimos: x·{y2} = {k_correct_rounded}")
                        print(f"3. Despejamos: x = {k_correct_rounded}/{y2} = {correct_value_rounded}")
                        print(f"4. Comprobamos: {x1}·{y1} = {correct_value_rounded}·{y2} = {k_correct_rounded} ✓")

                steps_completed += 1

        # Resultados del ejercicio
        print("\n" + "=" * 60)
        print("🏆 RESULTADO DEL EJERCICIO 🏆".center(60))
        print("=" * 60)

        if completed:
            print("\n¡Felicidades! Has completado el ejercicio correctamente.")
        else:
            print("\nSigue practicando. Recuerda que cada error es una oportunidad para aprender.")

        # Devolver estadísticas si se solicita seguimiento
        if track_progress:
            return {
                'steps_completed': steps_completed,
                'mistakes': mistakes,
                'help_requested': help_requested,
                'completed': completed
            }

        return None


class InequalityGenerator(ExerciseGenerator):
    def generate_exercise(self, level='principiante'):
        """Genera un ejercicio de desigualdades basado en el nivel de dificultad."""
        config = self.difficulty_levels.get(level, self.difficulty_levels['principiante'])
        max_value = config['max_value']

        # Estructura base para todos los niveles
        exercise = {
            'level': level,
            'inequality_type': '',
            'left_side': {},
            'right_side': {},
            'solution_range': [],
            'solution': 0
        }

        if level == 'principiante':
            # Desigualdad simple: ax > b o ax < b
            a = random.randint(1, 5)
            b = random.randint(1, max_value)

            # Elegir aleatoriamente entre > o <
            inequality_type = random.choice([">", "<"])

            # Generar solución basada en el tipo de desigualdad
            if inequality_type == ">":
                solution = (b // a) + 1  # Primer entero mayor que b/a
                solution_range = f"x > {b/a}"
            else:  # <
                solution = (b // a) - 1 if b % a != 0 else (b // a) - 1  # Primer entero menor que b/a
                solution_range = f"x < {b/a}"

            exercise['inequality_type'] = inequality_type
            exercise['left_side'] = {'coefficient': a, 'variable': 'x'}
            exercise['right_side'] = {'constant': b}
            exercise['solution_range'] = solution_range
            exercise['solution'] = solution

        elif level == 'intermedio':
            # Desigualdad con coeficientes en ambos lados: ax + b > cx o ax + b < cx
            a = random.randint(2, 6)
            b = random.randint(1, max_value)
            c = random.randint(1, a-1)  # c < a para que la desigualdad no cambie de sentido

            inequality_type = random.choice([">", "<"])

            # Cálculo de la solución: (ax + b > cx) → x > -b/(a-c) o x < -b/(a-c)
            division_result = -b / (a - c)

            if inequality_type == ">":
                # Si a-c es positivo, el sentido se mantiene
                if a - c > 0:
                    solution = math.ceil(division_result)
                    solution_range = f"x > {division_result}"
                else:
                    # Si a-c es negativo, el sentido se invierte
                    solution = math.floor(division_result)
                    solution_range = f"x < {division_result}"
            else:  # <
                # Si a-c es positivo, el sentido se mantiene
                if a - c > 0:
                    solution = math.floor(division_result)
                    solution_range = f"x < {division_result}"
                else:
                    # Si a-c es negativo, el sentido se invierte
                    solution = math.ceil(division_result)
                    solution_range = f"x > {division_result}"

            exercise['inequality_type'] = inequality_type
            exercise['left_side'] = {'coefficient': a, 'variable': 'x', 'constant': b}
            exercise['right_side'] = {'coefficient': c, 'variable': 'x'}
            exercise['solution_range'] = solution_range
            exercise['solution'] = solution

        else:  # avanzado
            # Desigualdad compleja: ax + b > cx + d o ax + b < cx + d
            a = random.randint(5, 10)
            b = random.randint(1, max_value)
            c = random.randint(1, a-2)  # c < a para mantener un caso estándar
            d = random.randint(1, max_value)

            inequality_type = random.choice([">", "<", "≥", "≤"])

            # Cálculo de la solución
            division_result = (d - b) / (a - c)

            # Determinar el rango de solución basado en el tipo de desigualdad
            if inequality_type in [">", "≥"]:
                # Si a-c es positivo, el sentido se mantiene
                if a - c > 0:
                    solution = math.ceil(division_result) if inequality_type == ">" else math.ceil(division_result - 0.5)
                    solution_range = f"x > {division_result}" if inequality_type == ">" else f"x ≥ {division_result}"
                else:
                    # Si a-c es negativo, el sentido se invierte
                    solution = math.floor(division_result) if inequality_type == ">" else math.floor(division_result + 0.5)
                    solution_range = f"x < {division_result}" if inequality_type == ">" else f"x ≤ {division_result}"
            else:  # < o ≤
                # Si a-c es positivo, el sentido se mantiene
                if a - c > 0:
                    solution = math.floor(division_result) if inequality_type == "<" else math.floor(division_result + 0.5)
                    solution_range = f"x < {division_result}" if inequality_type == "<" else f"x ≤ {division_result}"
                else:
                    # Si a-c es negativo, el sentido se invierte
                    solution = math.ceil(division_result) if inequality_type == "<" else math.ceil(division_result - 0.5)
                    solution_range = f"x > {division_result}" if inequality_type == "<" else f"x ≥ {division_result}"

            exercise['inequality_type'] = inequality_type
            exercise['left_side'] = {'coefficient': a, 'variable': 'x', 'constant': b}
            exercise['right_side'] = {'coefficient': c, 'variable': 'x', 'constant': d}
            exercise['solution_range'] = solution_range
            exercise['solution'] = solution

        return exercise

    def format_exercise(self, exercise):
        """Formatea el ejercicio de desigualdad para mostrarlo al usuario."""
        level = exercise['level']
        inequality_type = exercise['inequality_type']

        print("\n" + "=" * 60)
        print(f"🧮 EJERCICIO DE DESIGUALDADES - NIVEL: {level.upper()} 🧮".center(60))
        print("=" * 60)
        print("\n📝 Ejercicio: Resuelve la siguiente desigualdad")

        # Formatear la desigualdad según su nivel
        if level == 'principiante':
            a = exercise['left_side']['coefficient']
            b = exercise['right_side']['constant']
            print(f"\n{a}x {inequality_type} {b}")
        elif level == 'intermedio':
            a = exercise['left_side']['coefficient']
            b = exercise['left_side'].get('constant', 0)
            c = exercise['right_side']['coefficient']

            left_expression = f"{a}x"
            if b > 0:
                left_expression += f" + {b}"
            elif b < 0:
                left_expression += f" - {abs(b)}"

            right_expression = f"{c}x"

            print(f"\n{left_expression} {inequality_type} {right_expression}")
        else:  # avanzado
            a = exercise['left_side']['coefficient']
            b = exercise['left_side'].get('constant', 0)
            c = exercise['right_side']['coefficient']
            d = exercise['right_side'].get('constant', 0)

            left_expression = f"{a}x"
            if b > 0:
                left_expression += f" + {b}"
            elif b < 0:
                left_expression += f" - {abs(b)}"

            right_expression = f"{c}x"
            if d > 0:
                right_expression += f" + {d}"
            elif d < 0:
                right_expression += f" - {abs(d)}"

            print(f"\n{left_expression} {inequality_type} {right_expression}")

        print("\n💎 Encuentra un número entero que satisfaga la desigualdad.")
        return exercise

    def solve_interactive(self, exercise, track_progress=False):
        """Permite al usuario resolver el ejercicio de desigualdad interactivamente."""
        # Variables para tracking
        steps_completed = 0
        mistakes = 0
        help_requested = 0
        completed = False

        level = exercise['level']
        inequality_type = exercise['inequality_type']
        solution = exercise['solution']
        solution_range = exercise['solution_range']

        # Paso 1: Reorganizar la desigualdad
        print("\n🔍 PASO 1: Reorganiza la desigualdad para tener todos los términos con variable a un lado")

        if level == 'principiante':
            a = exercise['left_side']['coefficient']
            b = exercise['right_side']['constant']

            print(f"\nOriginal: {a}x {inequality_type} {b}")
            print("¿Cómo despejamos x? (escribe 'ayuda' para una pista)")

            user_response = input("> ").lower()

            if user_response == 'ayuda':
                help_requested += 1
                print(f"\n🔔 PISTA: Para despejar x, divide ambos lados por {a}.")
                print(f"Recuerda que al dividir por un número positivo, el sentido de la desigualdad se mantiene.")

            # Respuesta modelo
            print(f"\nSolución paso a paso:")
            print(f"1. Original: {a}x {inequality_type} {b}")
            print(f"2. Dividimos ambos lados por {a}: x {inequality_type} {b}/{a}")

            steps_completed += 1

        elif level == 'intermedio':
            a = exercise['left_side']['coefficient']
            b = exercise['left_side'].get('constant', 0)
            c = exercise['right_side']['coefficient']

            left_expression = f"{a}x"
            if b > 0:
                left_expression += f" + {b}"
            elif b < 0:
                left_expression += f" - {abs(b)}"

            right_expression = f"{c}x"

            print(f"\nOriginal: {left_expression} {inequality_type} {right_expression}")
            print("¿Cómo despejamos los términos con variable? (escribe 'ayuda' para una pista)")

            user_response = input("> ").lower()

            if user_response == 'ayuda':
                help_requested += 1
                print(f"\n🔔 PISTA: Resta {c}x a ambos lados para dejar todos los términos con x a la izquierda.")
                print(f"Luego, agrupa los términos semejantes.")

            # Respuesta modelo
            print(f"\nSolución paso a paso:")
            print(f"1. Original: {left_expression} {inequality_type} {right_expression}")
            print(f"2. Restamos {c}x a ambos lados: {a}x - {c}x", end="")
            if b > 0:
                print(f" + {b}", end="")
            elif b < 0:
                print(f" - {abs(b)}", end="")
            print(f" {inequality_type} 0")
            print(f"3. Agrupamos términos: {a-c}x", end="")
            if b > 0:
                print(f" + {b}", end="")
            elif b < 0:
                print(f" - {abs(b)}", end="")
            print(f" {inequality_type} 0")

            steps_completed += 1

        else:  # avanzado
            a = exercise['left_side']['coefficient']
            b = exercise['left_side'].get('constant', 0)
            c = exercise['right_side']['coefficient']
            d = exercise['right_side'].get('constant', 0)

            left_expression = f"{a}x"
            if b > 0:
                left_expression += f" + {b}"
            elif b < 0:
                left_expression += f" - {abs(b)}"

            right_expression = f"{c}x"
            if d > 0:
                right_expression += f" + {d}"
            elif d < 0:
                right_expression += f" - {abs(d)}"

            print(f"\nOriginal: {left_expression} {inequality_type} {right_expression}")
            print("¿Cómo reorganizamos la desigualdad? (escribe 'ayuda' para una pista)")

            user_response = input("> ").lower()

            if user_response == 'ayuda':
                help_requested += 1
                print(f"\n🔔 PISTA: Pasos para reorganizar:")
                print(f"1. Resta {c}x a ambos lados para dejar los términos con x a la izquierda.")
                print(f"2. Resta {d} a ambos lados para dejar los términos constantes a la derecha.")
                print(f"3. Agrupa términos semejantes.")

            # Respuesta modelo
            print(f"\nSolución paso a paso:")
            print(f"1. Original: {left_expression} {inequality_type} {right_expression}")
            print(f"2. Restamos {c}x a ambos lados: {a}x - {c}x", end="")
            if b > 0:
                print(f" + {b}", end="")
            elif b < 0:
                print(f" - {abs(b)}", end="")
            print(f" {inequality_type}", end="")
            if d > 0:
                print(f" {d}")
            elif d < 0:
                print(f" -{abs(d)}")
            else:
                print(" 0")

            print(f"3. Agrupamos términos con x: {a-c}x", end="")
            if b > 0:
                print(f" + {b}", end="")
            elif b < 0:
                print(f" - {abs(b)}", end="")
            print(f" {inequality_type}", end="")
            if d > 0:
                print(f" {d}")
            elif d < 0:
                print(f" -{abs(d)}")
            else:
                print(" 0")

            if b != 0 or d != 0:
                print(f"4. Movemos términos constantes: {a-c}x {inequality_type}", end="")
                if d > 0 and b > 0:
                    print(f" {d} - {b} = {d-b}")
                elif d > 0 and b < 0:
                    print(f" {d} + {abs(b)} = {d+abs(b)}")
                elif d < 0 and b > 0:
                    print(f" {d} - {b} = {d-b}")
                elif d < 0 and b < 0:
                    print(f" {d} + {abs(b)} = {d+abs(b)}")
                elif d == 0 and b > 0:
                    print(f" -{b}")
                elif d == 0 and b < 0:
                    print(f" {abs(b)}")
                elif d > 0 and b == 0:
                    print(f" {d}")
                elif d < 0 and b == 0:
                    print(f" {d}")

            steps_completed += 1

        # Paso 2: Despejar x
        print("\n🔍 PASO 2: Despeja la variable x")

        if level == 'principiante':
            print(f"\nTenemos x {inequality_type} {b}/{a}")
            print(f"Por lo tanto, el conjunto solución es {solution_range}")

        elif level == 'intermedio':
            print(f"\nTenemos {a-c}x", end="")
            if b > 0:
                print(f" + {b}", end="")
            elif b < 0:
                print(f" - {abs(b)}", end="")
            print(f" {inequality_type} 0")

            print(f"Despejando x:")

            # Verificar si (a-c) es positivo o negativo (afecta el sentido de la desigualdad)
            if a - c > 0:
                print(f"1. Restamos {b} a ambos lados: {a-c}x {inequality_type} -{b}")
                print(f"2. Dividimos por {a-c}: x {inequality_type} -{b}/{a-c}")
                print(f"3. Simplificando: x {inequality_type} {-b/(a-c)}")
            else:
                # Si a-c es negativo, el sentido de la desigualdad cambia al dividir
                opposite_inequality = "<" if inequality_type == ">" else ">"
                print(f"1. Restamos {b} a ambos lados: {a-c}x {inequality_type} -{b}")
                print(f"2. Dividimos por {a-c} (negativo, por lo que cambia el sentido): x {opposite_inequality} -{b}/{a-c}")
                print(f"3. Simplificando: x {opposite_inequality} {-b/(a-c)}")

            print(f"Por lo tanto, el conjunto solución es {solution_range}")

        else:  # avanzado
            final_right = d - b
            print(f"\nTenemos {a-c}x {inequality_type} {final_right}")
            print(f"Despejando x:")

            # Verificar si (a-c) es positivo o negativo
            if a - c > 0:
                print(f"1. Dividimos por {a-c}: x {inequality_type} {final_right}/{a-c}")
                print(f"2. Simplificando: x {inequality_type} {final_right/(a-c)}")
            else:
                # Si a-c es negativo, el sentido de la desigualdad cambia
                opposite_inequality = {">": "<", "<": ">", "≥": "≤", "≤": "≥"}[inequality_type]
                print(f"1. Dividimos por {a-c} (negativo, cambia el sentido): x {opposite_inequality} {final_right}/{a-c}")
                print(f"2. Simplificando: x {opposite_inequality} {final_right/(a-c)}")

            print(f"Por lo tanto, el conjunto solución es {solution_range}")

        steps_completed += 1

        # Paso 3: Encontrar una solución entera
        print("\n🔍 PASO 3: Encuentra un número entero que satisfaga la desigualdad")
        print(f"\n¿Cuál es un número entero que satisface {solution_range}? (escribe 'ayuda' para una pista)")

        user_answer = input("> ").lower()

        if user_answer == 'ayuda':
            help_requested += 1
            print(f"\n🔔 PISTA: Busca un número entero que cumpla {solution_range}.")
            print(f"Una posible respuesta es {solution}.")

            user_answer = input("\n¿Cuál es tu respuesta ahora? > ")

        try:
            user_value = int(user_answer)

            # Verificar si la respuesta satisface la desigualdad
            satisfied = False

            # Evaluar según el tipo de desigualdad y el valor límite
            limit_value = float(solution_range.split(" ")[2])
            inequality_symbol = solution_range.split(" ")[1]

            if inequality_symbol == ">":
                satisfied = user_value > limit_value
            elif inequality_symbol == "<":
                satisfied = user_value < limit_value
            elif inequality_symbol == "≥":
                satisfied = user_value >= limit_value
            elif inequality_symbol == "≤":
                satisfied = user_value <= limit_value

            if satisfied:
                print(f"\n✅ ¡Correcto! {user_value} satisface la desigualdad {solution_range}.")
                steps_completed += 1
                completed = True
            else:
                print(f"\n❌ Incorrecto. {user_value} no satisface la desigualdad {solution_range}.")
                print(f"Una posible respuesta correcta es {solution}.")
                mistakes += 1

        except ValueError:
            print("\n⚠️ Por favor, ingresa un número entero válido.")
            mistakes += 1

        # Resultados del ejercicio
        print("\n" + "=" * 60)
        print("🏆 RESULTADO DEL EJERCICIO 🏆".center(60))
        print("=" * 60)

        if completed:
            print("\n¡Felicidades! Has completado el ejercicio correctamente.")
        else:
            print("\nSigue practicando. Recuerda que cada error es una oportunidad para aprender.")

        # Devolver estadísticas si se solicita seguimiento
        if track_progress:
            return {
                'steps_completed': steps_completed,
                'mistakes': mistakes,
                'help_requested': help_requested,
                'completed': completed
            }

        return None


class ProportionalityConstantGenerator(ExerciseGenerator):
    def generate_exercise(self, level='principiante'):
        """Genera un ejercicio de constante de proporcionalidad basado en el nivel de dificultad."""
        config = self.difficulty_levels.get(level, self.difficulty_levels['principiante'])

        # Determinar si será proporcionalidad directa o inversa
        if level == 'principiante':
            # En nivel principiante solo proporcionalidad directa con números enteros
            is_direct = True
            k = random.randint(2, 5)  # Constante de proporcionalidad sencilla
            x1 = random.randint(1, config['max_value'])
            y1 = k * x1

            # Generar segundo punto
            x2 = random.randint(1, config['max_value'])
            while x2 == x1:  # Evitar valores repetidos
                x2 = random.randint(1, config['max_value'])
            y2 = k * x2

        elif level == 'intermedio':
            # En nivel intermedio puede ser directa o inversa
            is_direct = random.choice([True, False])

            if is_direct:
                # Constante de proporcionalidad puede ser fracción sencilla
                numerator = random.randint(1, 5)
                denominator = random.randint(1, 5)
                k = Fraction(numerator, denominator)

                x1 = random.randint(1, config['max_value'])
                y1 = k * x1

                x2 = random.randint(1, config['max_value'])
                while x2 == x1:
                    x2 = random.randint(1, config['max_value'])
                y2 = k * x2
            else:
                # Proporcionalidad inversa: y = k/x
                k = random.randint(12, 60)  # Valor que dé resultados enteros

                # Elegir valores de x que den resultados enteros para k
                divisors = [i for i in range(1, k+1) if k % i == 0]
                x1 = random.choice(divisors)
                y1 = k // x1

                x2 = random.choice([d for d in divisors if d != x1])
                y2 = k // x2

        else:  # avanzado
            # En nivel avanzado puede ser directa o inversa con valores más complejos
            is_direct = random.choice([True, False])

            if is_direct:
                # Constante puede ser decimal o fracción más compleja
                if random.choice([True, False]):
                    k = round(random.uniform(0.5, 5.0), 2)
                else:
                    numerator = random.randint(1, 10)
                    denominator = random.randint(2, 10)
                    k = Fraction(numerator, denominator)

                x1 = random.randint(1, config['max_value'])
                y1 = k * x1

                x2 = random.randint(1, config['max_value'])
                while x2 == x1:
                    x2 = random.randint(1, config['max_value'])
                y2 = k * x2

                # Añadir un tercer punto para verificación
                x3 = random.randint(1, config['max_value'])
                while x3 in [x1, x2]:
                    x3 = random.randint(1, config['max_value'])
                y3 = k * x3
            else:
                # Proporcionalidad inversa con valores más complejos
                k = random.randint(50, 200)

                # Elegir valores de x que den resultados razonables
                x1 = random.randint(2, 20)
                y1 = k / x1

                x2 = random.randint(2, 20)
                while x2 == x1:
                    x2 = random.randint(2, 20)
                y2 = k / x2

                # Tercer punto para verificación
                x3 = random.randint(2, 20)
                while x3 in [x1, x2]:
                    x3 = random.randint(2, 20)
                y3 = k / x3

        # Crear el ejercicio
        exercise = {
            'level': level,
            'is_direct': is_direct,
            'constant': k,
            'points': [
                {'x': x1, 'y': y1},
                {'x': x2, 'y': y2}
            ]
        }

        # Añadir tercer punto para nivel avanzado
        if level == 'avanzado':
            exercise['points'].append({'x': x3, 'y': y3})

        # Añadir contexto según el tipo de proporcionalidad
        contexts_direct = [
            {'name': 'Velocidad y tiempo', 'x_unit': 'horas', 'y_unit': 'kilómetros', 'description': 'distancia recorrida'},
            {'name': 'Precio por unidad', 'x_unit': 'unidades', 'y_unit': 'euros', 'description': 'costo total'},
            {'name': 'Consumo de combustible', 'x_unit': 'litros', 'y_unit': 'kilómetros', 'description': 'distancia recorrida'}
        ]

        contexts_inverse = [
            {'name': 'Velocidad y tiempo', 'x_unit': 'velocidad (km/h)', 'y_unit': 'tiempo (h)', 'description': 'tiempo necesario para recorrer una distancia fija'},
            {'name': 'Trabajadores y tiempo', 'x_unit': 'trabajadores', 'y_unit': 'días', 'description': 'tiempo para completar un trabajo'},
            {'name': 'Presión y volumen', 'x_unit': 'volumen (L)', 'y_unit': 'presión (atm)', 'description': 'relación entre presión y volumen a temperatura constante'}
        ]

        exercise['context'] = random.choice(contexts_direct if is_direct else contexts_inverse)

        return exercise

    def format_exercise(self, exercise):
        """Formatea el ejercicio de constante de proporcionalidad para mostrarlo al usuario."""
        level = exercise['level']
        is_direct = exercise['is_direct']
        points = exercise['points']
        context = exercise['context']

        prop_type = "directa" if is_direct else "inversa"

        print("\n" + "=" * 60)
        print(f"📊 EJERCICIO DE CONSTANTE DE PROPORCIONALIDAD - NIVEL: {level.upper()} 📊".center(60))
        print("=" * 60)

        print(f"\n📝 Contexto: {context['name']}")
        print(f"\nSe tiene una relación de proporcionalidad {prop_type} entre:")
        print(f"• Variable X: {context['x_unit']}")
        print(f"• Variable Y: {context['y_unit']} ({context['description']})")

        print("\nDados los siguientes pares de valores:")

        # Crear una tabla con los datos
        print("\n┌" + "─" * 30 + "┐")
        print("│  X (" + context['x_unit'].ljust(10) + ")│  Y (" + context['y_unit'].ljust(10) + ")│")
        print("├" + "─" * 30 + "┤")

        for point in points:
            x_val = point['x']
            y_val = point['y']

            # Formatear números para mejor visualización
            if isinstance(y_val, Fraction):
                y_display = str(y_val)
            elif isinstance(y_val, float):
                y_display = f"{y_val:.2f}" if y_val % 1 != 0 else str(int(y_val))
            else:
                y_display = str(y_val)

            if isinstance(x_val, Fraction):
                x_display = str(x_val)
            elif isinstance(x_val, float):
                x_display = f"{x_val:.2f}" if x_val % 1 != 0 else str(int(x_val))
            else:
                x_display = str(x_val)

            print("│  " + x_display.ljust(13) + "│  " + y_display.ljust(13) + "│")

        print("└" + "─" * 30 + "┘")

        print("\n🎯 Objetivos:")
        print("1. Determinar si se trata de una proporcionalidad directa o inversa")
        print("2. Calcular la constante de proporcionalidad (k)")
        print("3. Escribir la fórmula que relaciona X e Y")

        if level == 'avanzado':
            print("4. Verificar la fórmula con el tercer punto dado")

        return exercise

    def solve_interactive(self, exercise, track_progress=False):
        """Permite al usuario resolver el ejercicio interactivamente con seguimiento."""
        # Variables para tracking
        steps_completed = 0
        mistakes = 0
        help_requested = 0
        completed = False

        level = exercise['level']
        is_direct = exercise['is_direct']
        k = exercise['constant']
        points = exercise['points']
        context = exercise['context']

        # Paso 1: Identificar el tipo de proporcionalidad
        print("\n🔍 PASO 1: Identifica el tipo de proporcionalidad")

        print("\n¿Qué tipo de proporcionalidad existe entre X e Y?")
        print("A) Proporcionalidad directa")
        print("B) Proporcionalidad inversa")
        print("C) No hay proporcionalidad")
        print("\nEscribe tu respuesta (A, B, C o 'ayuda' para una pista):")

        user_answer = input("> ").upper()

        if user_answer == 'AYUDA':
            help_requested += 1
            print("\n🔔 PISTA:")
            print("• En proporcionalidad directa: si X aumenta, Y también aumenta; o si X disminuye, Y también disminuye.")
            print("• En proporcionalidad inversa: si X aumenta, Y disminuye; o si X disminuye, Y aumenta.")
            print("• Puedes verificar calculando Y/X (para directa) o X×Y (para inversa) y ver si es constante.")
            print("\n¿Cuál es tu respuesta ahora?")
            user_answer = input("> ").upper()

        correct_answer = "A" if is_direct else "B"

        if user_answer == correct_answer:
            print("\n✅ ¡Correcto! Has identificado correctamente el tipo de proporcionalidad.")
            steps_completed += 1
        else:
            print(f"\n❌ No es correcto. Se trata de una proporcionalidad {('directa' if is_direct else 'inversa')}.")
            mistakes += 1

            # Explicación del error
            if is_direct:
                print("\nExplicación: Observa que cuando X aumenta, Y también aumenta de manera proporcional.")
                print(f"Por ejemplo, si X pasa de {points[0]['x']} a {points[1]['x']}, Y pasa de {points[0]['y']} a {points[1]['y']}.")
            else:
                print("\nExplicación: Observa que cuando X aumenta, Y disminuye, o viceversa.")
                print(f"Por ejemplo, si X pasa de {points[0]['x']} a {points[1]['x']}, Y pasa de {points[0]['y']} a {points[1]['y']}.")

        # Paso 2: Calcular la constante de proporcionalidad
        print("\n🔍 PASO 2: Calcula la constante de proporcionalidad (k)")

        print("\n¿Cómo calcularías la constante de proporcionalidad?")
        if is_direct:
            print("A) k = X × Y")
            print("B) k = Y / X")
            print("C) k = X / Y")
        else:
            print("A) k = X + Y")
            print("B) k = Y / X")
            print("C) k = X × Y")

        print("\nEscribe tu respuesta (A, B, C o 'ayuda' para una pista):")
        user_answer = input("> ").upper()

        if user_answer == 'AYUDA':
            help_requested += 1
            print("\n🔔 PISTA:")
            if is_direct:
                print("• En proporcionalidad directa: Y = k × X, por lo tanto k = Y / X")
                print(f"• Prueba calculando Y/X para ambos pares de valores y verifica si obtienes el mismo resultado.")
            else:
                print("• En proporcionalidad inversa: Y = k / X, o equivalentemente X × Y = k")
                print(f"• Prueba calculando X×Y para ambos pares de valores y verifica si obtienes el mismo resultado.")
            print("\n¿Cuál es tu respuesta ahora?")
            user_answer = input("> ").upper()

        correct_formula = "B" if is_direct else "C"

        if user_answer == correct_formula:
            print("\n✅ ¡Correcto! Has identificado la fórmula correcta para calcular k.")
            steps_completed += 1
        else:
            print(f"\n❌ No es correcto. La fórmula correcta es la opción {correct_formula}.")
            mistakes += 1

            # Explicación del error
            if is_direct:
                print("\nExplicación: En proporcionalidad directa, Y = k × X, por lo tanto k = Y / X")
            else:
                print("\nExplicación: En proporcionalidad inversa, Y = k / X, o equivalentemente X × Y = k")

        # Paso 3: Calcular el valor de k
        print("\n🔍 PASO 3: Calcula el valor numérico de k")

        if is_direct:
            k1 = points[0]['y'] / points[0]['x']
            k2 = points[1]['y'] / points[1]['x']
            print(f"\nUsando el primer par de valores (X = {points[0]['x']}, Y = {points[0]['y']}):")
            print(f"k = Y / X = {points[0]['y']} / {points[0]['x']}")

            print(f"\nUsando el segundo par de valores (X = {points[1]['x']}, Y = {points[1]['y']}):")
            print(f"k = Y / X = {points[1]['y']} / {points[1]['x']}")
        else:
            k1 = points[0]['x'] * points[0]['y']
            k2 = points[1]['x'] * points[1]['y']
            print(f"\nUsando el primer par de valores (X = {points[0]['x']}, Y = {points[0]['y']}):")
            print(f"k = X × Y = {points[0]['x']} × {points[0]['y']}")

            print(f"\nUsando el segundo par de valores (X = {points[1]['x']}, Y = {points[1]['y']}):")
            print(f"k = X × Y = {points[1]['x']} × {points[1]['y']}")

        print("\n¿Cuál es el valor de k? (o escribe 'ayuda' para una pista)")
        user_k = input("> ").lower()

        if user_k == 'ayuda':
            help_requested += 1
            if is_direct:
                print(f"\n🔔 PISTA: k = Y / X = {points[0]['y']} / {points[0]['x']} = {k}")
            else:
                print(f"\n🔔 PISTA: k = X × Y = {points[0]['x']} × {points[0]['y']} = {k}")
            print("\n¿Cuál es tu respuesta ahora?")
            user_k = input("> ")

        # Verificar respuesta
        try:
            # Convertir la respuesta del usuario a fracción o flotante según corresponda
            if '/' in user_k:
                parts = user_k.split('/')
                user_k_value = float(parts[0]) / float(parts[1])
            else:
                user_k_value = float(user_k)

            # Convertir k a float para comparación si es una fracción
            k_value = float(k.numerator) / float(k.denominator) if isinstance(k, Fraction) else float(k)

            # Comprobar con un margen de error para decimales
            if abs(user_k_value - k_value) < 0.01:
                print("\n✅ ¡Correcto! Has calculado correctamente el valor de k.")
                steps_completed += 1
            else:
                print(f"\n❌ No es correcto. El valor de k es {k}")
                mistakes += 1
        except ValueError:
            print("\n⚠️ Por favor, ingresa un número válido.")
            mistakes += 1

        # Paso 4: Escribir la fórmula
        print("\n🔍 PASO 4: Escribe la fórmula que relaciona X e Y")

        if is_direct:
            print("\nEn una proporcionalidad directa, la fórmula general es:")
            print("Y = k × X")
        else:
            print("\nEn una proporcionalidad inversa, la fórmula general es:")
            print("Y = k / X")

        print("\nUtilizando el valor de k que calculaste, ¿cuál es la fórmula específica?")
        print("(Escribe la fórmula completa utilizando el valor de k, o 'ayuda' para una pista)")

        user_formula = input("> ").lower()

        if user_formula == 'ayuda':
            help_requested += 1
            if is_direct:
                print(f"\n🔔 PISTA: La fórmula es Y = {k} × X")
            else:
                print(f"\n🔔 PISTA: La fórmula es Y = {k} / X")
            print("\n¿Cuál es tu respuesta ahora?")
            user_formula = input("> ")

        # Verificar la fórmula (esta es una verificación flexible)
        correct_formula = False

        # Convertir k a string para comparación
        k_str = str(k)
        if isinstance(k, float):
            k_str = f"{k:.2f}".rstrip('0').rstrip('.') if k % 1 != 0 else str(int(k))

        # Normalizar la respuesta del usuario para comparación
        user_formula = user_formula.replace(" ", "").replace("*", "×").lower()

        if is_direct:
            expected_patterns = [
                f"y={k_str}x", f"y={k_str}*x", f"y={k_str}×x",
                f"y={k_str}·x", f"y={k_str}(x)"
            ]
        else:
            expected_patterns = [
                f"y={k_str}/x", f"y={k_str}÷x", f"y={k_str}:x",
                f"y={k_str}x^-1", f"y={k_str}/x"
            ]

        for pattern in expected_patterns:
            if pattern.replace(" ", "").lower() in user_formula:
                correct_formula = True
                break

        if correct_formula:
            print("\n✅ ¡Correcto! Has escrito la fórmula adecuadamente.")
            steps_completed += 1
        else:
            print("\n❌ No es correcto.")
            if is_direct:
                print(f"La fórmula correcta es: Y = {k} × X")
            else:
                print(f"La fórmula correcta es: Y = {k} / X")
            mistakes += 1

        # Paso 5 (opcional para nivel avanzado): Verificar con un tercer punto
        if level == 'avanzado':
            print("\n🔍 PASO 5: Verifica la fórmula con el tercer punto dado")

            x3 = points[2]['x']
            y3 = points[2]['y']

            print(f"\nTenemos el tercer par de valores: X = {x3}, Y = {y3}")
            print("Vamos a verificar si estos valores se ajustan a nuestra fórmula.")

            if is_direct:
                expected_y = k * x3
                print(f"\nSi Y = {k} × X, entonces para X = {x3}:")
                print(f"Y = {k} × {x3} = {expected_y}")
            else:
                expected_y = k / x3
                print(f"\nSi Y = {k} / X, entonces para X = {x3}:")
                print(f"Y = {k} / {x3} = {expected_y}")

            print(f"\n¿El valor calculado de Y ({expected_y}) coincide con el valor dado ({y3})? (sí/no)")
            user_verification = input("> ").lower()

            if user_verification in ['sí', 'si', 's', 'yes', 'y']:
                print("\n✅ ¡Correcto! Has verificado que la fórmula funciona para todos los puntos dados.")
                steps_completed += 1
            else:
                print("\n⚠️ Revisa tus cálculos. La fórmula debe funcionar para todos los puntos dados.")

        # Resultado final
        print("\n" + "=" * 60)
        print("📊 RESUMEN DEL EJERCICIO 📊".center(60))
        print("=" * 60)

        if is_direct:
            print(f"\n• Tipo de proporcionalidad: Directa")
            print(f"• Constante de proporcionalidad (k): {k}")
            print(f"• Fórmula: Y = {k} × X")
        else:
            print(f"\n• Tipo de proporcionalidad: Inversa")
            print(f"• Constante de proporcionalidad (k): {k}")
            print(f"• Fórmula: Y = {k} / X")

        print(f"\n• Contexto: {context['name']}")
        print(f"• Interpretación: {k} {context['y_unit']} por cada {context['x_unit'] if is_direct else '1/' + context['x_unit']}")

        # Establecer si se completó con éxito
        if steps_completed >= (4 if level != 'avanzado' else 5):
            completed = True

        # Devolver información de seguimiento si se solicitó
        if track_progress:
            return {
                'steps_completed': steps_completed,
                'mistakes': mistakes,
                'help_requested': help_requested,
                'completed': completed
            }

        return completed
    import os

# Crear archivo vacío para el modelo si no existe
if not os.path.exists('tutor_recommendation_model.joblib'):
    with open('tutor_recommendation_model.joblib', 'w') as f:
        pass
tutor_system = IntelligentTutorSystem()
tutor_system.start_session()
