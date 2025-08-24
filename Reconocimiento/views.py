from django.shortcuts import render, redirect
from django.contrib import messages
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.core.files.base import ContentFile
from .models import Person, RecognitionLog, extract_face_features, compare_faces_opencv
import cv2
import numpy as np
import pickle
import base64
import io
from PIL import Image
import os
from django.conf import settings

def index(request):
    return render(request, 'reconocerFaces/index.html')

def register_person(request):
    if request.method == 'POST':
        name = request.POST.get('name')
        email = request.POST.get('email', '')
        
        # Procesar imagen desde canvas
        if 'image_data' in request.POST:
            image_data = request.POST['image_data']
            
            # Decodificar imagen base64
            format, imgstr = image_data.split(';base64,')
            ext = format.split('/')[-1]
            
            # Crear archivo de imagen
            image_file = ContentFile(
                base64.b64decode(imgstr), 
                name=f'{name}.{ext}'
            )
            
            # Crear persona
            person = Person(
                name=name,
                email=email,
                photo=image_file
            )
            person.save()
            
            messages.success(request, f'Persona {name} registrada exitosamente!')
            return redirect('index')
        else:
            messages.error(request, 'No se pudo capturar la imagen')

    return render(request, 'reconocerFaces/register.html')

@csrf_exempt
def recognize_face(request):
    if request.method == 'POST':
        try:
            # Obtener imagen desde canvas
            image_data = request.POST['image_data']
            format, imgstr = image_data.split(';base64,')
            
            # Decodificar imagen
            image_bytes = base64.b64decode(imgstr)
            image = Image.open(io.BytesIO(image_bytes))
            
            # Convertir a array numpy para OpenCV
            image_np = np.array(image)
            
            # Convertir RGB a BGR para OpenCV
            if len(image_np.shape) == 3:
                image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            else:
                image_bgr = image_np
            
            # Extraer características del rostro
            current_features = extract_face_features(image_bgr)
            
            if current_features is None:
                print("❌ No se detectó ningún rostro en la imagen")
                return JsonResponse({'success': False, 'message': 'No se detectó ningún rostro'})
            
            print(f"✅ Características extraídas: {len(current_features)} features")
            
            # Obtener todas las personas registradas
            persons = Person.objects.exclude(face_encoding__isnull=True).exclude(face_encoding__exact=b'')
            
            if not persons:
                print("❌ No hay personas registradas con face_encoding")
                return JsonResponse({'success': False, 'message': 'No hay personas registradas'})
            
            print(f"👥 Comparando con {persons.count()} personas registradas")
            
            # Comparar con rostros conocidos
            best_match = None
            best_similarity = 0.0
            
            for person in persons:
                try:
                    # Cargar características almacenadas
                    stored_features = pickle.loads(person.face_encoding)
                    
                    # Calcular similitud
                    similarity = compare_faces_opencv(current_features, stored_features)
                    
                    print(f"👤 {person.name}: similitud = {similarity:.4f} ({'✅' if similarity > 0.75 else '❌'})")
                    
                    # Actualizar mejor coincidencia (umbral más estricto)
                    if similarity > best_similarity and similarity > 0.70:  # Bajamos el umbral a 70%
                        best_similarity = similarity
                        best_match = person
                        
                except Exception as e:
                    print(f"Error comparando con {person.name}: {e}")
                    continue
            
            print(f"Mejor coincidencia: {best_match.name if best_match else 'Ninguna'} - {best_similarity:.3f}")
            
            if best_match:
                print(f"🎯 RECONOCIDO: {best_match.name} con {best_similarity:.4f} de confianza")
                
                # Convertir numpy.float32 a float nativo de Python
                confidence_float = float(best_similarity)
                
                # Registrar reconocimiento
                recognition_log = RecognitionLog(
                    person=best_match,
                    confidence=confidence_float
                )
                
                # Guardar imagen del reconocimiento
                image_file = ContentFile(
                    image_bytes, 
                    name=f'recognition_{best_match.id}.jpg'
                )
                recognition_log.image = image_file
                recognition_log.save()
                
                return JsonResponse({
                    'success': True,
                    'name': best_match.name,
                    'confidence': round(confidence_float * 100, 2)
                })
            
            else:
                print(f"❌ NO RECONOCIDO - Mejor similitud: {best_similarity:.4f} (umbral: 0.70)")
            
            return JsonResponse({'success': False, 'message': 'Rostro no reconocido'})
            
        except Exception as e:
            print(f"Error en reconocimiento: {e}")
            return JsonResponse({'success': False, 'message': f'Error: {str(e)}'})

    return render(request, 'reconocerFaces/reconocer.html')

def recognition_history(request):
    logs = RecognitionLog.objects.all()[:50]  # Últimos 50 reconocimientos
    return render(request, 'reconocerFaces/history.html', {'logs': logs})

def person_list(request):
    persons = Person.objects.all()
    return render(request, 'reconocerFaces/personaList.html', {'persons': persons})

# Función auxiliar para debugging
def debug_face_recognition():
    """
    Función para depurar el sistema de reconocimiento facial
    """
    print("=== DEBUG RECONOCIMIENTO FACIAL ===")
    
    # Verificar personas registradas
    persons = Person.objects.all()
    print(f"Personas registradas: {persons.count()}")
    
    for person in persons:
        has_encoding = person.face_encoding is not None and len(person.face_encoding) > 0
        print(f"- {person.name}: {'✓' if has_encoding else '✗'} face_encoding")
        
        if has_encoding:
            try:
                features = pickle.loads(person.face_encoding)
                print(f"  Features: {len(features)} dimensiones")
            except Exception as e:
                print(f"  Error cargando features: {e}")
    
    print("=== FIN DEBUG ===")

# Función para reprocesar todas las imágenes
def reprocess_all_faces():
    """
    Reprocesa todas las imágenes de personas para extraer características
    """
    persons = Person.objects.all()
    
    for person in persons:
        if person.photo:
            print(f"Reprocesando {person.name}...")
            try:
                person.extract_face_features_opencv()
            except Exception as e:
                print(f"Error reprocesando {person.name}: {e}")
    
    print("Reprocesamiento completado")

# Función de prueba para testing
def test_opencv_detection():
    """
    Función de prueba para verificar que OpenCV está funcionando correctamente
    """
    try:
        # Crear una imagen de prueba
        test_image = np.zeros((200, 200, 3), dtype=np.uint8)
        cv2.rectangle(test_image, (50, 50), (150, 150), (255, 255, 255), -1)
        
        # Intentar detectar "rostros" (debería no encontrar ninguno)
        features = extract_face_features(test_image)
        
        print("OpenCV está funcionando correctamente")
        print(f"Características extraídas: {len(features) if features is not None else 0}")
        
        return True
    except Exception as e:
        print(f"Error en OpenCV: {e}")
        return False