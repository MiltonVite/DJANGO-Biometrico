from django.db import models
import cv2
import numpy as np
import pickle
from PIL import Image
import os

# Funciones de procesamiento de imágenes (movidas desde views.py)
def extract_face_features(image):
    """
    Extrae características faciales usando OpenCV
    Retorna un vector de características único para cada rostro
    """
    # Convertir a escala de grises
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Cargar clasificador Haar para detección de rostros
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Detectar rostros
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )
    
    if len(faces) == 0:
        return None
    
    # Tomar el primer rostro detectado
    (x, y, w, h) = faces[0]
    face_roi = gray[y:y+h, x:x+w]
    
    # Redimensionar a tamaño fijo para consistencia
    face_roi = cv2.resize(face_roi, (100, 100))
    
    # Crear vector de características usando múltiples métodos
    features = []
    
    # 1. Histograma de la imagen
    hist = cv2.calcHist([face_roi], [0], None, [256], [0, 256])
    features.extend(hist.flatten())
    
    # 2. LBP (Local Binary Patterns) simplificado
    lbp_features = calculate_lbp_features(face_roi)
    features.extend(lbp_features)
    
    # 3. Momentos de Hu
    moments = cv2.moments(face_roi)
    hu_moments = cv2.HuMoments(moments)
    features.extend(hu_moments.flatten())
    
    # 4. Características de textura usando filtros Gabor
    gabor_features = apply_gabor_filters(face_roi)
    features.extend(gabor_features)
    
    return np.array(features, dtype=np.float32)

def calculate_lbp_features(image):
    """Calcula características LBP (Local Binary Pattern) simplificado"""
    rows, cols = image.shape
    lbp = np.zeros((rows-2, cols-2))
    
    for i in range(1, rows-1):
        for j in range(1, cols-1):
            center = image[i, j]
            binary_string = ''
            
            # Comparar con 8 vecinos
            neighbors = [
                image[i-1, j-1], image[i-1, j], image[i-1, j+1],
                image[i, j+1], image[i+1, j+1], image[i+1, j],
                image[i+1, j-1], image[i, j-1]
            ]
            
            for neighbor in neighbors:
                binary_string += '1' if neighbor >= center else '0'
            
            lbp[i-1, j-1] = int(binary_string, 2)
    
    # Calcular histograma LBP
    hist_lbp = np.histogram(lbp, bins=256, range=(0, 256))[0]
    return hist_lbp.astype(np.float32)

def apply_gabor_filters(image):
    """Aplica filtros Gabor para extraer características de textura"""
    features = []
    
    # Parámetros del filtro Gabor
    angles = [0, 45, 90, 135]  # Diferentes orientaciones
    frequencies = [0.1, 0.3]   # Diferentes frecuencias
    
    for angle in angles:
        for freq in frequencies:
            # Crear kernel Gabor
            kernel = cv2.getGaborKernel((21, 21), 5, np.radians(angle), 2*np.pi*freq, 0.5, 0, ktype=cv2.CV_32F)
            
            # Aplicar filtro
            filtered = cv2.filter2D(image, cv2.CV_8UC3, kernel)
            
            # Extraer estadísticas
            features.append(np.mean(filtered))
            features.append(np.std(filtered))
    
    return features

def compare_faces_opencv(features1, features2):
    """
    Compara dos vectores de características faciales
    Retorna una puntuación de similitud (0-1, donde 1 es idéntico)
    """
    if features1 is None or features2 is None:
        return 0.0
    
    # Asegurar que ambos son arrays numpy
    features1 = np.array(features1, dtype=np.float32)
    features2 = np.array(features2, dtype=np.float32)
    
    # Normalizar características
    norm1 = np.linalg.norm(features1)
    norm2 = np.linalg.norm(features2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    features1_norm = features1 / norm1
    features2_norm = features2 / norm2
    
    # Calcular similitud coseno
    similarity = np.dot(features1_norm, features2_norm)
    
    # Convertir a rango 0-1 y asegurar que es float nativo
    similarity = float((similarity + 1) / 2)
    
    return max(0.0, min(1.0, similarity))


class Person(models.Model):
    name = models.CharField(max_length=100)
    email = models.EmailField(blank=True)
    photo = models.ImageField(upload_to='faces/')
    face_encoding = models.BinaryField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return self.name
    
    def save(self, *args, **kwargs):
        # Guardar primero para tener el archivo
        is_new = self.pk is None
        super().save(*args, **kwargs)
        
        # Procesar la imagen para extraer características faciales con OpenCV
        if self.photo and (is_new or not self.face_encoding):
            try:
                self.extract_face_features_opencv()
            except Exception as e:
                print(f"Error procesando imagen: {e}")
    
    def extract_face_features_opencv(self):
        """Extrae características faciales usando OpenCV"""
        try:
            # Cargar imagen
            image = cv2.imread(self.photo.path)
            if image is None:
                print("No se pudo cargar la imagen")
                return
            
            # Extraer características
            features = extract_face_features(image)
            
            if features is not None:
                # Serializar y guardar las características
                self.face_encoding = pickle.dumps(features)
                # Usar update para evitar recursión infinita
                Person.objects.filter(pk=self.pk).update(face_encoding=self.face_encoding)
                print(f"Características extraídas para {self.name}: {len(features)} features")
            else:
                print(f"No se detectó rostro en la imagen de {self.name}")
                
        except Exception as e:
            print(f"Error extrayendo características para {self.name}: {e}")

class RecognitionLog(models.Model):
    person = models.ForeignKey(Person, on_delete=models.CASCADE, null=True, blank=True)
    recognition_time = models.DateTimeField(auto_now_add=True)
    confidence = models.FloatField(default=0.0)
    image = models.ImageField(upload_to='recognitions/', blank=True)
    
    class Meta:
        ordering = ['-recognition_time']
    
    def __str__(self):
        if self.person:
            return f"{self.person.name} - {self.confidence:.2%} - {self.recognition_time}"
        return f"Desconocido - {self.recognition_time}"