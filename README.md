# Detección y Clasificación de Tumores Cerebrales con Deep Learning

Proyecto de Deep Learning para la detección y clasificación automática de tumores cerebrales en imágenes de Resonancia Magnética (MRI) utilizando Redes Neuronales Convolucionales.

## 📋 Descripción

Este proyecto implementa una Red Neuronal Convolucional (CNN) capaz de:
- **Detectar** la presencia de tumores cerebrales en imágenes MRI
- **Clasificar** el tipo de tumor en 4 categorías:
  - Glioma
  - Meningioma
  - Tumor Pituitario (Adenoma)
  - No Tumor

## 🎯 Contexto y Motivación

Anualmente se diagnostican aproximadamente 300,000 nuevos casos de tumores cerebrales en el mundo. En México, la incidencia es de 3.5 casos por cada 100,000 habitantes. La detección temprana y clasificación precisa de estos tumores es crucial para seleccionar el tratamiento más adecuado y mejorar las tasas de supervivencia de los pacientes.

## 📊 Dataset

El proyecto utiliza un conjunto público de datos disponible en Kaggle:
- **Fuente**: [Brain Tumor MRI Dataset](https://www.kaggle.com/masoudnickparvar/brain-tumor-mri-dataset)
- **Total de imágenes**: 7,023 MRIs reales del cerebro humano
- **Distribución**:
  - Entrenamiento: 5,712 imágenes (80% train, 20% validation)
  - Prueba: 1,311 imágenes
- **Clases**: 4 (glioma, meningioma, pituitary, no tumor)

## 🔧 Preprocesamiento

Todas las imágenes fueron procesadas para:
- Normalizar tamaño a **256×256 píxeles**
- Convertir a **escala de grises**
- Eliminar ruido y marcos
- Centrar las imágenes

Durante el entrenamiento se aplicó **Data Augmentation** mediante:
- Rotaciones
- Acercamientos (zoom)
- Volteos horizontales

## 🏗️ Arquitectura de la CNN

### Capas de Entrada
- Capa convolucional: 256 filtros (7×7), stride=2, activación ReLU
- Max Pooling: tamaño 3×3, stride=2

### Capas Intermedias
1. Conv2D: 256 → 256 filtros (ReLU)
2. Conv2D: 256 → 512 filtros (ReLU)
3. Conv2D: 512 → 256 filtros (ReLU)
4. Conv2D: 256 → 128 filtros (ReLU)

### Capas de Salida
- Global Average Pooling: 128 → 4
- Dense Layer: 4 clases con activación Softmax

**Parámetros totales**: ~15 millones

## 📈 Resultados

| Conjunto | Categorical Cross-Entropy | Accuracy |
|----------|---------------------------|----------|
| Entrenamiento | 0.43 | 84.1% |
| Validación | 0.34 | 89.4% |
| Prueba | 0.51 | 81.1% |

El modelo alcanza una precisión del **81.1%** en el conjunto de prueba, con una variación de 3.6% respecto al conjunto de entrenamiento.

## 🛠️ Tecnologías

- **Python 3**
- **TensorFlow 2.7.0**
- **Keras 2.7.0**
- **TFRecord** (formato de almacenamiento optimizado)

## 📝 Métricas de Evaluación

- **Categorical Cross-Entropy**: Cuantifica la diferencia entre las distribuciones de probabilidad predichas y reales
- **Categorical Accuracy**: Porcentaje de predicciones correctas sobre el total

## 🔮 Mejoras Futuras

- Entrenar por más épocas para mejorar las métricas en datos de prueba
- Agregar métricas de evaluación por categoría/clase
- Explorar diferentes arquitecturas de CNN (ResNet, EfficientNet, etc.)
- Aumentar el conjunto de datos de entrenamiento
- Implementar técnicas de regularización adicionales
- Desarrollar una aplicación web para uso clínico

## 👨‍💻 Autor

**Cristian Armando Flores Álvarez**

Proyecto Final - Diplomado en Ciencia de Datos  
DGTIC - UNAM  
Enero 2022

## 📚 Referencias

- [La guía definitiva de las Redes Neuronales Convolucionales](https://frogames.es/la-guia-definitiva-de-las-redes-neuronales-convolucionales/)
- [Categorical Crossentropy - Peltarion](https://peltarion.com/knowledge-center/documentation/modeling-view/build-an-ai-model/loss-functions/categorical-crossentropy)
- [Keras Accuracy Metrics](https://towardsdatascience.com/keras-accuracy-metrics)
- [UNAM - Boletín sobre tumores cerebrales](https://www.dgcs.unam.mx/boletin/bdboletin/2020_580.html)

## 📄 Licencia

Este proyecto fue desarrollado con fines educativos como parte del Diplomado en Ciencia de Datos de la UNAM.

---

⭐ Si este proyecto te resulta útil, no olvides darle una estrella!