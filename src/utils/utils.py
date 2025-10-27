import matplotlib.pyplot as plt
from math import floor, ceil, sqrt
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, MaxPooling2D
from tensorflow.keras.layers import GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.layers import RandomFlip, RandomRotation, RandomZoom
import os

def load_data(labels, path, str_to_index):
    imgs = []
    labels_ = []
    categories = []
    for e in labels:
        for f in os.listdir(path + '/' + e):
            imgs.append(path + '/' + e + '/' + f)
            labels_.append(e)
            categories.append(str_to_index[e])
    return imgs, labels_, categories

def muestra_imagenes(imgs, etiq, pred = False):
    t_muestra = imgs.shape[0]
    n_filas = floor(sqrt(t_muestra))
    n_cols = ceil(sqrt(t_muestra))
    fig, ax = plt.subplots(nrows=n_filas, ncols=n_cols, figsize=(2 * n_filas, 2 * n_cols))
    for i in range(n_filas):
        for j in range(n_cols):
            idx = i * n_filas + j
            ax[i, j].imshow(imgs[idx])
            ax[i, j].axis('off')
            ax[i, j].set_title(etiq[idx])
            if pred:
                c = 'darkblue' if etiq[idx] == pred[idx] else 'darkred'
                ax[i, j].text(5, 2, pred[idx], color = c)
    plt.tight_layout()
    plt.show()

def procesa_imagen_ent(ruta, categoria, imsize=(256,256)):
    imagen = tf.io.read_file(ruta)
    imagen = tf.image.decode_jpeg(imagen, channels=3)
    imagen = tf.image.resize(imagen, imsize)
    imagen /= 255.0
    imagen = tf.image.random_flip_left_right(imagen)
    imagen = tf.image.random_flip_up_down(imagen)
    imagen = tf.image.random_crop(imagen, size=[256, 256, 3])
    imagen = tf.image.rot90(imagen, tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))
    imagen = tf.clip_by_value(imagen, 0, 1)
    return imagen, categoria

def procesa_imagen_val(ruta, categoria, imsize=(256,256)):
    imagen = tf.io.read_file(ruta)
    imagen = tf.image.decode_jpeg(imagen, channels=3)
    imagen = tf.image.resize(imagen, imsize)
    imagen /= 255.0
    return imagen, categoria



### Lectura y procesamiento de images (funciones para leer y preprocesar nuestras imágenes de entrenamiento y validación como archivos TFRecords)

def valor_byte(valor):
    if isinstance(valor, type(tf.constant(0))):
        valor = valor.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[valor]))

def valor_int64(valor):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[valor]))

def serializa_ejemplo(ruta,etiqueta,image_size):
    imagen = tf.io.read_file(ruta)
    imagen = tf.image.decode_jpeg(imagen)
    imagen = tf.image.resize(imagen, (image_size,image_size))
    imagen /= 255.0
    cadena_img = tf.io.serialize_tensor(imagen)
    tam_img = imagen.shape
    caract = {
        'height': valor_int64(tam_img[0]),
        'width': valor_int64(tam_img[1]),
        'depth': valor_int64(tam_img[2]),
        'label': valor_int64(etiqueta),
        'image_raw': valor_byte(cadena_img),
        }
    return tf.train.Example(features=tf.train.Features(feature=caract))


# Podemos leer archivos serializados como tf.data.Dataset usando tf.data.TFRecordDataset. Para ello es necesario definir una funcion para deserializar cada ejemplo:

def deserializa_ejemplo(ej_proto):
    caract = {
        'height': tf.io.FixedLenFeature([], tf.int64),
        'width': tf.io.FixedLenFeature([], tf.int64),
        'depth': tf.io.FixedLenFeature([], tf.int64),
        'label': tf.io.FixedLenFeature([], tf.int64),
        'image_raw': tf.io.FixedLenFeature([], tf.string),
    }
    ej = tf.io.parse_single_example(ej_proto, caract)
    imagen = tf.io.parse_tensor(ej['image_raw'], out_type = float)
    img_shape = [ej['height'], ej['width'], ej['depth']]
    imagen = tf.reshape(imagen, img_shape)

    return imagen, ej['label']

# Gráfica de las frecuencias relativas de las categorías de cada muestra

def plot_labels_freq(df,sample,n_obs,label_col,freq_col):
    # sample size:
    # Abrimos espacio para la gráfica
    fig = plt.figure(figsize = (8, 5))
    plt.ylim(0, df[freq_col].max().round(2)+0.05)
    # Colores para cada categoría (son 4)
    glioma_color = 'steelblue'
    meningioma_color = 'lightsteelblue'
    noTumor_color = 'skyblue'
    pituitary_color = 'cornflowerblue'
    color = [glioma_color,meningioma_color,noTumor_color,pituitary_color]
    # Ponemos títulos e información importante
    plt.title('Tamaño de muestra: {0} MRIs'.format(str(n_obs)), loc='left', fontsize=10)
    plt.title('fuente: https://www.kaggle.com', loc='right', fontstyle='italic', fontsize=10)
    plt.suptitle('Muestra de {0} (% Categorías)'.format(sample),y = 1.0, fontsize=15)
    #  Bar plot
    plt.bar(df[label_col], df[freq_col],color=color,width = 0.6)
    # Damos formato a los valores de cada categoría
    for i, v in enumerate(df[freq_col].to_list()):
        plt.text(i-0.15,v+ 0.01, s = '{:1.1f}%'.format(v*100), c = 'dimgrey', weight = 'bold',fontsize=14)
    # Damos formato a los ejes y marcos
    ejes = fig.axes
    ejes[0].spines['right'].set_visible(False)
    ejes[0].spines['top'].set_visible(False)
    ejes[0].spines['left'].set_visible(False)
    ejes[0].get_yaxis().set_visible(False)
    plt.xticks(fontsize=15)
    plt.show()

## Definición de la CNN
## Ahora vamos a definir una red neuronal convolucional simple con bloques de tipo residual:

class Residual(tf.keras.Model):
    def __init__(self, n_filtros, aumenta = False):  
        super().__init__()
        self.conv1 = Conv2D(n_filtros, 3, padding = 'same')
        self.bn1 = BatchNormalization(axis = 3)
        self.conv2 = Conv2D(n_filtros, 3, padding = 'same')
        self.bn2 = BatchNormalization(axis = 3)
        self.conv1x1 = Conv2D(n_filtros, 1) if aumenta else None
    
    def call(self, x):
        y = self.conv1(x)
        y = self.bn1(y)
        y = tf.keras.activations.relu(y)
        y = self.conv2(y)
        y = self.bn2(y)
        if self.conv1x1:
            x = self.conv1x1(x)
        return tf.keras.activations.relu(x + y)

## Definimos un bloque ResNet con bloques residuales:

class BloqueResNet(tf.keras.Model):
    def __init__(self, mapas_ent, mapas_sal, k = 2):
        super().__init__()
        self.res = [Residual(mapas_sal, aumenta = mapas_ent != mapas_sal) for _ in range(k)]
    
    def call(self, x):
        for r in self.res:
            x = r(x)
        return x

## Finalmente definimos nuestra arquitectura ResNet18 usando los bloques ResNet.