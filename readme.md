<h1 style="text-align: center;">Apartado 1</h1>

Para habilitar el servicio local para detección de personas y coches basta con ejecutar:

```
$ docker build -t detector .
$ docker run -p 8080:8080 detector
```

<br>
Una vez esté funcionando el servicio, en otra consola, enviaremos la petición a traves de python. Cabe destacar que este script tiene dos argumentos:

1. Modo:
   * Para pasarlo por consola se utiliza --modo=original/modificado, y si no se le pasa el propio script pregunta al usuario por consola "¿Modo original o modificado? [O/M]:". 
   * El **modo original** es la solución rápida en términos de programación, pues consiste simplemente en utilizar el modelo preentrenado sin ningún tipo de modificación, es decir, con sus 80 clases. El propio repositorio permite al usuario lanzar la predicción del modelo centrándose únicamente en ciertas clases (_model.predict([ruta_imagen], classes=[0,2])_). A nivel interno en este proceso simplemente se descartan al final las predicciones de clases diferentes a las que se necesitan.
   * El **modo modificado** es una solución más sofisticada en la que se reprograma la arquitectura del modelo para que solamente tenga 2 clases, pero conservando los pesos originales para esas dos clases. Todo esto se realiza en la función _load_modified_model_, en la que se localiza el _head_ de detección y se sobreescriben las tres capas convolucionales encargadas de realizar la predicción final.

2. Ruta:
   * Para pasarlo por consola se utiliza --ruta=path/to/image, y si no se le pasa el propio script pregunta al usuario por consola "Ruta de la imagen:".
   * En este caso debemos especificar la ruta local de la imagen. Los formatos válidos son PNG, JPG y JPEG.

```python
python peticion.py --modo=original --ruta=path/to/image
```

Los resultados son publicados por pantalla.


<h1 style="text-align: center;">Apartado 2</h1>

## 1. Pasos necesarios a seguir

### Paso 1: Creación del dataset

- **Recopilación de imágenes**
- **Etiquetado**: uso de una herramienta, como por ejemplo [LabelImg](https://github.com/tzutalin/labelImg)
- **Procesado de etiquetas**: Procesado para convertir las etiquetas al formato que espera el trainer de la librería. Por un lado, la partición (train, val o test) del dataset debe tener una carpeta llamada "images" y otra "labels", que contienen respectivamente las imágenes y las etiquetas. La manera de relacionar una imagen y su etiqueta es porque ambas tienen el mismo nombre (o ruta) excepto la extensión, que en el caso de la etiqueta es ".txt".
<br>&nbsp; El formato de la etiqueta es:
<br>&nbsp;&nbsp;&nbsp;&nbsp; NumClase X Y W H
<br>&nbsp; donde:
<br>&nbsp;&nbsp;&nbsp;&nbsp; NumClase: clase numérica (comenzando en 0)
<br>&nbsp;&nbsp;&nbsp;&nbsp; X,Y: coordenadas normalizadas del centro del bounding box
<br>&nbsp;&nbsp;&nbsp;&nbsp; W,H: ancho y alto del bounding box
<br>
<br>&nbsp; Ejemplo del conjunto COCO (https://www.kaggle.com/datasets/ultralytics/coco128):
<br>&nbsp;&nbsp;&nbsp;&nbsp; Imagen: images\train2017\000000000081.jpg
<br>&nbsp;&nbsp;&nbsp;&nbsp; Label:  labels\train2017\000000000081.txt
<br>&nbsp;&nbsp;&nbsp;&nbsp; Contenido de la etiqueta (la clase 4, es decir, la quinta comenzando en 0, es avión): 4 0.516492 0.469388 0.912516 0.748282

### Paso 2: Configuración del dataset

- Consiste simplemente en crear un archivo `.yaml` que contenga la configuración necesaria para entrenar. En ella se especifican las rutas del dataset así como los nombres de las clases que incluye. <br>Referencia: [coco128.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco128.yaml)

### Paso 3: Lanzamiento

- El código se puede encontrar [aquí](https://docs.ultralytics.com/es/modes/train/#usage-examples).

## 2. Descripción de posibles problemas que puedan surgir y medidas para reducir el riesgo.

### 2.1 Errores en la ingesta de datos

Es el error más probable, ya que existen multitud de variables difíciles de prever: 
* Tipo de los datos (tanto para imagen como para etiqueta)
* Imagen cargada en rgb y lo correcto sea bgr
* Bounding boxes con una estructura diferente a los que se espera (por ejemplo xywh en vez de xyxy)
* Normalización o no de datos (tando para imagen como para etiqueta)
* ...

La primera medida de reducción de riesgo podría ser programar un script de visualización, que cargue las imágenes y las etiquetas, procese estas últimas y dibuje los bounding boxes sobre las imágenes, todo ello partiendo del conjunto de referencia (coco, coco128...). Una vez programado este algoritmo debería funcionar sin realizarle ninguna modificación sobre el dataset customizado.

### 2.2 Errores durante el forward

En este paso suelen ocurrir dos tipos de problema: por un lado podría ser que el modelo esté programado de forma adaptativa, es decir, que dependa de la forma de las imágenes y/o de las etiquetas para componer su arquitectura (por ejemplo, que el número de clases de salida se extraiga de las etiquetas); por otro lado, un problema muy común es cuando el modelo aplica poolings sucesivos y luego poolings inversos para restaurar el tamaño de la imagen, llevando a tamaños diferentes entre el tamaño inicial y el restaurado (por ejemplo, si se aplican 6 poolings lo recomendable es que el alto y el ancho de la imagen sean múltiplos de 2<sup>6</sup>, es decir, 64).
<br>
A la hora de lanzar el entrenamiento, se puede proceder de la misma manera: preparar el script adecuándolo al dataset de referencia para posteriormente, sin modificar nada, aplicarlo sobre el dataset customizado. En las pruebas iniciales, de ser posible, no modificar ningún hiperparámetro (obviamente si no es estrictamente necesario, como por ejemplo en el número de clases o la ubicación del dataset).

### 2.3 Errores durante el backward:

Especialmente en las fase inicial del entrenamiento, es común obtener pérdidas problemáticas, comunmente si obtienes NaN o pérdidas de un valor muy alto. En el caso de que la loss sea NaN, la medida preventiva debería ser analizar el caso particular, de modo que podría incluírse una excepción para que en este caso se publique información sobre el mismo. En el caso de pérdidas altas se mencionan un par de técnicas en el punto 4.

### 2.4 Otros

* Desbalanceo de clases: visualizar número de etiquetas por clase.
* Overfitting: visualizar las estadísticas sobre el conjunto de entrenamiento y validación. Una señal de que se está sobreentrenando es que mejoren las del entrenamiento mientras que validación se mantiene o sube. 
* Out of Memory: dado que es una limitación de hardware, poco se puede hacer aparte de reducir el tamaño de la imagen y mantener el tamaño del batch en 1. Por supuesto elegir un modelo más pequeño también es una solución plausible, especialmente en este caso que tenemos 5 arquitecturas por tamaño (n,s,m,l y x).
* Tamaño de las detecciones: visualizar estadísticas del tamaño de los bounding boxes, pues muchos modelos de detección están basados en anchors o en detecciones piramidales sobre diferentes mapas de características a distintos niveles y, en función de tu dataset, podrían sobrar o faltar tanto ciertos anchors como ciertos niveles.
* Imágenes sobre las que no se detecta nada: con mucha frecuencia esto genera un error
    
En algunas ocasiones, otra práctica recomendable es mantener el número de dispositivos a 1 (es decir, no entrenar en múltiples GPUs) así como el tamaño del batch en 1, para luego extenderlo hasta las capacidades permitidas por el hardware utilizado.

## 3. Estimación de cantidad de datos necesarios así como de los resultados, métricas, esperadas.

Hacer una estimación de métricas genérica me resulta bastante arriesgado, pues es demasiado dependiente de muchos factores: si las nuevas clases son muy pequeñas o muy grandes, si algunas son similares/confundibles entre sí, variabilidad en las condiciones durante la ejecución (a nivel lumínico, si hay diferentes cámaras, si el entorno es siempre el mismo o no...).
<br>
Las estadísticas de referencia del modelo preentrenado en su conjunto de entrenamiento original marcarán las métricas esperadas, de forma que generalmente si el número de clases se reduce éstas deben mejorar.
<br>
Respecto a la cantidad de datos, opino que un punto de partida podría ser unas 500 detecciones por clase para comenzar a ver cómo responde la red ante esa cantidad de datos. Posteriormente, a medida que se vaya aumentando el conjunto se puede ver la evolución de las métricas en función del tamaño y así afinar la estimación, decidiendo en cada momento si es o no rentable aumentar el conjunto.

## 4. Enumeración y pequeña descripción (2-3 frases) de técnicas que se pueden utilizar para mejorar el desempeño, las métricas del modelo en tiempo de entrenamiento y las métricas del modelo en tiempo de inferencia.

### 4.1 Durante el entrenamiento

* Loss baja y resultados malos: probar haciendo overfitting sobre pocas imágenes. Si todo funciona correctamente la red debería aprender los ejemplos de memoria y no fallar nada.
* Balanceo positivos/negativos: los modelos de detección suelen tener muchos más negativos que positivos, es decir, muchos más puntos en los que no hay objeto que en los que sí. Ésto suele estar solucionado en el trainer pero con respecto al conjunto de datos original, de modo que si los tamaños de tus clases personalizadas son muy diferentes o el número de detecciones por imagen es muy distinto quizá haya que considerar redefinirlo. Si subimos la proporción de positivos, el recall tenderá a aumentar mientras que la precisión a bajar, y viceversa.
* Losses iniciales altas/inestables: por un lado se puede establecer un período de "calentamiento" o "warmup", en el cual el learning rate va subiendo gradualmente hasta alcanzar el esperado, de manera que compensa las losses altas; por otro, se puede poner un tope al valor de la función loss (suele ser conocido como clip) para evitar que supere cierto valor. Otra técnica para evitarlo es entrenar la red por fases, es decir, entrenar primero congelando los pesos de todo el modelo excepto las últimas capas, y poco a poco ir permitiendo entrenamiento sobre más. Las técnicas no son excluyentes, sería recomendable usarlas al unísono.
* Capas de Batch Normalization: dado que esta capa tiene un funcionamiento diferente durante el entrenamiento que durante la inferencia, algo que me ha funcionado (muy) bien en el pasado es que durante las últimas épocas de entrenamiento se utilicen todas estas capas en modo inferencia. Especialmente recomendado si el batch es pequeño.
* Dropout: usualmente aplicado solo en las últimas capas (y siempre solo en fase de entrenamiento), esta técnica favorece la generalización de los pesos.
* Data augmented: especialmente si los datos son escasos, aplicar un aumento masivo de datos (mediante ruido, paso a escala de grises, desenfoques, giros...) suele tener muy buenos resultados según mi experiencia.
* Crop a las imágenes: durante el entrenamiento el consumo de memoria es mayor debido al paso de back propagation, de forma que hacer crop aleatorio a las imágenes para reducir su tamaño en este paso puede ser una técnica útil. De usarse, se suele incluir en el data augmented.

### 4.2 Durante la inferencia:
* NMS: el algoritmo de Non Max Supresion (el utilizado en este caso) tiene una serie de hiperparámetros relevante a la hora de establecer los resultados finales. Alguno de los más relevantes (presentes [aquí](https://docs.ultralytics.com/modes/predict/#inference-arguments)):
  * Umbral de confianza: confianza mínima para mostrar un resultado. También es posible establecer una confianza mínima para los bounding boxes previos al NMS, aunque no es un argumento elegible en la inferencia.
  * Umbral de IoU: umbral para que dos bounding boxes puedan ser considerados como el mismo.
  * Maximo de detecciones por imagen. Al igual que en el primer punto, también se puede establecer un número máximo (y/o mínimo) de candidatos para el NMS, pero no es elegible en los argumentos de inferencia.
* TTA: test-time augmentation, es una técnica que consiste en aplicar data augmented sobre la imagen en la que se hace la inferencia para generar varios resultados y combinarlos en uno más fiable. Elegible como argumento de inferencia.
* Conjunto de test: generalmente se tiene siempre el conjunto de entrenamiento y el de validación, siendo éste último el que se toma como referencia para elegir el mejor modelo. El problema es que, aunque es cierto que los pesos del modelo no se entrenan con dicho conjunto, muchas veces se tienda a sobreajustar los hiperparámetros a aquellos que favorecen las métricas de validación, y también hay cierto riesgo de overfitting. En estos casos, un segundo conjunto de validación, generalmente conocido como test, puede ser de gran ayuda.
* Ejecutar en FP16 o en INT8: mejora sustancialmente el tiempo de ejecución mientras que mantiene o empeora muy poco las métricas.
* Exportar el modelo: ONNX suele ser el formato desde el que se puede pasar a cualquier otro (y en concreto este repositorio cuenta con [exportación a ONNX](https://docs.ultralytics.com/es/modes/export/#usage-examples))
  * Desde OpenCV se puede ejecutar un ONNX en C, más rápido que Python, lo cual ya supone una mejora en el tiempo de ejecución sin pérdida en las métricas.
  * La propia librería ONNX ofrece posibilidades de optimización.
  * Otras librerías, como TensorRT (también disponible para este modelo: https://docs.ultralytics.com/es/modes/export/#export-formats) no solo ofrecen una optimización general del modelo (mediante fusión de capas y otras técnicas), sino que está especialmente diseñado para explotar todas las posibilidades del hardware desarrollado por NVIDIA (muy común en la computación en el borde).

<br><br><br>
<h1 style="text-align: center;">Apartado 3</h1>

Por un lado, para entrenar considero que siempre es mejor recopilar todos los datos a un sistema central donde se entrena un modelo, común para todos si hay más de un agente y que generalmente tendrá más capacidad que el borde.  Si hay varios agentes y no se pueden centralizar los datos, bien por cuestiones de privacidad (como en el área de salud) o por ancho de banda (como en el caso de vehículos autónomos), intentaría aplicar el Federated Learning. En caso de que haya solo un agente, exportado el modelo, considero que todo sería muy similar al apartado 2.

Lo más común es que solo sea necesario el modelo para realizar inferencias, reduciendo el tráfico de datos, la latencia... En este caso, simplemente consistiría en exportar el modelo e integrarlo en el proceso general que se necesite en el borde. Los problemas generalmente vendrán derivados de las diferencias en el hardware (de ahí que el formato al que se exporte sea lo más versátil posible, [como lo es en este caso](https://docs.ultralytics.com/es/modes/export/#usage-examples)) y por supuesto derivados de entornos de ejecución diferentes, lo cual podría simplificarse a través de dockers. 

Aún así, en mi opinión, tanto los pasos a seguir como los posibles problemas son muy dependientes del problema concreto y es demasiado abierto como para dar una respuesta genérica.
