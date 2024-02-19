from flask import Flask, request, jsonify
from pathlib import Path

app = Flask(__name__)

########################
## VARIABLES GLOBALES ##
########################

# Formatos permitidos para la imagen:
IMAGE_FORMATS = [
        '.jpg',
        '.jpeg',
        '.png',
    ]

##########################
## FUNCIONES AUXILIARES ##
##########################

def load_original_model(
    yaml_path='yolov8n.yaml',
    weights_path='yolov8n.pt'
):
    """
    Función que carga el modelo original
    """
    from ultralytics import YOLO
    return YOLO(yaml_path).load(weights_path)

def load_modified_model(classes=[0,2]):
    """
    Función que carga el modelo y modifica su arquitectura para que tenga únicamente dos clases, 
    manteniendo los pesos de dichas clases:
    """
    from torch.nn import Conv2d
    
    N = len(classes)
    model = load_original_model()

    # Vamos a modificar la arquitectura de la red para que la salida tenga únicamente las clases que nos interesan:
    # Localizamos el head de detección (ultralytics/nn/modules/head.py):
    head = model.model.model[22]

    # Modificamos el numero de clases:
    head.nc = N
    head.no = N + 64

    # Modificamos la arquitectura
    for i in range(head.nl):
        old_conv = model.model.model[22].cv3[i][2]
        new_conv = Conv2d(80, N, kernel_size=(1, 1), stride=(1, 1))
        for old_param,new_param in zip(old_conv.parameters(), new_conv.parameters()):
            new_param.data = old_param[classes]
        head.cv3[i][2] = new_conv

    return model

def json_parser(result):
    """
    El modelo genera el objeto "result", con esta función extraemos las detecciones en JSON 
    de forma customizada:
    """
    import json

    # Diccionario para pasar de etiqueta numérica al nombre de la categoría
    num2nam = {
        0: "persona", # En ambos modelos, la clase 0 es persona
        1: "coche",   # En el modelo modificado, la clase 1 es coche
        2: "coche"    # En el modelo original, la clase 2 es coche
    }
    
    # Creacion la estructura del json (como lista de diccionarios):
    data = []
    for cls,conf,(x1,y1,x2,y2) in zip(result.boxes.cls.numpy(),result.boxes.conf.numpy(),result.boxes.xyxy.numpy()):
        data.append({
            "cls": num2nam[int(cls)], 
            "conf": str(round(conf,2)),
            "bbs": [int(round(x1)),int(round(y1)),int(round(x2)),int(round(y2))]
        })

    return json.dumps(data)

def inferencia(ruta_imagen,modo):
    # Diferenciamos entre el método original y el que implica modificación de la arquitectura.
    # En ambos casos, simplemente se carga el modelo y se hace inferencia:
    if modo.startswith("ori"):
        model = load_original_model()
        result = model.predict([ruta_imagen], classes=[0,2])[0]
    elif modo.startswith("mod"):
        model = load_modified_model()
        result = model.predict([ruta_imagen])[0]
    else:
        raise ValueError(f"El argumento 'modo' debe ser 'original' o 'modificado', no {modo}")
    
    return json_parser(result)

#######################
## FUNCION PRINCIPAL ##
#######################

@app.route('/procesar', methods=['POST'])
def procesar():
    # Recepción y procesado de la imagen
    imagen = request.files['imagen']
    imagen.save('temp.jpg')
    ruta_temp = Path('temp.jpg')

    # Recepción y procesado del modo:
    modo = request.form['modo']

    # Generación del resultado de la inferencia
    resultado = inferencia(ruta_temp,modo)
    
    # Borrado del link temporal
    ruta_temp.unlink()

    return jsonify(resultado)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080)