import requests
import json
import argparse

def get_arguments():
    '''
    Función para procesar los argumentos de entrada.
    '''
    # Parseo de los argumentos:
    parser = argparse.ArgumentParser()
    parser.add_argument('--modo', type=str, default='', help='Si se quiere lanzar el modelo original o el modificado.')
    parser.add_argument('--ruta', type=str, default='', help='Ruta de la imagen sobre la que hacer inferencia.')
    args = parser.parse_args()

    # Si no se le pasa un modo, se pregunta por pantalla:
    if args.modo=='':
        modo = {
            'O': 'original',
            'o': 'original',
            'M': 'modificado',
            'm': 'modificado',
        }[input("¿Modo original o modificado? \n[O/M]: ")]
    else:
        modo = args.modo

    # Ídem con la ruta:
    if args.ruta=='':
        ruta_imagen = input("\nRuta de la imagen: ")
    else:
        ruta_imagen = args.ruta

    # URL del servicio
    url = 'http://localhost:8080/procesar'

    return url,modo,ruta_imagen

def get_response():
    # Obtenemos los argumentos:
    url,modo,ruta_imagen = get_arguments()

    # Abrir la imagen en modo binario
    with open(ruta_imagen, 'rb') as f:
        # Crear un diccionario con los datos de la imagen
        files = {'imagen': f}
        data = {'modo': modo}

        # Enviar la solicitud POST al servicio
        response = requests.post(url, files=files, data=data)

    return response

def main():
    # Obtenemos la respuesta:
    response = get_response()

    # Verificar el código de estado de la respuesta
    if response.status_code == 200:
        # Imprimir la respuesta JSON del servicio
        json_file = json.loads(response.json())
        for i,deteccion in enumerate(json_file):
            clase       = deteccion['cls']
            confianza   = int(round(100*float(deteccion['conf'])))
            x1,y1,x2,y2 = deteccion['bbs']
            print(f"  Objeto {i+1}:")
            print(f"    Clase:                      {clase}")
            print(f"    Confianza:                  {confianza}%")
            print(f"    Bounding Box (x1,y1,x2,y2): {x1}, {y1}, {x2}, {y2}")
            print("")
    else:
        # Si la solicitud falla, imprimir el código de estado
        print('Error:', response.status_code)

if __name__=="__main__":
    main()