import argparse
def parse_args():
    # Create the parser
    parser = argparse.ArgumentParser()

    # Add the arguments you know
    parser.add_argument("--input", type=str)
    parser.add_argument("--output", default="output",type=str)
    

    # Parse and return the arguments
    return parser.parse_known_args()


import cv2

def get_video_resolution(video_path):
    cap = cv2.VideoCapture(str(video_path))

    # Verificamos si el video se ha abierto correctamente
    if not cap.isOpened(): 
        raise Exception("No se pudo abrir el video")

    # Obtenemos la resolución del video
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    cap.release()

    # Devolvemos la resolución como cadena de texto
    return f"{width}:{height}"

