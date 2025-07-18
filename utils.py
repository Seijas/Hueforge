import os
import json
import numpy as np
from PIL import Image


def validate_image(image_path):
    """Valida que el archivo sea una imagen válida"""
    try:
        with Image.open(image_path) as img:
            img.verify()
        return True
    except Exception:
        return False


def get_image_info(image_path):
    """Obtiene información de la imagen"""
    try:
        with Image.open(image_path) as img:
            return {
                'format': img.format,
                'mode': img.mode,
                'size': img.size,
                'has_transparency': img.mode in ('RGBA', 'LA') or 'transparency' in img.info
            }
    except Exception:
        return None


def save_project_settings(settings, output_dir):
    """Guarda configuración del proyecto"""
    settings_file = os.path.join(output_dir, "project_settings.json")
    with open(settings_file, 'w') as f:
        json.dump(settings, f, indent=2)


def load_project_settings(output_dir):
    """Carga configuración del proyecto"""
    settings_file = os.path.join(output_dir, "project_settings.json")
    if os.path.exists(settings_file):
        with open(settings_file, 'r') as f:
            return json.load(f)
    return None


def estimate_print_time(layers, print_speed=50):
    """Estima tiempo de impresión aproximado"""
    total_area = sum(np.sum(layer['mask']) for layer in layers)
    # Estimación muy básica
    estimated_minutes = total_area / print_speed
    return estimated_minutes


def generate_gcode_header(layer_height=0.2, nozzle_temp=200, bed_temp=60):
    """Genera header básico para G-code"""
    return f"""
; HueForge Clone Generated G-code
; Layer Height: {layer_height}mm
; Nozzle Temperature: {nozzle_temp}°C
; Bed Temperature: {bed_temp}°C

G21 ; set units to millimeters
G90 ; use absolute coordinates
M82 ; use absolute distances for extrusion
G28 ; home all axes
M190 S{bed_temp} ; set bed temperature
M109 S{nozzle_temp} ; set nozzle temperature
"""


def create_output_directory(image_path, root_dir):
    """Crea la carpeta de salida siguiendo la estructura solicitada.

    Args:
        image_path (str): Ruta de la imagen original.
        root_dir (str): Carpeta base de salida.

    Returns:
        str: Ruta completa de la carpeta de salida generada.
    """
    from datetime import datetime
    from pathlib import Path

    date_folder = datetime.now().strftime("%y%m%d")
    base_dir = Path(root_dir) / date_folder
    base_dir.mkdir(parents=True, exist_ok=True)

    image_name = Path(image_path).stem
    output_dir = base_dir / image_name
    counter = 1
    while output_dir.exists():
        output_dir = base_dir / f"{image_name}({counter})"
        counter += 1

    output_dir.mkdir()
    return str(output_dir)
