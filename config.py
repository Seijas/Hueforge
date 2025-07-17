class Config:
    """Configuración global del programa"""

    # Configuración de impresión 3D
    DEFAULT_BASE_THICKNESS = 0.2  # mm
    DEFAULT_LAYER_HEIGHT = 0.2  # mm
    DEFAULT_MAX_HEIGHT = 5.0  # mm

    # Configuración de imagen
    DEFAULT_TARGET_WIDTH = 100  # píxeles
    DEFAULT_NUM_COLORS = 4  # número de colores a extraer

    # Configuración de escala
    DEFAULT_SCALE = 0.1  # escala para STL (mm por píxel)

    # Formatos de imagen soportados
    SUPPORTED_FORMATS = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']

    # Configuración de salida
    OUTPUT_DIR = "output_hueforge"
    VISUALIZATION_DPI = 300
