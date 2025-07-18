import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import struct
import os
from pathlib import Path
from config import Config


class HueForgeConverter:
    def __init__(self, base_thickness=0.2, layer_height=0.2, max_height=5.0):
        """
        Inicializa el conversor HueForge

        Args:
            base_thickness: Grosor base en mm
            layer_height: Altura de cada capa en mm
            max_height: Altura máxima total en mm
        """
        self.base_thickness = base_thickness
        self.layer_height = layer_height
        self.max_height = max_height

    def load_image(self, image_path, target_width_px=100):
        """Carga y redimensiona la imagen"""
        try:
            img = Image.open(image_path)
            # Convertir a RGB si es necesario
            if img.mode != 'RGB':
                img = img.convert('RGB')

            # Redimensionar manteniendo proporción
            aspect_ratio = img.height / img.width
            target_height = int(target_width_px * aspect_ratio)
            img = img.resize((target_width_px, target_height), Image.Resampling.LANCZOS)

            return np.array(img)
        except Exception as e:
            print(f"Error cargando imagen: {e}")
            return None

    def extract_colors(self, image_array, num_colors=8):
        """Extrae los colores principales usando K-means"""
        # Reshape para clustering
        pixels = image_array.reshape(-1, 3)

        # Aplicar K-means
        kmeans = KMeans(n_clusters=num_colors, random_state=42, n_init=10)
        kmeans.fit(pixels)

        colors = kmeans.cluster_centers_.astype(int)
        labels = kmeans.labels_

        # Reshape labels de vuelta a imagen
        color_map = labels.reshape(image_array.shape[:2])

        return colors, color_map

    def create_height_map(self, color_map, colors, brightness_based=True):
        """Crea mapa de alturas basado en colores"""
        height_map = np.zeros(color_map.shape, dtype=np.float32)

        if brightness_based:
            # Calcular altura basada en brillo de colores
            brightness = np.array([np.mean(color) for color in colors])
            # Normalizar brightness a rango de alturas
            min_brightness = np.min(brightness)
            max_brightness = np.max(brightness)

            if max_brightness > min_brightness:
                normalized_brightness = (brightness - min_brightness) / (max_brightness - min_brightness)
                heights = self.base_thickness + normalized_brightness * (self.max_height - self.base_thickness)
            else:
                heights = np.full_like(brightness, self.base_thickness)
        else:
            # Altura uniforme por color
            heights = np.linspace(self.base_thickness, self.max_height, len(colors))

        # Asignar alturas
        for i, height in enumerate(heights):
            height_map[color_map == i] = height

        return height_map, heights

    def create_layers(self, height_map, colors, color_map):
        """Crea capas para cada color/altura"""
        layers = []
        unique_heights = np.unique(height_map)

        for i, height in enumerate(unique_heights):
            # Crear máscara para esta altura
            mask = (height_map == height)

            # Verificar que exista al menos un píxel con esta altura
            if np.any(mask):
                # Obtener el color más común para esta altura
                color_indices = color_map[mask]
                if len(color_indices) > 0:
                    most_common_color = np.bincount(color_indices).argmax()
                    color = colors[most_common_color]
                else:
                    color = colors[0]
            else:
                color = colors[0]

            layers.append({
                'height': height,
                'mask': mask,
                'color': color,
                'layer_num': i
            })

        return sorted(layers, key=lambda x: x['height'])

    def create_mesh_vertices(self, mask, height, scale=1.0):
        """Crea vértices de mesh para una capa"""
        vertices = []
        faces = []

        h, w = mask.shape
        vertex_map = np.full((h, w), -1, dtype=int)
        vertex_count = 0

        # Crear vértices para píxeles activos
        for y in range(h):
            for x in range(w):
                if mask[y, x]:
                    # Vértice superior
                    vertices.append([x * scale, y * scale, height])
                    # Vértice inferior
                    vertices.append([x * scale, y * scale, 0])
                    vertex_map[y, x] = vertex_count
                    vertex_count += 2

        # Crear caras
        for y in range(h - 1):
            for x in range(w - 1):
                if mask[y, x] and mask[y + 1, x] and mask[y, x + 1] and mask[y + 1, x + 1]:
                    # Obtener índices de vértices
                    v1 = vertex_map[y, x]
                    v2 = vertex_map[y + 1, x]
                    v3 = vertex_map[y, x + 1]
                    v4 = vertex_map[y + 1, x + 1]

                    # Cara superior (z = height)
                    faces.append([v1, v3, v2])
                    faces.append([v2, v3, v4])

                    # Cara inferior (z = 0)
                    faces.append([v1 + 1, v2 + 1, v3 + 1])
                    faces.append([v2 + 1, v4 + 1, v3 + 1])

        return np.array(vertices), np.array(faces)

    def write_stl_binary(self, vertices, faces, filename):
        """Escribe archivo STL binario"""
        with open(filename, 'wb') as f:
            # Header (80 bytes)
            header = b'HueForge STL Generated' + b'\x00' * (80 - 22)
            f.write(header)

            # Número de triángulos
            f.write(struct.pack('<I', len(faces)))

            # Escribir cada triángulo
            for face in faces:
                if len(face) == 3 and all(idx < len(vertices) for idx in face):
                    # Calcular normal
                    v1, v2, v3 = vertices[face[0]], vertices[face[1]], vertices[face[2]]
                    normal = np.cross(v2 - v1, v3 - v1)
                    normal = normal / np.linalg.norm(normal) if np.linalg.norm(normal) > 0 else [0, 0, 1]

                    # Escribir normal
                    f.write(struct.pack('<fff', *normal))

                    # Escribir vértices
                    for vertex_idx in face:
                        f.write(struct.pack('<fff', *vertices[vertex_idx]))

                    # Atributo (2 bytes)
                    f.write(struct.pack('<H', 0))

    def create_combined_stl(self, layers, output_path, scale=0.1):
        """Crea un STL combinado con todas las capas"""
        all_vertices = []
        all_faces = []
        vertex_offset = 0

        for layer in layers:
            vertices, faces = self.create_mesh_vertices(layer['mask'], layer['height'], scale)

            if len(vertices) > 0 and len(faces) > 0:
                # Agregar vértices
                all_vertices.extend(vertices)

                # Ajustar índices de caras
                adjusted_faces = faces + vertex_offset
                all_faces.extend(adjusted_faces)

                vertex_offset += len(vertices)

        if all_vertices:
            self.write_stl_binary(np.array(all_vertices), np.array(all_faces), output_path)
            print(f"STL combinado guardado en: {output_path}")
        else:
            print("No se pudieron generar vértices")

    def create_individual_stls(self, layers, output_dir, scale=0.1):
        """Crea STL individual para cada capa"""
        os.makedirs(output_dir, exist_ok=True)

        for i, layer in enumerate(layers):
            vertices, faces = self.create_mesh_vertices(layer['mask'], layer['height'], scale)

            if len(vertices) > 0 and len(faces) > 0:
                filename = os.path.join(output_dir, f"layer_{i:02d}_h{layer['height']:.2f}.stl")
                self.write_stl_binary(vertices, faces, filename)
                print(f"Capa {i} guardada: {filename}")

    def visualize_layers(self, layers, colors, save_path=None):
        """Visualiza las capas generadas"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Imagen original reconstruida
        height_map = np.zeros(layers[0]['mask'].shape)
        color_image = np.zeros((*layers[0]['mask'].shape, 3), dtype=np.uint8)

        for layer in layers:
            height_map[layer['mask']] = layer['height']
            color_image[layer['mask']] = layer['color']

        # Mostrar altura
        im1 = axes[0, 0].imshow(height_map, cmap='viridis')
        axes[0, 0].set_title('Mapa de Alturas')
        axes[0, 0].axis('off')
        plt.colorbar(im1, ax=axes[0, 0], label='Altura (mm)')

        # Mostrar colores
        axes[0, 1].imshow(color_image)
        axes[0, 1].set_title('Colores por Capa')
        axes[0, 1].axis('off')

        # Histograma de alturas
        axes[1, 0].hist(height_map.flatten(), bins=len(layers), alpha=0.7)
        axes[1, 0].set_title('Distribución de Alturas')
        axes[1, 0].set_xlabel('Altura (mm)')
        axes[1, 0].set_ylabel('Frecuencia')

        # Paleta de colores
        color_patches = []
        for i, layer in enumerate(layers):
            color_patches.append(layer['color'] / 255.0)

        axes[1, 1].imshow([color_patches], aspect='auto')
        axes[1, 1].set_title('Paleta de Colores')
        axes[1, 1].set_xticks(range(len(layers)))
        axes[1, 1].set_xticklabels([f"L{i}" for i in range(len(layers))])
        axes[1, 1].set_yticks([])

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def process_image(self, image_path, output_dir, num_colors=8, target_width_mm=100, scale=Config.DEFAULT_SCALE, enforce_bw=False):
        """Procesa una imagen completa

        Args:
            image_path: Ruta de la imagen a procesar
            output_dir: Carpeta de salida
            num_colors: Número de colores a extraer
            target_width_mm: Ancho objetivo en milímetros
            scale: Milímetros por píxel
            enforce_bw: Si es True, fuerza la primera capa en negro y la última en blanco
        """
        print(f"Procesando imagen: {image_path}")

        # Crear directorio de salida
        os.makedirs(output_dir, exist_ok=True)

        # Convertir ancho en mm a píxeles
        target_width_px = max(1, int(target_width_mm / scale))

        # Cargar imagen
        image_array = self.load_image(image_path, target_width_px)
        if image_array is None:
            return

        # Extraer colores
        print("Extrayendo colores...")
        colors, color_map = self.extract_colors(image_array, num_colors)

        # Crear mapa de alturas
        print("Creando mapa de alturas...")
        height_map, heights = self.create_height_map(color_map, colors)

        # Crear capas
        print("Generando capas...")
        layers = self.create_layers(height_map, colors, color_map)

        if enforce_bw and layers:
            layers[0]['color'] = np.array([0, 0, 0])
            layers[-1]['color'] = np.array([255, 255, 255])

        # Visualizar
        print("Generando visualización...")
        viz_path = os.path.join(output_dir, "visualization.png")
        self.visualize_layers(layers, colors, viz_path)

        # Crear STLs
        print("Generando archivos STL...")
        combined_stl = os.path.join(output_dir, "combined_model.stl")
        self.create_combined_stl(layers, combined_stl, scale)

        individual_dir = os.path.join(output_dir, "individual_layers")
        self.create_individual_stls(layers, individual_dir, scale)

        # Guardar información de capas
        info_path = os.path.join(output_dir, "layer_info.txt")
        with open(info_path, 'w') as f:
            f.write("Información de Capas HueForge\n")
            f.write("=" * 30 + "\n\n")
            for i, layer in enumerate(layers):
                f.write(f"Capa {i}:\n")
                f.write(f"  Altura: {layer['height']:.2f} mm\n")
                f.write(f"  Color RGB: {layer['color']}\n")
                f.write(f"  Píxeles: {np.sum(layer['mask'])}\n\n")

        print(f"Procesamiento completado. Archivos guardados en: {output_dir}")


def main():
    """Ejemplo de uso"""
    # Configuración
    converter = HueForgeConverter(
        base_thickness=0.2,
        layer_height=0.2,
        max_height=3.0
    )

    # Procesar imagen
    image_path = "input_image.jpg"  # Cambiar por tu imagen
    output_dir = "output_hueforge"

    converter.process_image(
        image_path=image_path,
        output_dir=output_dir,
        num_colors=Config.DEFAULT_NUM_COLORS,
        target_width_mm=80,
        scale=Config.DEFAULT_SCALE,
    )


if __name__ == "__main__":
    # Verificar dependencias
    try:
        import numpy as np
        import matplotlib.pyplot as plt
        from PIL import Image
        from sklearn.cluster import KMeans

        print("Todas las dependencias están instaladas")
        main()
    except ImportError as e:
        print(f"Error: Falta instalar dependencia: {e}")
        print("Instala con: pip install numpy matplotlib pillow scikit-learn")
