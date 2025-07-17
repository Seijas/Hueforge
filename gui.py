import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinter import colorchooser
import os
from pathlib import Path


class HueForgeGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("HueForge Clone - Conversor de Imagen a STL")
        self.root.geometry("600x500")

        self.converter = None
        self.image_path = None

        self.create_widgets()

    def create_widgets(self):
        # Frame principal
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Selección de imagen
        ttk.Label(main_frame, text="Seleccionar Imagen:").grid(row=0, column=0, sticky=tk.W, pady=5)

        img_frame = ttk.Frame(main_frame)
        img_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)

        self.img_label = ttk.Label(img_frame, text="Ninguna imagen seleccionada")
        self.img_label.grid(row=0, column=0, sticky=tk.W)

        ttk.Button(img_frame, text="Examinar", command=self.select_image).grid(row=0, column=1, padx=10)

        # Configuración
        config_frame = ttk.LabelFrame(main_frame, text="Configuración", padding="10")
        config_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)

        # Número de colores
        ttk.Label(config_frame, text="Número de colores:").grid(row=0, column=0, sticky=tk.W)
        self.num_colors_var = tk.StringVar(value="8")
        ttk.Entry(config_frame, textvariable=self.num_colors_var, width=10).grid(row=0, column=1, sticky=tk.W, padx=5)

        # Ancho objetivo
        ttk.Label(config_frame, text="Ancho objetivo (px):").grid(row=1, column=0, sticky=tk.W)
        self.width_var = tk.StringVar(value="100")
        ttk.Entry(config_frame, textvariable=self.width_var, width=10).grid(row=1, column=1, sticky=tk.W, padx=5)

        # Grosor base
        ttk.Label(config_frame, text="Grosor base (mm):").grid(row=2, column=0, sticky=tk.W)
        self.base_thickness_var = tk.StringVar(value="0.2")
        ttk.Entry(config_frame, textvariable=self.base_thickness_var, width=10).grid(row=2, column=1, sticky=tk.W,
                                                                                     padx=5)

        # Altura máxima
        ttk.Label(config_frame, text="Altura máxima (mm):").grid(row=3, column=0, sticky=tk.W)
        self.max_height_var = tk.StringVar(value="5.0")
        ttk.Entry(config_frame, textvariable=self.max_height_var, width=10).grid(row=3, column=1, sticky=tk.W, padx=5)

        # Botón de procesamiento
        ttk.Button(main_frame, text="Procesar Imagen", command=self.process_image).grid(row=3, column=0, columnspan=2,
                                                                                        pady=20)

        # Área de progreso
        self.progress_var = tk.StringVar(value="Listo para procesar")
        ttk.Label(main_frame, textvariable=self.progress_var).grid(row=4, column=0, columnspan=2, pady=5)

        self.progress_bar = ttk.Progressbar(main_frame, mode='indeterminate')
        self.progress_bar.grid(row=5, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)

    def select_image(self):
        file_path = filedialog.askopenfilename(
            title="Seleccionar imagen",
            filetypes=[
                ("Imágenes", "*.jpg *.jpeg *.png *.bmp *.tiff *.webp"),
                ("Todos los archivos", "*.*")
            ]
        )

        if file_path:
            self.image_path = file_path
            self.img_label.config(text=os.path.basename(file_path))

    def process_image(self):
        if not self.image_path:
            messagebox.showerror("Error", "Por favor selecciona una imagen primero")
            return

        try:
            # Obtener parámetros
            num_colors = int(self.num_colors_var.get())
            target_width = int(self.width_var.get())
            base_thickness = float(self.base_thickness_var.get())
            max_height = float(self.max_height_var.get())

            # Crear conversor
            from hueforge_converter import HueForgeConverter
            converter = HueForgeConverter(
                base_thickness=base_thickness,
                layer_height=0.2,
                max_height=max_height
            )

            # Iniciar procesamiento
            self.progress_var.set("Procesando imagen...")
            self.progress_bar.start()
            self.root.update()

            # Procesar
            output_dir = f"output_{Path(self.image_path).stem}"
            converter.process_image(
                image_path=self.image_path,
                output_dir=output_dir,
                num_colors=num_colors,
                target_width=target_width
            )

            self.progress_bar.stop()
            self.progress_var.set("Procesamiento completado")

            messagebox.showinfo("Éxito", f"Procesamiento completado.\nArchivos guardados en: {output_dir}")

        except Exception as e:
            self.progress_bar.stop()
            self.progress_var.set("Error en procesamiento")
            messagebox.showerror("Error", f"Error durante el procesamiento: {str(e)}")


def run_gui():
    root = tk.Tk()
    app = HueForgeGUI(root)
    root.mainloop()


if __name__ == "__main__":
    run_gui()
