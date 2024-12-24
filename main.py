import customtkinter as ctk
from tkinter import filedialog, messagebox, StringVar, Menu, DoubleVar
from PIL import Image, ImageTk, ImageEnhance, ImageFilter
import numpy as np
import torch
import random
import multiprocessing
import cv2

# Klasse für die Verbindung zwischen Knoten
class Connection:
    def __init__(self, target_node, weight=None):
        self.target_node = target_node
        self.weight = weight if weight is not None else random.uniform(0.1, 1.0)
        self.weight_history = []

# Klasse für den Knoten im neuronalen Netzwerk
class Node:
    def __init__(self, label):
        self.label = label
        self.connections = []
        self.activation = 0.0
        self.activation_history = []

    def add_connection(self, target_node, weight=None):
        self.connections.append(Connection(target_node, weight))

    def propagate_signal(self, input_signal):
        self.activation = max(0, input_signal)  # Keine negativen Aktivierungen
        self.activation_history.append(self.activation)
        for connection in self.connections:
            connection.target_node.activation += self.activation * connection.weight
            connection.weight_history.append(connection.weight)

# Klasse für den Bildknoten
class ImageNode(Node):
    def __init__(self, label):
        super().__init__(label)
        self.image = None

    def generate_image(self, category_nodes, original_image, brightness_factor, contrast_factor):
        self.image = self.generate_image_from_categories(category_nodes, original_image, brightness_factor, contrast_factor)

    def generate_image_from_categories(self, category_nodes, original_image, brightness_factor, contrast_factor):
        image_array = np.array(original_image) / 255.0  # Normalisiere Originalbild auf [0, 1]
        image_tensor = torch.tensor(image_array, dtype=torch.float32).permute(2, 0, 1)  # Zu Tensor konvertieren

        # Verwenden Sie Multiprocessing, um die Pixelverarbeitung zu parallelisieren
        pool = multiprocessing.Pool()
        chunk_size = image_tensor.shape[1] // multiprocessing.cpu_count()
        chunks = [(image_tensor, i, chunk_size, category_nodes, brightness_factor, contrast_factor) for i in range(0, image_tensor.shape[1], chunk_size)]
        results = pool.map(self.process_chunk, chunks)

        pool.close()
        pool.join()

        for start_row, modified_chunk in results:
            image_tensor[:, start_row:start_row + modified_chunk.shape[1], :] = modified_chunk

        return image_tensor

    def process_chunk(self, args):
        image_tensor, start_row, chunk_size, category_nodes, brightness_factor, contrast_factor = args
        modified_chunk = image_tensor[:, start_row:start_row + chunk_size, :].clone()
        for x in range(modified_chunk.shape[1]):
            for y in range(modified_chunk.shape[2]):
                pixel = modified_chunk[:, x, y]
                # Anpassen der Helligkeit und des Kontrasts basierend auf den Aktivierungen der Knoten
                brightness = float(brightness_factor)
                contrast = float(contrast_factor)
                for node in category_nodes:
                    brightness += node.activation * 0.1  # Anpassen der Helligkeit
                    contrast += node.activation * 0.1  # Anpassen des Kontrasts
                pixel = torch.clamp((pixel - 0.5) * contrast + 0.5 + brightness, 0, 1)
                modified_chunk[:, x, y] = pixel
        return start_row, modified_chunk

    def get_color_from_label(self, label):
        color_map = {
            "Rot": torch.tensor([1, 0, 0]),
            "Grün": torch.tensor([0, 1, 0]),
            "Blau": torch.tensor([0, 0, 1]),
            "Gelb": torch.tensor([1, 1, 0]),
            "Cyan": torch.tensor([0, 1, 1]),
            "Magenta": torch.tensor([1, 0, 1])
        }
        return color_map.get(label, torch.tensor([1, 1, 1]))

# Funktion zum Extrahieren der Hauptfarbwerte aus dem Bild
def extract_main_colors(image):
    image_array = np.array(image)
    colors = {}
    for i in range(image_array.shape[0]):
        for j in range(image_array.shape[1]):
            color = tuple(image_array[i, j])
            if sum(color) > 30:  # Vermeiden von sehr dunklen Farben
                if color in colors:
                    colors[color] += 1
                else:
                    colors[color] = 1
    # Sortieren nach Häufigkeit und die häufigsten Farben auswählen
    sorted_colors = sorted(colors.items(), key=lambda item: item[1], reverse=True)
    main_colors = [color for color, count in sorted_colors[:6]]  # Nehmen Sie die 6 häufigsten Farben

    # Skalieren der Farben auf einen helleren Bereich
    scaled_colors = [(min(c[0] + 50, 255), min(c[1] + 50, 255), min(c[2] + 50, 255)) for c in main_colors]
    return scaled_colors

# Funktion zum Erstellen des neuronalen Netzwerks
def create_neural_network(main_colors):
    category_nodes = [Node(label) for label in ["Rot", "Grün", "Blau", "Gelb", "Cyan", "Magenta"]]
    for node in category_nodes:
        for target_node in category_nodes:
            if node != target_node:
                node.add_connection(target_node, weight=random.uniform(0.01, 8.0))
    return category_nodes

# Funktion zum Speichern des Bildes
def save_image(image_tensor, filename, resolution):
    resolutions = {
        "HD": (1280, 720),
        "Full HD": (1920, 1080),
        "2K": (2048, 2048),
        "4K": (5760, 3240),
        "8K": (10670, 6000),
        "Cover": (1024, 1024)
    }
    width, height = resolutions.get(resolution, (1920, 1080))
    image = Image.fromarray((image_tensor.numpy().transpose(1, 2, 0) * 255).astype(np.uint8))
    image = image.resize((width, height), Image.Resampling.LANCZOS)
    image.save(filename, format='PNG')  # Speichern als PNG
    print(f"Bild erfolgreich gespeichert als {filename} mit Auflösung {resolution}")

# Funktion zum Schärfen des Bildes
def sharpen_image(image):
    enhancer = ImageEnhance.Sharpness(image)
    sharpened_image = enhancer.enhance(1.5)  # Schärfe um 1.5 erhöhen
    return sharpened_image

# Funktion zum Anpassen von Helligkeit und Kontrast
def match_histogram(source, template):
    source = cv2.cvtColor(np.array(source), cv2.COLOR_RGB2LAB).astype("float32")
    template = cv2.cvtColor(np.array(template), cv2.COLOR_RGB2LAB).astype("float32")

    (l, a, b) = cv2.split(source)
    (lH, aH, bH) = cv2.split(template)

    # Anwenden von CLAHE nur auf den L-Kanal
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply((l * 255).astype(np.uint8)) / 255.0

    # Anwenden von Histogram-Matching auf die a- und b-Kanäle
    a = cv2.equalizeHist((a * 255).astype(np.uint8)) / 255.0
    b = cv2.equalizeHist((b * 255).astype(np.uint8)) / 255.0

    result = cv2.merge((l, a, b))
    result = cv2.cvtColor(result.astype("uint8"), cv2.COLOR_LAB2RGB)

    return Image.fromarray(result)

# GUI-Funktion zum Laden des Bildes und Einstellungen vornehmen
def load_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp;*.gif")])
    if file_path:
        image = Image.open(file_path)
        image = image.resize((1920, 1080), Image.Resampling.LANCZOS)
        img_tk = ImageTk.PhotoImage(image)
        image_label.configure(image=img_tk)
        image_label.image = img_tk
        global loaded_image
        loaded_image = image
        print(f"Bild geladen: {file_path}")

# GUI-Funktion zum Generieren des Bildes
def generate_image():
    if loaded_image:
        # Extrahieren der Hauptfarbwerte aus dem Bild
        main_colors = extract_main_colors(loaded_image)
        print(f"Hauptfarbwerte: {main_colors}")

        # Validierung: Sicherstellen, dass mindestens sechs Farben vorhanden sind
        if len(main_colors) < 6 or all(sum(color) < 150 for color in main_colors):
            print("Hauptfarben zu dunkel oder zu wenige Farben, Standardfarben werden verwendet.")
            main_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255)]  # Standardfarben

        # Konvertiere die Farben in normale Aktivierungswerte zwischen 0 und 1
        main_colors = [(r / 255.0, g / 255.0, b / 255.0) for r, g, b in main_colors]

        # Erstellen des neuronalen Netzwerks
        category_nodes = create_neural_network(main_colors)

        # Setze initiale Aktivierungen basierend auf den Hauptfarben
        for node, color in zip(category_nodes, main_colors):
            node.activation = sum(color) / 3  # Durchschnittliche Helligkeit als Aktivierung
            print(f"Knoten {node.label}: Aktivierung = {node.activation}, Farbe = {color}")

        # Erstellen und Verarbeiten des Bildes
        brightness_factor = brightness_var.get()
        contrast_factor = contrast_var.get()
        image_node = ImageNode("Image")
        image_node.generate_image(category_nodes, loaded_image, brightness_factor, contrast_factor)

        # Anzeigen des generierten Bildes
        generated_image = Image.fromarray((image_node.image.numpy().transpose(1, 2, 0) * 255).astype(np.uint8))
        generated_image = sharpen_image(generated_image)  # Schärfen des Bildes
        generated_image = match_histogram(generated_image, loaded_image)  # Helligkeit und Kontrast anpassen
        generated_image = generated_image.resize((64, 64), Image.Resampling.LANCZOS)
        img_tk = ImageTk.PhotoImage(generated_image)
        generated_image_label.configure(image=img_tk)
        generated_image_label.image = img_tk

        # Speichern des generierten Bildes
        resolution = resolution_var.get()
        save_image(image_node.image, "kunst.png", resolution)  # Speichern als PNG

        print("Bild erfolgreich generiert und angezeigt.")
    else:
        messagebox.showwarning("Warnung", "Bitte laden Sie zuerst ein Bild.")

# GUI-Funktion zum Starten der GUI
def start_gui():
    global image_label, generated_image_label, loaded_image, resolution_var, brightness_var, contrast_var
    loaded_image = None

    ctk.set_appearance_mode("dark")  # Setzen Sie das Erscheinungsbildmodus auf "dark"
    ctk.set_default_color_theme("blue")  # Setzen Sie das Standardfarbthema auf "blue"

    root = ctk.CTk()
    root.title("Bildgenerierung")
    root.geometry("550x400")

    # Menü hinzufügen
    menubar = Menu(root)
    root.config(menu=menubar)

    file_menu = Menu(menubar, tearoff=0)
    menubar.add_cascade(label="Datei", menu=file_menu)
    file_menu.add_command(label="Bild laden", command=load_image)
    file_menu.add_command(label="Bild generieren", command=generate_image)
    file_menu.add_separator()
    file_menu.add_command(label="Beenden", command=root.quit)

    resolution_var = StringVar(value="Full HD")
    resolution_label = ctk.CTkLabel(root, text="Bildauflösung:")
    resolution_label.pack(pady=5)
    resolution_combobox = ctk.CTkComboBox(root, variable=resolution_var, values=["HD", "2K", "Full HD", "4K", "8K", "Cover"])
    resolution_combobox.pack(pady=5)

    brightness_var = DoubleVar(value=0.0)
    brightness_label = ctk.CTkLabel(root, text="Helligkeit:")
    brightness_label.pack(pady=5)
    brightness_scale = ctk.CTkSlider(root, from_=-1.0, to=1.0, number_of_steps=20, orientation=ctk.HORIZONTAL, variable=brightness_var)
    brightness_scale.pack(pady=5)

    contrast_var = DoubleVar(value=1.0)
    contrast_label = ctk.CTkLabel(root, text="Kontrast:")
    contrast_label.pack(pady=5)
    contrast_scale = ctk.CTkSlider(root, from_=0.5, to=2.0, number_of_steps=15, orientation=ctk.HORIZONTAL, variable=contrast_var)
    contrast_scale.pack(pady=5)

    image_label = ctk.CTkLabel(root)
    image_label.pack(pady=10)

    generated_image_label = ctk.CTkLabel(root)
    generated_image_label.pack(pady=10)

    root.mainloop()

if __name__ == "__main__":
    start_gui()
