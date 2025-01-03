import gradio as gr
import numpy as np
import torch
import random
import cv2
import time
from PIL import Image, ImageEnhance
import os

# Geräteauswahl (CUDA wenn verfügbar)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Verwendetes Gerät: {device}")

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
        image_tensor = torch.tensor(image_array, dtype=torch.float32).permute(2, 0, 1).to(device)  # Zu Tensor konvertieren und auf das Gerät verschieben

        modified_image_tensor = self.process_image(image_tensor, category_nodes, brightness_factor, contrast_factor)
        return modified_image_tensor

    def process_image(self, image_tensor, category_nodes, brightness_factor, contrast_factor):
        modified_image_tensor = image_tensor.clone()
        for x in range(modified_image_tensor.shape[1]):
            for y in range(modified_image_tensor.shape[2]):
                pixel = modified_image_tensor[:, x, y]
                brightness = float(brightness_factor)
                contrast = float(contrast_factor)
                for node in category_nodes:
                   brightness += node.activation * 0.1
                   contrast += node.activation * 0.1
                pixel = torch.clamp((pixel - 0.5) * contrast + 0.5 + brightness, 0, 1)
                modified_image_tensor[:, x, y] = pixel
        return modified_image_tensor

    def get_color_from_label(self, label):
        color_map = {
            "Rot": torch.tensor([1, 0, 0], device=device),
            "Grün": torch.tensor([0, 1, 0], device=device),
            "Blau": torch.tensor([0, 0, 1], device=device),
            "Gelb": torch.tensor([1, 1, 0], device=device),
            "Cyan": torch.tensor([0, 1, 1], device=device),
            "Magenta": torch.tensor([1, 0, 1], device=device)
        }
        return color_map.get(label, torch.tensor([1, 1, 1], device=device))

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
def save_image(image_tensor, filename, resolution, original_size=None):
    resolutions = {
        "HD": (1280, 720),
        "Full HD": (1920, 1080),
        "2K": (2048, 2048),
        "4K": (5760, 3240),
        "8K": (10670, 6000),
        "Cover": (1024, 1024),
        "Original (2K)": (2048,2048)
    }

    if resolution == "Original (2K)" and original_size:
        width, height = original_size
    else:
        width, height = resolutions.get(resolution, (1920, 1080))

    image = Image.fromarray((image_tensor.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8))
    image = image.resize((width, height), Image.Resampling.LANCZOS)
    image.save(filename, format='webp', lossless=True, quality=80, dpi=(300, 300))  # Speichern als WEBP mit 300 DPI
    print(f"Bild erfolgreich gespeichert als {filename} mit Auflösung {resolution} und 300 DPI")


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

def calculate_brightness(image):
    image_array = np.array(image).astype(float)
    mean_brightness = np.mean(image_array) / 255.0  # Normalisieren auf den Bereich [0, 1]
    # Verschieben des Bereichs von [0, 1] auf [-1, 1] für den Slider
    return (mean_brightness * 2) -1

def calculate_contrast(image):
    image_array = np.array(image).astype(float)
    mean_brightness = np.mean(image_array)

    # Berechnung der Standardabweichung und des Kontrasts basierend auf der Helligkeitsvariation
    std_dev = np.std(image_array)
    contrast = (std_dev / 128.0)  # Skalierung des Kontrasts auf einen Bereich um 1

    # Stärkere Kontraste nach oben, schwächere nach unten anpassen
    contrast = (contrast * 1.0) + 1.0
    contrast = max(0.5, min(2.0, contrast)) # Begrenzung des Kontrasts

    return contrast

# Funktion zum Generieren des Bildes (angepasst für Gradio)
def generate_and_display_image(image, brightness_factor, contrast_factor, resolution):
    if image is None:
        return None, None
    start_time = time.time()

    # Extrahieren der Hauptfarbwerte aus dem Bild
    main_colors = extract_main_colors(image)
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
    image_node = ImageNode("Image")
    image_node.generate_image(category_nodes, image, brightness_factor, contrast_factor)

    # Anzeigen des generierten Bildes
    generated_image = Image.fromarray((image_node.image.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8))
    generated_image = sharpen_image(generated_image)  # Schärfen des Bildes
    generated_image = match_histogram(generated_image, image)  # Helligkeit und Kontrast anpassen

     # Auflösung anpassen, BEVOR das Bild angezeigt wird
    resolutions = {
        "HD": (1280, 720),
        "Full HD": (1920, 1080),
        "2K": (2048, 2048),
        "4K": (5760, 3240),
        "8K": (10670, 6000),
        "Cover": (1024, 1024),
        "Original (2K)": (2048,2048)
    }

    if resolution == "Original (2K)":
        width, height = image.size
    else:
       width, height = resolutions.get(resolution, (1920, 1080))

    generated_image = generated_image.resize((width,height), Image.Resampling.LANCZOS)

    end_time = time.time()  # Endzeit messen
    generation_time = end_time - start_time
    print(f"Bild erfolgreich generiert und angezeigt. Generierungszeit: {generation_time:.2f} Sekunden")

    save_image(image_node.image, "kunst.webp", resolution, original_size=image.size if resolution == "Original (2K)" else None) # Speichern als WEBP
    return image, "kunst.webp"

def process_inputs(image, brightness, contrast, resolution):
    if image is not None:
      return generate_and_display_image(image, brightness, contrast, resolution)
    else:
      return None, None

# Funktion zum Laden des Bildes aus dem Dateisystem
def load_image_from_file(file_path):
    if os.path.exists(file_path):
        return Image.open(file_path)
    else:
        return None

# Gradio Interface erstellen
if __name__ == "__main__":
    with gr.Blocks() as iface:
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(label="Eingabebild")
                brightness_slider = gr.Slider(minimum=-1, maximum=1, value=0.0, step=0.05, label="Helligkeit")
                contrast_slider = gr.Slider(minimum=0.5, maximum=2.0, value=1.0, step=0.05, label="Kontrast")
                resolution_dropdown = gr.Dropdown(choices=["HD", "2K", "Full HD", "4K", "8K", "Cover", "Original (2K)"], value="Full HD", label="Auflösung")
                generate_button = gr.Button("Generiere Bild")

            with gr.Column():
                original_output = gr.Image(label="Eingabebild")
                generated_output = gr.Image(label="Generiertes Bild")

        input_params = [input_image, brightness_slider, contrast_slider, resolution_dropdown]
        output_params = [original_output, generated_output]

        generate_button.click(process_inputs, inputs=input_params, outputs=output_params)

        # Laden des generierten Bildes aus dem Dateisystem und Anzeigen in der Weboberfläche
        generated_image_path = "kunst.webp"
        generated_image = load_image_from_file(generated_image_path)
        if generated_image:
            generated_output.value = generated_image

    iface.launch()
