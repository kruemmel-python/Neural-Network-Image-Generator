# Neural Network Image Generator
Demo:
https://huggingface.co/spaces/Rkruemmel/Neural_Network_Image_Generator

Dieses Repository enthält ein neuronales Netzwerk, das Bilder basierend auf Farbaktivierungen generiert und anpasst. Die Anwendung kombiniert Computer Vision, Deep Learning und Weboberflächen-Elemente, um eine interaktive Umgebung für die Bildverarbeitung bereitzustellen.

This repository contains a neural network that generates and adjusts images based on color activations. The application combines computer vision, deep learning, and web interface elements to provide an interactive environment for image processing.

![image](https://github.com/user-attachments/assets/8911a9fe-0432-4f5e-ad40-e36c8e90066c)

![image](https://github.com/user-attachments/assets/ab79067d-2aed-4f7e-878e-7a4405ea04d8)

---

## Funktionen | Features

### **Deutsch:**
- **Bildverarbeitung mit neuronalen Netzwerken:**
  - Knoten und Verbindungen modellieren Aktivierungen und Gewichtungen.
  - Anpassung von Helligkeit und Kontrast basierend auf Aktivierungen.

- **Interaktive Bildanpassung:**
  - Hauptfarben eines Bildes werden extrahiert.
  - Farben beeinflussen die Knotenaktivierungen.
  - Dynamische Generierung von Bildern.

- **Neues Bild statt Skalierung:**
  - Das generierte Bild wird **Pixel für Pixel komplett neu erstellt**.
  - Bei einer Vergrößerung auf 4K oder höhere Auflösungen erfolgt keine Hochrechnung wie in herkömmlichen Bildeditoren, sondern es wird ein völlig neues Bild auf Basis der ursprünglichen Daten generiert. Dadurch bleibt die **Bildqualität unverändert**, unabhängig von der Zielauflösung.

- **Weboberfläche für einfache Nutzung:**
  - Lade Bilder direkt über die Weboberfläche.
  - Passe Helligkeit, Kontrast und Bildauflösung an.
  - Speichere die bearbeiteten Bilder.

- **Fortschrittliche Bildverarbeitung:**
  - Parallele Verarbeitung mit `multiprocessing`.
  - Schärfen und Histogrammanpassung für verbesserte Bildqualität.

---

### **English:**
- **Image Processing with Neural Networks:**
  - Nodes and connections model activations and weights.
  - Adjust brightness and contrast based on activations.

- **Interactive Image Adjustment:**
  - Extract main colors of an image.
  - Colors influence node activations.
  - Dynamically generate new images.

- **New Image Instead of Scaling:**
  - The generated image is **completely rebuilt pixel by pixel**.
  - Enlarging to 4K or higher resolutions does not involve upscaling, as in traditional editors. Instead, a completely new image is generated based on the original data, ensuring **unchanged image quality**, regardless of target resolution.

- **User-Friendly Web Interface:**
  - Load images directly via the web interface.
  - Adjust brightness, contrast, and image resolution.
  - Save the edited images.

- **Advanced Image Processing:**
  - Parallel processing with `multiprocessing`.
  - Sharpening and histogram adjustment for enhanced image quality.

---

## Voraussetzungen | Prerequisites

### **Deutsch:**
Stelle sicher, dass folgende Pakete installiert sind:
- Python 3.8+
- `gradio`
- `Pillow`
- `torch`
- `numpy`
- `opencv-python`
- `multiprocessing`

Installiere die Abhängigkeiten mit:
```bash
pip install gradio pillow torch numpy opencv-python
```

### **English:**
Ensure the following packages are installed:
- Python 3.8+
- `gradio`
- `Pillow`
- `torch`
- `numpy`
- `opencv-python`
- `multiprocessing`

Install dependencies using:
```bash
pip install gradio pillow torch numpy opencv-python
```

---

## Nutzung | Usage

### **Deutsch:**
1. **Starte die Weboberfläche:**
   ```bash
   python main.py
   ```
2. **Lade ein Bild:**
   - Gehe zu `Datei > Bild laden`.
3. **Passe Parameter an:**
   - Wähle Helligkeit, Kontrast und Auflösung.
4. **Generiere ein Bild:**
   - Gehe zu `Datei > Bild generieren`.
   - Das generierte Bild wird unter `kunst.webp` gespeichert.

### **English:**
1. **Start the Web Interface:**
   ```bash
   python main.py
   ```
2. **Load an Image:**
   - Navigate to `File > Load Image`.
3. **Adjust Parameters:**
   - Choose brightness, contrast, and resolution.
4. **Generate an Image:**
   - Navigate to `File > Generate Image`.
   - The generated image will be saved as `kunst.webp`.

---

## Funktionsweise | Functionality

### **Deutsch:**
- **Neuronales Netzwerk:**
  - Jeder Knoten repräsentiert eine Farbe (z. B. Rot, Grün, Blau).
  - Aktivierungen basieren auf Farbhelligkeit und beeinflussen die Bildverarbeitung.

- **Bildgenerierung:**
  1. Hauptfarben extrahieren: Farben mit hoher Helligkeit werden priorisiert.
  2. Netzwerk erstellen: Knotenverbindungen mit zufälligen Gewichtungen.
  3. Pixelanpassung: Parallele Verarbeitung, um Helligkeit und Kontrast dynamisch zu modifizieren.
  4. Bild neu erstellen: Jedes Pixel wird neu berechnet, um maximale Qualität sicherzustellen.

### **English:**
- **Neural Network:**
  - Each node represents a color (e.g., red, green, blue).
  - Activations are based on color brightness and influence image processing.

- **Image Generation:**
  1. Extract main colors: Prioritize colors with high brightness.
  2. Create network: Connect nodes with random weights.
  3. Pixel adjustment: Use parallel processing to dynamically modify brightness and contrast.
  4. Rebuild image: Each pixel is recalculated to ensure maximum quality.

---

## Beispiel | Example

### **Deutsch:**
- **Originalbild:**
  ![Original Image](https://github.com/user-attachments/assets/fa99bed8-eb9e-40fb-b31e-df190fe9c0d0)

- **Generiertes Bild:**
  ![Generated Image](https://github.com/user-attachments/assets/63d477dc-5c02-4ec6-bf50-70c6575c6e48)

### **English:**
- **Original Image:**
  ![Original Image](https://github.com/user-attachments/assets/fa99bed8-eb9e-40fb-b31e-df190fe9c0d0)

- **Generated Image:**
  ![Generated Image](https://github.com/user-attachments/assets/63d477dc-5c02-4ec6-bf50-70c6575c6e48)

---

## Entwicklerhinweise | Developer Notes

### **Deutsch:**
- **Erweiterungsmöglichkeiten:**
  - Füge mehr Farben hinzu, um spezifische Anpassungen zu ermöglichen.
  - Implementiere ein Tool, um Aktivierungen und Gewichtungen grafisch darzustellen.
  - Optimiere die Performance für Echtzeitverarbeitung.

- **Debugging:**
  - Überprüfe Aktivierungswerte und Gewichtungen während der Bildgenerierung.

### **English:**
- **Extension Possibilities:**
  - Add more colors for more specific adjustments.
  - Implement a tool to visualize activations and weights graphically.
  - Optimize performance for real-time processing.

- **Debugging:**
  - Check activation values and weights during image generation.

---

## Lizenz | License

Dieses Projekt ist unter der MIT-Lizenz veröffentlicht. Siehe [LICENSE](LICENSE) für weitere Details.
This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## Autoren | Authors

Entwickelt von [Ralf Krümmel].
Developed by [Ralf Krümmel].
