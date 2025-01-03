# **Dokumentation: Neural Network Image Generator**

---

## **Projektbeschreibung**

### **Deutsch:**
Der Neural Network Image Generator ist eine Anwendung, die ein neuronales Netzwerk nutzt, um Bilder basierend auf Farbaktivierungen zu generieren und anzupassen. Benutzer können ein Bild laden, die Hauptfarben extrahieren und diese zur dynamischen Erstellung eines neuen Bildes verwenden. Die Anwendung bietet eine Weboberfläche, die es einfach macht, Helligkeit, Kontrast und Bildauflösung einzustellen.

### **English:**
The Neural Network Image Generator is an application that uses a neural network to generate and adjust images based on color activations. Users can load an image, extract its main colors, and use them to dynamically create a new image. The application provides a web interface for easy adjustment of brightness, contrast, and image resolution.

---

## **Funktionen und Merkmale**

### **Deutsch:**
1. **Neurales Netzwerk-Modell:**
   - Simulation von Knoten und Verbindungen mit Aktivierungs- und Gewichtshistorie.
2. **Hauptfarbenextraktion:**
   - Identifikation der sechs häufigsten Farben eines Bildes.
3. **Bildgenerierung:**
   - Anpassung von Helligkeit und Kontrast basierend auf den Aktivierungen der Hauptfarbenknoten.
4. **Weboberfläche:**
   - Einfache Bedienung über Menüs und Schieberegler.
5. **Parallele Verarbeitung:**
   - Verwendung von Multiprocessing für schnelle Bildbearbeitung.
6. **Auflösungsauswahl:**
   - Unterstützung für mehrere Auflösungen, einschließlich HD, Full HD, 2K, 4K, 8K und quadratischer „Cover“-Größe.
7. **Bildspeicherung:**
   - Export der generierten Bilder im WEBP-Format.

### **English:**
1. **Neural Network Model:**
   - Simulation of nodes and connections with activation and weight history.
2. **Main Color Extraction:**
   - Identification of the six most frequent colors in an image.
3. **Image Generation:**
   - Adjustment of brightness and contrast based on main color node activations.
4. **Web Interface:**
   - Easy operation via menus and sliders.
5. **Parallel Processing:**
   - Use of multiprocessing for fast image processing.
6. **Resolution Selection:**
   - Support for multiple resolutions, including HD, Full HD, 2K, 4K, 8K, and square "Cover" size.
7. **Image Saving:**
   - Export of generated images in WEBP format.

---

## **Benutzung**

### **Deutsch:**
1. **Bild laden:**
   - Wähle ein Bild aus, das bearbeitet werden soll.
2. **Helligkeit und Kontrast anpassen:**
   - Nutze die Schieberegler, um die Helligkeit und den Kontrast des Bildes zu ändern.
3. **Auflösung auswählen:**
   - Wähle die gewünschte Auflösung aus den verfügbaren Optionen (HD, Full HD, 2K, 4K, 8K, Cover).
4. **Bild generieren:**
   - Erstelle ein neues Bild basierend auf den Hauptfarben des geladenen Bildes.
5. **Bild speichern:**
   - Speichere das generierte Bild im WEBP-Format.

### **English:**
1. **Load an Image:**
   - Select an image to process.
2. **Adjust Brightness and Contrast:**
   - Use sliders to modify the image's brightness and contrast.
3. **Select Resolution:**
   - Choose the desired resolution from the available options (HD, Full HD, 2K, 4K, 8K, Cover).
4. **Generate Image:**
   - Create a new image based on the main colors of the loaded image.
5. **Save Image:**
   - Save the generated image in WEBP format.

---

## **Technische Details**

### **Deutsch:**
- **Bibliotheken:** `gradio`, `torch`, `PIL`, `numpy`, `multiprocessing`, `cv2`
- **Neurales Netzwerk:**
  - Jeder Knoten repräsentiert eine Farbe, und Verbindungen zwischen den Knoten simulieren Interaktionen zwischen den Farben.
  - Aktivierungen und Gewichtungen werden basierend auf den Farbmerkmalen angepasst.

### **English:**
- **Libraries:** `gradio`, `torch`, `PIL`, `numpy`, `multiprocessing`, `cv2`
- **Neural Network:**
  - Each node represents a color, and connections between nodes simulate interactions between colors.
  - Activations and weights are adjusted based on the color characteristics.

---

## **Installation und Ausführung**

### **Deutsch:**
1. **Voraussetzungen:**
   - Python 3.8 oder höher
   - Installiere die Abhängigkeiten mit:
     ```bash
     pip install gradio torch pillow numpy opencv-python
     ```
2. **Ausführung:**
   - Starte die Anwendung mit:
     ```bash
     python main.py
     ```

### **English:**
1. **Prerequisites:**
   - Python 3.8 or higher
   - Install dependencies using:
     ```bash
     pip install gradio torch pillow numpy opencv-python
     ```
2. **Run:**
   - Start the application with:
     ```bash
     python main.py
     ```

---

## **Mögliche Verbesserungen**

### **Deutsch:**
- Unterstützung für mehr Farbkategorien.
- Optimierung der parallelen Verarbeitung für höhere Effizienz.

### **English:**
- Support for more color categories.
- Optimization of parallel processing for higher efficiency.
