
# Neural Network Image Generator

Dieses Repository enthält ein neuronales Netzwerk, das Bilder basierend auf Farbaktivierungen generiert und anpasst. Die Anwendung kombiniert Computer Vision, Deep Learning und GUI-Elemente, um eine interaktive Umgebung für die Bildverarbeitung bereitzustellen.

## Funktionen

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

- **GUI für einfache Nutzung:**
  - Lade, bearbeite und speichere Bilder direkt über eine grafische Benutzeroberfläche.
  - Konfiguriere Helligkeit, Kontrast und Bildauflösung.

- **Fortschrittliche Bildverarbeitung:**
  - Parallele Verarbeitung mit `multiprocessing`.
  - Schärfen und Histogrammanpassung für verbesserte Bildqualität.

## Voraussetzungen

Stelle sicher, dass folgende Pakete installiert sind:

- Python 3.8+
- `customtkinter`
- `Pillow`
- `torch`
- `numpy`
- `opencv-python`
- `multiprocessing`

Installiere die Abhängigkeiten mit:
```bash
pip install customtkinter pillow torch numpy opencv-python
```

## Nutzung

1. **Starte die GUI:**
   ```bash
   python main.py
   ```
2. **Lade ein Bild:**
   - Gehe zu `Datei > Bild laden`.
3. **Passe Parameter an:**
   - Wähle Helligkeit, Kontrast und Auflösung.
4. **Generiere ein Bild:**
   - Gehe zu `Datei > Bild generieren`.
   - Das generierte Bild wird unter `kunst.png` gespeichert.

## Funktionsweise

### Neuronales Netzwerk
- Jeder Knoten repräsentiert eine Farbe (z. B. Rot, Grün, Blau).
- Aktivierungen basieren auf Farbhelligkeit und beeinflussen die Bildverarbeitung.

### Bildgenerierung
1. **Hauptfarben extrahieren:**
   - Farben mit hoher Helligkeit werden priorisiert.
2. **Netzwerk erstellen:**
   - Knotenverbindungen mit zufälligen Gewichtungen.
3. **Pixelanpassung:**
   - Parallele Verarbeitung, um Helligkeit und Kontrast dynamisch zu modifizieren.
4. **Bild wird neu erstellt:**
   - Jedes Pixel wird auf Basis der Aktivierungen und Parameter neu berechnet. Das Bild wird vollständig neu generiert, was eine maximale Qualität gewährleistet.

### Bildqualität
- **Schärfen:** Verbessert die Kantenschärfe.
- **Histogrammanpassung:** Optimiert die Helligkeitsverteilung.
- **Neue Auflösung ohne Qualitätsverlust:** Durch die Neuberechnung aller Pixel bleiben Details und Klarheit bei jeder Zielauflösung unverändert.

## Beispiel

### Originalbild:
![1](https://github.com/user-attachments/assets/fa99bed8-eb9e-40fb-b31e-df190fe9c0d0)

### Generiertes Bild:
![kunst](https://github.com/user-attachments/assets/63d477dc-5c02-4ec6-bf50-70c6575c6e48)

## Entwicklerhinweise

### Erweiterungsmöglichkeiten
- **Neue Knotenfarben:** Füge mehr Farben hinzu, um spezifische Anpassungen zu ermöglichen.
- **Datenvisualisierung:** Implementiere ein Tool, um Aktivierungen und Gewichtungen grafisch darzustellen.
- **Echtzeitverarbeitung:** Optimiere die Performance für große Bilder.

### Debugging
- Prüfe Aktivierungswerte und Gewichtungen während der Bildgenerierung mit Debug-Logs.

## Lizenz

Dieses Projekt ist unter der MIT-Lizenz veröffentlicht. Siehe [LICENSE](LICENSE) für weitere Details.

## Autoren

Entwickelt von [Ralf Krümmel].
```

