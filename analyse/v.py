from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
import cv2

# Pfade zu den Bildern
original_path = "bild4.jpg"
neural_path = "kunst.png"

# Lade die Bilder
original_image = Image.open(original_path).convert("RGB")
neural_image = Image.open(neural_path).convert("RGB")

# Größe des Originalbildes anpassen, falls nötig
neural_image_resized = neural_image.resize(original_image.size, resample=Image.LANCZOS)

# 1. Histogramm-Vergleich
original_hist, _ = np.histogram(np.array(original_image).flatten(), bins=256, range=(0, 255))
neural_hist, _ = np.histogram(np.array(neural_image_resized).flatten(), bins=256, range=(0, 255))

plt.figure(figsize=(12, 6))
plt.bar(range(256), original_hist, color='blue', alpha=0.5, label='Original')
plt.bar(range(256), neural_hist, color='red', alpha=0.5, label='Neural')
plt.title("Histogram Comparison")
plt.xlabel("Pixel Intensity")
plt.ylabel("Frequency")
plt.legend()
plt.savefig("histogram_comparison.png")
plt.close()

# 2. Kanten-Detektion
original_gray = original_image.convert("L")
neural_gray_resized = neural_image_resized.convert("L")

edges_original = cv2.Canny(np.array(original_gray), threshold1=100, threshold2=200)
edges_neural = cv2.Canny(np.array(neural_gray_resized), threshold1=100, threshold2=200)

fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes[0].imshow(edges_original, cmap="gray")
axes[0].set_title("Edges - Original Image")
axes[0].axis("off")

axes[1].imshow(edges_neural, cmap="gray")
axes[1].set_title("Edges - Neural Network Image")
axes[1].axis("off")

plt.tight_layout()
plt.savefig("edge_detection.png")
plt.close()

# 3. Differenzanalyse
original_array = np.array(original_image)
neural_array = np.array(neural_image_resized)
difference = np.abs(original_array.astype(int) - neural_array.astype(int))
difference_highlighted = np.clip(difference * 5, 0, 255).astype(np.uint8)

plt.figure(figsize=(8, 8))
plt.imshow(difference_highlighted)
plt.title("Pixel Difference (Highlighted)")
plt.axis("off")
plt.savefig("pixel_difference.png")
plt.close()

# 4. SSIM-Analyse
original_gray_cv = cv2.cvtColor(np.array(original_image), cv2.COLOR_RGB2GRAY)
neural_gray_cv = cv2.cvtColor(np.array(neural_image_resized), cv2.COLOR_RGB2GRAY)

similarity_index, diff = ssim(original_gray_cv, neural_gray_cv, full=True)
diff = (diff * 255).astype(np.uint8)

plt.figure(figsize=(8, 8))
plt.imshow(diff, cmap="gray")
plt.title(f"SSIM Difference Map (Index: {similarity_index:.4f})")
plt.axis("off")
plt.savefig("ssim_difference.png")
plt.close()

# 5. Frequenzanalyse
def plot_frequency_spectrum(image, title, save_path):
    # Fourier-Transformation
    image_array = np.array(image.convert("L"))
    f_transform = np.fft.fft2(image_array)
    f_shift = np.fft.fftshift(f_transform)
    magnitude_spectrum = 20 * np.log(np.abs(f_shift) + 1)

    # Darstellung
    plt.figure(figsize=(8, 8))
    plt.imshow(magnitude_spectrum, cmap="gray")
    plt.title(title)
    plt.axis("off")
    plt.savefig(save_path)
    plt.close()

# Analyse des Original- und Neural-Bildes
plot_frequency_spectrum(original_gray, "Frequenzspektrum - Originalbild", "frequency_spectrum_original.png")
plot_frequency_spectrum(neural_gray_resized, "Frequenzspektrum - Neural generiertes Bild", "frequency_spectrum_neural.png")

# 6. Vergleich der Frequenzbereiche
def decompose_frequency(image, title, save_path):
    # Fourier-Transformation
    image_array = np.array(image.convert("L"))
    f_transform = np.fft.fft2(image_array)
    f_shift = np.fft.fftshift(f_transform)

    # Maskierung
    rows, cols = f_shift.shape
    crow, ccol = rows // 2, cols // 2

    # Niedrige Frequenzen
    low_pass = np.copy(f_shift)
    low_pass[crow-50:crow+50, ccol-50:ccol+50] = 0

    # Hohe Frequenzen
    high_pass = f_shift - low_pass

    # Rücktransformation
    low_image = np.abs(np.fft.ifft2(np.fft.ifftshift(low_pass)))
    high_image = np.abs(np.fft.ifft2(np.fft.ifftshift(high_pass)))

    # Darstellung
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(low_image, cmap="gray")
    plt.title(f"Niedrige Frequenzen - {title}")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(high_image, cmap="gray")
    plt.title(f"Hohe Frequenzen - {title}")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# Analyse des Original- und Neural-Bildes
decompose_frequency(original_gray, "Originalbild", "frequency_decomposition_original.png")
decompose_frequency(neural_gray_resized, "Neural generiertes Bild", "frequency_decomposition_neural.png")

# Erweiterung: Vergleich der Farbfrequenzen
def compare_color_frequency(image1, image2, save_path):
    # Zerlege beide Bilder in RGB-Komponenten
    image1_array = np.array(image1)
    image2_array = np.array(image2)

    fig, axes = plt.subplots(3, 1, figsize=(10, 8))
    colors = ['red', 'green', 'blue']
    for i, color in enumerate(colors):
        hist1, _ = np.histogram(image1_array[..., i].flatten(), bins=256, range=(0, 255))
        hist2, _ = np.histogram(image2_array[..., i].flatten(), bins=256, range=(0, 255))
        axes[i].bar(range(256), hist1, color=color, alpha=0.5, label='Original')
        axes[i].bar(range(256), hist2, color=color, alpha=0.5, label='Neural')
        axes[i].set_title(f"{color.capitalize()} Channel Comparison")
        axes[i].legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# Erweiterung: Local Binary Pattern (LBP) für Texturen
def compute_lbp(image, radius=1, points=8):
    from skimage.feature import local_binary_pattern
    gray_image = np.array(image.convert("L"))
    lbp = local_binary_pattern(gray_image, points, radius, method="uniform")
    return lbp

def compare_lbp(image1, image2, save_path):
    lbp1 = compute_lbp(image1)
    lbp2 = compute_lbp(image2)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(lbp1, cmap="gray")
    axes[0].set_title("LBP - Original Image")
    axes[0].axis("off")

    axes[1].imshow(lbp2, cmap="gray")
    axes[1].set_title("LBP - Neural Image")
    axes[1].axis("off")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# Erweiterung: Gradient Magnitude (Unschärfeprüfung)
def compute_gradient_magnitude(image):
    gray_image = np.array(image.convert("L"), dtype=float)
    grad_x = np.gradient(gray_image, axis=1)
    grad_y = np.gradient(gray_image, axis=0)
    grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    return grad_magnitude

def compare_gradients(image1, image2, save_path):
    grad1 = compute_gradient_magnitude(image1)
    grad2 = compute_gradient_magnitude(image2)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(grad1, cmap="hot")
    axes[0].set_title("Gradient Magnitude - Original Image")
    axes[0].axis("off")

    axes[1].imshow(grad2, cmap="hot")
    axes[1].set_title("Gradient Magnitude - Neural Image")
    axes[1].axis("off")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# Farbfrequenzvergleich
compare_color_frequency(original_image, neural_image_resized, "color_frequency_comparison.png")

# LBP-Analyse
compare_lbp(original_image, neural_image_resized, "lbp_comparison.png")

# Gradientenvergleich
compare_gradients(original_image, neural_image_resized, "gradient_comparison.png")

# Ergebnisse in eine Textdatei speichern
with open("comparison_results.txt", "w") as f:
    f.write("Histogram Comparison:\n")
    f.write(f"Original Histogram: {original_hist}\n")
    f.write(f"Neural Histogram: {neural_hist}\n\n")

    f.write("SSIM Analysis:\n")
    f.write(f"Similarity Index: {similarity_index:.4f}\n\n")

    f.write("Color Frequency Comparison:\n")
    for i, color in enumerate(['red', 'green', 'blue']):
        hist1, _ = np.histogram(np.array(original_image)[..., i].flatten(), bins=256, range=(0, 255))
        hist2, _ = np.histogram(np.array(neural_image_resized)[..., i].flatten(), bins=256, range=(0, 255))
        f.write(f"{color.capitalize()} Channel:\n")
        f.write(f"Original: {hist1}\n")
        f.write(f"Neural: {hist2}\n\n")

    f.write("LBP Analysis:\n")
    lbp1 = compute_lbp(original_image)
    lbp2 = compute_lbp(neural_image_resized)
    f.write(f"LBP Original: {lbp1}\n")
    f.write(f"LBP Neural: {lbp2}\n\n")

    f.write("Gradient Magnitude Comparison:\n")
    grad1 = compute_gradient_magnitude(original_image)
    grad2 = compute_gradient_magnitude(neural_image_resized)
    f.write(f"Gradient Original: {grad1}\n")
    f.write(f"Gradient Neural: {grad2}\n")

print("Analyse abgeschlossen. Ergebnisse sind gespeichert.")
