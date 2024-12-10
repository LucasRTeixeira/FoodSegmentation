import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance
from pathlib import Path
from scipy import ndimage
import numpy as np
from skimage.exposure import equalize_adapthist
from skimage.filters import sobel, unsharp_mask
from cv2 import bilateralFilter
from skimage.restoration import denoise_nl_means, estimate_sigma

base_dir = Path(__file__).parent 
images_dir = base_dir / "assets/images"

def criaMascara(img):

    def plotar_imagens(img_rgb , img_processada):
        fig, axes = plt.subplots(ncols=2, sharex=True, sharey=True, figsize=(8, 8))

        # Mostrando a imagem original
        axes[0].imshow(img_rgb)
        axes[0].set_title('Imagem Original')
        axes[0].axis('off')

        # Mostrando a imagem original sem contraste
        axes[1].imshow(img_processada, cmap='gray')
        axes[1].set_title('Imagem Original Cinza')
        axes[1].axis('off')

        plt.tight_layout()
        plt.show()

    # Carregando e convertendo imagem
    img_rgb = Image.open(images_dir / "imagem3.jpg")
    img = img_rgb.convert('L')

    # Ajuste de contraste
    enhancer = ImageEnhance.Contrast(img)
    img_contraste = enhancer.enhance(1.3)
    plotar_imagens(img_rgb, img_contraste)

    # Remoção de ruídos (Filtro Bilateral)
    img_bilateral = bilateralFilter(np.array(img_contraste), d=9, sigmaColor=75, sigmaSpace=75)
    plotar_imagens(img_rgb, img_bilateral)

    # Equalização adaptativa do histograma
    img_equalized = equalize_adapthist(img_bilateral, clip_limit=0.03)
    plotar_imagens(img_rgb, img_equalized)

    # Suavização com filtro Gaussiano
    img_gaussian = ndimage.gaussian_filter(img_equalized, sigma=1)
    plotar_imagens(img_rgb, img_gaussian)

    # Realce de bordas (Sobel)
    img_bordas = sobel(img_gaussian)
    plotar_imagens(img_rgb, img_bordas)

    # Nitidez (Unsharp Mask)
    img_final = unsharp_mask(img_bordas, radius=1, amount=1.5)
    plotar_imagens(img_rgb, img_final)