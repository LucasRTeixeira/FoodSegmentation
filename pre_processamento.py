import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance
from pathlib import Path
from scipy import ndimage
import numpy as np
from skimage.metrics import adapted_rand_error
from skimage.filters import sobel, gabor
from skimage.morphology import binary_closing, binary_opening, binary_dilation, binary_erosion, skeletonize, thin, disk

def criaMascara(img):

    base_dir = Path(__file__).parent 

    
    images_dir = base_dir / "assets/images"

    ######################################################################################
    #Funções para avaliar a qualidade da máscara

    #IoU - sobreposição
    def calcular_iou(mascara_obtida, mascara_ideal):
        if mascara_obtida.shape != mascara_ideal.shape:
            raise ValueError(f"Dimensões diferentes: mascara_obtida {mascara_obtida.shape}, mascara_ideal {mascara_ideal.shape}")

        interseccao = np.logical_and(mascara_obtida, mascara_ideal)
        uniao = np.logical_or(mascara_obtida, mascara_ideal)
        iou = np.sum(interseccao) / np.sum(uniao)
        return iou

    #Dice Coefficient - sobreposição com mais peso na interseção
    def calcular_dice(mascara_obtida, mascara_ideal):
        interseccao = np.logical_and(mascara_obtida, mascara_ideal)
        dice = 2 * np.sum(interseccao) / (np.sum(mascara_obtida) + np.sum(mascara_ideal))
        return dice

    #Mean Squared Error - Erro Quadrático Médio
    def calcular_mse(mascara_obtida, mascara_ideal):
        mascara_obtida_int = mascara_obtida.astype(np.float32)
        mascara_ideal_int = mascara_ideal.astype(np.float32)
        mse = np.mean((mascara_obtida_int - mascara_ideal_int) ** 2)
        return mse

    #Rand Index - similaridade entre grupos de pixel
    def calcular_rand_index(mascara_obtida, mascara_ideal):
        error, precision, recall = adapted_rand_error(mascara_ideal, mascara_obtida)
        return error, precision, recall

    ######################################################################################

    # Função para plotar imagens
    def plotar_imagens(img_rgb , img1, img2, img3, img4, img5, title1 , title2, title3, title4):
        fig, axes = plt.subplots(nrows=2, ncols=3, sharex=True, sharey=True, figsize=(8, 8))

        # Mostrando a imagem original
        axes[0, 0].imshow(img_rgb)
        axes[0, 0].set_title('Imagem Original')
        axes[0, 0].axis('off')

        # Mostrando a imagem original sem contraste
        axes[0, 1].imshow(img1, cmap='gray')
        axes[0, 1].set_title('Imagem Original Cinza')
        axes[0, 1].axis('off')

        # Mostrando a imagem modificada
        axes[0, 2].imshow(img2, cmap='gray')
        axes[0, 2].set_title(title1)
        axes[0, 2].axis('off')

        # Mostrando a imagem modificada
        axes[1, 0].imshow(img3, cmap='gray')
        axes[1, 0].set_title(title2)
        axes[1, 0].axis('off')

        # Mostrando a imagem modificada
        axes[1, 1].imshow(img4, cmap='gray')
        axes[1, 1].set_title(title3)
        axes[1, 1].axis('off')

        # Mostrando a imagem modificada
        axes[1, 2].imshow(img5, cmap='gray')
        axes[1, 2].set_title(title4)
        axes[1, 2].axis('off')

        # Ajustando o layout
        plt.tight_layout()
        plt.show()

    # Função para aplicar morfologia
    # Use 'closing', 'opening', 'dilation', 'erosion', 'skeletonize' ou 'thin'
    def aplicar_morfologia(img_binaria, operacao="closing", elemento_estruturante=2):
        selem = disk(elemento_estruturante)  # Elemento estruturante em formato de disco
        
        if operacao == "closing":
            img_morf = binary_closing(img_binaria, selem)   #talvez o melhor resultado
        elif operacao == "opening":
            img_morf = binary_opening(img_binaria, selem)   #bom resultado
        elif operacao == "dilation":
            img_morf = binary_dilation(img_binaria, selem)  #bom resultado
        elif operacao == "erosion":
            img_morf = binary_erosion(img_binaria, selem)   #resultado mais ou menos
        elif operacao == "skeletonize":
            img_morf = skeletonize(img_binaria)             #tem fundamento, mas não vai servir
        elif operacao == "thin":
            img_morf = thin(img_binaria)                    #tem fundamento, mas não vai servir
        else:
            raise ValueError("Operação desconhecida. Use 'closing', 'opening', 'dilation', 'erosion', 'skeletonize' ou 'thin'.")
        
        return img_morf

    # Carregando a imagem
    img_rgb = Image.open(images_dir / "imagem10.jpg")
    img = img.convert('L')  # Convertendo para escala de cinza / 2 dim

    mascara_ideal_rgb = Image.open(images_dir / "images mask/imagem1-label-mascara.jpg")
    mascara_ideal = mascara_ideal_rgb.convert('L')
    mascara_ideal = np.array(mascara_ideal) > 128  # Transformar para binário

    # Aumentando o contraste da imagem
    enhancer = ImageEnhance.Contrast(img)  # 1.2 aumenta o contraste; 1.0 seria o padrão
    img_contraste0 = enhancer.enhance(1.2)  # 2º melhor (em cinza)
    img_contraste1 = enhancer.enhance(1.3)  # 1º melhor (em cinza)
    img_contraste2 = enhancer.enhance(1.4)  # 3º melhor (em cinza)
    img_contraste3 = enhancer.enhance(1.5)  # 4º melhor (em cinza)

    plotar_imagens(img_rgb , img, img_contraste0, img_contraste1, img_contraste2, img_contraste3, 'Contraste Aumentado 1.2' , 'Contraste Aumentado 1.3', 'Contraste Aumentado 1.4' , 'Contraste Aumentado 1.5')

    # Aplicando filtro de média
    img_media0 = ndimage.median_filter(img_contraste1, size=2)  # 4º melhor (em cinza)
    img_media1 = ndimage.median_filter(img_contraste1, size=3)  # 2º melhor (em cinza)
    img_media2 = ndimage.median_filter(img_contraste1, size=4)  # 1º melhor (em cinza)
    img_media3 = ndimage.median_filter(img_contraste1, size=5)  # 3º melhor (em cinza)

    #plotar_imagens(img_rgb , img, img_media0, img_media1, img_media2, img_media3, 'Filtro de media size 2', 'Filtro de media size 3', 'Filtro de media size 4', 'Filtro de media size 5')

    ######################################################################################

    # Aplicando filtro gaussiano
    img_gaussian0 = ndimage.gaussian_filter(img_media2, sigma=0.5)  # 2º melhor (em cinza)
    img_gaussian1 = ndimage.gaussian_filter(img_media2, sigma=1)  # 1º melhor (em cinza)
    img_gaussian2 = ndimage.gaussian_filter(img_media2, sigma=1.5)  # 3º melhor (em cinza)
    img_gaussian3 = ndimage.gaussian_filter(img_media2, sigma=2)  # 4º melhor (em cinza)

    plotar_imagens(img_rgb , img, img_gaussian0, img_gaussian1, img_gaussian2, img_gaussian3, 'Filtro gaussiano sigma 0.5', 'Filtro gaussiano sigma 1', 'Filtro gaussiano sigma 1.5', 'Filtro gaussiano sigma 2')

    ######################################################################################

    # Aplicando sobel (com limiar)
    sobel_thresh = 0.0046       #pode ser que mexendo nisso a máscara melhore. Está bom, mas pode melhorar
    img_sobel = sobel(img_gaussian1)
    img_sobel_limiar = np.where(img_sobel > sobel_thresh, img_sobel, 0)

    fig, ax = plt.subplots(ncols=1, sharex=True, sharey=True, figsize=(6, 6))
    ax.imshow(img_sobel_limiar, cmap='gray')
    ax.set_title('Imagem com Sobel (Limiar aplicado)')
    ax.axis('off')
    plt.tight_layout()
    plt.show()

    # Aplicando morfologia às imagens após o limiar de Sobel
    img_sobel_morf = aplicar_morfologia(img_sobel_limiar, operacao="closing", elemento_estruturante=2)

    fig, ax = plt.subplots(ncols=1, sharex=True, sharey=True, figsize=(6, 6))
    ax.imshow(img_sobel_morf, cmap='gray')
    ax.set_title('Imagem com Sobel (Morfologia aplicada)')
    ax.axis('off')
    plt.tight_layout()
    plt.show()

    ######################################################################################

    # Aplicando filtro de Gabor (com limiar)
    gabor_thresh = 0.001       #pode ser que mexendo nisso a máscara melhore. Está bom, mas pode melhorar
    filt_real0, img_gabor0 = gabor(img_gaussian1, frequency=0.7)  # 1º melhor (em cinza)
    filt_real1, img_gabor1 = gabor(img_gaussian1, frequency=0.9)  # 2º melhor (em cinza)
    filt_real2, img_gabor2 = gabor(img_gaussian1, frequency=1.1)  # 3º melhor (em cinza)
    filt_real3, img_gabor3 = gabor(img_gaussian1, frequency=1.3)  # 4º melhor (em cinza)

    # Aplicando o limiar ao filtro de Gabor
    img_gabor0_limiar = np.where(img_gabor0 > gabor_thresh, img_gabor0, 0)
    img_gabor1_limiar = np.where(img_gabor1 > gabor_thresh, img_gabor1, 0)
    img_gabor2_limiar = np.where(img_gabor2 > gabor_thresh, img_gabor2, 0)
    img_gabor3_limiar = np.where(img_gabor3 > gabor_thresh, img_gabor3, 0)

    # Plotando as imagens resultantes com o limiar aplicado
    plotar_imagens(img_rgb , img, img_gabor0_limiar, img_gabor1_limiar, img_gabor2_limiar, img_gabor3_limiar, 'Gabor (0.7) com Limiar', 'Gabor (0.9) com Limiar', 'Gabor (1.1) com Limiar', 'Gabor (1.3) com Limiar')

    img_gabor0_morf = aplicar_morfologia(img_gabor0_limiar, operacao="closing", elemento_estruturante=2)
    img_gabor1_morf = aplicar_morfologia(img_gabor1_limiar, operacao="closing", elemento_estruturante=2)
    img_gabor2_morf = aplicar_morfologia(img_gabor2_limiar, operacao="closing", elemento_estruturante=2)
    img_gabor3_morf = aplicar_morfologia(img_gabor3_limiar, operacao="closing", elemento_estruturante=2)

    plotar_imagens(img_rgb , img, img_gabor0_morf, img_gabor1_morf, img_gabor2_morf, img_gabor3_morf, 'Gabor (0.7) com Morfologia', 'Gabor (0.9) com Morfologia', 'Gabor (1.1) com Morfologia', 'Gabor (1.3) com Morfologia')

    ######################################################################################
    # Plotando a máscara final e os resultados intermediários
    mascara_final = np.maximum(img_sobel_morf, img_gabor0_morf)

    # Redimensionando a mascara_final para ter o mesmo tamanho da mascara_ideal
    mascara_final_resized = Image.fromarray(mascara_final).resize(mascara_ideal.shape[::-1], Image.NEAREST)
    mascara_final_resized = np.array(mascara_final_resized)

    # Nem todas as imagens estão rodando certinho
    iou = calcular_iou(mascara_final_resized, mascara_ideal)
    dice = calcular_dice(mascara_final_resized, mascara_ideal)
    mse = calcular_mse(mascara_final_resized, mascara_ideal)
    rand_error, precision, recall = calcular_rand_index(mascara_final_resized, mascara_ideal)

    mascara_original = Image.fromarray(mascara_final_resized).resize(mascara_final.shape[::-1], Image.NEAREST)
    mascara_original = np.array(mascara_original)

    print(f"IoU: {iou}, Dice: {dice}, MSE: {mse}")
    print(f"Rand Error: {rand_error}, Precision: {precision}, Recall: {recall}")

    plotar_imagens(img_rgb, img, img_sobel_morf, img_gabor0_morf, mascara_original, img_gabor1_morf,
                    'Sobel com Morfologia', 'Gabor com Morfologia', 'Mascara Final', 'IGNORAR')

    return mascara_original