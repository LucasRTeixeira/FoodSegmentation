from matplotlib import pyplot as plt
from skimage import color
from skimage.segmentation import mark_boundaries
from pathlib import Path
import numpy as np
from skimage.segmentation import slic
from skimage.io import imread
from skimage.transform import resize

base_dir = Path(__file__).parent 

images_dir = base_dir / "assets/images"

# Carregue sua imagem de referência (imagem "gabarito") e a imagem que você deseja segmentar

imagem_referencia = imread(images_dir / "imagem1-labels.jpg")
imagem_segmentar = imread(images_dir / "imagem1.jpg")

# Redimensione a imagem que você deseja segmentar para ter as mesmas dimensões da imagem de referência
imagem_segmentar = resize(imagem_segmentar, imagem_referencia.shape[:2], mode='constant')

# Realize a segmentação na imagem que você deseja segmentar usando o método SLIC
segmentos = slic(imagem_segmentar, n_segments=100, compactness=10)

# Obtenha a lista de valores únicos dos segmentos
valores_segmentos = np.unique(segmentos)

# Inicialize o somatório das sobreposições e o contador de segmentos
soma_sobreposicoes = 0
contador_segmentos = 0

# Percorra cada segmento
for valor_segmento in valores_segmentos:
    # Crie uma máscara para o segmento atual
    mascara_segmento = segmentos == valor_segmento

    # Calcule a área do segmento na imagem de referência
    area_referencia = np.sum(imagem_referencia[mascara_segmento])

    # Calcule a área do segmento na imagem segmentada
    area_segmentar = np.sum(imagem_segmentar[mascara_segmento])

    # Calcule a área da interseção entre o segmento na imagem de referência e na imagem segmentada
    area_intersecao = np.sum(imagem_referencia[mascara_segmento] * imagem_segmentar[mascara_segmento])

    # Verifique se a área de interseção é maior que zero antes de calcular o IoU
    if area_intersecao > 0:
        # Calcule o Índice de Jaccard (IoU) para o segmento atual
        iou = area_intersecao / (area_referencia + area_segmentar - area_intersecao)

        # Adicione o IoU ao somatório total
        soma_sobreposicoes += iou

        # Incremente o contador de segmentos
        contador_segmentos += 1

# Verifique se pelo menos um segmento teve sobreposição válida
if contador_segmentos > 0:
    # Calcule a média dos IoUs para obter a sobreposição média
    sobreposicao_media = soma_sobreposicoes / contador_segmentos
else:
    # Caso contrário, defina a sobreposição média como zero
    sobreposicao_media = 0

print("Sobreposição média (IoU):", sobreposicao_media)



#
# fig, ax = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(20, 20))
# ax1, ax2, ax3 = ax.ravel()
# ax1.set_title("Imagem Original")
# ax1.imshow(imagem_segmentar)
# ax2.set_title("Imagem SLIC")
# ax2.imshow(image_slic)
# ax3.set_title("Imagem Gabarito")
# ax3.imshow(imagem_referencia)
# plt.show()
