# bibliotecas necessarias
from matplotlib import pyplot as plt
from pathlib import Path
from skimage.segmentation import slic
import skimage.graph as graph
from skimage.metrics import adapted_rand_error
from sklearn.metrics import jaccard_score
import numpy as np
from PIL import Image

base_dir = Path(__file__).parent 

images_dir = base_dir / "assets/images"

#IoU implementado (Rychard)
def iou_coef(y_true, y_pred):
    axes = (0, 1)
    intersection = np.sum(np.logical_and(y_pred, y_true), axis=axes)
    union = np.sum(np.logical_or(y_pred, y_true), axis=axes)
    smooth = .001
    iou = (intersection + smooth) / (union + smooth)
    iou = np.mean(iou)
    return iou


# Carregando as imagens filtradas
img = Image.open(images_dir / "imagem3.jpg")
img_rgb = img
img = np.array(img)

img_labels = Image.open(images_dir / "imagem3-labels-teste.jpg")
img_labels = np.array(img_labels)
img_labels = img_labels[:, :, 0] + img_labels[:, :, 1] + img_labels[:, :, 2]
flatten_labels = img_labels.flatten()

#variaveis para guardar no txt
precision = 0
iou_jaccard = 0
iou_implementado = 0

#segmentacao
segments = slic(img, compactness=25, n_segments=50, start_label=1, sigma=0.78999)

# Primeiro NCut
threshold1 = 0.001
threshold2 = 0.01
if threshold1 > 0:
    g1 = graph.rag_mean_color(img, segments, mode='similarity')
    segments = graph.cut_normalized(segments, g1, thresh=threshold1)
        
    #Verifique tem nós suficientes no grafo antes de prosseguir com o segundo corte
    if len(np.unique(segments)) > 1:
        # Segundo NCut
        if threshold2 > 0:
            g2 = graph.rag_mean_color(img, segments, mode='similarity')
            segments = graph.cut_normalized(segments, g2, thresh=threshold2)

# Verificando as variáveis img_labels e segments
#print(f"img_labels shape: {img_labels.shape}, unique values: {np.unique(img_labels)}")
#print(f"segments shape: {segments.shape}, unique values: {np.unique(segments)}")

# Convertendo as segmentações para binário para IoU
y_true_bin = img_labels > 0
y_pred_bin = segments > 0

# Verificando os binários
#print(f"y_true_bin shape: {y_true_bin.shape}, unique values: {np.unique(y_true_bin)}")
#print(f"y_pred_bin shape: {y_pred_bin.shape}, unique values: {np.unique(y_pred_bin)}")

# Calculando os resultados
error, precision, recall = adapted_rand_error(img_labels, segments)
flatten_segments = segments.flatten()
iou_jaccard = jaccard_score(flatten_labels, flatten_segments, average='weighted')
iou_implementado = iou_coef(y_true_bin, y_pred_bin)

print(f'IoU Jaccard: {iou_jaccard}')
print(f'IoU Implementado: {iou_implementado}')
print(f'Precisao: {precision}')

#guardando os resultados
#dados.iat[linha.Index, 5] = iou_jaccard
#dados.iat[linha.Index, 6] = iou_implementado
#dados.iat[linha.Index, 7] = precision

plt.figure(figsize=(10, 10))
plt.imshow(segments)
plt.title(f"Jaccard: {iou_jaccard}; IoU: {iou_implementado}; Precision: {precision}")
plt.axis('off')
plt.show()

# Criando um dicionário para armazenar as segmentações com rótulos
segmented_images = {}

# Atribuindo rótulos e criando máscaras binárias para cada segmentação
for label in np.unique(segments):
    segmented_images[label] = (segments == label).astype(np.uint8)

# Exibindo as segmentações individualmente
for label, mask in segmented_images.items():
    plt.figure()
    plt.imshow(mask, cmap='gray')
    plt.title(f"Segmentação {label}")
    plt.axis('off')
    plt.show()
