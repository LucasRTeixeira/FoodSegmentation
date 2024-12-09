# bibliotecas necessarias
from matplotlib import pyplot as plt
from skimage import io, color, morphology
from skimage.segmentation import slic, mark_boundaries
from skimage import graph
from pathlib import Path
from skimage.measure import label, regionprops
from skimage.metrics import adapted_rand_error, variation_of_information
from sklearn.metrics import jaccard_score
from skimage.filters import gabor, sobel
import numpy as np
from PIL import Image


base_dir = Path(__file__).parent 

images_dir = base_dir / "assets/images"

# Carregando as imagens
img = Image.open(images_dir / "imagem1.jpg")
img = np.array(img)         #converte o objeto "img" para um vetor manipulável pelo numpy

img_labels = Image.open(images_dir / "imagem1-labels.jpg")
img_labels = np.array(img_labels)         #converte o objeto "img_labels" para um vetor manipulável pelo numpy
print(img_labels.shape)                   #retorna uma tupla de inteiros que mostra o tamanho da imagem (por enquanto é tridimensional)

img_labels = img_labels[:, :, 0] + img_labels[:, :, 1] + img_labels[:, :, 2] #cria uma imagem cinza ao mesmo tempo que deixa a tupla com 2 dimen
print(img_labels.shape)
# np.reshape(img_labels, )
'''

Compactness = 10.0;
N_Segments = 300.0;
Sigma = 0.89;
Threshold_NCut1 = 0.01;
Threshold_NCut2 = 0.01;
IoU_Jaccard = 8.963743053707783e-07;
IoU_Implementado = 0.7921312879335708;
RI_Precision = 1.0

10.0;300.0;0.89;0.01;0.01;8.963743053707783e-07;0.7921312879335708;1.0

'''

# segmentation
compact = 10.0      #parâmetro compactness 
segs = int(300.0)   #parâmetro de N segmentos
sig = 0.89          #parâmetro sigma

segments = slic(img , n_segments = segs , compactness = compact , sigma = sig , start_label = 1)
#(imagem 2d ou 3d , nº de labels no output , proximidade de cor e espaço: quanto maior mais cúbico o segmento , suavizar a dimensão da imagem: 0 é sem suavidade , começo de indexação: 0 ou 1)

image_slic = color.label2rgb(segments, img, kind='avg') #retorna uma imagem rgb que tem color-coded labels pintadas sobre a imagem
#(vetor de labels com shape = ao de img , a própria imagem , 'avg' pinta cada segmento da sua cor média para ficar homogêneo)

image_slic = mark_boundaries(image_slic, segments, color = (0, 0, 0)) #retorna a imagem com os limites entre as regiões rotuladas
#(imagem trabalhada , vetor de labels onde as regiões são marcadas por diferentes valores , cores rgb que os limites terão)


# Primeiro NCut
g1 = graph.rag_mean_color(img, segments, mode='similarity') #representa a imagem na forma de um grafo onde os pixels são os vertices e as adjacências são as arestas. Elas são mescladas pelo peso das arestas
segments_norm1 = graph.cut_normalized(segments, g1, thresh=0.01) #continua o processo de divisão e mesclagem da imagem com o ncut
image_norm1 = color.label2rgb(segments_norm1, img, kind='avg', bg_label=0) #converte os novos rótulos m uma imagem colorida
image_norm1 = mark_boundaries(image_norm1, segments_norm1, (0, 0, 0)) # destaca os contornos dos segmentos da nova imagem trabalhada

# Segundo NCut
g2 = graph.rag_mean_color(img, segments_norm1, mode='similarity')
segments_norm2 = graph.cut_normalized(segments_norm1, g2, thresh=0.01)
image_norm2 = color.label2rgb(segments_norm2, img, kind='avg', bg_label=0)
image_norm2 = mark_boundaries(image_norm2, segments_norm2, (0, 0, 0))

# Mostrando o output
fig, ax = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(20, 20)) #comando que permite colocar várias imagens em uma única
ax1, ax2, ax3, ax4 = ax.ravel()
ax1.set_title("Imagem Original")
ax1.imshow(img)
ax2.set_title("Imagem SLIC")
ax2.imshow(image_slic)
ax3.set_title("Imagem 1 NCut")
ax3.imshow(image_norm1)
ax4.set_title("Imagem 2 NCut")
ax4.imshow(image_norm2)
plt.show()

# Calculando Resultados (IoU e RandIndex)


def iou_coef(y_true, y_pred):
    axes = (0, 1)
    intersection = np.sum(np.logical_and(y_pred, y_true), axis=axes) #logical_and é uma função que aplica a operação lógica "E"
    union = np.sum(np.logical_or(y_pred, y_true), axis=axes) #logical_or é a mesma coisa acima, porém com a operação "OU"
    smooth = .001
    iou = (intersection + smooth) / (union + smooth)
    iou = np.mean(iou) #média aritmética
    return iou


flatten_labels = img_labels.flatten()

print("---------------------- Segments: SLIC -------------------------")
error, precision, recall = adapted_rand_error(img_labels, segments)
flatten_segments = segments.flatten()
print(f"Adapted Rand error: {error}")
print(f"Adapted Rand precision: {precision}")
print(f"Adapted Rand recall: {recall}")
print(f"IOU Jaccard : {jaccard_score(flatten_labels, flatten_segments, average='weighted')}")
print(f"IOU implementado : {iou_coef(img_labels, segments)}")


print("---------------------- Segments: NCut 1 -------------------------")
error, precision, recall = adapted_rand_error(img_labels, segments_norm1)
flatten_segmentsNorm1 = segments_norm1.flatten()
print(f"Adapted Rand error: {error}")
print(f"Adapted Rand precision: {precision}")
print(f"Adapted Rand recall: {recall}")
print(f"IOU Jaccard : {jaccard_score(flatten_labels, flatten_segmentsNorm1, average='weighted')}")
print(f"IOU implementado : {iou_coef(img_labels, segments_norm1)}")

print("---------------------- Segments: NCut 2 -------------------------")
error, precision, recall = adapted_rand_error(img_labels, segments_norm2)
flatten_segmentsNorm2 = segments_norm2.flatten()
print(f"Adapted Rand error: {error}")
print(f"Adapted Rand precision: {precision}")
print(f"Adapted Rand recall: {recall}")
print(f"IOU Jaccard : {jaccard_score(flatten_labels, flatten_segmentsNorm2, average='weighted')}")
print(f"IOU implementado : {iou_coef(img_labels, segments_norm2)}")
