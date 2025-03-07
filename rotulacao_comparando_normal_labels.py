'''
O código a seguir é o código principal, ele segmenta a imagem e plota os segmentos comparando com os segmentos do labels verificando métricas.

Problema: preciso extrair os segmentos da imagem labels de alguma maneira para poder comparar os segmentos obtido pelo algoritmo da maneira correta.
'''

from matplotlib import pyplot as plt
from skimage import color
from skimage.filters import gabor
from skimage.segmentation import slic, mark_boundaries
import skimage.graph as graph
from skimage.metrics import adapted_rand_error
from sklearn.metrics import jaccard_score
import numpy as np
from PIL import Image, ImageEnhance
from scipy import ndimage

# IoU implementado (Rychard)
def iou_coef(y_true, y_pred):
    axes = (0, 1)
    intersection = np.sum(np.logical_and(y_pred, y_true), axis=axes)
    union = np.sum(np.logical_or(y_pred, y_true), axis=axes)
    smooth = .001
    iou = (intersection + smooth) / (union + smooth)
    iou = np.mean(iou)
    return iou

# Carregando as imagens filtradas
img = Image.open(r"C:\Users\Lucas\OneDrive\Documentos\UERJ- MATÉRIAS\INICIAÇÃO CIENTÍFICA\pythonProject\images\imagem3.jpg")
img_rgb = img

enhancer = ImageEnhance.Contrast(img)
sharp = ImageEnhance.Sharpness(img)
img = enhancer.enhance(1.3)
img = sharp.enhance(1.3)
img = ndimage.median_filter(img, size=4)
img = ndimage.gaussian_filter(img, sigma=1)

fig, ax = plt.subplots(ncols=1, sharex=True, sharey=True, figsize=(6, 6))
ax.imshow(img, cmap='gray')
ax.set_title('Imagem com melhorias')
ax.axis('off')
plt.tight_layout()
plt.show()

img = np.array(img)
#img = img[:, :, 0] + img[:, :, 1] + img[:, :, 2] #caso queira usar gabor

#Caso queira aplicar Gabor
#gabor_filter_real, gabor_filter_imag = gabor(img, frequency=0.6)
#img = np.sqrt(gabor_filter_real**2 + gabor_filter_imag**2)

img_labels = Image.open(r"C:\Users\Lucas\OneDrive\Documentos\UERJ- MATÉRIAS\INICIAÇÃO CIENTÍFICA\pythonProject\images\imagem3-labels.jpg")
img_labels = np.array(img_labels)
#img_labels = img_labels[:, :, 0] + img_labels[:, :, 1] + img_labels[:, :, 2]

# Segmentação img normal
#img_lab = color.rgb2lab(img) #talvez seja uma boa ideia para diferenciar o arroz do prato
#segments = slic(img, compactness=25, n_segments=280, start_label=1, sigma=0.789999999, channel_axis = None) #se tiver com Gabor

segments = slic(img, compactness=25, n_segments=280, start_label=1, sigma=0.789999999)
out1 = color.label2rgb(segments, img, kind='avg', bg_label=0)
out1 = mark_boundaries(out1, segments, (0, 0, 0))

# Primeiro NCut
threshold1 = 0.001
threshold2 = 0.01
threshold3 = 0.1

g1 = graph.rag_mean_color(img, segments, mode='similarity')
segments = graph.cut_normalized(segments, g1, thresh=threshold1, max_edge=0.2)

# Verifique se há nós suficientes no grafo antes de prosseguir com o segundo corte
if len(np.unique(segments)) > 1:
    # Segundo NCut
    g2 = graph.rag_mean_color(img, segments, mode='similarity')
    segments = graph.cut_normalized(segments, g2, thresh=threshold2, max_edge=0.2)
    if len(np.unique(segments)) > 1:
    #Terceiro NCut
        g3 = graph.rag_mean_color(img, segments, mode='similarity')
        segments = graph.cut_normalized(segments, g3, thresh=threshold3, max_edge=0.2)

out2 = color.label2rgb(segments, img, kind='avg', bg_label=0)
out2 = mark_boundaries(out2, segments, (0, 0, 0))


# Segmentação img labels
segments_labels = slic(img_labels, compactness=25, n_segments=280, start_label=1, sigma=0.789999999)

# Primeiro NCut
g1_labels = graph.rag_mean_color(img_labels, segments_labels, mode='similarity')
segments_labels = graph.cut_normalized(segments_labels, g1_labels, thresh=threshold1)
    
# Verifique se há nós suficientes no grafo antes de prosseguir com o segundo corte
if len(np.unique(segments_labels)) > 1:
    # Segundo NCut
    g2_labels = graph.rag_mean_color(img_labels, segments_labels, mode='similarity')
    segments_labels = graph.cut_normalized(segments_labels, g2_labels, thresh=threshold2)
    if len(np.unique(segments)) > 1:
    #Terceiro NCut
        g3_labels = graph.rag_mean_color(img_labels, segments_labels, mode='similarity')
        segments_labels = graph.cut_normalized(segments_labels, g3_labels, thresh=threshold3)


# Plotando e calculando métricas para cada segmento separadamente
unique_segments, counts = np.unique(segments, return_counts=True)
sorted_indices = np.argsort(-counts)  # Ordena em ordem decrescente de frequência
sorted_segments = unique_segments[sorted_indices]
sorted_counts = counts[sorted_indices]

print(f"Segmentos unicos em ordem de frequencia da imagem original: {sorted_segments}")
print(f"Frequencia correspondente: {sorted_counts}")

unique_segments_labels, counts_labels = np.unique(segments_labels, return_counts=True)
sorted_indices_labels = np.argsort(-counts_labels)  # Ordena em ordem decrescente de frequência
sorted_segments_labels = unique_segments_labels[sorted_indices_labels]
sorted_counts_labels = counts_labels[sorted_indices_labels]

print(f"\nSegmentos unicos em ordem de frequencia do labels: {sorted_segments_labels}")
print(f"Frequencia correspondente: {sorted_counts_labels}\n")

if len(unique_segments) >= 4:
    # Inicializando listas para armazenar as métricas de cada segmento
    iou_jaccard_por_segmento = []
    iou_implementado_por_segmento = []
    precision_por_segmento = []

    # Garantir que estamos comparando o menor número de segmentos (caso haja diferença)
    num_segments = min(len(unique_segments), len(unique_segments_labels))

    # Loop para comparar cada segmento da imagem original com o correspondente da imagem rotulada
    for idx in range(num_segments):
        seg_id = sorted_segments[idx]
        seg_label_id = sorted_segments_labels[idx]
        
        # Criar máscara para o segmento da imagem original
        mask = segments == seg_id
        
        # Criar máscara para o segmento correspondente da imagem rotulada
        mask_label = segments_labels == seg_label_id

        # Convertendo o segmento atual para binário
        y_pred_segment_bin = mask
        
        # Convertendo o label correspondente para binário
        y_true_segment_bin = mask_label
        
        # Calculando as métricas IoU e Rand Index para o segmento
        if np.any(y_true_segment_bin) or np.any(y_pred_segment_bin):  # Verifica se o segmento existe no label
            iou_jaccard = jaccard_score(y_true_segment_bin.flatten(), y_pred_segment_bin.flatten(), average='binary', zero_division=0)
            error, precision, recall = adapted_rand_error(y_true_segment_bin, y_pred_segment_bin)
            iou_segment = iou_coef(y_true_segment_bin, y_pred_segment_bin)
        else:
            iou_jaccard, precision, iou_segment = 0, 0, 0  # Se o segmento não existir no rótulo verdadeiro
        
        # Armazenar os valores para cada segmento
        iou_implementado_por_segmento.append((seg_id, iou_segment))
        precision_por_segmento.append((seg_id, precision))
        iou_jaccard_por_segmento.append((seg_id, iou_jaccard))
        
        # Exibir as métricas para cada segmento
        print(f'Segmento {seg_id} vs {seg_label_id}: Jaccard = {iou_jaccard}, IoU = {iou_segment}, Precision = {precision}')

        # Criar uma figura com 2 subplots lado a lado
        fig, axes = plt.subplots(1, 2, figsize=(8, 5))

        # Plotar o segmento da imagem original
        axes[0].imshow(img * np.dstack([mask] * 3))  # Multiplicar pela máscara para mostrar apenas o segmento
        axes[0].set_title(f"Segmento Original: {seg_id}")
        axes[0].axis('off')  # Remover os eixos

        # Plotar o segmento correspondente da imagem rotulada
        axes[1].imshow(img_labels * np.dstack([mask_label] * 3))  # Mostrar apenas o segmento rotulado
        axes[1].set_title(f"Segmento Rotulado: {seg_label_id}")
        axes[1].axis('off')  # Remover os eixos

        # Ajustar o layout para evitar sobreposição
        plt.tight_layout()

        # Exibir a janela única com as duas imagens
        plt.show()

    # Exibindo os resultados gerais (médias)
    jaccard_medio = np.mean([jac for _, jac in iou_jaccard_por_segmento])
    iou_medio = np.mean([iou for _, iou in iou_implementado_por_segmento])
    precision_media = np.mean([prec for _, prec in precision_por_segmento])

    print(f'Media Jaccard por segmento: {jaccard_medio}')
    print(f'Media IoU por segmento: {iou_medio}')
    print(f'Media Precision por segmento: {precision_media}')

else:
    print('Segmentacao nao vai servir')


fig, ax = plt.subplots(1, 3)

ax[0].imshow(img)
ax[0].set_title('Imagem Original')

ax[1].imshow(out1)
ax[1].set_title('Imagem SLIC')

ax[2].imshow(out2)
ax[2].set_title('Imagem NCUT')

plt.tight_layout()
plt.show()