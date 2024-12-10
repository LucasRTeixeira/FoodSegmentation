from matplotlib import pyplot as plt
from skimage import color
from skimage.segmentation import slic
import skimage.graph as graph
from pathlib import Path
from skimage.metrics import adapted_rand_error
from sklearn.metrics import jaccard_score
import numpy as np
from PIL import Image

base_dir = Path(__file__).parent 

images_dir = base_dir / "assets/images"

# IoU implementado (Rychard)
def iou_coef(y_true, y_pred):
    axes = (0, 1)
    intersection = np.sum(np.logical_and(y_pred, y_true), axis=axes)
    union = np.sum(np.logical_or(y_pred, y_true), axis=axes)
    smooth = .001
    iou = (intersection + smooth) / (union + smooth)
    iou = np.mean(iou)
    if iou <= 9.592633205077928e-07:
        iou = 0.0
    return iou

# Carregando as imagens filtradas
img = Image.open(images_dir / "imagem3.jpg")
img_rgb = img
img = np.array(img)

img_labels = Image.open(images_dir / "imagem3-labels-teste.jpg")
img_labels = np.array(img_labels)
#img_labels = img_labels[:, :, 0] + img_labels[:, :, 1] + img_labels[:, :, 2]
flatten_labels = img_labels.flatten()

# Segmentação img normal
segments = slic(img, compactness=30, n_segments=100, start_label=1, sigma=0.78999, min_size_factor=0.3)
#min size pode ser um parâmetro a se considerar, padrão = 0.5
out1 = color.label2rgb(segments, img, kind='avg', bg_label=0)

# Primeiro NCut
threshold1 = 0.001
threshold2 = 0.1
if threshold1 > 0:
    g1 = graph.rag_mean_color(img, segments, mode='similarity')
    segments = graph.cut_normalized(segments, g1, thresh=threshold1)
    
    # Verifique se há nós suficientes no grafo antes de prosseguir com o segundo corte
    if len(np.unique(segments)) > 1:
        # Segundo NCut
        if threshold2 > 0:
            g2 = graph.rag_mean_color(img, segments, mode='similarity')
            segments = graph.cut_normalized(segments, g2, thresh=threshold2)
            out2 = color.label2rgb(segments, img, kind='avg', bg_label=0)


# Segmentação img labels
segments_labels = slic(img_labels, compactness=30, n_segments=100, start_label=1, sigma=0.78999, min_size_factor=0.3)

# Primeiro NCut
threshold1 = 0.001
threshold2 = 0.01
if threshold1 > 0:
    g1 = graph.rag_mean_color(img_labels, segments_labels, mode='similarity')
    segments_labels = graph.cut_normalized(segments_labels, g1, thresh=threshold1)
    
    # Verifique se há nós suficientes no grafo antes de prosseguir com o segundo corte
    if len(np.unique(segments_labels)) > 1:
        # Segundo NCut
        if threshold2 > 0:
            g2 = graph.rag_mean_color(img_labels, segments_labels, mode='similarity')
            segments_labels = graph.cut_normalized(segments_labels, g2, thresh=threshold2)

# Mapeamento de rótulos para as segmentações
rotulos_segmentos = {
    1: "ERRO",
    4: "Feijao",
    5: "Arroz e prato",
    6: "ERRO",
    9: "ERRO",
    12: "Cenoura",
    27: "Chuchu",
    53: "Molho",
    55: "ERRO",
    56: "ERRO"
    # Adicione mais rótulos conforme necessário
}

# Plotando e calculando métricas para cada segmento separadamente
unique_segments = np.unique(segments)
print(f"Segmentos unicos encontrados na imagem original: {unique_segments}")

unique_segments_labels = np.unique(segments_labels)
print(f"Segmentos unicos encontrados na imagem rotulada: {unique_segments_labels}")

# Inicializando listas para armazenar as métricas de cada segmento
iou_jaccard_por_segmento = []
iou_implementado_por_segmento = []
precision_por_segmento = []

# Garantir que estamos comparando o menor número de segmentos (caso haja diferença)
num_segments = min(len(unique_segments), len(unique_segments_labels))

# Loop para comparar cada segmento da imagem original com o correspondente da imagem rotulada
for idx in range(num_segments):
    seg_id = unique_segments[idx]
    seg_label_id = unique_segments_labels[idx]
    
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

        
    # Plotar o segmento da imagem original
    plt.imshow(img * np.dstack([mask] * 3))  # Multiplicar pela máscara para mostrar apenas o segmento
    plt.title(f"Segmento Original: {seg_id}")
    plt.axis('off')
    plt.show()

    # Plotar o segmento correspondente da imagem rotulada
    plt.imshow(img_labels * np.dstack([mask_label] * 3))  # Mostrar apenas o segmento rotulado
    plt.title(f"Segmento Rotulado: {seg_label_id}")
    plt.axis('off')
    plt.show()

# Exibindo os resultados gerais (médias)
jaccard_medio = np.mean([jac for _, jac in iou_jaccard_por_segmento])
iou_medio = np.mean([iou for _, iou in iou_implementado_por_segmento])
precision_media = np.mean([prec for _, prec in precision_por_segmento])

print(f'Media Jaccard por segmento: {jaccard_medio}')
print(f'Media IoU por segmento: {iou_medio}')
print(f'Media Precision por segmento: {precision_media}')