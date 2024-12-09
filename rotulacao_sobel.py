from matplotlib import pyplot as plt
from skimage import color
from pathlib import Path
from skimage.filters import sobel
from skimage.segmentation import slic
import skimage.graph as graph
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
    return iou

# Carregando as imagens filtradas
img_rgb = Image.open(images_dir / "imagem3.jpg")
img = img_rgb.convert('L')  # Convertendo para escala de cinza
img = np.array(img)
img_sobel = sobel(img)
out0 = color.label2rgb(img_sobel, img, kind='avg', bg_label=0)

img_labels = Image.open(images_dir / "imagem3-labels-teste.jpg")
img_labels = np.array(img_labels)
img_labels = img_labels[:, :, 0] + img_labels[:, :, 1] + img_labels[:, :, 2]

# Segmentação
segments = slic(img_sobel, compactness=25, n_segments=275, start_label=1, sigma=0.789999999, channel_axis=None)
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
print(f"Segmentos unicos encontrados: {unique_segments}")

# Inicializando listas para armazenar as métricas de cada segmento
iou_jaccard_por_segmento = []
iou_implementado_por_segmento = []
precision_por_segmento = []

for seg_id in unique_segments:
    # Criar máscara para o segmento
    mask = segments == seg_id
    
    # Rótulo do segmento
    rotulo = rotulos_segmentos.get(seg_id, f"Segmento {seg_id}")  # Se o rótulo não estiver mapeado, use o ID
    
    # Converter a imagem sobel de cinza para RGB (expandindo para 3 canais)
    img_sobel_rgb = np.dstack([img_sobel] * 3)  # Expandir para 3 canais para usar na multiplicação
    
    # Plotar o segmento com a máscara
    plt.imshow(img_sobel_rgb * np.dstack([mask] * 3))  # Multiplicar pela máscara para mostrar apenas o segmento
    plt.title(f"Segmento: {rotulo}")
    plt.axis('off')
    plt.show()

    
    # Convertendo o segmento atual para binário
    y_pred_segment_bin = mask
    
    # Convertendo o label correspondente para binário (somente a parte do label que corresponde ao segmento)
    y_true_segment_bin = (img_labels == seg_id)
    
    # Calculando as métricas IoU e Rand Index para o segmento
    if np.any(y_true_segment_bin) or np.any(y_pred_segment_bin):  # Verifica se o segmento existe no label
        iou_jaccard = jaccard_score(y_true_segment_bin, y_pred_segment_bin, average='weighted', zero_division = 0)
        error, precision, recall = adapted_rand_error(y_true_segment_bin, y_pred_segment_bin)
        iou_segment = iou_coef(y_true_segment_bin, y_pred_segment_bin)
    else:
        jaccard, precision, iou_segment = 0, 0, 0  # Se o segmento não existir no rótulo verdadeiro
    
    # Armazenar os valores para cada segmento
    iou_implementado_por_segmento.append((rotulo, iou_segment))
    precision_por_segmento.append((rotulo, precision))
    iou_jaccard_por_segmento.append((rotulo, iou_jaccard))
    
    # Exibir as métricas para cada segmento
    print(f'Segmento {rotulo}: Jaccard = {iou_jaccard}, IoU = {iou_segment}, Precision = {precision}')

# Exibindo os resultados gerais (médias)
jaccard_medio = np.mean([jac for _, jac in iou_jaccard_por_segmento])
iou_medio = np.mean([iou for _, iou in iou_implementado_por_segmento])
precision_media = np.mean([prec for _, prec in precision_por_segmento])

print(f'Media Jaccard por segmento: {jaccard_medio}')
print(f'Media IoU por segmento: {iou_medio}')
print(f'Media Precision por segmento: {precision_media}')

fig, ax = plt.subplots(1, 4)

ax[0].imshow(img_rgb)
ax[0].set_title('Imagem Original')

ax[1].imshow(out0)
ax[1].set_title('Imagem com filtro Sobel')

ax[2].imshow(out1)
ax[2].set_title('Imagem modificada SLIC')

ax[3].imshow(out2)
ax[3].set_title('Imagem modificada NCUT')

plt.tight_layout()
plt.show()