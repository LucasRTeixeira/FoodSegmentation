'''
O código a seguir segmenta a imagem e plota os segmentos comparando com os segmentos do labels verificando métricas. Ele extrai os segmentos da imagem rotulada e compara com cada segmento da imagem original através de uma comparação no IoU

O código vai plotar os segmentos extraídos do algoritmo e da imagem rotulada e comparar um com o outro. Os segmentos da imagem rotulada estão cúbicos, mas podemos ter uma boa ideia de como funciona.

Segmentos da imagem original estão armazenados em "sorted_segments" e segmentos da imagem rotulada estão em "results"

- Caso a saída seja "Segmentação não vai servir", pare e rode o programa novamente até obter uma segmentação funcional com alguns segmentos.
- O algoritmo de segmentação nem sempre retorna a segmentação ideal. Até agora, caso seja uma segmentação ruim, eu rodo o algoritmo novamente até vir uma segmentação boa.
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

# A parte comentada abaixo é a função anterior, que roda mais rápido mas retorna os segmentos menos refinados

'''def extract_labeled_segments(labeled_image):
    # Reshape para 2D array de cores
    pixels = labeled_image.reshape(-1, labeled_image.shape[2])
    
    # Encontra cores únicas de forma mais eficiente
    unique_colors = np.unique(pixels, axis=0)
    
    # Pré-aloca o dicionário com tamanho conhecido
    segments_dict = {}
    
    # Cria uma matriz 3D de comparação
    labeled_image_expanded = labeled_image[..., np.newaxis, :]
    unique_colors_expanded = unique_colors[np.newaxis, np.newaxis, :, :]
    
    # Compara todas as cores de uma vez
    masks = np.all(labeled_image_expanded == unique_colors_expanded, axis=3)
    
    # Cria o dicionário de segmentos
    for i, color in enumerate(unique_colors):
        segments_dict[i] = {
            'mask': masks[:, :, i],
            'color': color
        }
    
    return segments_dict'''

def extract_labeled_segments(labeled_image):
    # Converte a imagem para float para maior precisão
    labeled_image = labeled_image.astype(np.float32) / 255.0
    
    # Threshold para considerar pixels similares (ajuste conforme necessário)
    color_threshold = 0.1
    
    # Encontra cores únicas com mais precisão
    pixels = labeled_image.reshape(-1, labeled_image.shape[2])
    unique_colors = np.unique(pixels, axis=0)
    
    segments_dict = {}
    
    for i, color in enumerate(unique_colors):
        # Calcula a diferença euclidiana para cada pixel
        diff = np.sqrt(np.sum((labeled_image - color) ** 2, axis=2))
        
        # Cria máscara com threshold adaptativo
        mask = diff <= color_threshold
        
        # Fecha pequenos buracos
        mask = ndimage.binary_fill_holes(mask)
        
        # Suaviza as bordas
        mask = ndimage.binary_opening(mask, structure=np.ones((3,3)))
        mask = ndimage.binary_closing(mask, structure=np.ones((3,3)))
        
        # Remove pequenos objetos e ruídos
        mask = ndimage.binary_opening(mask)
        
        # Adiciona ao dicionário apenas se o segmento não for muito pequeno
        if np.sum(mask) > 100:  # ajuste este valor conforme necessário
            segments_dict[i] = {
                'mask': mask,
                'color': color
            }
    
    return segments_dict

def find_best_match(segment_mask, labeled_segments):
    # Converter segment_mask para 1D array
    segment_mask_flat = segment_mask.flatten()
    
    # Pré-alocar arrays para IoU
    ious = np.zeros(len(labeled_segments))
    masks = []
    
    # Preparar todas as máscaras de uma vez
    for label, segment in labeled_segments.items():
        masks.append(segment['mask'].flatten())
    
    # Converter para array numpy para processamento vetorizado
    masks = np.array(masks)
    
    # Calcular interseção e união vetorizado
    intersection = np.logical_and(segment_mask_flat, masks).sum(axis=1)
    union = np.logical_or(segment_mask_flat, masks).sum(axis=1)
    
    # Calcular IoU vetorizado
    ious = intersection / (union + 1e-6)
    
    # Encontrar o melhor match
    best_idx = np.argmax(ious)
    best_iou = ious[best_idx]
    
    # Obter a chave correta correspondente ao índice encontrado
    best_label = list(labeled_segments.keys())[best_idx]
    return best_label, labeled_segments[best_label]['mask'], best_iou


# Dicionário de rótulos (ainda a melhorar)
label_names = {
    0: "Arroz",
    1: "Feijao",
    2: "Alface",
    3: "Carne",
    4: "Cenoura"
}

# Extrair segmentos rotulados
print("... Extraindo segmentos - extract_labeled_segments() iniciada ...\n")
labeled_segments = extract_labeled_segments(img_labels)
print("Segmentos extraidos - extract_labeled_segments() finalizada\n")

# Analisar segmentos da imagem original e criar results
results = []
unique_segments, counts = np.unique(segments, return_counts=True)
sorted_indices = np.argsort(-counts)
sorted_segments = unique_segments[sorted_indices]
print(f"Segmentos unicos em ordem de frequencia da imagem original: {sorted_segments}")

# Caso haja menos de 4 segmentos, o algoritmo não vai rodar
if len(unique_segments) >= 4:
    for seg_id in sorted_segments:  # Para cada segmento
        segment_mask = segments == seg_id
        best_label, best_match, iou = find_best_match(segment_mask, labeled_segments)
        
        results.append({
            'segment_id': seg_id,                                   #Id do segmento
            'matched_label': best_label,                            # Melhor rótulo
            'label_name': label_names.get(best_label, "Unknown"),   # Nome do segmento (ainda em processo)
            'iou_score': iou,                                       # Valor do IoU
            'size': np.sum(segment_mask)                            # Tamanho do segmento
        })

        print(f'Segmento {seg_id} processado')

    # Ordenar results por tamanho do segmento (similar à ordenação original)
    results = sorted(results, key=lambda x: x['size'], reverse=True)

    # Calcular e mostrar métricas para cada segmento
    print("\nAnalise dos segmentos:")
    iou_jaccard_por_segmento = []
    iou_implementado_por_segmento = []
    precision_por_segmento = []

    for result in results:
        seg_id = result['segment_id']
        mask = segments == seg_id
        mask_label = labeled_segments[result['matched_label']]['mask']
        
        # Calculando métricas
        iou_jaccard = jaccard_score(mask_label.flatten(), mask.flatten(), average='binary', zero_division=0)
        error, precision, recall = adapted_rand_error(mask_label, mask)
        iou_segment = iou_coef(mask_label, mask)
        
        # Armazenar os valores
        iou_implementado_por_segmento.append((seg_id, iou_segment))
        precision_por_segmento.append((seg_id, precision))
        iou_jaccard_por_segmento.append((seg_id, iou_jaccard))
        
        print(f'Segmento {seg_id} ({result["label_name"]}): Jaccard = {iou_jaccard:.3f}, IoU = {iou_segment:.3f}, Precision = {precision:.3f}')

        # Visualização dos pares de segmentos
        fig, axes = plt.subplots(1, 2, figsize=(8, 5))
        
        axes[0].imshow(img * np.dstack([mask] * 3))
        axes[0].set_title(f"Segmento Original: {seg_id}")
        axes[0].axis('off')
        
        axes[1].imshow(img_labels * np.dstack([mask_label] * 3))
        axes[1].set_title(f"Segmento Rotulado: {result['label_name']}")
        axes[1].axis('off')
        
        plt.tight_layout()
        plt.show()

    # Cálculo das médias
    jaccard_medio = np.mean([jac for _, jac in iou_jaccard_por_segmento])
    iou_medio = np.mean([iou for _, iou in iou_implementado_por_segmento])
    precision_media = np.mean([prec for _, prec in precision_por_segmento])

    print(f'\nMetricas medias:')
    print(f'Media Jaccard: {jaccard_medio:.3f}')
    print(f'Media IoU: {iou_medio:.3f}')
    print(f'Media Precision: {precision_media:.3f}')

else:
    print('\nSegmentacao nao vai servir')

fig, ax = plt.subplots(1, 3)

ax[0].imshow(img)
ax[0].set_title('Imagem Original')

ax[1].imshow(out1)
ax[1].set_title('Imagem SLIC')

ax[2].imshow(out2)
ax[2].set_title('Imagem NCUT')

plt.tight_layout()
plt.show()