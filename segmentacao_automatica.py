'''
Este código faz a segmentação (apenas com slic) da imagem original e compara com os segmentos extraídos da imagem rotulada pela função
extract_labeled_segments(). Ela verifica os segmentos extraídos que mais combinam com os segmentos obtidos pelo SLIC automaticamente
'''

from matplotlib import pyplot as plt
from skimage import color
from skimage.util import img_as_float
from skimage.segmentation import slic, mark_boundaries
import skimage.graph as graph
from skimage.metrics import adapted_rand_error
from sklearn.metrics import jaccard_score
import numpy as np
from PIL import Image
import pprint

def extract_labeled_segments(labeled_image):
    # Encontra valores únicos na imagem rotulada (cada cor representa um segmento)
    unique_values = np.unique(labeled_image.reshape(-1, labeled_image.shape[2]), axis=0)
    
    segments_dict = {}
    for i, color in enumerate(unique_values):
        # Cria máscara binária para cada cor única
        mask = np.all(labeled_image == color, axis=2)
        segments_dict[i] = {
            'mask': mask,
            'color': color
        }
    return segments_dict

def find_best_match(segment_mask, labeled_segments):
    best_iou = -1
    best_match = None
    best_label = None
    
    for label, labeled_segment in labeled_segments.items():
        iou = jaccard_score(segment_mask.flatten(), 
                        labeled_segment['mask'].flatten(), 
                        average='binary')
        if iou > best_iou:
            best_iou = iou
            best_match = labeled_segment['mask']
            best_label = label
    
    return best_label, best_match, best_iou

# Carregando as imagens
img = Image.open(r"C:\Users\Lucas\OneDrive\Documentos\UERJ- MATÉRIAS\INICIAÇÃO CIENTÍFICA\pythonProject\images\imagem3.jpg")
img = np.array(img)
img = img_as_float(img)

img_labels = Image.open(r"C:\Users\Lucas\OneDrive\Documentos\UERJ- MATÉRIAS\INICIAÇÃO CIENTÍFICA\pythonProject\images\imagem3-labels.jpg")
img_labels = np.array(img_labels)

# Extrai segmentos da imagem rotulada
labeled_segments = extract_labeled_segments(img_labels)

# Segmentação
segments = slic(img, n_segments=11, compactness=11, start_label=0)
out1 = color.label2rgb(segments, img, kind='avg', bg_label=0)
out1 = mark_boundaries(out1, segments, (0, 0, 0))

fig, ax = plt.subplots(1, 2)

ax[0].imshow(img)
ax[0].set_title('Imagem Original')

ax[1].imshow(out1)
ax[1].set_title('Imagem SLIC')

plt.tight_layout()
plt.show()

label_names = {
    0: "Arroz",
    1: "Feijao",
    2: "Alface",
    3: "Carne",
    4: "Cenoura"
}

# Análise de cada segmento
unique_segments = np.unique(segments)
results = []

for seg_id in unique_segments:
    # Criar máscara para o segmento atual
    segment_mask = segments == seg_id
    
    # Encontrar melhor correspondência nos segmentos rotulados
    best_label, best_match, iou = find_best_match(segment_mask, labeled_segments)
    
    results.append({
        'segment_id': seg_id,                                   #número do segmento
        'matched_label': best_label,                            #numero do rótulo
        'label_name': label_names.get(best_label, "Unknown"),   #nome do rótulo, dado por label_names
        'iou_score': iou                                        #valor do IoU em relação ao seu segmento extraído
    })
    
    # Visualização
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    
    ax1.imshow(img * np.dstack([segment_mask] * 3))
    ax1.set_title(f'Segmento Original {seg_id}')
    ax1.axis('off')
    
    ax2.imshow(img_labels * np.dstack([best_match] * 3))
    ax2.set_title(f'Segmento Rotulado {label_names.get(best_label, "Unknown")}')
    ax2.axis('off')
    
    plt.show()
    print(f'Segmento {seg_id} corresponde a {label_names.get(best_label, "Unknown")} com IoU = {iou:.3f}')

# Calcular métricas médias
average_iou = np.mean([r['iou_score'] for r in results])
print(f'\nIoU médio: {average_iou:.3f}')
pprint.pprint(results)