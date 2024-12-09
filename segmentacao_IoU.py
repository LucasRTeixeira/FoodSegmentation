import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import cv2
from skimage import segmentation
from sklearn.cluster import KMeans
from skimage.metrics import adapted_rand_error as arand_index

base_dir = Path(__file__).parent 

images_dir = base_dir / "assets/images"

# Cálculo do IoU
def calculate_iou(ground_truth, prediction):
    intersection = np.logical_and(ground_truth, prediction)
    union = np.logical_or(ground_truth, prediction)
    smooth = 0.001
    iou = (np.sum(intersection) + smooth) / (np.sum(union) + smooth)
    return iou

# Carregue as imagens
def color_to_label(image, color_map):
    label_image = np.zeros(image.shape[:2], dtype=np.int32)
    for label, color in color_map.items():
        mask = np.all(image == color, axis=-1)
        label_image[mask] = label
    return label_image

def plot_images(segmented_image, annotated_image, kmeans_labels):
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    
    # Imagem original segmentada
    axs[0].imshow(segmented_image.squeeze(), cmap='gray')
    axs[0].set_title('Segmented Image')
    
    # Imagem ground truth anotada
    axs[1].imshow(annotated_image)
    axs[1].set_title('Ground Truth')

    # Segmentação refinada com KMeans
    axs[2].imshow(kmeans_labels, cmap='tab20')
    axs[2].set_title('KMeans Segmentation')
    
    # Remove eixos
    for ax in axs:
        ax.axis('off')
    
    plt.show()

# Exemplo de um mapa de cores (valores RGB) para rótulos
color_map = {
    1: [255, 255, 255],   # Branco -> Label 1 -> Prato
    2: [0, 0, 0],   # Preto -> Label 2 -> Fundo
    3: [24, 105, 0],   # Verde escuro -> Label 3 -> Arroz
    4: [108, 10, 9],  # Marrom -> Label 4 -> Feijão
    5: [41, 255, 1],  # Verde claro -> Label 5 -> Alface
    6: [255, 71, 0],  # Laranja -> Label 6 -> Cenoura
    7: [3, 0, 255]  # Azul -> Label 7 -> Carne
}

annotated_image = cv2.imread(images_dir / 'imagem1-labels.jpg')
##############################################################################################################################
#Se as cores não forem previamente conhecidas, pode-se aplicar o mesmo modelo para segmentar as imagens originais, para 
#segmentar a imagem anotada com as diferentes colores indicadas pelo anotador.
###############################################################################################################################
label_image = color_to_label(annotated_image, color_map)
segmented_image = cv2.imread(images_dir / 'imagem1.jpg')  # Imagem segmentada
segmented_image = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB)  #Passar a imagem de BGR para RGB
segmented_image = np.expand_dims(segmented_image, axis=-1)

# Aplique SLIC na imagem segmentada
slic_segments = segmentation.slic(segmented_image, compactness=25, n_segments=275, start_label=1, sigma=0.789999999)

# Use KMeans para refinamento da segmentação
slic_flat = slic_segments.reshape(-1, 1)
kmeans = KMeans(n_clusters=4, random_state=0).fit(slic_flat)
kmeans_labels = kmeans.labels_.reshape(slic_segments.shape)

# Comparação com a imagem anotada
iou_scores = []
for i in range(1, 7):  # Assumindo 6 segmentos
    gt_mask = annotated_image == i  #shape = 3 dimensoes
    pred_mask = kmeans_labels == i  #shape = 2 dimensoes
    iou = calculate_iou(gt_mask, pred_mask)
    iou_scores.append(iou)

mean_iou = np.mean(iou_scores)
print(f'Mean IoU: {mean_iou}')

# (Opcional) Métrica Adapted Rand Index para comparação geral
error, precision, recall = arand_index(annotated_image, kmeans_labels)
print(f'Adapted Rand Error: {error}, Precision: {precision}, Recall: {recall}')

plot_images(segmented_image, annotated_image, kmeans_labels)