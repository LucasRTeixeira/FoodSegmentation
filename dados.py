# bibliotecas necessarias
from matplotlib import pyplot as plt
from skimage import io, color, morphology
from skimage.segmentation import slic, mark_boundaries
from skimage import graph
from skimage.measure import label, regionprops
from skimage.metrics import adapted_rand_error, variation_of_information
from sklearn.metrics import jaccard_score
from skimage.filters import gabor, sobel
import numpy as np
from pathlib import Path
from PIL import Image
import pandas as pd

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


# Carregando as imagens
img = Image.open(images_dir / "imagem1.jpg")
img = np.array(img)

img_labels = Image.open(images_dir / "imagem1-labels.jpg")
img_labels = np.array(img_labels)
img_labels = img_labels[:, :, 0] + img_labels[:, :, 1] + img_labels[:, :, 2]
flatten_labels = img_labels.flatten()

#importando o csv
dados = pd.read_csv(base_dir / 'data' / 'raw' / 'dados_segments.csv', sep=';')
#print(dados.head())r


for linha in dados.itertuples():
    print(linha.Index)

    #variaveis para guardar no df
    precision = 0
    iou_jaccard = 0
    iou_implementado = 0

    #segmentacao
    segments = slic(img, compactness=linha.Compactness, n_segments=linha.N_Segments, start_label=1, sigma=float(linha.Sigma))

    # Primeiro NCut
    threshold1 = float(linha.Threshold_NCut1)
    threshold2 = float(linha.Threshold_NCut2)
    if threshold1 > 0:
        g1 = graph.rag_mean_color(img, segments, mode='similarity')
        segments = graph.cut_normalized(segments, g1, thresh=threshold1)

        # Segundo NCut
        if threshold2 > 0:
            g2 = graph.rag_mean_color(img, segments, mode='similarity')
            segments = graph.cut_normalized(segments, g2, thresh=threshold2)

    #Calculando os resultados
    error, precision, recall = adapted_rand_error(img_labels, segments)
    flatten_segments = segments.flatten()
    iou_jaccard = jaccard_score(flatten_labels, flatten_segments, average='weighted')
    iou_implementado = iou_coef(img_labels, segments)

    #guardando os resultados
    dados.iat[linha.Index, 5] = iou_jaccard
    dados.iat[linha.Index, 6] = iou_implementado
    dados.iat[linha.Index, 7] = precision

#exportando um novo csv
dados.to_csv('dados_segments_image1.csv', sep=';', index=False)

