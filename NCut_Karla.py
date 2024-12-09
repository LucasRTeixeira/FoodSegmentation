# bibliotecas necessarias
from matplotlib import pyplot as plt
import networkx as nx
from skimage import io, color, morphology
from skimage.segmentation import slic, mark_boundaries
import skimage.graph as graph
from pathlib import Path
from skimage.measure import label, regionprops
from skimage.metrics import adapted_rand_error, variation_of_information
from sklearn.metrics import jaccard_score
from skimage.filters import gabor, sobel
import numpy as np
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
img = Image.open(images_dir / "imagem10.jpg")
img = np.array(img)

img_labels = Image.open(images_dir / "imagem10-labels.jpg")
img_labels = np.array(img_labels)
img_labels = img_labels[:, :, 0] + img_labels[:, :, 1] + img_labels[:, :, 2]
flatten_labels = img_labels.flatten()

#importando o csv
dados = pd.read_csv(base_dir / 'data' / 'raw'/ 'dados_segments_Arthur.csv', sep=';')
#print(dados.head())

arquivo = open('saidaDTC11.txt','a')
arquivo.seek(0, 2)
arquivo.write("Compactness;N_Segments;Sigma;Threshold_NCut1;Threshold_NCut2;IoU_Jaccard;IoU_Implementado;RI_Precision")
arquivo.write('\n')

for linha in dados.iloc[2340:].itertuples():
#for linha in dados.itertuples():
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
            

            
        #Verifique tem nós suficientes no grafo antes de prosseguir com o segundo corte
        if len(np.unique(segments)) > 1:
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
    #dados.iat[linha.Index, 5] = iou_jaccard
    #dados.iat[linha.Index, 6] = iou_implementado
    #dados.iat[linha.Index, 7] = precision
    
    arquivo.write(str(dados.iat[linha.Index, 0])+";")
    arquivo.write(str(dados.iat[linha.Index, 1])+";")
    arquivo.write(str(dados.iat[linha.Index, 2])+";")
    arquivo.write(str(dados.iat[linha.Index, 3])+";")
    arquivo.write(str(dados.iat[linha.Index, 4])+";")                  
    arquivo.write(str(iou_jaccard)+";")
    arquivo.write(str(iou_implementado)+";")
    arquivo.write(str(precision))
    arquivo.write('\n')
#exportando um novo csv
#dados.to_csv('dados_segments_image1.csv', sep=';', index=False)
arquivo.close()