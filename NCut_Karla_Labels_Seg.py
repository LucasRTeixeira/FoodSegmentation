# bibliotecas necessarias
from matplotlib import pyplot as plt
from skimage import io, color, morphology
from skimage.segmentation import slic, mark_boundaries
import skimage.graph as graph
import scipy.sparse.linalg as spla
from skimage.feature import canny
from skimage.measure import label, regionprops
from skimage.metrics import adapted_rand_error, variation_of_information
from sklearn.metrics import jaccard_score
from skimage.filters import gabor, sobel, roberts, unsharp_mask
import numpy as np
from pathlib import Path
from PIL import Image
import pandas as pd

base_dir = Path(__file__).parent 

images_dir = base_dir / "assets/images"

#IoU implementado (Rychard)
def iou_coef(y_true, y_pred):
    axes = (0, 1)
    intersection = np.sum(np.logical_and(y_pred, y_true), axis=axes) #sempre dá o mesmo resultado (67770)
    union = np.sum(np.logical_or(y_pred, y_true), axis=axes) #sempre dá o mesmo resultado (85554)
    smooth = .001
    iou = (intersection + smooth) / (union + smooth)
    iou = np.mean(iou)
    return iou


# Carregando as imagens filtradas
img = Image.open(images_dir / "imagem3.jpg")
img_rgb = img
img = np.array(img)
#img = img[:, :, 0] + img[:, :, 1] + img[:, :, 2]

#Carregando a filtragem das imagens
#img = sobel(img)    #sobel
#img = canny(img, sigma=2)  #canny
#img = roberts(img)  #roberts
#img = unsharp_mask(img, radius=0.5, amount=2)   #unsharp mask
#img, phase = gabor(img, frequency=0.6)    #gabor
#img = unsharp_mask(img, radius=1.0, amount=1.5)
#img = color.rgb2lab(img)


img_labels = Image.open(images_dir / "imagem3-labels-teste.jpg")
img_labels = np.array(img_labels)
#img_labels = img_labels[:, :, 0] + img_labels[:, :, 1] + img_labels[:, :, 2]

threshold1_labels, threshold2_labels = 0.001 , 0.00001

segments_labels = slic(img_labels, compactness=25, n_segments=100, start_label=1, sigma=0.99)
# Primeiro NCut para img_labels
if threshold1_labels > 0:
    g1_labels = graph.rag_mean_color(img_labels, segments_labels, mode='similarity')
    segments_labels = graph.cut_normalized(segments_labels, g1_labels, thresh=threshold1_labels)

    # Verificar se há nós suficientes no grafo antes de prosseguir com o segundo corte
    if len(np.unique(segments_labels)) > 1:
        # Segundo NCut para img_labels
        if threshold2_labels > 0:
            g2_labels = graph.rag_mean_color(img_labels, segments_labels, mode='similarity')
            segments_labels = graph.cut_normalized(segments_labels, g2_labels, thresh=threshold2_labels)


#importando o csv
dados = pd.read_csv(base_dir / 'data' / 'raw'/ 'dados_segments_Arthur.csv', sep=';')

inter1 = dados.iloc[1:11]
inter2 = dados.iloc[1000:1011]
#inter3 = dados.iloc[2000:2011]
inter4 = dados.iloc[3000:3011]
inter5 = dados.iloc[4000:4011]
inter6 = dados.iloc[5000:5011]

interTotal = pd.concat([inter1 , inter2 , inter4 , inter5 , inter6])

arquivo = open('testeIoU_Novo_Labels.txt','a')
arquivo.seek(0, 2)
arquivo.write("Compactness;N_Segments;Sigma;Threshold_NCut1;Threshold_NCut2;IoU_Jaccard;IoU_Implementado;RI_Precision\n")

for linha in interTotal.itertuples():       #segments provavelmente está deixando o IoU fixo
#for linha in dados.itertuples():
    print(linha.Index)

    #variaveis para guardar no txt
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
        try:
            segments = graph.cut_normalized(segments, g1, thresh=threshold1)
        except spla.ArpackError as e:
            print(f"ARPACK error in line {linha.Index}: {e}")
            continue  # Pular para o próximo loop em caso de erro
        
        #Verifique tem nós suficientes no grafo antes de prosseguir com o segundo corte
        if len(np.unique(segments)) > 1:
            # Segundo NCut
            if threshold2 > 0:
                g2 = graph.rag_mean_color(img, segments, mode='similarity')
                try:
                    segments = graph.cut_normalized(segments, g2, thresh=threshold2)
                except spla.ArpackError as e:
                    print(f"ARPACK error in line {linha.Index}: {e}")
                    continue  # Pular para o próximo loop em caso de erro

    unique_segments = np.unique(segments)
print(f"Segmentos únicos encontrados: {unique_segments}")

# Inicializando listas para armazenar as métricas de cada segmento
#iou_jaccard_por_segmento = []
iou_implementado_por_segmento = []
precision_por_segmento = []

for seg_id in unique_segments:
    # Criar máscara para o segmento
    mask = segments == seg_id
    
    # Plotar o segmento com a máscara
    plt.imshow(img * np.dstack([mask] * 3))  # Multiplicar pela máscara para mostrar apenas o segmento
    plt.title(f"Segmento: {seg_id}")
    plt.axis('off')
    plt.show()
    
    # Convertendo o segmento atual para binário
    y_pred_segment_bin = mask
    
    # Convertendo o label correspondente para binário (somente a parte do label que corresponde ao segmento)
    y_true_segment_bin = (img_labels == seg_id)
    
    # Calculando as métricas IoU e Rand Index para o segmento
    if np.any(y_true_segment_bin) or np.any(y_pred_segment_bin):  # Verifica se o segmento existe no label
        #iou_jaccard = jaccard_score(y_true_segment_bin, y_pred_segment_bin, average='weighted')
        error, precision, recall = adapted_rand_error(y_true_segment_bin, y_pred_segment_bin)
        iou_segment = iou_coef(y_true_segment_bin, y_pred_segment_bin)
    else:
        jaccard, precision, iou_segment = 0, 0, 0  # Se o segmento não existir no rótulo verdadeiro
    
    # Armazenar os valores para cada segmento
    iou_implementado_por_segmento.append((seg_id, iou_segment))
    precision_por_segmento.append((seg_id, precision))
    #iou_jaccard_por_segmento.append((rotulo, iou_jaccard))
    
    # Exibir as métricas para cada segmento
    #print(f'Segmento {seg_id}: Jaccard = {iou_jaccard}, IoU = {iou_segment}, Precision = {precision}')
    print(f'Segmento {seg_id}: IoU = {iou_segment}, Precision = {precision}')

# Exibindo os resultados gerais (médias)
#jaccard_medio = np.mean([jac for _, jac in iou_jaccard_por_segmento])
iou_medio = np.mean([iou for _, iou in iou_implementado_por_segmento])
precision_media = np.mean([prec for _, prec in precision_por_segmento])

#print(f'Média Jaccard por segmento: {jaccard_medio}')
print(f'Média IoU por segmento: {iou_medio}')
print(f'Média Precision por segmento: {precision_media}')

#exportando um novo csv
#dados.to_csv('dados_segments_image1.csv', sep=';', index=False)
arquivo.close()