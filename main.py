# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 22:00:06 2021

@author: rycba
"""
import operator
from scipy import ndimage as ndi
from pathlib import Path
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from skimage.metrics import (adapted_rand_error, variation_of_information)
from sklearn.metrics import jaccard_score
from skimage import data, io, segmentation, color, morphology, filters
from skimage import graph
from skimage.measure import label, regionprops
from pydicom import dcmread
from skimage.morphology import disk, ball, closing, square
from skimage.filters import gabor, sobel, threshold_otsu, threshold_multiotsu
from os import listdir
import matplotlib.cm as cm
import cv2
from skimage.segmentation import mark_boundaries
from skimage.filters.rank import entropy
from PIL import Image
import numpy as np
from skimage.filters import median, gaussian
from matplotlib import pyplot as plt
from scipy import ndimage, misc
from scipy import ndimage as ndi
from keras import backend as K
import tensorflow as tf

base_dir = Path(__file__).parent 

images_dir = base_dir / "assets/images"

def iou_coef(y_true, y_pred):
    axes = (0, 1)
    intersection = np.sum(np.logical_and(y_pred, y_true), axis=axes)
    union = np.sum(np.logical_or(y_pred, y_true), axis=axes)
    smooth = .001
    iou = (intersection + smooth) / (union + smooth)
    iou = np.mean(iou)
    return iou


def dice_coef(y_true, y_pred):
    axes = (0, 1)
    intersection = np.sum(np.logical_and(y_pred, y_true), axis=axes)
    mask_sum = np.sum(np.abs(y_true), axis=axes) + np.sum(np.abs(y_pred), axis=axes)
    smooth = .001
    dice = 2 * (intersection + smooth) / (mask_sum + smooth)
    dice = np.mean(dice)
    return dice


def kmeans_test(n_clusters, X):
    clusterer = KMeans(n_clusters=n_clusters, init='k-means++', n_init=10, random_state=0, max_iter=500)
    cluster_labels = clusterer.fit(X)
    labels = cluster_labels.labels_
    return labels


def change_contrast(img):
    factor = (259 * (5 + 255)) / (255 * (259 - 5))

    def contrast(c):
        return 128 + factor * (c - 128)

    return img.point(contrast)


# img = Image.open(r"C:\Users\rycba\Downloads\pf8.jpg")
img = Image.open(images_dir / "imagem2.jpg")
img = change_contrast(img)

# true_label = Image.open(r"C:\Users\rycba\OneDrive\Imagens\tcc\2Rick.jpg")
true_label = Image.open(images_dir / "imagem2-labels.jpg")
true_label = np.array(true_label)
#print(true_label.shape) = 3
true_label_original = np.array(true_label)

img_real = np.array(img)
original = np.array(img)

img = color.rgb2gray(img_real)
img_nec = color.rgb2gray(img_real)
img = ndimage.median_filter(img, size=3)
img = ndimage.gaussian_filter(img, sigma=1)

img_filt, real = gabor(img, frequency=0.9)
real = real < 0.001
mask2 = mask = morphology.remove_small_holes(
    morphology.remove_small_objects(real > 0.9, 7000), 7000)
edges1 = sobel(img)
edges = edges1 < 0.0046

mask = morphology.remove_small_holes(
    morphology.remove_small_objects(edges < 0.9, 700), 700)

mask2 = ~(mask ^ mask2)
mask = mask2
maskk = (1) * mask

img_real[:, :, 0] = img_real[:, :, 0] * maskk
img_real[:, :, 1] = img_real[:, :, 1] * maskk
img_real[:, :, 2] = img_real[:, :, 2] * maskk

true_label_original[:, :, 0] = true_label_original[:, :, 0] * maskk
true_label_original[:, :, 1] = true_label_original[:, :, 1] * maskk
true_label_original[:, :, 2] = true_label_original[:, :, 2] * maskk
img_nec = img_nec * maskk
#print(true_label.shape)=3
true_label = true_label_original[:, :, 0] + true_label_original[:, :, 1] + true_label_original[:, :, 2]
#print(true_label.shape)=1


def segment():
    clusters = 14
    fig, ax = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(25, 25))
    ax1, ax2, ax3, ax4 = ax.ravel()  # ax1, ax2,

    labels1 = segmentation.slic(img_real, compactness=35, n_segments=170, mask=mask, start_label=1,
                                sigma=0.7)  # , enforce_connectivity=True)
    outt = color.label2rgb(labels1, original, kind='avg', bg_label=0)
    outt = mark_boundaries(outt, labels1, (0, 0, 0))

    g = graph.rag_mean_color(img_real, labels1, mode='similarity')
    labels2 = graph.cut_normalized(labels1, g, thresh=0.00002)  # 0.004 #0.0001 OK limiar
    out = color.label2rgb(labels2, original, kind='avg', bg_label=0)
    out = mark_boundaries(out, labels2, (0, 0, 0))

    ga = graph.rag_mean_color(img_real, labels2, mode='similarity')
    labels3 = graph.cut_normalized(labels2, ga, thresh=0.0000005)  # 0.00016/0.0005 -- 0.00005 -> 0.006
    outFinal = color.label2rgb(labels3, original, kind='avg', bg_label=0)
    outFinal = mark_boundaries(outFinal, labels3, (0, 0, 0))
    labelFinal = labels3

    props = regionprops(labels3, intensity_image=img_nec)
    array_centroid = []
    array_main = []

    for i in props:
        x1, y1 = i.centroid
        x2 = int(x1)
        y2 = int(y1)
        aux_out = int(original[x2, y2, 0] * 255)
        array_main.append([aux_out])
        array_centroid.append((x2, y2))

    error, precision, recall = adapted_rand_error(true_label, labels3)
    splits, merges = variation_of_information(true_label, labels3)
    iou = iou_coef(true_label, labels3)
    dice = dice_coef(true_label, labels3)
    flatten_img = true_label.flatten()
    flatten_pred = labels3.flatten()
    iou_real = jaccard_score(flatten_img, flatten_pred, average='weighted')



    print("---------------------- labels3 -------------------------")

    print(true_label.shape)
    print(labels2.shape)

    print(f"Adapted Rand error: {error}")
    print(f"Adapted Rand precision: {precision}")
    print(f"Adapted Rand recall: {recall}")
    print(f"False Splits: {splits}")
    print(f"False Merges: {merges}")
    print(f"iou : {iou}")
    print(f"dice : {dice}")
    print(f"IOU REAL : {iou_real}")

    print("---------------------- labels2 -------------------------")
    error, precision, recall = adapted_rand_error(true_label, labels2)
    splits, merges = variation_of_information(true_label, labels2)
    iou = iou_coef(true_label, labels2)
    dice = dice_coef(true_label, labels2)
    flatten_pred = labels2.flatten()
    iou_real = jaccard_score(flatten_img, flatten_pred, average='weighted')

    print(f"Adapted Rand error: {error}")
    print(f"Adapted Rand precision: {precision}")
    print(f"Adapted Rand recall: {recall}")
    print(f"False Splits: {splits}")
    print(f"False Merges: {merges}")
    print(f"iou : {iou}")
    print(f"dice : {dice}")
    print(f"IOU REAL : {iou_real}")

    if (len(props) > 13):
        kmeans = kmeans_test(clusters, array_main)
        label_int = int(255 / clusters)
        for i in range(len(array_centroid)):
            aux = labels3[array_centroid[i]]
            labels3[labels3 == aux] = kmeans[i] * label_int
        print("---------------------- kmeans -------------------------")
        error, precision, recall = adapted_rand_error(true_label, labels3)
        splits, merges = variation_of_information(true_label, labels3)
        iou = iou_coef(true_label, labels3)
        dice = dice_coef(true_label, labels3)
        flatten_pred = labels3.flatten()
        iou_real = jaccard_score(flatten_img, flatten_pred, average='weighted')

        outK = color.label2rgb(labels3, original, kind='avg', bg_label=0)
        outK = mark_boundaries(outK, labels3, (0, 0, 0))

        print(f"Adapted Rand error: {error}")
        print(f"Adapted Rand precision: {precision}")
        print(f"Adapted Rand recall: {recall}")
        print(f"False Splits: {splits}")
        print(f"False Merges: {merges}")
        print(f"iou : {iou}")
        print(f"dice : {dice}")
        print(f"IOU REAL : {iou_real}")

    ax1.set_title("Imagem Original")
    ax1.imshow(original)
    # ax2.set_text("Imagem segmentada com máscara")
    # ax2.imshow(outt)
    # ax1.imshow(original)
    # ax2.imshow(segmentation.mark_boundaries(original, labels1, (255, 0, 0)))
    # ax.set_title("Imagem segmentada após o primeiro Ncut", fontsize=30)
    # ax.set_title("Imagem segmentada após o segundo Ncut" ,fontsize = 30)
    ax2.set_title("Imagem Segmentada com Slic")
    ax2.imshow(outt)
    ax3.set_title("Imagem após o primeiro Ncut")
    ax3.imshow(out)
    ax4.set_title("Imagem após o segundo Ncut")
    ax4.imshow(outFinal)
    # ax3.imshow(segmentation.mark_boundaries(original, labels2, (255, 0, 0)))
    # ax4.imshow(segmentation.mark_boundaries(original, labelFinal, (255, 0, 0)))
    plt.show()


segment()

# index, value = max(enumerate(iou_array), key=operator.itemgetter(1))




