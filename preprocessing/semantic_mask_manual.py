#!/usr/bin/env python
# coding: utf-8

import os
import cv2
import numpy as np
from skimage import measure, color, feature, morphology
from PIL import Image
import json
from pycocotools import mask as mask_utils
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import datetime


def coco_to_mask(ann, height, width):
    '''Converts data from our COCO annotations into a binary mask

    This function takes in COCO segmentation annotation and converts
    it into a binary mask given an input height and width (ideally the
    same height and width as the raw image). The function handles segmentation
    data in both RLE and polygon-based formats.

    Parameters
    ----------
    ann : list(str)
        A list of file paths to COCO format JSON files that need to be merged.

    height : int
        Height of the raw input image

    width : int
        Width of the raw input image

    Returns
    -------
    mask: numpy.ndarray
        A binary mask of shape (height, width). Pixels within the segmented region
        are set to 1, and all other pixels are set to 0.
    '''
    if isinstance(ann['segmentation'], dict):
        # Here we check if segmentation is RLE
        rle = mask_utils.frPyObjects(ann['segmentation'], height, width)
        mask = mask_utils.decode(rle)
    else:
        # Here we check if segemention is Polygon (Note that all images in this example, e.g. manual annotations, use Polygons)
        mask = np.zeros((height, width), dtype=np.uint8)
        for seg in ann['segmentation']:
            poly = np.array(seg).reshape((len(seg)//2, 2))
            cv2.fillPoly(mask, [poly.astype(np.int32)], 1)
    return mask


def create_semantic_mask(json_file, image_height, image_width):
    with open(json_file, 'r') as f:
        data = json.load(f)
    semantic_mask = np.zeros((image_height, image_width), dtype=np.uint8)
    for ann in data['annotations']:
        instance_mask = coco_to_mask(ann, image_height, image_width)
        semantic_mask = np.logical_or(semantic_mask, instance_mask).astype(np.uint8)
    return semantic_mask


def process_all_images_and_jsons(image_dir, json_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    all_images = [file for file in os.listdir(image_dir) if file.endswith(('.png', '.jpg', '.jpeg'))]
    for filename in all_images:
        image_path = os.path.join(image_dir, filename)
        json_file = os.path.join(json_dir, os.path.splitext(filename)[0] + '.json')
        output_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_mask.png")

        if not json_file:
            print(f"Skip this {filename} and figure out what's wrong")
            continue

        image = cv2.imread(image_path)
        image_height, image_width = image.shape[:2]
        semantic_mask = create_semantic_mask(json_file, image_height, image_width)
        output_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_mask.png")
        cv2.imwrite(output_path, semantic_mask * 255)

        if semantic_mask.shape[:2] != image.shape[:2]:
            print(f"Error: Semantic mask size ({semantic_mask.shape[:2]}) does not match image size ({image.shape[:2]}) for {filename}")
            continue

        plt.figure(figsize=(10, 5))
        plt.subplot(121)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title('Original Image')
        plt.axis('off')
        plt.subplot(122)
        plt.imshow(semantic_mask, cmap='gray')
        plt.title('Semantic Mask')
        plt.axis('off')
        plt.tight_layout()
        plt.show()
        plt.close()

        print(f"Processed {filename}")


def main():
    '''
    Main function that executes the functions above and creates visualizations for semantic masks
    '''
    dirname = os.path.dirname(os.path.abspath('__file__'))
    image_dir = os.path.join(dirname, "manual_annotations")
    json_dir = os.path.join(dirname, "manual_annotations")
    output_dir = os.path.join(dirname, "semantic_segmentation_output")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    process_all_images_and_jsons(image_dir, json_dir, output_dir)


if __name__ == "__main__":
    main()

