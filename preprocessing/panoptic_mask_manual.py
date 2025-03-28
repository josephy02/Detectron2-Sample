#!/usr/bin/env python
# coding: utf-8


import cv2
import numpy as np
from pycocotools import mask as mask_utils
import torch
import json
import os
import pandas as pd
import zipfile
import matplotlib.pyplot as plt

# # Panoptic Masks


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
        rle = mask_utils.frPyObjects(ann['segmentation'], height, width)
        mask = mask_utils.decode(rle)
    else:
        # check if segemention is Polygon (Note that pretty much all images in this example use Polygons)
        mask = np.zeros((height, width), dtype=np.uint8)
        for seg in ann['segmentation']:
            poly = np.array(seg).reshape((len(seg)//2, 2))
            cv2.fillPoly(mask, [poly.astype(np.int32)], 1)
    return mask


def main():
    '''
    Main function that executes function above and creates visualizations for panoptic masks
    '''
    dirname = os.path.dirname(os.path.abspath('__file__'))
    data_dir = os.path.join(dirname, "manual_annotations")
    output_dir = os.path.join(dirname, "panoptic_segmentation_output")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for filename in os.listdir(data_dir):
        if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".PNG"):
            image_path = os.path.join(data_dir, filename)
            json_path = os.path.join(data_dir, os.path.splitext(filename)[0] + ".json")

            if not os.path.exists(json_path):
                print(f"No JSON found for {filename}, skipping.")
                continue
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            height, width = image.shape[:2]

            with open(json_path, 'r') as f:
                coco_data = json.load(f)

            mask = np.zeros((height, width, 3), dtype=np.uint8)

            for ann in coco_data['annotations']:
                category_id = ann['category_id']
                instance_mask = coco_to_mask(ann, height, width)

                color = np.random.randint(0, 255, size=3, dtype=np.uint8)
                mask[instance_mask > 0] = color

            output_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_mask.png")
            cv2.imwrite(output_path, cv2.cvtColor(mask, cv2.COLOR_RGB2BGR))

            print(f"Processed {filename}")

    print("Mask visualization completed.")


if __name__ == "__main__":
    main()

