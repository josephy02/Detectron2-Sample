#!/usr/bin/env python
# coding: utf-8

# # Augmentations with Imagaug


import numpy as np
import os
import cv2
import json
from imgaug import augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
from imgaug.augmentables.polys import Polygon, PolygonsOnImage
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage


def load_image_and_masks(image_path, panoptic_mask_path, semantic_mask_path):
    '''
    This function loads in an image and its corresponding panoptic and semantic masks.

    Parameters
    ----------
        image_path (str): Path to the raw input image.
        panoptic_mask_path (str): Path to the input panoptic mask.
        semantic_mask_path (str): Path to the input semantic mask.

    Returns
    -------
        (image, panoptic_mask, semantic_mask) --> (tuple(numpy.ndarray, numpy.ndarray, numpy.ndarray)):
        Contains the loaded image, panoptic, and semantic masks respectively.

    '''
    image = cv2.imread(image_path)
    panoptic_mask = cv2.imread(panoptic_mask_path, cv2.IMREAD_GRAYSCALE)
    semantic_mask = cv2.imread(semantic_mask_path, cv2.IMREAD_GRAYSCALE)
    return image, panoptic_mask, semantic_mask


def load_json(json_path):
    '''
    This function simply loads in an input json

    Parameters
    ----------
        json_path (str): Path to the JSON file.

    Returns
    -------
        (dict): The loaded JSON data as a Python dictionary.
    '''
    with open(json_path, 'r') as f:
        return json.load(f)



def create_polygons_and_bounding_boxes(coco_json_data):
    '''
    Create polygons and bounding boxes from COCO JSON annotations.

    Parameters
    ----------
        coco_json_data (dict): COCO format JSON data containing annotations.

    Returns
    -------
        (polygons, bboxes) --> tuple(list, list): Contains a list of Polygon objects and BoundingBox.
    '''
    polygons = []
    bboxes = []
    for ann in coco_json_data['annotations']:
        poly = Polygon(np.array(ann['segmentation']).reshape(-1, 2))
        polygons.append(poly)
        x, y, w, h = ann['bbox']
        bboxes.append(BoundingBox(x, y, x+w, y+h))
    return polygons, bboxes



def mod_augmentation_sequence():
    '''
    Create a modiefied augmentation sequence (see imagaug documentation).

    Returns
    -------
        seq (iaa.Sequential): An imgaug Sequential object containing a series of image augmentation operations.
    '''
    seq = iaa.Sequential([
        iaa.Fliplr(0.5),
        iaa.LinearContrast((0.75, 1.5)),
        iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
        iaa.Multiply((0.8, 1.2), per_channel=0.2)
        ],random_order=True)
    return seq


def perform_augmentations(image, panoptic_mask, semantic_mask, polygons, bboxes, seq):
    '''
    Create a modiefied augmentation sequence (see imagaug documentation).

    Parameters
    ----------
        image (numpy.ndarray): Raw input image.
        panoptic_mask (numpy.ndarray): The panoptic mask.
        semantic_mask (numpy.ndarray): The semantic mask.
        polygons (list): List of Polygon objects.
        bboxes (list): List of Bounding Box objects.
        seq (iaa.Sequential): An imgaug Sequential object containing a series of image augmentation operations.

    Returns
    -------
        tuple(image_aug, panoptic_mask_aug, semantic_mask_aug, polygons_aug, bboxes_aug):
            image_aug: Augmented raw image.
            panoptic_mask_aug: Augmented panoptic mask.
            semantic_mask_aug: Augmented semantic mask.
            polygons_aug: Augmented polygon list.
            bboxes_aug: Augmented bounding box list.
    '''
    polygons_on_image = PolygonsOnImage(polygons, shape=image.shape)
    bboxes_on_image = BoundingBoxesOnImage(bboxes, shape=image.shape)
    segmap_panoptic = SegmentationMapsOnImage(panoptic_mask, shape=image.shape)
    segmap_semantic = SegmentationMapsOnImage(semantic_mask, shape=image.shape)

    image_aug, segmaps_aug, polygons_aug, bboxes_aug = seq(
        image=image,
        segmentation_maps=[segmap_panoptic, segmap_semantic],
        polygons=polygons_on_image,
        bounding_boxes=bboxes_on_image
    )

    panoptic_mask_aug = segmaps_aug[0].get_arr()
    semantic_mask_aug = segmaps_aug[1].get_arr()

    return image_aug, panoptic_mask_aug, semantic_mask_aug, polygons_aug, bboxes_aug



def new_json_data(coco_json_data, polygons_aug, bboxes_aug, image_aug, output_filename):
    '''
    Update the COCO JSON with the augmented information

    Parameters
    ----------
        coco_json_data (dict): Original COCO JSON data.
        polygons_aug (PolygonsOnImage): Augmented polygons.
        bboxes_aug (BoundingBoxesOnImage): Augmented bounding boxes.
        image_aug (numpy.ndarray): Augmeneted image.
        ouput_filename (str): Filename for the outputted image.

    Returns
    -------
        coco_json_data (dict):
            Updated COCO JSON that includes all the augmented information.

    '''
    for ann, poly_aug, bbox_aug in zip(coco_json_data['annotations'], polygons_aug, bboxes_aug):
        segmentation = poly_aug.exterior.reshape(-1).tolist()
        ann['segmentation'] = [segmentation]
        ann['bbox'] = [bbox_aug.x1, bbox_aug.y1, bbox_aug.width, bbox_aug.height]
        ann['area'] = poly_aug.area

    coco_json_data['images'][0]['file_name'] = output_filename
    coco_json_data['images'][0]['width'] = image_aug.shape[1]
    coco_json_data['images'][0]['height'] = image_aug.shape[0]

    return coco_json_data



def save_augmented_data(image_aug, panoptic_mask_aug, semantic_mask_aug, coco_json_data, output_dir, base_filename):
    '''
    This function saves the augmented images and masks and dumps the quantitative data into a COCO formatted JSON file

     Parameters
     ----------
        image_aug (numpy.ndarray): The augmented image.
        panoptic_mask_aug (numpy.ndarray): The augmented panoptic mask.
        semantic_mask_aug (numpy.ndarray): The augmented semantic mask.
        coco_json_data (dict): Updated COCO JSON data.
        output_dir (str): Directory to save the output files.
        base_filename (str): Base filename for saving the outputs.

    Returns
    -------
        None
    '''
    image_dir = os.path.join(output_dir, "raw_images")
    panoptic_mask_dir = os.path.join(output_dir, "panoptic_masks")
    semantic_mask_dir = os.path.join(output_dir, "semantic_masks")
    json_dir = os.path.join(output_dir, "augmented_jsons")

    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(panoptic_mask_dir, exist_ok=True)
    os.makedirs(semantic_mask_dir, exist_ok=True)
    os.makedirs(json_dir, exist_ok=True)

    cv2.imwrite(os.path.join(image_dir, f"{base_filename}.jpg"), image_aug)
    cv2.imwrite(os.path.join(panoptic_mask_dir, f"{base_filename}_panoptic_mask.png"), panoptic_mask_aug)
    cv2.imwrite(os.path.join(semantic_mask_dir, f"{base_filename}_semantic_mask.png"), semantic_mask_aug)
    with open(os.path.join(json_dir, f"{base_filename}.json"), 'w') as f:
        json.dump(coco_json_data, f)



def process_images(image_path, panoptic_mask_path, semantic_mask_path, coco_json_path, output_dir, num_augmentations):
    '''
    This function processes images and applies custom augmentations.

    Parameters
    ----------
        image_path (str): Path to the input image.
        panoptic_mask_path (str): Path to the panoptic mask.
        semantic_mask_path (str): Path to the semantic mask.
        coco_json_path (str): Path to the COCO JSON file.
        output_dir (str): Directory to save the augmented outputs.
        num_augmentations (int): Number of augmentations to perform per image.

    Returns
    -------
        None
    '''
    image, panoptic_mask, semantic_mask = load_image_and_masks(image_path, panoptic_mask_path, semantic_mask_path)
    coco_json_data = load_json(coco_json_path)
    polygons, bboxes = create_polygons_and_bounding_boxes(coco_json_data)
    seq = mod_augmentation_sequence()
    base_filename = os.path.splitext(os.path.basename(image_path))[0]

    for aug in range(num_augmentations):
        image_aug, panoptic_mask_aug, semantic_mask_aug, polygons_aug, bboxes_aug = perform_augmentations(
            image, panoptic_mask, semantic_mask, polygons, bboxes, seq
        )
        output_filename = f"augmented_{base_filename}_{aug+1}.jpg"
        coco_data_aug = new_json_data(coco_json_data.copy(), polygons_aug, bboxes_aug, image_aug, output_filename)
        save_augmented_data(image_aug, panoptic_mask_aug, semantic_mask_aug, coco_data_aug, output_dir, f"augmented_{base_filename}_{aug+1}")



def main():
    '''
    Main function that sets up paths and processes images. The function is set up to iterate through all the input
    images (along with their masks) to apply the augmentations.
    '''
    dirname = os.path.dirname(os.path.abspath('__file__'))
    image_dir = os.path.join(dirname, "images")
    coco_json_dir = os.path.join(dirname, "manual_annotations")
    panoptic_mask_dir = os.path.join(dirname, "panoptic_segmentation_output")
    semantic_mask_dir = os.path.join(dirname, "semantic_segmentation_output")
    output_dir = os.path.join(dirname, "imagaug_data")
    num_augmentations = 10

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(image_dir):
        if filename.endswith(".jpg"):
            base_filename = os.path.splitext(filename)[0]
            image_path = os.path.join(image_dir, filename)
            panoptic_mask_path = os.path.join(panoptic_mask_dir, f"{base_filename}_mask.png")
            semantic_mask_path = os.path.join(semantic_mask_dir, f"{base_filename}_mask.png")
            coco_json_path = os.path.join(coco_json_dir, f"{base_filename}.json")

            if all(os.path.exists(path) for path in [image_path, panoptic_mask_path, semantic_mask_path, coco_json_path]):
                process_images(image_path, panoptic_mask_path, semantic_mask_path, coco_json_path, output_dir, num_augmentations)



if __name__ == "__main__":
    main()


# ### NOTE: Take a closer look at the documentation for `imagaug` to increase the robustness of the augmentations of the images. Be sure to merge the JSON files prior to training using merge.py
