#!/usr/bin/env python
# coding: utf-8

from pycocotools.coco import COCO
import json
import os


def merge_coco_json(json_files, output_file):
    ''' Merges multiple COCO format JSON files into a single COCO format JSON file.

    This function take a list of COCO formatted JSON files and combines them into a single
    COCO formatted JSON file, taking into account unique ID conflicts by adding
    offset values to the image, annotation, and category ID keys.

    Parameters
    ----------
    json_files : list(str)
        A list of file paths to COCO format JSON files that need to be merged.

    output_file : str
        The file path where the merged COCO format JSON file will be saved

    Returns
    -------
    None
        The function writes the merged JSON data directly to the specified output file.
    '''
    merged_annotations = {
        "info": {},
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": []
    }

    image_id_offset = 0
    annotation_id_offset = 0
    category_id_offset = 0
    existing_category_ids = set()

    for idx, file in enumerate(json_files):
        coco = COCO(file)

        # Update image IDs to avoid conflicts
        for image in coco.dataset['images']:
            image['id'] += image_id_offset
            merged_annotations['images'].append(image)

        # Update annotation IDs to avoid conflicts
        for annotation in coco.dataset['annotations']:
            annotation['id'] += annotation_id_offset
            annotation['image_id'] += image_id_offset
            merged_annotations['annotations'].append(annotation)

        # Update categories and their IDs to avoid conflicts
        for category in coco.dataset['categories']:
            if category['id'] not in existing_category_ids:
                category['id'] += category_id_offset
                merged_annotations['categories'].append(category)
                existing_category_ids.add(category['id'])

        image_id_offset = len(merged_annotations['images'])
        annotation_id_offset = len(merged_annotations['annotations'])
        category_id_offset = len(merged_annotations['categories'])

    # Save merged annotations to output file
    with open(output_file, 'w') as f:
        json.dump(merged_annotations, f)


def main():
    '''
    Function to take parameters and call the merge_coco_json function above.
    '''
    dirname = os.path.dirname(os.path.abspath('__file__'))
    all_jsons = os.path.join(dirname, "manual_annotations")
    # List of paths to COCO JSON files to merge
    json_files = [os.path.join(all_jsons, pos_json) for pos_json in os.listdir(all_jsons) if pos_json.endswith('.json')]
    # Output file path for merged annotations
    output_file = os.path.join(dirname, "panoptic_segmentation_output.json")

    # Merge COCO JSON files
    merge_coco_json(json_files, output_file)

    print("Merged COCO JSON files saved to", output_file)


if __name__ == "__main__":
    main()




# LINK: https://stackoverflow.com/questions/68650460/how-to-merge-multiple-coco-json-files-in-python

