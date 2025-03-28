# Panoptic Segmentation using Detectron2

This folder contains scripts for handling COCO-formatted datasets, performing image augmentations, and generating panoptic and semantic segmentation masks for computer vision tasks (e.g segmentation, classification, etc). Additionally, it includes a Jupyter Notebook, `detectron2.ipynb`, which demonstrates training and evaluation of panoptic image segmentation for predefined classes. While the default configuration specifies a model and classes, these can be customized. Note that the behavior of the outputs in the notebook is influenced by the selected model and class definitions. Below is a detailed description of each file and its functionality. NOTE: will be updated this with additional supplementary files.
---

## Files and Descriptions

### 1. **merge.py**

This script merges multiple COCO format JSON files into a single COCO format JSON file. It resolves ID conflicts for images, annotations, and categories by applying unique offsets.

#### Key Functions:
- `merge_coco_json(json_files, output_file)`: Merges multiple COCO JSON files into one while maintaining unique IDs.
- `main()`: Automatically detects COCO JSON files in the `manual_annotations` directory and merges them into `panoptic_segmentation_output.json`.

#### Usage:
Run the script to combine multiple JSON files in the `manual_annotations` directory.
```bash
cd preprocessing
python merge.py
```

---

### 2. **training_image_augmentations.py**

This script uses the `imgaug` library to apply custom augmentations to images, panoptic masks, semantic masks, and their corresponding COCO JSON annotations.

#### Key Functions:
- `load_image_and_masks(image_path, panoptic_mask_path, semantic_mask_path)`: Loads an image and its associated masks.
- `create_polygons_and_bounding_boxes(coco_json_data)`: Converts COCO segmentation data into polygons and bounding boxes.
- `mod_augmentation_sequence()`: Defines a sequence of augmentations (e.g., flipping, contrast adjustment, noise addition).
- `perform_augmentations(...)`: Applies augmentations to images, masks, polygons, and bounding boxes.
- `new_json_data(...)`: Updates COCO JSON annotations with augmented data.
- `save_augmented_data(...)`: Saves augmented images, masks, and JSON files.
- `process_images(...)`: Processes multiple images and generates augmented datasets.

#### Usage:
Run the script to augment images and their associated metadata. Ensure input directories (`images`, `manual_annotations`, `panoptic_segmentation_output`, `semantic_segmentation_output`) are correctly set up.
```bash
cd preprocessing
python training_image_augmentations.py
```

---

### 3. **semantic_mask_manual.py**

This script generates semantic segmentation masks from COCO JSON annotations by combining instance masks into a single binary mask.

#### Key Functions:
- `coco_to_mask(ann, height, width)`: Converts COCO annotations to binary masks.
- `create_semantic_mask(json_file, image_height, image_width)`: Generates semantic masks by combining all instance masks.
- `process_all_images_and_jsons(image_dir, json_dir, output_dir)`: Processes all images and JSON files in the input directory and generates semantic masks.

#### Usage:
Run the script to generate semantic masks for all images in the `manual_annotations` directory.
```bash
cd preprocessing
python semantic_mask_manual.py
```

---

### 4. **panoptic_mask_manual.py**

This script generates panoptic segmentation masks by assigning unique colors to each instance in an image based on COCO annotations.

#### Key Functions:
- `coco_to_mask(ann, height, width)`: Converts COCO annotations to binary masks.
- `main()`: Processes all images and JSON files in the `manual_annotations` directory and generates colored panoptic masks.

#### Usage:
Run the script to create panoptic masks.
```bash
cd preprocessing
python panoptic_mask_manual.py
```

---

## Running the Notebook

This Jupyter Notebook (`detectron2.ipynb`) demonstrates the integration of Detectron2 for training and evaluating segmentation models using the generated data. It includes:
- Preprocessing steps.
- Model training using the COCO dataset.
- Visualization of results.

#### Usage:
Open the notebook and execute cells sequentially.
```bash
jupyter notebook detectron2.ipynb
```

---

## Current Directory Structure

Ensure the following directory structure before running the scripts:
```
project/
|-- images/                   # Input raw images
|-- manual_annotations/       # COCO JSON files and masks
|-- panoptic_segmentation_output/  # Panoptic masks (output)
|-- semantic_segmentation_output/  # Semantic masks (output)
|-- imagaug_data/             # Augmented data (output)
|-- preprocessing/
  |-- merge.py
  |-- training_image_augmentations.py
  |-- semantic_mask_manual.py
  |-- panoptic_mask_manual.py
|-- detectron2.ipynb
```

## Requirements

Install the required Python libraries:
```bash
pip install opencv-python numpy matplotlib imgaug pycocotools scikit-image
```

## Notes

- Ensure all paths are correctly set in the scripts before running.
- Use `merge.py` to combine COCO JSON files before training.
- Modify augmentation parameters in `training_image_augmentations.py` to suit your dataset.
- Test each script independently to validate outputs.

---

## Contact
For any questions or clarification don't be afraid to reach out! Email: josephyared0@gmail.com