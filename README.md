# Deep Learning: Graphics - Final Project

Final team project for Deep Learning: Graphics at Barnard, Spring 2025

*README updated on 5/7/2025*

## Contents

- [Project Overview](#overview)
- [Implementation Details](#details)
- [Sample Results](#sample-results)
- [Prior Conceptualization](#prior-conceptualization)
- [Directory Structure](#directory-structure)

## Overview

The goal of this project is to develop a generative adversarial network model that is capable
of turning a simple line sketch drawing into a full-fledged painting with a color palette
specified by the user.  One can think of it in simplistic terms as as a color-by-numbers 
project - taking a fixed palette of any set of colors along with a sketch or outline drawing
and having the GAN generate a painting using both the sketch and the color palette.

While most image synthesis projects are focused on text-based prompts, this project has as its
aim art students who may not be advanced enough in their skillset to convert a line drawing
into a painting but who would like to see what such a visualization might look like given a
custom color palette.  

The user provides a sketch and some colors, and the model paints the sketch as though it were
a painting using the colors the user specifies.

## Details

To train the model, a dataset of 6000 artworks is split 80/20 into training and validation
sets.  The GAN is trained on the training set for 100 epochs and then run on the validation
set.  To see the results of the image synthesis on the validation set, navigate to the
results directory and the val_results subdirectory.

The trained model was then run on a set of 20,000 simple sketches of various objects, each
given its own random set of 3 colors for a palette.  This resulted in 20,000 colorized/painted
versions of the simple sketches.  The results of this test set can be found in results/simple_sketch_results.

## Sample Results

## Prior Conceptualization

This project initially began as an object detection model with the aim of using labeled objects
detected in artworks as inputs into a GAN to rearrange the objects in a newly synthesized image.
There was, however, an overreliance on pretrained Detectron2 and Detic models, so the project was
reconceptualized into its current form.

The original idea was to use the Detectron2 along with the COCO dataset to detect objects in an 
artwork dataset, drawing bounding boxes with labels around the objects on the object, then
extracting the objects using masks.  

The original model was a pretrained Detectron2 model with inputs being both the COCO dataset
and a stylized version of the COCO dataset to better approximate the art styles found in artworks.
The COCO dataset proved too limited in its classifications to detect many objects, so the model
was switched to use a pretrained Detic model with a custom vocabulary of 1500 words drawn from
WordNet.  This detected many more objects in the artworks, resulting in many more bounding
boxes and mask crops.  However, given the overreliance on pretrained models and subar results in
training a GAN using these masks as input, the project was reconceptualized.

## Directory Structure

The directories are as ordered as follows:

```
- datasets: All original datasets used for training and validation
    >  train_A:  Contour+palette combination images used as input into the GAN (5,191 contours)
    >  train_B:  The original artworks corresponding to the files in train_A (5,191 artworks)
    >    val_A:  Contour+palette combination images used for testing the GAN (1,299 contours)
    >    val_B:  Original artworks corrseponding to files in val_A (1,299 artworks)
                 (val_B not used for testing or training; comparison only)
    >  val_simple_sketches:  Simple sketches dataset (20,000 sketches) used to test trained model
    >  prelim_projects_datasets:  Datasets that were used in original project conception
          * coco_val2017:  Common Objects in Context validation 2017 dataset used for training
          * coco_stylized:  Stylized versions of COCO val2017 dataset
    >  binary_masks(unused):  Black/white bounding boxes representing where certain objects
            are located in the original painting to tell the GAN to focus on these areas.
            This is currently not used in the currently trained model.
```

```
- results: Results from running the trained GAN on specific sketch datasets
     > val_results:  Validation results from original artworks dataset 
         * val_contour:  The contour+palette images used for testing (1,299 contour images)
         * val_original:  The original artwork images (1,299 artworks) not used except for comparison
         * val_synthesized:  The synthesized GAN model results (1,299 novel artworks)
     > simple_sketch_results:  The results after testing the GAN on 20,000 simple sketch images
         * ss_contour:  The simple sketch images with random color palettes (20,000 contours)
         * ss_original:  The original simple sketch images used for comparison only (20,000 works)
         * ss_synthesized:  The colorized/painted GAN conceptualization of the sketches with the
                given color palette (20,000 synthesized images)
    > prelim_proj_results:  Artwork object detection bounding boxes and crops from original
            project conception.
```

```
- samples: Sample triplet images showing a small subset of the results with original/contour/synthesized
     images placed in juxtaposition for comparison to see how the model performs
```

```
- models: 200 trained models (.pth) - 100 discriminator and 100 generator.  Each model corresponds to
     one epoch and is labeled with the epoch number as the suffix in the file name.  Only the final trained
     model is included in this directory.  The remaining are used merely to generate sample reference images 
     from all aspects of the training process and can be ignored and aren't included here due to space
     constraints.
```

