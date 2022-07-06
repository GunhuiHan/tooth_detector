# Wisdom tooth(사랑니) detector
  - Using Tensorflow object detection API
  - model = SSD ResNet50 V1 FPN 640x640

# version
  - python 3.0
  - tensorflow 2.5.0

# Directory Structure

* preprocess/
    * preprocessing.py
        - change wrong folder and file names
        - unify image type to .bmp
        - delete unnecessary files (only use original file and panorama/*0_1.bmp)
        - make json file (format : {"bbox" : [xmin, ymin, xmax, ymax] , "filename"})
    
    * partition_dataset.py : divide dataset to train, test (9:1)
    * generate_tfrecord.py : generate tfrecord for each train, test dataset for feeding as custom dataset of Tensorflow object detection API
    * labem_map.pbtx : only one class 'wisdom'

* Tensorflow/ : Tensorflow object detection API (out of size for uploading)
    - https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md
    - use SSD ResNet50 V1 FPN 640x640 with config {num_classes: 1, batch_size: 8, shuffle_buffer_size: 800, num_steps:25000}

* output/ : output images from trained model


# Evaluation results

- Total Dataset : 990
- Training dataset : 895
- Test dataset : 95

 - Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.284
 - Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.610
 - Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.216
 - Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = -1.000
 - Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = -1.000
 - Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.284
 - Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.350
 - Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.596
 - Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.615
 - Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = -1.000
 - Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = -1.000
 - Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.615
- INFO:tensorflow:        Eval metrics at step 25000
- INFO:tensorflow:        + DetectionBoxes_Precision/mAP: 0.283663
- INFO:tensorflow:        + DetectionBoxes_Precision/mAP@.50IOU: 0.610300
- INFO:tensorflow:        + DetectionBoxes_Precision/mAP@.75IOU: 0.216358
- INFO:tensorflow:        + DetectionBoxes_Precision/mAP (small): -1.000000
- INFO:tensorflow:        + DetectionBoxes_Precision/mAP (medium): -1.000000
- INFO:tensorflow:        + DetectionBoxes_Precision/mAP (large): 0.283663
- INFO:tensorflow:        + DetectionBoxes_Recall/AR@1: 0.350000
- INFO:tensorflow:        + DetectionBoxes_Recall/AR@10: 0.595833
- INFO:tensorflow:        + DetectionBoxes_Recall/AR@100: 0.614583
- INFO:tensorflow:        + DetectionBoxes_Recall/AR@100 (small): -1.000000
- INFO:tensorflow:        + DetectionBoxes_Recall/AR@100 (medium): -1.000000
- INFO:tensorflow:        + DetectionBoxes_Recall/AR@100 (large): 0.614583
- INFO:tensorflow:        + Loss/localization_loss: 0.178654
- INFO:tensorflow:        + Loss/classification_loss: 0.236925
- INFO:tensorflow:        + Loss/regularization_loss: 2.124194
- INFO:tensorflow:        + Loss/total_loss: 2.539773