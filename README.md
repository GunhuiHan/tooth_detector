# Wisdom tooth(사랑니) detector (진행중)
  - Using Tensorflow object detection API
  - model = SSD ResNet50 V1 FPN 640x640

# version
  - python 3.9
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

output example
[output_example.pdf](https://github.com/GunhuiHan/tooth_detector/files/9136844/output_example.pdf)
