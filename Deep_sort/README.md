# Deep Sort

This is a Deep Sort implementation repository. Download the pre-trained weights from the drive links provided.
The apporach uses YOLOv4 for object detection and the original deep sort model trained by the authors.

## Key Libraries used:

```
Keras                     2.4.0
matplotlib                3.5.1 
numpy                     1.19.5
opencv-python             4.4.0.42 
tensorflow                2.6.0 
```

## Testing
1. Download the pre-trained YOLOv4 weights from **[here](https://github.com/Mohit-robo/Deep_Learning/tree/main/Deep_sort/data/vehicle-detector)**
2. Deep Sort model from **[here](https://github.com/Mohit-robo/Deep_Learning/tree/main/Deep_sort/model_data)**
3. YOLOv4 and YOLOv4-tiny weights are been provided in the drive folder.

Once the files have been setup run the ob_detection.py file
          
    python lp_detection.py
    
The paths to video, weights, the objects to be tracked and other constants are been declared in the ./core/config.py file.
The dependency functions are been declared in the ./core/dependencies.py and ./core/Object_dependencies.py do check them out for modifications.


Main deep sort **[github](https://github.com/nwojke/deep_sort)** repo.
