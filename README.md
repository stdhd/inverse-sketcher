# inverse-sketcher


## How to train
In order to start model training, follow these steps:
1. Make sure, the dataset folder sticks to the structure explained below.
2. Place ``.yaml`` file in the ``params`` folder describing the training parameters. In ``example.yaml`` you can find a valid configuration.
3. Call ``python3 train.py modelname.yaml``
4. You can find the resulting training statics and model parameter after training has finished in the ``saves_models`` directory.

## Necessary structure of dataset folder
<details>
 
```
dataset
 |dataset1
   |photo
     |class1
       |image files...
     |class2
       |image files...
     |...
     |classN
   |sketch
     |class1
       |image files...
     |class2
       |image files...
     |...
     |classN
```
 <summary>Click to expand!</summary>
 </details>
 
 ## How to generate images from test split of the dataset
After the training of a model has finished, the used random train-test split is saved to the model parameter file, as well. When evaluating or generating images using a trained model, the corresponding test images will be used in order to maintain a clean data split. Follow these steps for generating images from a trained model: 
1. Search for the name of the model you want to use in the ``saved_models``directory.
2. Use this model name to call ``python3 generator.py modelname --generate``.
3. The generated images are stores in the ``generator/modelname``directory.
