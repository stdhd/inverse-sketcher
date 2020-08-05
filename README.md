# inverse-sketcher

## Prerequisites
Make sure to install all dependencies listed in ``requirements.txt``.

## How to train
In order to start model training, follow these steps:
1. Make sure, the dataset folder sticks to the structure explained below.
2. Place ``.yaml`` file in the ``params`` folder describing the training parameters. In ``example.yaml`` you can find a valid configuration.
3. The model name ist defined in the ``.yaml`` file, as well. This name will be used to identifiy stored models, as well.
4. Call ``python3 train.py modelname.yaml``.
5. You can find the resulting training statistics and model parameter files after training has finished in the ``saved_models`` directory.

## Necessary structure of dataset folder
The dataset folder needs to be placed inside the project root directory. This folder then splits into different sub-directories naming all the different datasets that can be used while training and evaluation. Inside the dataset folders, there is one photo folder and one sketch folder, containing each directories for image categories (for shoes, there is only one category).
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
3. The generated images are stored in the ``generator/modelname``directory.
