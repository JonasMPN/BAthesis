# BA thesis
Vortex detection in a 2-dimensional flow field using deep neuronal networks.  
NOTE:  
The files coco_eval, coco_utils, engine, transforms and utils have not been created by me.

## data_evaluation.py  
This file provides functionalities to:  
-  calculate IoU for bounding boxes
-  calculate center distances between bounding boxes
-  calculate the average precision of predictions
-  filter true positives  

## data_post.py  
This file provides functionalities to:  
- create a parallel coordinates plot of the information contained in a protocol file (a file that keeps track of all parameter combinations and their resuluts)
- turn all information from a protocol file to a format that above mentioned plotting functionality requires  

## data_prep.py  
This file provides functionalities to:  
- create pictures of velocity fields containing one or arbitrary many Hamel-Oseen-Vortices (class HamelOseenVortexCreator)
- translate experimental data into usable pictures (class ExperimentalData)
- Create parameter combinations, and create and handle a database (class Orders). This class has many features, is fast, inhernently prevents douplicates, and is independent of all the other code.  

## data_visualise.py  
This file contains functionalities to:
- plot the truth and predicted bounding boxes as well as their centers  

## experiment.py  
This file provides functionalities to:
- set parameters
- start creating data for the models, start training, testing, plotting the predictions, evaluating, and visualising results  

## helper_functions.py  
This file provides basic functionalities needed in all other files.  

## structure_evaluation.py  
Class EvaluationRig is a wrapper for the functionalities from data_evaluation.py to make the functionalities easier to use.  

## structure_test_rig.py  
This file provides functionalities to:
- test a set of parameter combinations
- test a certain parameter combination
- test a dataset for validation (hence there is no need to safe all data that is being safed in the two points above)  

## structure_train_rig.py
This file provides functionilites to:
- train a model on a set of parameter combinations database
- use my version of k-fold cross validation while training
- create a custom dataset for pytorch's models
- use my own early stopper during training  


