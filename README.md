# Layer-Wise Divergence Control Model
This repository contains the codes for our proposed Layer-Wise Divergence Control defense mechanism, 
and also code for TRADES model for benchmarking. 

## Prerequisites
The `environment.yml` file includes all dependencies.   
You may run the following command to create a conda environment with all dependencies needed. 
```
conda env create -f environment.yml
```
If running on Google Colab, you only need to install dependencies 
`pytorch-lightning` and `advtorch`.

## Usage 
By simply running the `activation_invariant_training.py` with intended arguments, the model will start
training, validating and testing. It will print out the training accuracy, validation accuracy for each epoch
and test accuracy once after training is done.   

If no argument is specified, the model will run with all defaults values:  
attack = none,  model = VGG16,  epsilon = 8/255, ... All default values are specified in `activation_invariant_training.py`  

If run with no argument is specified as below, the model will be trained with no adversarial attack 
(i.e. only on clean dataset):
```
python activation_invariant_training.py'
```
To train the model with a specific attack method, use the `--attack = ... ` and `--eps = ...` to specify. 
For example, the following command train the model using `PGD` with L2 norm with epsilon 1.0
```
python activation_invariant_training.py --attack='pgdl2' --eps=1.0'
```
To train the TRADES model for benchmarking, one can add `--TRADES` to the command line to explicitly run the TRADES model
under intended attack model:
```
python activation_invariant_training.py --TRADES --attack='pgdl2' --eps=1.0'
```
To test the trained model and generated results under attack, one should run `run_attack.pu`, specify the `--model_path` to a saved model that you want to test 
on and specify the attack method `--attack=`, one can evaluate classification accuracy of the saved model. For example, we trained our model with pgd-inf 
adversarial attack and then test with PGD-L2 adversarial attack and without normalization to evaluate it.   

```
python run_attack.py --attack='pgdl2' --no_normalization --model_path='exp_0/checkpoints'
``` 

Some of other optional arguments:  
`--logdir`: specify directory to save log files. Default is current directory.  
`--nepochs`: specify the maximum number of epochs to train. Note that the training will stop with stagnating validation accuracy.
Default is 100.  
`--z_criterion`: specify the type of distance you want to use for calculating the divergence between clean and 
perturbed data at specific layers (specified using `--layer_idx`). Default is cosine similarity.  
`--layer_idx`: specify the layers indicies you would like to compute the divergence between the layer output of clean data 
and adversarial data. Default is all layers.  
`--cln_loss_wt`: specify the weight we use for adding the loss on clean dataset to the full loss.  
`--adv_loss_wt`: specify the weight we use for adding the loss on adversarial dataset to the full loss.  
`--layer_weighting`: specify the way you want to assign weight for adding each layer's divergence to the full loss.
  The weight of layer divergence can be constant, linearly assigned or exponentailly assigned. Default is all layers' divergence
  have weight=1.  
`--layer_indices`: specify which layers to use for layer-wise divergence.

 


## Contact
Contact: {shaohuam,mrosnero,jinchenx,yamany}@andrew.cmu.edu  
Project Link: https://github.com/xjcjessie/11785_project
