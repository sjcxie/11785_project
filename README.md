# 11785_project
Adversarial Attack Project for 11785 Intro to Deep Learning.


## attack_classifier.py
Generate attacks and predictions. 
#### ** Attacker Class **
It has method `generate_examples()` to generate perturbed dataset.  
And method `eval()` to evaluate the accuracy of perturbed data on attacked model.
#### ** whitebox_attack() **



## iterative_projected_gradient.py
Contains PGD method classes and other one-step attack methods like FastFeatureAttack.


## extract_and_plot_embeddings.py
#### ** get_embeddings() **  
The "distance" between clean data embeddings and perturbed data embeddings is calculated using 
`pairwise_distance = np.sqrt()`. Then the pairwise distance is averaged,   
TODO: we want to experiement with other distance measures here.   
The preds are generated in `attack_classifier.py`    


#### ** _get_embeddings() ** 
with `model(_, store_intermediate=True)`, saves the intermediate embeddings of each layers. 
Concatenate them together and return for using later in `get_embeddings()`.

### activation_invariance_training.py

#### TRADESTrainer
TRADES: TRadeoff-inspired Adversarial DEfense via Surrogate-loss minimization

