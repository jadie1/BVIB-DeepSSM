# BVIB-DeepSSM
Implementation of [Fully Bayesian VIB-DeepSSM](https://arxiv.org/abs/2305.05797) to be presented at MICCAI 2023.
Please cite the paper if you use this code in research.

To run the supershapes experiment, first download and unzip the supershapes dataset: [SS_data](https://drive.google.com/file/d/1oegp1RZVFJfx8s7aL5GgX55UKGhRgAZn/view?usp=drive_link).
The supershape file names specificy the number of shape lobes and the degree to which the image is blurred. 

Next generate torch loaders for the train, validation and test sets by calling `SS_loaders.py`. Specifcy the desired size of the training set (up to 1000). For example:
```
python SS_loaders.py --size 1000
```
This will create an `SS_loaders/` folder with the loaders. 

To train a model, call ```trainer.py``` with a config file. The config files with parameters used for experiments reported in the paper are in ```SS_experiments/```.
For example:
```
python trainer.py -c SS_experiments/vib.json
```
When the model is done training it will add the path to the trained model to the config file. Then to evaluate the performance on the test set, call:
```
python eval.py -c SS_experiments/vib.json -n 4
```
Where `-n` defines the number of samples to use in testing. 

For naive ensembling, train multiple models and then use the `dropout_ensemble.py` script to get predicitions.
