# BVIB-DeepSSM
Implementation of [Fully Bayesian VIB-DeepSSM](https://arxiv.org/abs/2305.05797) to be presented at MICCAI 2023.
Please cite the paper if you use this code in research.

To run the SuperShapes example, first download the data loader from here: [SS_loaders](https://drive.google.com/drive/folders/1fb4_l98GPDTzqjJjAZ9oBDZfj1UIhq3H?usp=drive_link)

Unzip into the ```SS_loaders/``` folder. 

To train a model, call ```trainer.py``` with a config file. The config files with parameters used for experiments reported in the paper are in ```SS_experiments/```.
For example:
```
python trainer.py -c SS_experiments/size_1000/vib.json
```
When the model is done training it will add the path to the trained model to the config file. Then to evaluate the performance on the test set, call:
```
python eval.py -c SS_experiments/size_1000/vib.json -n 4
```
Where `-n` defines the number of samples to use in testing. 

For naive ensembling, train multiple models and then use the `dropout_ensemble.py` script to get predicitions.
