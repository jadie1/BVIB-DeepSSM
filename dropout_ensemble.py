import os
import json
import numpy as np
from numpy import matlib
import torch
from torch.utils.data import DataLoader
import re
from scipy.stats import pearsonr
import bayesian_deepssm
import json

def get_samples(experiment_dir, config_file, loader_names, samples=30):
	# Get params
	with open(config_file) as json_file: 
		parameters = json.load(json_file)
	model_dir = experiment_dir + parameters["model_name"]+ '/'
	sample_dir = model_dir + 'predictions/samples/'
	if not os.path.exists(sample_dir):
		os.makedirs(sample_dir)
	if parameters["use_best_model"]:
		model_path = model_dir + 'best_model.torch'
	else:
		model_path = model_dir + 'final_model.torch'
	loader_dir = parameters["paths"]["loader_dir"].replace("med","kep")
	num_latent = parameters["num_latent_dim"]
	# initalizations
	print("Loading model " + model_path + "...")
	# print("Epochs", parameters["best_model_epochs"])
	model = bayesian_deepssm.BayesianDeepSSMNet(config_file)
	model.load_state_dict(torch.load(model_path, map_location="cuda:0"))
	model.cuda()
	model.eval()
	with torch.no_grad():
		enable_dropout(model)
		# enable_mixup(model)
		for name in loader_names:
			test_loader(model, name, loader_dir, sample_dir, samples, parameters)

def enable_dropout(model):
	count = 0
	for m in model.modules():
		if 'Dropout' in m.__class__.__name__:
			count += 1
			m.train()
	print("Turned on", count, "dropout layers.")
	model_dict = model.state_dict()
	drop_keys = [key for key in model_dict.keys() if "p_logit" in key]
	if drop_keys:
		Ps = torch.empty(10)
		for i in range(len(drop_keys)):
			Ps[i] = torch.sigmoid(model_dict[drop_keys[i]])
		print("Dropout probs: ", Ps.numpy())

def test_loader(model, name, loader_dir, sample_dir, samples, parameters):
	# train_loader = torch.load(loader_dir + name)
	train_loader = torch.load((loader_dir + name).replace("kep","med"))
	with open(loader_dir + "../data_info.json") as json_file:
		data_info = json.load(json_file) 
	for img, pca, corr, nm in train_loader:
		nm = nm[0]
		num_particles = corr.shape[1]
		corr = corr.flatten(start_dim=1)
		# Epistemic - Avg over dropout masks
		sampled_y_mus = []
		for i in range(samples):
			pred_z, pred_y, _ = model(img.to(model.device), 0, use_dropout=True)
			sampled_y_mus.append(pred_y[0].detach().cpu().numpy())
		save_samples = np.array(sampled_y_mus).reshape((samples, num_particles,3))
		np.save(sample_dir + nm + ".npy", save_samples)

def predict_ensemble(experiment_dir, model_names, ensemble_name, test_loader="combo"):
	for model_name in model_names:
		get_samples(experiment_dir, experiment_dir+model_name+".json", [test_loader])

	# Create out dirs
	out_dir = experiment_dir + ensemble_name + '/'
	pred_dir = out_dir + 'predictions/'
	if not os.path.exists(pred_dir):
		os.makedirs(pred_dir)
	for folder in ['particles/', 'aleatoric/', 'epistemic/']:
		if not os.path.exists(pred_dir + folder):
			os.makedirs(pred_dir + folder)
	for file in sorted(os.listdir(experiment_dir+model_names[0]+'/'+'predictions/particles/')):
		pred = []
		ale = []
		samples = []
		for model in model_names:
			pred.append(np.loadtxt(experiment_dir+model+'/'+'predictions/particles/'+file))
			ale.append(np.loadtxt(experiment_dir+model+'/'+'predictions/aleatoric/'+file.replace("particles", 'npy')))
			samples.append(np.load(experiment_dir+model+'/'+'predictions/samples/'+file.replace("particles", 'npy')))
		samples = np.array(samples)
		samples = samples.reshape((samples.shape[0]*samples.shape[1], samples.shape[2], samples.shape[3]))
		np.savetxt(pred_dir+'particles/'+file, np.mean(np.array(pred),axis=0))
		np.savetxt(pred_dir+'aleatoric/'+file.replace("particles", 'npy'), np.mean(np.array(ale),axis=0))
		np.savetxt(pred_dir+'epistemic/'+file.replace("particles", 'npy'), np.var(np.array(samples),axis=0))

if __name__ == '__main__':
	experiment_dir = 'SS_experiments/size_1000/'
	models = ['concrete_vib', 'concrete_vib_2', 'concrete_vib_3', 'concrete_vib_4']
	ensmeble_name = 'ensemble_concrete_vib'
	predict_ensemble(experiment_dir, models, ensmeble_name, test_loader='test')