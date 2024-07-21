"""
Part code from (c) AAPM DGM-Image Challenge Organizers. 
Author:Zhikai Yang
This source-code is licensed under the terms outlined in the accompanying LICENSE file.
if this code is useful for you, please cite our paper.
"""
import argparse
import numpy as np
from scipy import stats
import os
import os.path as p
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import ks_2samp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from utils import convert_to_image_domain, convert_to_frequency_domain
from utils import load_data, load_lowfreq_data,load_lowres_data
from utils import extract_frequency_features

pd.set_option('use_inf_as_na', True) # All infs throughout are set to nans

parser = argparse.ArgumentParser(conflict_handler='resolve')
parser.add_argument("--results_dir",    type=str,   default='metric_results', help="Results directory")
# data arguments
parser.add_argument("--data_name",      type=str,   default='downsized_victre_xray_objects2',   help="Data name (innermost folder name used for stylegan training)")
parser.add_argument("--path_to_reals",  type=str,   default='',                                 help="Path to the folder containing the .png images (real or fake)")
parser.add_argument("--path_to_fakes",  type=str,   default='',                                 help="Path to the folder containing the .png images (real or fake)")
parser.add_argument("--num_images",     type=int,   default=1000,                               help="Number of images to load")
# summary arguments
parser.add_argument("--num_random_runs",          default=100000,   type=int,                   help='Number of samples to draw from while measuring the cosine distance.')
parser.add_argument("--pca_components",           default=9,        type=int,                   help='Number of PCA components to use.')
parser.add_argument("--to_save_pca_plots",        default=1,        type=lambda b:bool(int(b)), help="save pca plots")
parser.add_argument("--to_save_cosine_plots",     default=0,        type=lambda b:bool(int(b)), help="save cosine plots. Requires seaborn installed.")
args = parser.parse_args()

if args.results_dir != '':
    os.makedirs(args.results_dir, exist_ok=True)

# load real or fake data
real_data = {}
fnames, imgs = load_data(args.path_to_reals, args.num_images)
real_data['fnames'] = fnames
real_data['data'] = imgs
print("Real data loaded")

fake_data = {}
# fnames, imgs = load_data(args.path_to_fakes, args.num_images)
# fnames, imgs = load_lowfreq_data(args.path_to_fakes, args.num_images)
fnames, imgs = load_lowres_data(args.path_to_fakes, args.num_images)

fake_data['fnames'] = fnames
fake_data['data'] = imgs

low_freq_features_set1 = []
high_freq_features_set1 = []
low_freq_features_set2 = []
high_freq_features_set2 = []

crop_ratio= 0.3
for image in real_data['data']:
    low_freq_features, high_freq_features = extract_frequency_features(image, crop_ratio)
    low_freq_features_set1.append(low_freq_features)
    high_freq_features_set1.append(high_freq_features)

for image in fake_data['data'] :
    low_freq_features, high_freq_features = extract_frequency_features(image, crop_ratio)
    low_freq_features_set2.append(low_freq_features)
    high_freq_features_set2.append(high_freq_features)
# print(len(low_freq_features_set1))
low_freq_features_set1 = np.array(low_freq_features_set1)
high_freq_features_set1 = np.array(high_freq_features_set1)
high_freq_features_set1 = np.nan_to_num(high_freq_features_set1)
high_freq_features_set1[np.isfinite(high_freq_features_set1)] = 0.0001

low_freq_features_set2 = np.array(low_freq_features_set2)
high_freq_features_set2 = np.array(high_freq_features_set2)
high_freq_features_set2 = np.nan_to_num(high_freq_features_set2)
high_freq_features_set2[np.isfinite(high_freq_features_set2)] = 0.0001
    # Combine the feature sets
all_low_freq_features = np.concatenate((low_freq_features_set1, low_freq_features_set2), axis=0)
all_high_freq_features = np.concatenate((high_freq_features_set1, high_freq_features_set1), axis=0)
print("all_low_freq_features",all_low_freq_features.shape)
print("all_high_freq_features",all_high_freq_features.shape)
# Apply PCA to the low-frequency features
n_components = 4  # Number of principal components
pca_low_freq = PCA(n_components=n_components)
pca_low_freq.fit(all_low_freq_features)
transformed_low_freq_set1 = np.abs(pca_low_freq.transform(low_freq_features_set1))
transformed_low_freq_set2 = np.abs(pca_low_freq.transform(low_freq_features_set2))
transformed_low_freq_set1_pd = pd.DataFrame(transformed_low_freq_set1)
transformed_low_freq_set2_pd = pd.DataFrame(transformed_low_freq_set2)
real_cosines = []
mixed_cosines = []
for i in range(args.num_random_runs):
    # print(f'{i}/{args.num_random_runs}', end='\r')
    sampled = transformed_low_freq_set1_pd.sample(n=2, replace=False).to_numpy()
    real_cosines.append( np.dot(sampled[0,:], sampled[1,:]) / (np.linalg.norm(sampled[0,:])*np.linalg.norm(sampled[1,:])) )
    sampledr = transformed_low_freq_set1_pd.sample(n=1, replace=False).to_numpy()
    sampledf = transformed_low_freq_set2_pd.sample(n=1, replace=False).to_numpy()
    mixed_cosines.append( np.dot(sampledr[0,:], sampledf[0,:]) / (np.linalg.norm(sampledr[0,:])*np.linalg.norm(sampledf[0,:])) )
        
print("Compute low freq KS statistic")
ks_stat_low, _ = ks_2samp(real_cosines, mixed_cosines) # This will slightly vary over multiple runs, probably in the second decimal place.
print("KS statistic : ", ks_stat_low)
    
# Apply PCA to the high-frequency features
pca_high_freq = PCA(n_components=n_components)
pca_high_freq.fit(all_high_freq_features)
transformed_high_freq_set1 = np.abs(pca_high_freq.transform(high_freq_features_set1))
transformed_high_freq_set2 = np.abs(pca_high_freq.transform(high_freq_features_set2))
transformed_high_freq_set1_pd = pd.DataFrame(transformed_high_freq_set1)
transformed_high_freq_set2_pd = pd.DataFrame(transformed_high_freq_set2)
real_cosines = []
mixed_cosines= []
for i in range(args.num_random_runs):
    # print(f'{i}/{args.num_random_runs}', end='\r')
    sampled = transformed_high_freq_set1_pd.sample(n=2, replace=False).to_numpy()
    real_cosines.append( np.dot(sampled[0,:], sampled[1,:]) / (np.linalg.norm(sampled[0,:])*np.linalg.norm(sampled[1,:])) )
    sampledr = transformed_high_freq_set1_pd.sample(n=1, replace=False).to_numpy()
    sampledf = transformed_high_freq_set2_pd.sample(n=1, replace=False).to_numpy()
    mixed_cosines.append( np.dot(sampledr[0,:], sampledf[0,:]) / (np.linalg.norm(sampledr[0,:])*np.linalg.norm(sampledf[0,:])) )

    # Compute KS statistic on the reals and the mixed distributions of cosine similarities.
print("Compute high freq KS statistic")
ks_stat_high, _ = ks_2samp(real_cosines, mixed_cosines) # This will slightly vary over multiple runs, probably in the second decimal place.
print("KS statistic : ", ks_stat_high)
    # Plot the transformed low-frequency and high-frequency feature sets