import cv2
import numpy as np
from sklearn.decomposition import PCA
from scipy.stats import ks_2samp
from threading import Thread
import glob
import os
import imageio as io
from PIL import Image

def load_data(path, num_images):
    data = np.zeros((num_images, 512, 512), dtype=np.uint8)
    fnames = sorted(glob.glob(os.path.join(path, '*.png')))[:num_images]
    for i,f in enumerate(fnames):
        print(f"Loading data {i}/{num_images}", end='\r')
        data[i] = io.imread(f)
    data = data.astype(int)
    return fnames, data


def load_lowfreq_data(path, num_images):
    data = np.zeros((num_images, 512, 512), dtype=np.uint8)
    fnames = sorted(glob.glob(p.join(path, '*.png')))[:num_images]
    for i,f in enumerate(fnames):
        print(f"Loading data {i}/{num_images}", end='\r')
        image = io.imread(f)
        data[i]= preserve_low_frequency(image,0.7)
    data = data.astype(int)
    return fnames, data

def load_lowres_data(path, num_images):
    data = np.zeros((num_images, 512, 512), dtype=np.uint8)
    fnames = sorted(glob.glob(os.path.join(path, '*.png')))[:num_images]
    for i,f in enumerate(fnames):
        print(f"Loading data {i}/{num_images}", end='\r')
        image = Image.open(f)
        image = image.resize((256,256))
        image = image.resize((512,512))
        # print("image.size",image.size)
        image = np.array(image)
        if len(image.shape)==2:
            # data[i]= image
            # img  = np.repeat(image[:,:,np.newaxis],3,axis=1)
            # img = np.squeeze(img)
            print("image.shape",image.shape)
            data[i]= image
        else:
            data[i]= image[:,:,0]
    data = data.astype(int)
    return fnames, data

def convert_to_frequency_domain(image):
    # Convert image to grayscale
    gray_image = image
    # gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Perform FFT
    frequency_domain = np.fft.fft2(gray_image)
    shifted_frequency_domain = np.fft.fftshift(frequency_domain)
    return shifted_frequency_domain

def convert_to_image_domain(freq):
    f_ishift = np.fft.ifftshift(freq)
    image_filtered = np.fft.ifft2(f_ishift)
    image_filtered = np.abs(image_filtered)

    # Convert the filtered image back to uint8 format
    image_filtered = image_filtered.astype(np.uint8)
    return image_filtered


def preserve_low_frequency(image, crop_ratio):
    frequency_domain = convert_to_frequency_domain(image)
    # Get the size of the image
    rows, cols = frequency_domain.shape    
    # Calculate the center of the image
    crow, ccol = int(rows / 2), int(cols / 2)
    crop_pixels = int(min(crow, ccol) * crop_ratio)

    # Create a mask to preserve low-frequency components
    low_freq_mask = np.zeros(frequency_domain.shape, np.uint8)
    low_freq_mask[crow - crop_pixels:crow + crop_pixels, ccol - crop_pixels:ccol + crop_pixels] = 1

    high_freq_mask = np.zeros(frequency_domain.shape, np.uint8)
    high_freq_mask[crow - crop_pixels:crow + crop_pixels, ccol - crop_pixels:ccol + crop_pixels] = 1
    high_freq_mask = 1 - high_freq_mask    
    low_freq_features = frequency_domain * low_freq_mask
    high_freq_features = frequency_domain * high_freq_mask

    # low_freq_features = low_freq_features[crow - crop_pixels:crow + crop_pixels, ccol - crop_pixels:ccol + crop_pixels]
    # mask = np.zeros((rows, cols), dtype=np.uint8)
    # mask[center_row - cutoff_frequency:center_row + cutoff_frequency,
    #      center_col - cutoff_frequency:center_col + cutoff_frequency] = 1
    # # Apply the mask to the frequency domain representation
    # fshift_filtered = fshift * mask
    image_filtered = convert_to_image_domain(high_freq_features)
    return image_filtered

def extract_frequency_features(image, crop_ratio):
    frequency_domain = convert_to_frequency_domain(image)
    # Get the center coordinates of the frequency domain
    rows, cols = frequency_domain.shape
    crow, ccol = int(rows / 2), int(cols / 2)

    # Calculate the number of pixels to crop based on the crop ratio
    crop_pixels = int(min(crow, ccol) * crop_ratio)
    # print("crop_pixels",crop_pixels)
    # Create masks to separate high-frequency and low-frequency components
    low_freq_mask = np.zeros(frequency_domain.shape, np.uint8)
    low_freq_mask[crow - crop_pixels:crow + crop_pixels, ccol - crop_pixels:ccol + crop_pixels] = 1

    high_freq_mask = np.zeros(frequency_domain.shape, np.uint8)
    high_freq_mask[crow - crop_pixels:crow + crop_pixels, ccol - crop_pixels:ccol + crop_pixels] = 1
    high_freq_mask = 1 - high_freq_mask
    # print("high_freq_mask",high_freq_mask)
    # Apply masks to the frequency domain
    low_freq_features = frequency_domain * low_freq_mask
    low_freq_features = low_freq_features[crow - crop_pixels:crow + crop_pixels, ccol - crop_pixels:ccol + crop_pixels]
    # print("low_freq_features.shape",low_freq_features.shape)
    low_freq_features = np.log(low_freq_features)
    high_freq_features = frequency_domain * high_freq_mask
    high_freq_features = np.abs(high_freq_features.flatten())    
    high_freq_features = abs(np.sort(-high_freq_features))
    # high_freq_features.sort()
    # print(high_freq_features)
    high_freq_features = high_freq_features[:512**2 - (crop_pixels*2)**2]
    # print("high_freq_features.shape",high_freq_features.shape)
    high_freq_features = np.log(high_freq_features)
    
    # print("high_freq_features",high_freq_features)
    # high_freq_features = high_freq_features[:-100]
    # print("low_freq_features",np.min(low_freq_features))
    # print("high_freq_features",np.max(high_freq_features))
    return np.abs(low_freq_features.flatten()), np.abs(high_freq_features.flatten())