import numpy as np
import h5py
import matplotlib.pyplot as plt
from numpy.fft import fft, ifft, ifftshift, fftshift
from scipy.fft import ifft2, fft2
import torch
import sys
sys.path.append('/home/parth/Projects/UCSD/Medical_Imaging/fastMRI')
import fastmri
# from fastmri.fftc import *
# from fastmri.data import transforms as T

class PartialFourier:

	def __init__(self, filepath):
		with h5py.File(filepath, "r") as hf:
			self.image_file = np.array(hf["kspace"][()])

	def getFourier(self, image, mode='fastmri'):
		if mode == 'fastmri':
			return fastmri.fft2c(image)
		else:
			return fftshift(fft2(fftshift(image, [-2, -1])), [-2, -1])

	def getFourierInverse(self, k_image, mode='fastmri'):
		if mode == 'fastmri':
			return fastmri.ifft2c(k_image)
		else:
			return ifftshift(ifft2(ifftshift(k_image)))

	def maskFourier(self, k_image, mode='fastmri'):
		masked_k_image = k_image
		if mode == 'fastmri':
			masked_k_image[:, 320:, :, :] = torch.flip(masked_k_image[:, :320, :, :], [0])
		else:
			masked_k_image[:, 320:, :] = np.flip(masked_k_image[:, :320, :], [0])
		return masked_k_image

	def plot(self, image, mode='fastmri'):
		if mode == 'fastmri':
			slice_image_abs = fastmri.complex_abs(image)
		else:
			slice_image_abs = np.abs(image)
		plt.imshow(slice_image_abs, cmap='Greys_r')
		plt.show()

if __name__ == '__main__':
	P = PartialFourier('/media/parth/DATA/datasets/fastMRI/singlecoil_challenge/file1000054.h5')
	reconstructed = P.getFourierInverse(P.image_file, mode='numpy')
	reconstructed_fourier = P.getFourier(reconstructed, mode='numpy')
	print(reconstructed_fourier[15, :, :])
	masked_fourier = P.maskFourier(reconstructed_fourier, mode='numpy')
	reconstructed_real = P.getFourierInverse(reconstructed_fourier, mode='numpy')
	print(reconstructed_real == reconstructed)
	P.plot(reconstructed_real[15], mode='numpy')
