import numpy as np
import h5py
import matplotlib.pyplot as plt
from numpy.fft import fft2, ifft2, ifftshift, fftshift
# from scipy.fft import ifft2, fft2
import torch
import sys
sys.path.append('/home/parth/Projects/UCSD/Medical_Imaging/fastMRI')
import fastmri
# from fastmri.fftc import *
# from fastmri.data import transforms as T

class PartialFourier:

	def __init__(self, filepath, mode='fastmri'):
		self.mode = mode

		with h5py.File(filepath, "r") as hf:
			self.image_file = np.array(hf["kspace"][()])
		if self.mode == 'fastmri':
			self.image_file = T.to_tensor(self.image_file)
		else:
			self.image_file = np.array(self.image_file)

	def getFourier(self, image):
		if self.mode == 'fastmri':
			return fastmri.fft2c(image)
		else:
			return fftshift(fft2(fftshift(image)))

	def getFourierInverse(self, k_image):
		if self.mode == 'fastmri':
			return fastmri.ifft2c(k_image)
		else:
			return ifftshift(ifft2(ifftshift(k_image)))

	def maskFourier(self, k_image, mode='complex_conjugate'):
		masked_k_image = k_image.copy()

		if mode == 'zero_fill':
			masked_k_image[:, masked_k_image.shape[1]//2, :] = 0.
		else:
			for i in range(1, masked_k_image.shape[1]//2):
				for j in range(-masked_k_image.shape[2]//2 + 1, masked_k_image.shape[2]//2):
					masked_k_image[:, 320 + i, masked_k_image.shape[2]//2 + j] = np.conjugate(masked_k_image[:, 320 - i, masked_k_image.shape[2]//2 - j])
		return masked_k_image

	def plot(self, image):
		if self.mode == 'fastmri':
			slice_image_abs = fastmri.complex_abs(image)
		else:
			slice_image_abs = np.abs(image)
		plt.imshow(slice_image_abs, cmap='Greys_r')
		plt.show()

	def getPartialFourier(self, image):
		fourier = self.getFourier(image)
		masked_fourier = self.maskFourier(fourier, mode='zero_fill')
		reconstructed = self.getFourierInverse(masked_fourier)
		return reconstructed, self.getDifference(image, reconstructed)

	def getDifference(self, image1, image2):
		return image1 - image2, np.linalg.norm(image1 - image2)

if __name__ == '__main__':
	P = PartialFourier('/media/parth/DATA/datasets/fastMRI/singlecoil_challenge/file1000054.h5', 
					   mode = 'numpy')
	image = P.getFourierInverse(P.image_file)
	real_image = np.real(image)

	reconstructed_image, diff = P.getPartialFourier(image)
	reconstructed_real_image, real_diff = P.getPartialFourier(real_image)
	
	print(diff[1], real_diff[1])

	P.plot(diff[0][15])
