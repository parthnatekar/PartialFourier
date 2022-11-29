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
		print(self.image_file.shape)

	def getFourier(self, image):
		if self.mode == 'fastmri':
			return fastmri.fft2c(image)
		else:
			return fft2(image)

	def getFourierInverse(self, k_image):
		if self.mode == 'fastmri':
			return fastmri.ifft2c(k_image)
		else:
			return ifft2(k_image)

	def maskFourier(self, k_image):
		masked_k_image = k_image.copy()
		if self.mode == 'fastmri':
			masked_k_image[:, 320:, :, :] = torch.flip(masked_k_image[:, :320, :, :], [0])
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

if __name__ == '__main__':
	P = PartialFourier('/media/parth/DATA/datasets/fastMRI/singlecoil_challenge/file1000054.h5', 
					   mode = 'numpy')
	reconstructed = np.real(P.getFourierInverse(P.image_file))
	# P.plot(reconstructed[15])
	reconstructed_fourier = P.getFourier(reconstructed)
	print(reconstructed_fourier[15, 320+10, 184+10], reconstructed_fourier[15, 320-10, 184-10])
	# print(fftshift(reconstructed_fourier, [-2, -1]))
	masked_fourier = P.maskFourier(reconstructed_fourier)
	plt.imshow(np.abs(reconstructed_fourier[15] - masked_fourier[15]))
	plt.show()
	print(masked_fourier[15, 320+10, 184+10], masked_fourier[15, 320-10, 184-10])
	reconstructed_real = P.getFourierInverse(masked_fourier)
	# print(np.where(reconstructed_real != reconstructed))
	print(np.linalg.norm(reconstructed-reconstructed_real))
	P.plot(reconstructed_real[15])
