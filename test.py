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

	def maskFourier(self, k_image, p = 0.3, mode='complex_conjugate'):
		masked_k_image = k_image.copy()

		masked_k_image[:, (int((1-p)*masked_k_image.shape[1]) + 1):, :] = 0.

		if mode == 'complex_conjugate':
			for i in range(1, int(p*masked_k_image.shape[1])):
				for j in range(-masked_k_image.shape[2]//2 + 1, masked_k_image.shape[2]//2):
					masked_k_image[:, int((1-p)*masked_k_image.shape[1]) + i, masked_k_image.shape[2]//2 + j] = np.conjugate(masked_k_image[:, int(p*masked_k_image.shape[1]) - i, masked_k_image.shape[2]//2 - j])
		elif mode == 'homodyne':
			masked_k_image = self.getHomodyne(masked_k_image, p)
		return masked_k_image

	def getHomodyne(self, k_image, p):
		homodyne_filter = np.ones(k_image.shape, dtype=np.complex_)

		homodyne_filter[:, :(int(p*k_image.shape[1])), :] = 2.
		homodyne_filter[:, (int((1-p)*k_image.shape[1]) + 1):, :] = 0.

		return k_image*homodyne_filter

	def getPhase(self, k_image, p):
		phase = k_image.copy()

		phase[:, :(int(p*k_image.shape[1])), :] = 0. + 0j

		def hann(N, k_image):
			window = k_image.copy()
			window[:, k_image.shape[1]//2-N//2:k_image.shape[1]//2+N//2, :] = np.repeat(np.repeat(0.5*(1-np.cos(np.conjugate(2*np.pi*np.arange(0, N))/(N)))[None, ..., None], 
																							 k_image.shape[0], axis=0), k_image.shape[2], axis=2)
			return window

		hann_window = hann(int(p*k_image.shape[1]), phase)
		
		phase = phase * hann_window

		phase = np.exp(-1j * np.angle(self.getFourierInverse(phase)))

		return phase

	def plot(self, image):
		if self.mode == 'fastmri':
			slice_image_abs = fastmri.complex_abs(image)
		else:
			slice_image_abs = np.abs(image)
		plt.imshow(slice_image_abs, cmap='Greys_r')
		plt.show()

	def getPartialFourier(self, image, p=0.4, mode='complex_conjugate'):
		fourier = self.getFourier(image)
		masked_fourier = self.maskFourier(fourier, p=p, mode=mode)
		# plt.imshow(np.abs(fourier[15]) - np.abs(masked_fourier[15]))
		# plt.colorbar()
		# plt.show()
		reconstructed = self.getFourierInverse(masked_fourier)
		if mode == 'homodyne':
			phase = self.getPhase(masked_fourier, p)
			# reconstructed = np.real(phase * reconstructed)
			# reconstructed[reconstructed < 0.] = 0.
		return reconstructed, self.getDifference(image, reconstructed)

	def getDifference(self, image1, image2):
		return image1 - image2, np.linalg.norm(image1 - image2)

if __name__ == '__main__':
	P = PartialFourier('/media/parth/DATA/datasets/fastMRI/singlecoil_challenge/file1000054.h5', 
					   mode = 'numpy')
	image = P.getFourierInverse(P.image_file)
	real_image = np.real(image)

	mode = 'homodyne'
	proportion = 0.4

	reconstructed_image, diff = P.getPartialFourier(image, p = proportion, mode = mode)
	reconstructed_real_image, real_diff = P.getPartialFourier(real_image, p = proportion, mode=mode)
	
	print(diff[1], real_diff[1])

	P.plot(diff[0][15])
