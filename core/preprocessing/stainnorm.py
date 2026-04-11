import numpy as np
import cv2
from matplotlib import pyplot as plt

class StainNormalizer:
    """
    Class to normalize staining appearance of H&E stained images.
    Based on Macenko et al., ISBI 2009 and Vink et al., J Microscopy, 2013.
    """
    
    def __init__(self, Io=240, alpha=1, beta=0.15):
        """Initialize with default parameters."""
        self.Io = Io
        self.alpha = alpha
        self.beta = beta
        self.HERef = np.array([[0.5626, 0.2159],
                               [0.7201, 0.8012],
                               [0.4062, 0.5581]])
        self.maxCRef = np.array([1.9705, 1.0308])

    def read_image(self, path):
        """Reads and converts an image to RGB."""
        img = cv2.imread(path)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    def rgb_to_od(self, img):
        """Converts RGB image to Optical Density (OD)."""
        img = img.reshape((-1, 3))
        OD = -np.log10((img.astype(np.float32) + 1) / self.Io)
        return OD
    
    def remove_transparent_pixels(self, OD):
        """Removes pixels with OD intensity less than beta."""
        return OD[~np.any(OD < self.beta, axis=1)]
    
    def compute_svd(self, ODhat):
        """Computes SVD to find stain vectors."""
        eigvals, eigvecs = np.linalg.eigh(np.cov(ODhat.T))
        return eigvals, eigvecs
    
    def find_stain_vectors(self, eigvecs, ODhat):
        """Finds the hematoxylin and eosin stain vectors."""
        That = ODhat.dot(eigvecs[:, 1:3])
        phi = np.arctan2(That[:, 1], That[:, 0])

        minPhi = np.percentile(phi, self.alpha)
        maxPhi = np.percentile(phi, 100 - self.alpha)

        vMin = eigvecs[:, 1:3].dot(np.array([np.cos(minPhi), np.sin(minPhi)]).T)
        vMax = eigvecs[:, 1:3].dot(np.array([np.cos(maxPhi), np.sin(maxPhi)]).T)

        if vMin[0] > vMax[0]:
            return np.array((vMin, vMax)).T
        else:
            return np.array((vMax, vMin)).T

    def separate_stains(self, OD, HE):
        """Separates the image into Hematoxylin and Eosin components."""
        Y = np.reshape(OD, (-1, 3)).T
        C = np.linalg.lstsq(HE, Y, rcond=None)[0]

        maxC = np.array([np.percentile(C[0, :], 99), np.percentile(C[1, :], 99)])
        C2 = np.divide(C, (maxC / self.maxCRef)[:, np.newaxis])
        return C2

    def recreate_image(self, C2):
        """Recreates the normalized image from the separated components."""
        Inorm = np.multiply(self.Io, np.exp(-self.HERef.dot(C2)))
        Inorm[Inorm > 255] = 254
        return np.reshape(Inorm.T, (self.h, self.w, 3)).astype(np.uint8)

    def extract_H_E(self, C2):
        """Extracts Hematoxylin and Eosin stain images."""
        H = np.multiply(self.Io, np.exp(np.expand_dims(-self.HERef[:, 0], axis=1).dot(np.expand_dims(C2[0, :], axis=0))))
        E = np.multiply(self.Io, np.exp(np.expand_dims(-self.HERef[:, 1], axis=1).dot(np.expand_dims(C2[1, :], axis=0))))

        H[H > 255] = 254
        E[E > 255] = 254

        H = np.reshape(H.T, (self.h, self.w, 3)).astype(np.uint8)
        E = np.reshape(E.T, (self.h, self.w, 3)).astype(np.uint8)

        return H, E

    def process(self, img_path):
        """Main workflow to normalize image and extract stains."""
        if isinstance(img_path, str):
            img = self.read_image(img_path)
        else:
            img = img_path
        self.h, self.w, _ = img.shape

        OD = self.rgb_to_od(img)
        ODhat = self.remove_transparent_pixels(OD)
        _, eigvecs = self.compute_svd(ODhat)

        HE = self.find_stain_vectors(eigvecs, ODhat)
        C2 = self.separate_stains(OD, HE)

        Inorm = self.recreate_image(C2)
        H, E = self.extract_H_E(C2)

        return Inorm, H, E