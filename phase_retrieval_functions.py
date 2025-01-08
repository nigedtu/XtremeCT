import os
import numpy as np
import matplotlib.pyplot as plt
import math
import h5py
import scipy.ndimage as ndimage
from scipy.signal import find_peaks
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn
import pandas
from sklearn.cluster import KMeans
from PIL import Image
import opencv as cv
import tifffile

import numpy as np
from scipy import special
import pyfftw
import time
import scipy
from scipy import ndimage, misc


from scipy.fft import fft, fftshift
from scipy.signal import convolve

#import cv2 as cv

#os.chdir(r"C:\Users\nige\OneDrive - Danmarks Tekniske Universitet\Dokumenter\PhD\XRR_Fitting_tool")
#from optical_constants import oc

def convert_pixel_size_to_actual_distance(pixel_size_nm, magnification):
    """
    Convert detector pixel size to actual physical distance in nanometers.

    Parameters:
    - pixel_size_nm: Size of one pixel on the detector in nanometers.
    - magnification: Magnification factor.

    Returns:
    - Actual physical distance in nanometers.
    """
    # Calculate the actual distance in the object plane
    actual_distance_nm = pixel_size_nm / magnification

    return actual_distance_nm

def wavelength2energy(wavelength):
    # wavelength is in nm - conversion to energy in eV 
    eV = 1.60217662*1e-19 # [J] per eV
    h = 6.62607004*1e-34 # [m^2*kg/s]
    c = 299792458 # [m/s]
    
    E = h*c/(wavelength*1e-9)
    energy = E/eV 
    return energy
    
def energy2wavelength(energy):
    # energy is in eV - conversion to wavelength in nm 
    eV = 1.60217662*1e-19 # [J] per eV
    h = 6.62607004*1e-34 # [m^2*kg/s]
    c = 299792458 # [m/s]
    
    E = energy*eV # converting from [eV] to [J]
    wavelength = h*c/E*1e9 # [nm] Instrument wavelength / energy
    return wavelength


class IndexTracker(object):
    def __init__(self, ax, X):
        self.ax = ax
        ax.set_title('use scroll wheel to navigate images')

        self.X = X
        #rows, cols, self.slices = X.shape
        self.slices, rows, cols = X.shape
        self.ind = self.slices//2

        self.im = ax.imshow(self.X[self.ind,:, :])
        self.update()

    def onscroll(self, event):
        print("%s %s" % (event.button, event.step))
        if event.button == 'up':
            self.ind = (self.ind + 1) % self.slices
        else:
            self.ind = (self.ind - 1) % self.slices
        self.update()

    def update(self):
        self.im.set_data(self.X[self.ind,:, :])
        self.ax.set_ylabel('slice %s' % self.ind)
        self.im.axes.figure.canvas.draw()

values = []

def draw_parallel_lines(matrix, start, end, num_lines, spacing=2, width=1):
    """
    Draws parallel lines to the central line defined by start and end points.
    
    Parameters:
    - matrix: The grid where the lines will be drawn (2D numpy array).
    - start: (x0, y0) starting point of the central line.
    - end: (x1, y1) ending point of the central line.
    - num_lines: The number of parallel lines to draw (including the central line).
    - spacing: The spacing between each parallel line.
    - width: The width of each line.
    
    Returns:
    - all_line_points: A list of lists, where each sublist contains the points of one line.
    - all_xxx: A list of lists, where each sublist contains the x-coordinates of one line.
    - all_yyy: A list of lists, where each sublist contains the y-coordinates of one line.
    """

    def bresenham_line(matrix, start, end, width=1):
        """
        Bresenham's algorithm to draw a line between start and end points on the matrix.
        Also returns the x and y coordinates of the points.
        """
        x0, y0 = start
        x1, y1 = end
        
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        
        err = dx - dy
        
        line_points = []  # List to store the coordinates of the line
        xxx = []  # To store x-coordinates
        yyy = []  # To store y-coordinates
        
        while True:
            for i in range(-(width // 2), (width // 2) + 1):
                if dx > dy:
                    new_point = (y0 + i, x0)
                else:
                    new_point = (y0, x0 + i)
                    
                if 0 <= new_point[0] < matrix.shape[0] and 0 <= new_point[1] < matrix.shape[1]:
                    matrix[new_point] = 1
                    line_points.append(new_point)  # Store the coordinate
                    xxx.append(new_point[0])  # Store the x-coordinate
                    yyy.append(new_point[1])  # Store the y-coordinate
                    
            if x0 == x1 and y0 == y1:
                break
                
            e2 = 2 * err
            
            if e2 > -dy:
                err -= dy
                x0 += sx
            
            if e2 < dx:
                err += dx
                y0 += sy
        
        return line_points, xxx, yyy

    def calculate_angle(start, end):
        """Calculate the angle of the line between start and end points."""
        delta_y = end[1] - start[1]
        delta_x = end[0] - start[0]
        return np.arctan2(delta_y, delta_x)

    def shift_parallel(start, end, distance, angle):
        """Shift the start and end points to create parallel lines."""
        dx_perp = np.sin(angle) * distance
        dy_perp = -np.cos(angle) * distance
        
        new_start = (start[0] + dx_perp, start[1] + dy_perp)
        new_end = (end[0] + dx_perp, end[1] + dy_perp)
        
        return new_start, new_end

    # Calculate the angle of the central line
    angle = calculate_angle(start, end)

    # Store all line points, xxx, and yyy
    all_line_points = []
    all_xxx = []
    all_yyy = []

    # Draw the central line and parallel lines
    for i in range(-(num_lines//2), (num_lines//2) + 1):
        # Shift the line by i * spacing in the perpendicular direction
        start_shifted, end_shifted = shift_parallel(start, end, i * spacing, angle)
        
        # Draw the shifted line and store the points
        line_points, xxx, yyy = bresenham_line(matrix, (int(start_shifted[0]), int(start_shifted[1])), 
                                               (int(end_shifted[0]), int(end_shifted[1])), width=width)
        
        # Append the results for each line
        all_line_points.append(line_points)
        all_xxx.append(xxx)
        all_yyy.append(yyy)

    return all_line_points, all_xxx, all_yyy

#def zoom(img, zoom_factor=2):
#    return cv.resize(img, None, fx=zoom_factor, fy=zoom_factor)

def fom(A, B):
    #fom = np.sum(np.abs(A - B)) # sum of absolute differences (SAD),
    #fom = np.abs(np.nansum(A-B))
    fom = np.nansum(np.abs(A-B))
    return fom

def normalize_array(arr):
    norm_arr = (arr - np.min(arr)) / (np.max(arr) - np.min(arr))
    return norm_arr

def draw_concentric_circles(matrix, center, radius, num_circles, width=1):
    """
    Draws concentric circles around a specified center point and returns the original matrix values
    at those points along with their coordinates, ordered in a clockwise direction around the circle.
    
    Parameters:
    - matrix: The grid where the circles will be drawn (2D numpy array).
    - center: (center_row, center_col) center point of the circles.
    - radius: The radius of the circles.
    - num_circles: The number of parallel circles to draw (including the central circle).
    - width: The width of each circle.
    
    Returns:
    - circle_info_matrix: A 2D NumPy array where each row corresponds to (x, y, original_value)
                          for each circle point, ordered in a clockwise direction.
    """

    all_circle_info = []

    # Draw the concentric circles
    for i in range(num_circles):
        current_radius = radius + i  # Increment the radius for each circle

        # Collect points in a clockwise manner using parametric equations
        for angle in np.linspace(0, 2 * np.pi, num=360, endpoint=False):
            x = int(center[1] + current_radius * np.cos(angle))  # Column index (x)
            y = int(center[0] + current_radius * np.sin(angle))  # Row index (y)
            
            # Ensure the point is within the matrix bounds
            if 0 <= y < matrix.shape[0] and 0 <= x < matrix.shape[1]:
                original_value = matrix[y, x]  # Get original value
                matrix[y, x] = 1  # Mark the circle
                all_circle_info.append((x, y, original_value))  # Store (x, y, original_value)

    # Convert the collected information to a NumPy array
    circle_info_matrix = np.array(all_circle_info)

    return circle_info_matrix


values = []
def draw_line(mat, x0, y0, x1, y1, inplace=False):
    if not (0 <= x0 < mat.shape[0] and 0 <= x1 < mat.shape[0] and
            0 <= y0 < mat.shape[1] and 0 <= y1 < mat.shape[1]):
        raise ValueError('Invalid coordinates.')
    if not inplace:
        mat = mat.copy()
    if (x0, y0) == (x1, y1):
        print('yay')
        values.append(mat[x0, y0])
        mat[x0, y0] = 3
        return mat if not inplace else None
    # Swap axes if Y slope is smaller than X slope
    transpose = abs(x1 - x0) < abs(y1 - y0)
    if transpose:
        mat = mat.T
        x0, y0, x1, y1 = y0, x0, y1, x1
    # Swap line direction to go left-to-right if necessary
    if x0 > x1:
        x0, y0, x1, y1 = x1, y1, x0, y0
    # Write line ends # and save in current value
    #values.append(mat[x0, y0])
    values.append(mat[x1, y1])
    #values = mat[x0, y0]
    mat[x0, y0] = 3
    mat[x1, y1] = 3
    # Compute intermediate coordinates using line equation
    x = np.arange(x0 + 1, x1)
    y = np.round(((y1 - y0) / (x1 - x0)) * (x - x0) + y0).astype(x.dtype)
    # Write intermediate coordinates
    #values.append(mat[x, y])
    #print(mat[x,y])
    bla = [x,y]
    mat[x, y] = 3
    if not inplace:
        return mat , bla if not transpose else mat.T
    
    

class IndexTracker(object):
    def __init__(self, ax, X):
        self.ax = ax
        ax.set_title('use scroll wheel to navigate images')

        self.X = X
        #rows, cols, self.slices = X.shape
        self.slices, rows, cols = X.shape
        self.ind = self.slices//2

        self.im = ax.imshow(self.X[self.ind,:, :])
        self.update()

    def onscroll(self, event):
        print("%s %s" % (event.button, event.step))
        if event.button == 'up':
            self.ind = (self.ind + 1) % self.slices
        else:
            self.ind = (self.ind - 1) % self.slices
        self.update()

    def update(self):
        self.im.set_data(self.X[self.ind,:, :])
        self.ax.set_ylabel('slice %s' % self.ind)
        self.im.axes.figure.canvas.draw()

    
def remove_outliers(data, dezinger, dezinger_threshold):
    """Remove outliers (dezinger)
    Parameters
    ----------
    data : ndarray
        Input 3D array
    dezinger: int
        Radius for the median filter
    dezinger_threshold: float
        Threshold for outliers
    
    Returns
    -------
    res : ndarray
        Output array    
    """
    res = data.copy()
    if (int(dezinger) > 0):
        w = int(dezinger)
        # print(data.shape)
        fdata = ndimage.median_filter(data, [w, w]) # [1,w, w] changed from this
        res[:] = np.where(np.logical_and(
            data > fdata, (data - fdata) > dezinger_threshold), fdata, data)
    return res


def rotate(x, y, angle):
    angle = np.deg2rad(angle)
    x_rot = x * np.cos(angle) - y * np.sin(angle)
    y_rot = x * np.sin(angle) + y * np.cos(angle)
    return x_rot, y_rot


def _upsampled_dft(data, ups,
                   upsample_factor=1, axis_offsets=None):

    im2pi = 1j * 2 * np.pi
    tdata = data.copy()
    kernel = (np.tile(np.arange(ups), (data.shape[0], 1))-axis_offsets[:, 1:2])[
        :, :, None]*np.fft.fftfreq(data.shape[2], upsample_factor)
    kernel = np.exp(-im2pi * kernel)
    tdata = np.einsum('ijk,ipk->ijp', kernel, tdata)
    kernel = (np.tile(np.arange(ups), (data.shape[0], 1))-axis_offsets[:, 0:1])[
        :, :, None]*np.fft.fftfreq(data.shape[1], upsample_factor)
    kernel = np.exp(-im2pi * kernel)
    rec = np.einsum('ijk,ipk->ijp', kernel, tdata)

    return rec


#registration_shift:
#src_image=rdata_scaled[k,0:1]
#target_image=rdata_scaled[0,0:1]
#upsample_factor=10
#space="real"
def registration_shift(src_image, target_image, upsample_factor=1, space="real"):

    # assume complex data is already in Fourier space
    if space.lower() == 'fourier':
        src_freq = src_image
        target_freq = target_image
    # real data needs to be fft'd.
    elif space.lower() == 'real':
        src_freq = np.fft.fft2(src_image)
        target_freq = np.fft.fft2(target_image)

    # Whole-pixel shift - Compute cross-correlation by an IFFT
    shape = src_freq.shape
    image_product = src_freq * target_freq.conj()
    cross_correlation = np.fft.ifft2(image_product)
    A = np.abs(cross_correlation)
    maxima = A.reshape(A.shape[0], -1).argmax(1)
    maxima = np.column_stack(np.unravel_index(maxima, A[0, :, :].shape))

    midpoints = np.array([np.fix(axis_size / 2)
                          for axis_size in shape[1:]])

    shifts = np.array(maxima, dtype=np.float64)
    ids = np.where(shifts[:, 0] > midpoints[0])
    shifts[ids[0], 0] -= shape[1]
    ids = np.where(shifts[:, 1] > midpoints[1])
    shifts[ids[0], 1] -= shape[2]
    
    if upsample_factor > 1:
        # Initial shift estimate in upsampled grid
        shifts = np.round(shifts * upsample_factor) / upsample_factor
        upsampled_region_size = np.ceil(upsample_factor * 1.5)
        # Center of output array at dftshift + 1
        dftshift = np.fix(upsampled_region_size / 2.0)

        normalization = (src_freq[0].size * upsample_factor ** 2)
        # Matrix multiply DFT around the current shift estimate

        sample_region_offset = dftshift - shifts*upsample_factor
        cross_correlation = _upsampled_dft(image_product.conj(),
                                                upsampled_region_size,
                                                upsample_factor,
                                                sample_region_offset).conj()
        cross_correlation /= normalization
        # Locate maximum and map back to original pixel grid
        A = np.abs(cross_correlation)
        maxima = A.reshape(A.shape[0], -1).argmax(1)
        maxima = np.column_stack(
            np.unravel_index(maxima, A[0, :, :].shape))

        maxima = np.array(maxima, dtype=np.float64) - dftshift

        shifts = shifts + maxima / upsample_factor
           
    return shifts

def apply_shift(psi, p,n):
    """Apply shift for all projections."""
    psi = np.array(psi)
    p = np.array(p)
    tmp = np.pad(psi,((0,0),(n//2,n//2),(n//2,n//2)), 'symmetric')
    [x, y] = np.meshgrid(np.fft.rfftfreq(2*n),
                         np.fft.fftfreq(2*n))
    shift = np.exp(-2*np.pi*1j *    
                   (x*p[:, 1, None, None]+y*p[:, 0, None, None]))
    res0 = np.fft.irfft2(shift*np.fft.rfft2(tmp))
    res = res0[:, n//2:3*n//2, n//2:3*n//2]#.get()
    return res

# For prefocus:
# Define a model for a decaying curve that stabilizes (asymptotic decay)
def asymptotic_decay(x, a, b, c):
    return a * np.exp(-b * x) + c







# Methods start here


def CTFPurePhase(rads, wlen, dists, fx, fy, alpha):
   """
   weak phase approximation from Cloetens et al. 2002




   Parameters
   ----------
   rad : 2D-array
       projection.
   wlen : float
       X-ray wavelentgth assumes monochromatic source.
   dist : float
       Object to detector distance (propagation distance) in mm.
   fx, fy : ndarray
       Fourier conjugate / spatial frequency coordinates of x and y.
   alpha : float
       regularization factor.
       
   Return
   ------
   phase retrieved projection in real space
   """    
   numerator = 0
   denominator = 0    
   for j in range(0, len(dists)):    
       rad_freq = np.fft.fft2(rads[j])
       taylorExp = np.sin(np.pi*wlen*dists[j]*(fx**2+fy**2)) 
       numerator = numerator + taylorExp * (rad_freq)
       denominator = denominator + 2*taylorExp**2 
   numerator = numerator / len(dists)
   denominator = (denominator / len(dists)) + alpha
   phase = np.real(  np.fft.ifft2(numerator / denominator) )
   return phase


def CTFPurePhase_new(rads, wlen, dists, delta, beta, fx, fy, Rm, alpha):
    """
    weak phase approximation from Cloetens et al. 2002


    Parameters
    ----------
    rad : 2D-array
        projection.
    wlen : float
        X-ray wavelentgth assumes monochromatic source.
    dist : float
        Object to detector distance (propagation distance) in mm.
    delta : float    
        refractive index decrement
    beta : float    
        absorption index
    fx, fy : ndarray
        Fourier conjugate / spatial frequency coordinates of x and y.
    alpha : float
        regularization factor.
        
    Return
    ------

    phase retrieved projection in real space
    """    

    numerator = 0
    denominator = 0    
    for j in range(0, len(dists)):    
        rad_freq = pyfftw.interfaces.numpy_fft.fft2(rads[j])
        taylorExp = np.sin(np.pi*wlen*dists[j]*(fx**2+fy**2)) 
        numerator = numerator + taylorExp * (rad_freq)
        denominator = denominator + 2*taylorExp**2 

    numerator = numerator / len(dists)
    denominator = (denominator / len(dists)) + alpha

    #phase = np.log(np.real(  pyfftw.interfaces.numpy_fft.ifft2(numerator / denominator) ))
    phase = np.real(  np.fft.ifft2(numerator / denominator) )
    phase = (delta/beta) * 0.5 * phase

    return phase




def CTF(rads, wlen, dists, fx, fy, Rm, alpha): # Rm
    """
    Phase retrieval method based on Contrast Transfer Function.    This 
    method assumes weak absoprtion and slowly varying phase shift.
    Derived from Langer et al., 2008: Quantitative comparison of direct
    phase retrieval algorithms.

    Parameters
    ----------
    rads : list of 2D-array
        Elements of the list correspond to projections of the sample
        taken at different distance. One projection per element.
    wlen : float
        X-ray wavelentgth assumes monochromatic source.
    dists : list of float
        Object to detector distance (propagation distance) in mm. One 
        distance per element.
    fx, fy : ndarray
        Fourier conjugate / spatial frequency coordinates of x and y.
    alpha : float
        regularization factor.
        
    Return
    ------

    phase retrieved projection in real space

    """

    A = np.zeros((rads[0].shape[0], rads[0].shape[1]))
    B = np.zeros((rads[0].shape[0], rads[0].shape[1]))
    C = np.zeros((rads[0].shape[0], rads[0].shape[1]))
    E = np.zeros((rads[0].shape[0], rads[0].shape[1]))
    F = np.zeros((rads[0].shape[0], rads[0].shape[1]))

    for j in range(0,len(dists)):
        sin = 2*np.sin(np.pi*wlen*dists[j]*(fx**2+fy**2)) * Rm[:,:,j]
        cos = 2*np.cos(np.pi*wlen*dists[j]*(fx**2+fy**2)) * Rm[:,:,j]
        A = A + sin * cos
        B = B + sin * sin
        C = C + cos * cos
        rad_freq = pyfftw.interfaces.numpy_fft.fft2(rads[j])
        E = E + rad_freq * sin
        F = F + rad_freq * cos
    A = A / len(dists)
    B = B / len(dists)
    C = C / len(dists)    
    Delta = B * C - A**2
    
    phase = (C * E - A * F)    * (1 / (2*Delta+alpha)) 
    phase[0,0] = 0. + 0.j
    phase = pyfftw.interfaces.numpy_fft.ifft2(phase)
    phase = np.real(phase)

    return phase

def homoCTF(rads, wlen, dists, delta, beta, fx, fy, Rm, alpha):# 
    """



    Parameters
    ----------
    rad : 2D-array
        projection.
    wlen : float
        X-ray wavelentgth assumes monochromatic source.
    dist : float
        Object to detector distance (propagation distance) in mm.
    delta : float    
        refractive index decrement
    beta : float    
        absorption index
    fx, fy : ndarray
        Fourier conjugate / spatial frequency coordinates of x and y.
    alpha : float
        regularization factor.
        
    Return
    ------

    phase retrieved projection in real space
    """    
    delta_dirac = np.bitwise_and(fx==0,fy==0).astype(np.double) #Aditya: Discretized Dirac Delta function
    numerator = 0
    denominator = 0
    for j in range(0, len(dists)):    
        rad_freq = pyfftw.interfaces.numpy_fft.fft2(rads[j])
        cos = np.cos(np.pi*wlen*dists[j]*(fx**2+fy**2))
        sin = np.sin(np.pi*wlen*dists[j]*(fx**2+fy**2)) 
        taylorExp = cos*Rm[:,:,j] + (delta/beta) * sin*Rm[:,:,j] # what is Rm????
        #taylorExp = cos + (delta/beta) * sin
        numerator = numerator + taylorExp * (rad_freq - delta_dirac)
        denominator = denominator + taylorExp**2

    numerator = numerator / len(dists)
    denominator = (denominator / len(dists)) + alpha
    
    phase = numerator / denominator    
    phase = np.real(  pyfftw.interfaces.numpy_fft.ifft2(phase) )
    phase = (delta/beta) * 0.5 * phase
    #phase = (delta/beta) * phase

    
    return phase

def CTFPurePhaseWithAbs(rads, wlen, dists, delta, beta, fx, fy, Rm, alpha):
    """
    weak phase approximation from Cloetens et al. 2002


    Parameters
    ----------
    rad : 2D-array
        projection.
    wlen : float
        X-ray wavelentgth assumes monochromatic source.
    dist : float
        Object to detector distance (propagation distance) in mm.
    delta : float    
        refractive index decrement
    beta : float    
        absorption index
    fx, fy : ndarray
        Fourier conjugate / spatial frequency coordinates of x and y.
    alpha : float
        regularization factor.
        
    Return
    ------

    phase retrieved projection in real space
    """    
    argMin = np.argmin(dists)
    numerator = 0
    denominator = 0    
    for j in range(0, len(dists)):    
        rad_freq = pyfftw.interfaces.numpy_fft.fft2(rads[j]/rads[argMin])
        taylorExp = np.sin(np.pi*wlen*dists[j]*(fx**2+fy**2)) 
        numerator = numerator + taylorExp * (rad_freq)
        denominator = denominator + 2*taylorExp**2 

    numerator = numerator / len(dists)
    denominator = (denominator / len(dists)) + alpha

    phase = np.log(np.real(  pyfftw.interfaces.numpy_fft.ifft2(numerator / denominator) ))
    phase = (delta/beta) * 0.5 * phase

    return phase

def multiPaganin(rads, wlen, dists, delta, beta, fx, fy, Rm, alpha):
    """
    Phase retrieval method based on Contrast Transfer Function. This 
    method relies on linearization of the direct problem, based  on  the
    first  order  Taylor expansion of the transmittance function.
    Found in Yu et al. 2018 and adapted from Cloetens et al. 1999


    Parameters
    ----------
    rad : 2D-array
        projection.
    wlen : float
        X-ray wavelentgth assumes monochromatic source.
    dist : float
        Object to detector distance (propagation distance) in mm.
    delta : float    
        refractive index decrement
    beta : float    
        absorption index
    fx, fy : ndarray
        Fourier conjugate / spatial frequency coordinates of x and y.
    alpha : float
        regularization factor.
        
    Return
    ------

    phase retrieved projection in real space
    """    
    numerator = 0
    denominator = 0    
    for j in range(0, len(dists)):    
        rad_freq = pyfftw.interfaces.numpy_fft.fft2(rads[j])    
        taylorExp = 1 + wlen * dists[j] * np.pi * (delta/beta) * (fx**2+fy**2)
        numerator = numerator + taylorExp * (rad_freq)
        denominator = denominator + taylorExp**2 

    numerator = numerator / len(dists)
    denominator = (denominator / len(dists)) + alpha

    phase = np.log(np.real(  pyfftw.interfaces.numpy_fft.ifft2(numerator / denominator) ))    
    phase = (delta/beta) * 0.5 * phase

    
    return phase


def Paganin(rad, wlen, dist, delta, beta, fx, fy, Rm):            
    rad_freq = pyfftw.interfaces.numpy_fft.fft2(rad)

    '''from Paganin et al., 2002'''
    #~ mu = (4 * np.pi * beta) / wlen
    #~ phase = (rad_freq * mu) / (alpha+delta*dist*4*(np.pi**2)*(fx**2+fy**2)*Rm+mu) # 4 * pi^2 not explicit in manuscript
    #~ phase = np.real(pyfftw.interfaces.numpy_fft.ifft2(phase))
    #~ phase = (1/mu)*np.log(phase)
    #~ phase = (2*np.pi*delta/wlen)*phase

    '''from ANKA - Weitkamp et al., 2011'''
    filtre =  1 + (wlen*dist*delta*4*(np.pi**2)*(fx**2+fy**2) / (4*np.pi*beta)) # 4 * pi^2 not explicit in manuscript
    trans_func = np.log(np.real( pyfftw.interfaces.numpy_fft.ifft2( rad_freq / filtre)))
    phase = (delta/(2*beta)) * trans_func
        
    #~ phase = phase *(-wlen)/(2*np.pi*delta)
    return phase    

def sglDstCTF(rad, wlen, dist, delta, beta, fx, fy, Rm, alpha):
    """
    Phase retrieval method based on Contrast Transfer Function.    This 
    method relies on linearization of the direct problem, based  on  the
    first  order  Taylor expansion of the transmittance function.
    Found in Yu et al. 2018 and adapted from Cloetens et al. 1999


    Parameters
    ----------
    rad : 2D-array
        projection.
    wlen : float
        X-ray wavelentgth assumes monochromatic source.
    dist : float
        Object to detector distance (propagation distance) in mm.
    delta : float    
        refractive index decrement
    beta : float    
        absorption index
    fx, fy : ndarray
        Fourier conjugate / spatial frequency coordinates of x and y.
    alpha : float
        regularization factor.
        
    Return
    ------

    phase retrieved projection in real space
    """    
    delta_dirac = np.bitwise_and(fx==0,fy==0).astype(np.double) #Aditya: Discretized Dirac Delta function
    rad_freq = pyfftw.interfaces.numpy_fft.fft2(rad)
    filtre = np.cos(np.pi*wlen*dist*(fx**2+fy**2)) + (delta/beta) * np.sin(np.pi*wlen*dist*(fx**2+fy**2))
    phase = (delta/beta) * 0.5 * ((rad_freq - delta_dirac) / filtre)
    phase = np.real(pyfftw.interfaces.numpy_fft.ifft2(phase))

    return phase


def invLaplacian(rads,pix_width,fx,fy,alpha):    
    """
    calculate inverse laplacian according to equation (21) from Langer 
    et al., 2008: Quantitative comparison of direct phase retrieval 
    algorithms.

    Parameters
    ----------    
    rads : 2D-array
        2D projection in real space.
    pix_width : float
        Pixel width in mm
    fx, fy : ndarray
        Fourier conjugate / spatial frequency coordinates of x and y.
    alpha : float
        regularization factor.
        
    Return
    ------

    phase retrieved projection in real space

    """
        
    # ~ rads_freq = np.fft.fft2(rads,pix_width)
    rads_freq = pyfftw.interfaces.numpy_fft.fft2(rads)
    res = rads_freq/(fx**2+fy**2+alpha)
    res[0,0] = 0. + 0.j    
    # ~ res = -(1/(4*np.pi**2))*get_inv_fft(res).real
    res = -(1/(4*np.pi**2))*pyfftw.interfaces.numpy_fft.ifft2(res).real
    return res





def TIE(rads,wlen,dists,pix_width,fx,fy,Rm,alpha):        
    """
    Transport of Intensity Equation
    Derived from Langer et al., 2008: Quantitative comparison of direct
    phase retrieval algorithms.

    Parameters
    ----------
    rads : 2D-array
        Images of the projections in real space taken at the first and 
        second propagation distance.    
    wlen : float
        X-ray wavelentgth assumes monochromatic source.
    dists : list of float
        Object to detector distance (propagation distance) in mm. One 
        distance per element.
    pix_width : float
        Pixel width in mm
    fx, fy : ndarray
        Fourier conjugate / spatial frequency of x and y.
    alpha : float
        regularization factor.
            
    Return
    ------

    phase retrieved projection in real space

    """
    rad0 = rads[0]
    rad1 = rads[1]
    
    rad0[rad0==0] = 1e-6 #avoid division by 0 of some pixel
    res = (rad1-rad0) / (dists[1]-dists[0])
    res = invLaplacian(res,pix_width,fx,fy,alpha)
    res_y, res_x = np.gradient(res)
    res_x = res_x/rad0
    res_y = res_y/rad0
    res = np.gradient(res_x, axis=1) + np.gradient(res_y, axis=0)
    res = res/(pix_width**2) #Aditya: Should divide by pixel width for each gradient
    res = invLaplacian(res,pix_width,fx,fy,alpha)
    res = (-2*np.pi/wlen)*res    
    return res





def WTIE(rads,wlen,dists,pix_width,fx,fy,Rm,alpha):
    """
    TIE for weak absorption. Similar method to the TIE but combining
    phase retrieval and inverse Radon transform in one step. 
    Derived from Langer et al., 2008: Quantitative comparison of direct
    phase retrieval algorithms.

    Parameters
    ----------
    rads : 2D-array
        Images of the projections in real space taken at the first and 
        second propagation distance.    
    wlen : float
        X-ray wavelentgth assumes monochromatic source.
    dists : list of float
        Object to detector distance (propagation distance) in mm. One 
        distance per element.
    pix_width : float
        Pixel width in mm
    fx, fy : ndarray
        Fourier conjugate / spatial frequency of x and y.
    alpha : float
        regularization factor.
            
    Return
    ------

    phase retrieved projection in real space

    """
    rad0 = rads[0]
    rad1 = rads[1]
    
    rad0 = rad0 + 1e-6 #avoid division by 0
    res = (rad1/rad0) - 1
    res = -((2*np.pi)/(wlen*(dists[1])))*invLaplacian(res,pix_width,fx,fy,alpha)
    res[0,0] = 0.
    res = res.real
    return res

def calculate_resolution(image, detector_pixelsize, magnification, filterwidth=5, highfrq=2.0, nblfac=2.0):
    """
    Calculate the resolution in both row and column directions based on the given image.
    
    Parameters:
    - image: 2D numpy array representing the image.
    - detector_pixelsize: The size of each pixel in meters.
    - magnification: The magnification factor.
    - filterwidth: The width of the convolution filter (default: 5).
    - highfrq: The frequency threshold for resolution estimation (default: 2.0).
    - nblfac: The factor used for resolution estimation (default: 2).
    - ax: The axis object for plotting (optional, default: None).

    Returns:
    - row_resolution: The resolution in the row direction.
    - row_uns: The uncertainty in the row resolution.
    - col_resolution: The resolution in the column direction.
    - col_uns: The uncertainty in the column resolution.
    """
    
    # Define the region of interest (ROI) based on user input
    def defineroi_func():
        # Placeholder function for defining ROI, adjust as needed
        return np.s_[:], np.s_[:]  # Select all for now, replace with actual ROI selection
    
    yroi, xroi = defineroi_func()
    data = image[yroi, xroi]

    # Rows processing (zeilen)
    N = data.shape[1]
    k = 2 * np.pi / N * np.arange(-N // 2, N // 2)  # Sample values in k-space
    PowerFdata = np.abs(fftshift(fft(data, axis=1), axes=1)) ** 2
    PowerFdata = PowerFdata[:, k > 0]
    k = k[k > 0]

    # Convolution and resolution estimation for rows
    ConvPowerFdline = np.zeros_like(PowerFdata)
    filter = np.ones(filterwidth)  # Length should be odd
    filter = filter / np.sum(filter)
    xres_row = np.zeros(PowerFdata.shape[0])
    uxres_row = np.zeros(PowerFdata.shape[0])

    for is_ in range(PowerFdata.shape[0]):
        zw = convolve(PowerFdata[is_, :], filter, mode='same')
        ConvPowerFdline[is_, :] = zw
        nbl = np.mean(ConvPowerFdline[is_, k > highfrq])
        mink = k[np.min(np.where(ConvPowerFdline[is_, :] <= nblfac * nbl))]
        maxk = k[np.max(np.where(ConvPowerFdline[is_, :] >= nblfac * nbl))]
        if np.isnan(mink) or np.isnan(maxk):
            mink = np.inf
            maxk = np.inf
        kres = np.mean([mink, maxk])
        ukres = 0.5 * (maxk - mink)

        xres_row[is_] = 2 * np.pi / kres
        uxres_row[is_] = 2 * np.pi / kres ** 2 * ukres

    # Columns processing (spalten)
    N = data.shape[0]
    k = 2 * np.pi / N * np.arange(-N // 2, N // 2)
    PowerFdata = np.abs(fftshift(fft(data, axis=0), axes=0)) ** 2
    PowerFdata = PowerFdata[k > 0, :]
    k = k[k > 0]

    # Convolution and resolution estimation for columns
    ConvPowerFdline = np.zeros_like(PowerFdata)
    filter = np.ones(filterwidth)  # Length should be odd
    filter = filter / np.sum(filter)
    xres_col = np.zeros(PowerFdata.shape[1])
    uxres_col = np.zeros(PowerFdata.shape[1])

    for is_ in range(PowerFdata.shape[1]):
        zw = convolve(PowerFdata[:, is_], filter, mode='same')
        ConvPowerFdline[:, is_] = zw
        nbl = np.mean(ConvPowerFdline[k > highfrq, is_])
        mink = k[np.min(np.where(ConvPowerFdline[:, is_] <= nblfac * nbl))]
        maxk = k[np.max(np.where(ConvPowerFdline[:, is_] >= nblfac * nbl))]
        kres = np.mean([mink, maxk])
        ukres = 0.5 * (maxk - mink)

        xres_col[is_] = 2 * np.pi / kres
        uxres_col[is_] = 2 * np.pi / kres ** 2 * ukres

    # Calculate final row and column resolutions
    row_resolution = (np.mean(xres_row) * detector_pixelsize) / magnification
    row_unc = (np.std(xres_row) * detector_pixelsize) / magnification
    col_resolution = (np.mean(xres_col) * detector_pixelsize) / magnification
    col_unc = (np.std(xres_col) * detector_pixelsize) / magnification

    # Return results as a dictionary
    output = {
        "row_resolution [m]": row_resolution,
        "row_uncertainty": row_unc,
        "col_resolution": col_resolution,
        "col_uncertainty": col_unc
    }

    return output,row_resolution,row_unc,col_resolution,col_unc


def calculate_average_height(matrix):
    # Ensure the input is a numpy array
    matrix = np.array(matrix)
    
    # Step 1: Calculate the height (max - min) for each row
    row_heights = []
    for row in matrix:
        row_max = np.max(row)  # Maximum value in the row
        row_min = np.min(row)  # Minimum value in the row
        row_height = row_max - row_min  # Height as the difference
        row_heights.append(row_height)
    
    # Calculate the overall average height of rows
    overall_average_height_row = np.mean(row_heights)
    overall_std_height_row = np.std(row_heights)  # Standard deviation of row heights
    
    # Step 2: Calculate the height (max - min) for each column
    column_heights = []
    for col in range(matrix.shape[1]):  # Iterate through each column
        col_max = np.max(matrix[:, col])  # Maximum value in the column
        col_min = np.min(matrix[:, col])  # Minimum value in the column
        col_height = col_max - col_min  # Height as the difference
        column_heights.append(col_height)
    
    # Calculate the overall average height of columns
    overall_average_height_column = np.mean(column_heights)
    overall_std_height_column = np.std(column_heights)  # Standard deviation of column heights
    
    return (overall_average_height_row, overall_std_height_row,
         overall_average_height_column, overall_std_height_column)



from sklearn.cluster import AgglomerativeClustering
def calculate_average_cluster_height(matrix):
    # Ensure the input is a numpy array
    matrix = np.array(matrix)
    
    # Step 1: Cluster each row into two clusters and calculate the mean cluster difference
    row_cluster_differences = []
    for row in matrix:
        # Reshape the row to a 2D array for clustering (AgglomerativeClustering expects 2D input)
        row_reshaped = row.reshape(-1, 1)
        
        # Perform Agglomerative Clustering with 2 clusters
        clustering = AgglomerativeClustering(n_clusters=2).fit(row_reshaped)
        
        # Get the labels of the clusters (0 or 1)
        labels = clustering.labels_
        
        # Split the row into two clusters based on the labels
        cluster1 = row[labels == 0]
        cluster2 = row[labels == 1]
        
        # Calculate the difference between the two clusters' means
        cluster_diff = abs(np.median(cluster1) - np.median(cluster2))
        
        row_cluster_differences.append(cluster_diff)
    
    # Calculate the overall average cluster difference of rows
    overall_average_height_row = np.median(row_cluster_differences)
    overall_std_height_row = np.std(row_cluster_differences)  # Standard deviation of row cluster differences
    
    # Step 2: Cluster each column into two clusters and calculate the mean cluster difference
    column_cluster_differences = []
    for col in range(matrix.shape[1]):  # Iterate through each column
        col_reshaped = matrix[:, col].reshape(-1, 1)
        
        # Perform Agglomerative Clustering with 2 clusters
        clustering = AgglomerativeClustering(n_clusters=2).fit(col_reshaped)
        
        # Get the labels of the clusters (0 or 1)
        labels = clustering.labels_
        
        # Split the column into two clusters based on the labels
        cluster1 = matrix[:, col][labels == 0]
        cluster2 = matrix[:, col][labels == 1]
        
        # Calculate the difference between the two clusters' means
        cluster_diff = abs(np.median(cluster1) - np.median(cluster2))
        
        column_cluster_differences.append(cluster_diff)
    
    # Calculate the overall average cluster difference of columns
    overall_average_height_column = np.median(column_cluster_differences)
    overall_std_height_column = np.std(column_cluster_differences)  # Standard deviation of column cluster differences
    
    return (overall_average_height_row, overall_std_height_row,
         overall_average_height_column, overall_std_height_column)