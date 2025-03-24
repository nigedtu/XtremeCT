import numpy as np
import matplotlib.pyplot as plt

# First plot flatfield, and the gray value, raw projection, and the corrected projection
#distances = [31, 31.2, 31.5, 32.5, 38.0, 44, 48.69, 50, 52, 54.1] # Sample – focus distances:
distances = [31, 32.5, 38.0, 44, 48.69] # Sample – focus distances: # mm

    
    
    
    
    
PLANCK_CONSTANT = 4.135667696e-18  # [keV*s]
SPEED_OF_LIGHT = 299792458  # [m/s]

ndist = len(distances)
n = 2156 # number of pixels. or 2688detector_pixelsize = 0.55e-6 # m  #NIS LOOK HERE 220824. 
energy = 19.9  # [keV] x-ray energy    
wavelength = PLANCK_CONSTANT * SPEED_OF_LIGHT / energy # [m]
focusToDetectorDistance = 0.45086 # [m] 
detector_pixelsize = 0.55e-6  # m  3.84e-8   0.55e-6   3.78e-8


sx0 = 0 # 3.7e-4# [m] motor offset from the focal spot
#z1 = np.array([4.584e-3,4.765e-3,5.488e-3,6.9895e-3])-sx0 # positions of the sample (1 position is enough when using several codes)
z1 = np.array(distances)*1e-3 #np.array([15.2e-3, 20.0e-3, 25.2e-3, 40e-3]) # m (focus and sample)

z1=z1[:ndist]
z2 = focusToDetectorDistance-z1 # propagation distance between the code and detector
magnifications = (z1+z2)/z1 # magnification when propagating from the sample to the code
norm_magnifications = magnifications/magnifications[0] # normalized magnifications
distances = (z1*z2)/(z1+z2) # propagation distances after switching from the point source wave to plane wave,
distances = distances*norm_magnifications**2 # scaled propagation distances due to magnified probes [m]


voxelsize = detector_pixelsize/magnifications[0]  # object voxel size

    
fx = np.fft.fftfreq(n,d=voxelsize)
#[fx,fy] = np.meshgrid(fx,fx)
#rf=np.sqrt(fx**2+fy**2)
#rf = np.sum(rf,axis=0)
taylorExp =[]
wlen = PLANCK_CONSTANT * SPEED_OF_LIGHT/energy # [m] 19.9 keV
for k in range(0,ndist):
    distances_rec = (distances/norm_magnifications**2)#[:k]
    

for k in range(len(distances_rec)):   
    taylorExp.append( (np.sin(np.pi*wlen*distances_rec[k]*fx**2)**2))#(fx**2+fy**2)) 
    
fmin = 1/(np.sqrt(2*wlen*max(distances_rec))) # from Optimization of phase contrast imaging using hard x rays S. Zabler

fmax = 1/(2*detector_pixelsize) # from Optimization of phase contrast imaging using hard x rays S. Zabler
print(fmin, fmax)
#%%
#for j in range(0, len(dists)): 
#dists = dists []
#taylorExp = np.sin(np.pi*wlen*dists[j]*fx**2)#(fx**2+fy**2)) 
     
#%%
fs=15
lw=3
xx=75
fig, (ax1) = plt.subplots(nrows=1, ncols=1,figsize=(6,5))
fig.tight_layout(pad=3.0)
ax1.plot(fx[0:xx],taylorExp[0][0:xx],'--', color='red',linewidth=1,label=r'$D_1$')
ax1.plot(fx[0:xx],taylorExp[1][0:xx],'--', color='black',linewidth=1,label=r'$D_2$')

ax1.plot(fx[0:xx],taylorExp[2][0:xx],'--', color='black',linewidth=1,label=r'$D_3$')
ax1.plot(fx[0:xx],taylorExp[3][0:xx],'--', color='black',linewidth=1,label=r'$D_4$')
ax1.plot(fx[0:xx],taylorExp[4][0:xx],'--', color='blue',linewidth=1,label=r'$D_5$')
#ax1.plot(fx[0:700],taylorExp[6][0:700],'--', color='black',linewidth=1,label='Data')
#ax1.plot(fx[0:700],taylorExp[8][0:700],'--', color='black',linewidth=1,label='Data')
#ax1.plot(fx[0:xx],taylorExp[9][0:xx],'--', color='blue',linewidth=1,label='D_9')
ax1.plot(fx[0:xx],(np.sum(taylorExp,axis=0)/np.max(np.sum(taylorExp,axis=0)))[0:xx],linewidth=lw,label=r'$∑ D_1 - D_5$', color='black')
#∑ (D_1 - D_5)

plt.axvline(fmin, color = 'blue', label = r'$f_{min}$')
plt.axvline(fmax, color = 'red', label = r'$f_{max}$')


#ax1.set_ylabel("Counts", fontsize = fs)
ax1.set_ylabel(r"$\sin(\pi \lambda D f^2)$", fontsize = fs)
ax1.set_xlabel("f (m$^{-1}$)", fontsize = fs)
plt.legend()
ax1.set_ylim(-0.02,1.02)
ax1.set_xlim(0.0,1.0*1e6)
ax1.grid()