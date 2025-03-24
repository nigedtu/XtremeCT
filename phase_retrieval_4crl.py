import os
# Change to your local directory:
os.chdir(r"C:\Users\nige\OneDrive - Danmarks Tekniske Universitet\Dokumenter\Postdoc\Projects\Phase retrieval\GitHub")



from phase_retrieval_functions import *
from optical_constants import oc
os.chdir(r"siemens_star")
#fnames = ['scan-0646.h5','scan-0647.h5','scan-0648.h5','scan-0649.h5','scan-0650.h5','scan-0651.h5','scan-0652.h5','scan-0653.h5','scan-0654.h5','scan-0655.h5'] # 4 CRL
fnames = ['scan-0646.h5','scan-0649.h5','scan-0650.h5','scan-0651.h5','scan-0652.h5'] # 4 CRL

df = 'scan-0594.h5' # dark field 4 CRL
ff = 'scan-0645.h5' # flat field 4 CRL

#First plot flatfield, and the gray value, raw projection, and the corrected projection
distances = [31, 32.5, 38.0, 44, 48.69] # Sample – focus distances: # mm
n = 2156 # number of pixels. or 2688
ndist=len(distances)
ntheta = 1




scans = []
# Get dark field data
with h5py.File(df, 'r') as file:
    orca_data= file['entry']['measurement']['orca']
    df = np.mean(orca_data[:, 0:n,0:n],axis=0) # dark field, 50 fra,es 1 sec exposure
dark0 = df
 
# Get flat field data
with h5py.File(ff, 'r') as file:
    orca_data= file['entry']['measurement']['orca']
    ff = np.mean(orca_data[:, 0:n,0:n], axis=0)-dark0 #  flat field field, 20 frames 1 sec exposure, subtracted df
rref = ff   
for i in range(len(fnames)):
    with h5py.File(fnames[i], 'r') as file:
        orca_data= file['entry']['measurement']['orca']
        #scans.append(np.ndarray.sum(orca_data[:,0:n,0:n],axis=0))
        scans.append(np.mean(orca_data[0:1,0:n,0:n],axis=0)-dark0) # mean of 10 frames, 1 sec exposure, subtracted darkfield
        
rdata=np.array(scans) # Corrected projection initialisation
rref[rref<0] = 0 # no negative values in flatfield
rdata[rdata<0]=0 # no negative values in raw projection
rdata/=(rref+1e-9) # subtract flatfield from rawprojection = corrected projection

for k in range(ndist):
    radius = 3
    threshold = 1400000000
    rdata[k] = remove_outliers(rdata[k], radius, threshold)
    #rref = remove_outliers(rref, radius, threshold)  

# Correct first image, so all CRL match
shifts = np.array([[8.19999981, -59.70000076]])
a = apply_shift([rdata[0]],-shifts,n)[0]# note first shift then magnification
matrix = np.array(a[a.shape[0]//2-n//2:a.shape[0]//2+n//2,a.shape[1]//2-n//2:a.shape[1]//2+n//2]  )
rdata[0] = matrix
#plt.imshow( matrix[:n-200,200:n])



'''Scale data using magnification calculated from distances'''
energy = 19.9  # [keV] x-ray energy    
PLANCK_CONSTANT = 4.135667696e-18  # [keV*s]
SPEED_OF_LIGHT = 299792458  # [m/s]
wavelength = PLANCK_CONSTANT * SPEED_OF_LIGHT / energy # [m]

detector_pixelsize = 0.55e-6 # m  
focusToDetectorDistance = 0.45086 # [m] 
sx0 = 0 # 3.7e-4# [m] motor offset from the focal spot
#z1 = np.array([4.584e-3,4.765e-3,5.488e-3,6.9895e-3])-sx0 # positions of the sample (1 position is enough when using several codes)
z1 = np.array(distances)*1e-3 #np.array([15.2e-3, 20.0e-3, 25.2e-3, 40e-3]) # m (focus and sample)
z1=z1[:ndist]
z2 = focusToDetectorDistance-z1 # propagation distance between the code and detector
magnifications = (z1+z2)/z1 # magnification when propagating from the sample to the code
norm_magnifications = magnifications/magnifications[0] # normalized magnifications
distances = (z1*z2)/(z1+z2) # propagation distances after switching from the point source wave to plane wave,
distances = distances*norm_magnifications**2 # scaled propagation distances due to magnified probes
voxelsize = detector_pixelsize/magnifications[0]  # object voxel size
plot_magnification = False

rdata_scaled = rdata.copy()
for k in range(ndist):    
    a = ndimage.zoom(rdata[k],1/norm_magnifications[k])
    rdata_scaled[k] = a[a.shape[0]//2-n//2:a.shape[0]//2+n//2,a.shape[1]//2-n//2:a.shape[1]//2+n//2]
    
if plot_magnification:
    # Plot raw projections
    fig, ax = plt.subplots(1, 1)
    X = np.array(scans)
    tracker = IndexTracker(ax, X)
    fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
    plt.show()
    
    # Plot corrected projections
    fig, ax = plt.subplots(1, 1)
    X = np.array(rdata)
    tracker = IndexTracker(ax, X)
    fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
    plt.show()
    
    for k in range(ndist):
        fig, axs = plt.subplots(1, 3, figsize=(12, 4))
        im=axs[0].imshow(rdata_scaled[0],cmap='gray',vmin = 0.9,vmax=1.2 )
        axs[0].set_title(f'rdata_scaled for dist {0}')
        fig.colorbar(im)
        im=axs[1].imshow(rdata_scaled[k],cmap='gray',vmin = 0.9,vmax=1.2 )
        axs[1].set_title(f'rdata_scaled for dist {k}')
        fig.colorbar(im)        
        im=axs[2].imshow(rdata_scaled[k]-rdata_scaled[0],cmap='gray',vmin =-0.1,vmax=0.1 )
        axs[2].set_title(f'difference')
        fig.colorbar(im) 
#%%
#%%
os.chdir(r"C:\Users\nige\OneDrive - Danmarks Tekniske Universitet\Dokumenter\PhD\XRR_Fitting_tool")
from optical_constants import oc
from unit_conversion import wavelength2energy, energy2wavelength, thickness2energy, energy2thickness, convert_pixel_size_to_actual_distance
energy = 19.9  # [keV] x-ray energy   
density_Au = 19.3 # g/cm**3
density_SiN = 3.44 # g/cm**3

elements_Au = ['Au'] # 
elements_SiN = ['Si','N'] # from CXRO transmission calculator

z_Au = 1600 # Thickness of gold [nm]    or 500 Å  = 0.05 mu, maybe 1.6 mu # plus minus 10 %
z_SiN = 1000 # Thickness of silicon [nm]

number_of_atoms_Au = [1]
number_of_atoms_SiN = [3,4]
oc_source = "oc_source/NIST/"

wavelength = energy2wavelength(energy*10**3) # nm

n_Au = oc(wavelength,density_Au,number_of_atoms_Au,elements_Au, oc_source)     
beta = n_Au.imag
delta = n_Au.real

n_SiN = oc(wavelength,density_SiN,number_of_atoms_SiN,elements_SiN, oc_source)    

phi_Au = (-2*np.pi*(1-n_Au.real)*(z_Au*1e-09))/(wavelength*1e-9) #- (n_Au.imag*z_Au) #  [m]
phi_SiN = (-2*np.pi*(1-n_SiN.real)*(z_SiN*1e-09))/(wavelength*1e-9) #- (n_SiN.imag*z_SiN)#   [m]
print("Phase difference for Au:"+str(phi_Au))
#%%

# Plot flatfield
fs = 15
fig, axs = plt.subplots(1,2,figsize=(21, 6))
im=axs[0].imshow(rref)#,cmap='gray',vmax=0.05,vmin=-0.05) # cmap='Greys',
axs[0].set_ylabel("Distance (pixels)", fontsize = fs)
axs[0].set_xlabel("Distance (pixels)", fontsize = fs)

divider = make_axes_locatable(axs[0])
cax = divider.append_axes("right", size="5%", pad=0.05)  # Adjust size and padding as needed
fig.colorbar(im, cax=cax) # Create the colorbar in the new axis
#%% Apply shift using Fourier transformed cross-correlation method, from Holotomo toolbox
plot_shifts = True
shifts_dist = np.zeros([ndist,ntheta,2],dtype='float32')
for k in range(ndist):
    shifts_dist[k] = registration_shift([rdata_scaled[k]],[rdata_scaled[0]],upsample_factor=10)
    print(f'{k}: {shifts_dist[k,0]}')
shifts = shifts_dist*norm_magnifications[:,np.newaxis,np.newaxis]


rdata_scaled = rdata.copy()
for k in range(ndist):    
    a = apply_shift(rdata[k:k+1],-shifts[k:k+1,0],n)[0]# note first shift then magnification
    a = ndimage.zoom(a,1/norm_magnifications[k])
    rdata_scaled[k] = a[a.shape[0]//2-n//2:a.shape[0]//2+n//2,a.shape[1]//2-n//2:a.shape[1]//2+n//2]        

if plot_shifts:
    med = np.median(rdata_scaled[0])
    for k in range(ndist):
        fig, axs = plt.subplots(1, 3, figsize=(12, 4))
        im=axs[0].imshow(rdata_scaled[0],cmap='gray',vmin = med-0.1,vmax=med+0.1 )
        axs[0].set_title(f'First image')
        #fig.colorbar(im)
        im=axs[1].imshow(rdata_scaled[k],cmap='gray',vmin = med-0.1,vmax=med+0.1 )
        axs[1].set_title(f'Scaled and shifted for dist {k}')
        #fig.colorbar(im)        
        im=axs[2].imshow(rdata_scaled[k]-rdata_scaled[0],cmap='gray',vmin =-0.1,vmax=0.1 )
        axs[2].set_title(f'Difference')
    
        #fig.colorbar(im)  
        
rdata_scaled= rdata_scaled[:,:n-200,200:n] # Cut away the outer part, so all images are aligned. 
#%% Plot all scaled 
n = np.shape(rdata_scaled[0])[0]
real_world_pixel_size = detector_pixelsize / magnifications[0]
x_mu = np.arange(0, n) * (real_world_pixel_size*1e6) # micrometer

fig, ax = plt.subplots(1, 1)
X = np.array(rdata_scaled)
tracker = IndexTracker(ax, X)
fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
plt.show()



#%% Make all CTFPurePhase for five distances
'''
alpha = [1e-2,1e-2,1e-2,1e-2,1e-2,1e-2,1e-2,1e-2,1e-2,1e-2] # 
CTFrec = []
for k in range(1,ndist+1):

    rads = rdata_scaled[:k]
    distances_rec = (distances/norm_magnifications**2)[:k]
    fx = np.fft.fftfreq(n,d=voxelsize)
    [fx,fy] = np.meshgrid(fx,fx)

    wlen = PLANCK_CONSTANT * SPEED_OF_LIGHT/energy
    recCTFPurePhase = CTFPurePhase(rads, wlen, distances_rec, fx, fy, alpha[k-1])
    CTFrec.append(recCTFPurePhase)
    
# Plot all CTFpurephase
fig, ax = plt.subplots(1, 1)
X = np.array(CTFrec)
tracker = IndexTracker(ax, X)
fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
plt.show() 
'''
#%% Make all CTFPurePhase for five distances

rads = rdata_scaled
distances_rec = (distances/norm_magnifications**2)
fx = np.fft.fftfreq(n,d=voxelsize)
[fx,fy] = np.meshgrid(fx,fx)
alpha = 1e-4#1e-24
wlen = PLANCK_CONSTANT * SPEED_OF_LIGHT/energy
recCTFPurePhase = CTFPurePhase(rads, wlen, distances_rec, fx, fy, alpha)
#recCTFPurePhase = CTFPurePhase_new(rads, wlen, distances_rec, delta, beta, fx, fy, Rm, alpha)
                        
fs = 15
fig, ax = plt.subplots(1,1)
im=ax.imshow(recCTFPurePhase,extent=[x_mu.min(), x_mu.max(), x_mu.min(), x_mu.max()])
ax.set_ylabel(r"Distance ($\mathrm{\mu}$m)", fontsize = fs)
ax.set_xlabel(r"Distance ($\mathrm{\mu}$m)", fontsize = fs)
# Add the colorbar
cbar = fig.colorbar(im, ax=ax)
# Set the colorbar label
cbar.set_label('Phase (rads)', fontsize=fs, labelpad=-40)
# Optionally, you can set the ticks on the colorbar to show the min and max values
cbar.set_ticks([recCTFPurePhase.min(), recCTFPurePhase.max()])

result = calculate_resolution(recCTFPurePhase, detector_pixelsize, magnifications[0])
print('Resolution [nm]:'+str(((result[1]+result[3])/2)*10**9))
phase = calculate_average_height(recCTFPurePhase)
print('Phase='+str(phase))


#%% Make all CTF for five distances
rads = rdata_scaled
distances_rec = (distances/norm_magnifications**2)
#Rm = np.zeros((n, n, len(distances_rec))) 
wlen = PLANCK_CONSTANT * SPEED_OF_LIGHT/energy
fx = np.fft.fftfreq(n,d=voxelsize)
[fx,fy] = np.meshgrid(fx,fx)
alpha = 1e-2#1
Rm = np.zeros((n, n, len(distances_rec))) 
degCoh = np.ones((n, n, len(distances_rec)))
OptTrnFunc = np.ones((n, n))
for i in range(0,len(distances_rec)):
    Rm[:,:,i] = degCoh[:,:,i] * OptTrnFunc
    
CTF_1 =CTF(rads, wlen, distances_rec, fx, fy, Rm, alpha)

fig, ax = plt.subplots(1,1)
im=ax.imshow(CTF_1,extent=[x_mu.min(), x_mu.max(), x_mu.min(), x_mu.max()])
ax.set_ylabel(r"Distance ($\mathrm{\mu}$m)", fontsize = fs)
ax.set_xlabel(r"Distance ($\mathrm{\mu}$m)", fontsize = fs)
# Add the colorbar
cbar = fig.colorbar(im, ax=ax)
# Set the colorbar label
cbar.set_label('Phase (rads)', fontsize=fs, labelpad=-40)
# Optionally, you can set the ticks on the colorbar to show the min and max values
cbar.set_ticks([CTF_1.min(), CTF_1.max()])

result = calculate_resolution(CTF_1, detector_pixelsize, magnifications[0])
print('Resolution [nm]:'+str(((result[1]+result[3])/2)*10**9))

phase = calculate_average_height(CTF_1)
print('Phase='+str(phase))

#%% Make all homoCTF for five distances
# img_fft = pyfftw.interfaces.numpy_fft.fft2(image)
#Rm = np.zeros((img_fft.shape[0], img_fft.shape[1], len(dists))) 
rads = rdata_scaled
Rm = np.zeros((n, n, len(distances_rec))) 
degCoh = np.ones((n, n, len(distances_rec)))
OptTrnFunc = np.ones((n, n))
for i in range(0,len(distances_rec)):
    Rm[:,:,i] = degCoh[:,:,i] * OptTrnFunc
    
wlen = PLANCK_CONSTANT * SPEED_OF_LIGHT/energy
fx = np.fft.fftfreq(n,d=voxelsize)
[fx,fy] = np.meshgrid(fx,fx)
alpha = 1e-9
beta = n_Au.imag
delta = 1-n_Au.real

CTF_homo= homoCTF(rads, wlen, distances_rec, delta, beta, fx, fy, Rm, alpha)
    
fig, ax = plt.subplots(1,1)
im=ax.imshow(CTF_homo,extent=[x_mu.min(), x_mu.max(), x_mu.min(), x_mu.max()])
ax.set_ylabel(r"Distance ($\mathrm{\mu}$m)", fontsize = fs)
ax.set_xlabel(r"Distance ($\mathrm{\mu}$m)", fontsize = fs)
# Add the colorbar
cbar = fig.colorbar(im, ax=ax)
# Set the colorbar label
cbar.set_label('Phase (rads)', fontsize=fs, labelpad=-30)
# Optionally, you can set the ticks on the colorbar to show the min and max values
cbar.set_ticks([CTF_homo.min(), CTF_homo.max()])
# Move the ticks to the top and bottom of the colorbar
cbar.ax.tick_params(labeltop=True, labelbottom=True)


result = calculate_resolution(CTF_homo, detector_pixelsize, magnifications[0])
print('Resolution [nm]:'+str(((result[1]+result[3])/2)*10**9))

#phase = calculate_average_height(CTF_homo)
#print('Max Min Phase='+str(phase))

#result_2 =calculate_average_cluster_height(CTF_homo)
#print('Cluster Phase='+str(result_2))


phase_map_area = CTF_homo[200:1750,200:1000]
#plt.imshow(phase_map_area)
result = calculate_resolution(phase_map_area , detector_pixelsize, magnifications[0])
print('Resolution, area 1 [nm]:'+str(((result[1]+result[3])/2)*10**9))
phase = calculate_average_height(phase_map_area)
print('Max Min Phase, area 1='+str(phase))
#result_2 =calculate_average_cluster_height(phase_map_area)
#print('Cluster Phase area ='+str(result_2))

new_phase = calculate_matrix_height_difference(phase_map_area)
print(new_phase)


#%% Plot line profile of phase map at three positions. Can we see the effect of pre-focus?

# First spot
image = np.copy(CTF_homo)
matrix = np.copy(image)
start_point = (391,357) # (y,x)  (354,356)
end_point = ( 516,362) #(y,x)
line_width = 91 # Specify the width of the line
#start_point = (430,252) # (y,x)
#end_point = ( 792,272) #(y,x)
#line_width = 91 # Specify the width of the line
# Call the function to draw parallel lines
line_points, x, y = draw_parallel_lines(matrix, start_point, end_point, line_width)
line_values= np.mean(image[y[:],x[:]],axis=0)

# Crosshair properties
line_length = 250          # Length of the crosshair lines
rotation_angle = -2       # Rotation angle in degrees
center_row_1 = 463           # Define the center row position of the crosshair
center_col_1 = 360           # Define the center column position of the crosshair


# Define the coordinates of the crosshair before rotation
horizontal_line = [(-line_length / 2, 0), (line_length / 2, 0)]
vertical_line = [(0, -line_length / 2), (0, line_length / 2)]
# Apply rotation to the crosshair lines
rotated_horizontal_line = [rotate(x, y, rotation_angle) for x, y in horizontal_line]
rotated_vertical_line = [rotate(x, y, rotation_angle) for x, y in vertical_line]



fs = 15
fig, axs = plt.subplots(1,3,figsize=(21, 6))
im=axs[0].imshow(image)#,cmap='gray',vmax=0.05,vmin=-0.05) # cmap='Greys',
axs[0].set_title(f'Reconstruction from {k+1} distances')
axs[0].plot(x,y, 'r.' )
axs[0].set_ylabel("Distance (pixels)", fontsize = fs)
axs[0].set_xlabel("Distance (pixels)", fontsize = fs)

divider = make_axes_locatable(axs[0])
cax = divider.append_axes("right", size="5%", pad=0.05)  # Adjust size and padding as needed
fig.colorbar(im, cax=cax) # Create the colorbar in the new axis


im=axs[1].imshow(image)#,cmap='gray',vmax=0.05,vmin=-0.05)
axs[1].set_ylim([850, 350])
axs[1].set_xlim([50, 500])
x_spot1,y_spot1 = x,y
axs[1].plot(x,y, 'r.' )
axs[1].set_ylabel("Distance (pixels)", fontsize = fs)
axs[1].set_xlabel("Distance (pixels)", fontsize = fs)
# Plot the rotated horizontal line of the crosshair
axs[1].plot([center_col_1 + rotated_horizontal_line[0][0], center_col_1 + rotated_horizontal_line[1][0]],
         [center_row_1 + rotated_horizontal_line[0][1], center_row_1 + rotated_horizontal_line[1][1]],
         color='blue', linewidth=2)
# Plot the rotated vertical line of the crosshair
axs[1].plot([center_col_1 + rotated_vertical_line[0][0], center_col_1 + rotated_vertical_line[1][0]],
         [center_row_1 + rotated_vertical_line[0][1], center_row_1 + rotated_vertical_line[1][1]],
         color='blue', linewidth=2)
divider = make_axes_locatable(axs[1])
cax = divider.append_axes("right", size="5%", pad=0.05)  # Adjust size and padding as needed
fig.colorbar(im, cax=cax) # Create the colorbar in the new axis


data = line_values
data_reshaped = data.reshape(-1, 1)# Reshape the data for K-Means
kmeans = KMeans(n_clusters=2, random_state=0)# Fit K-Means with 2 clusters
kmeans.fit(data_reshaped)
clustered_data = kmeans.cluster_centers_[kmeans.labels_].flatten()# Replace each data point with the center of its cluster
x = np.arange(len(data)) # Generate the indices for the x-axis
slope_data = np.diff(data) # Calculate the first derivative (slope) of the data
# Find the indices where the slope is consistently positive (e.g., upward trend)
start_idx = 105 # np.where(slope_data < 0)[0][0]  # Start of upward trend
end_idx = 120 #  np.where(slope_data > 0)[0][-1]   # End of upward trend
slope, intercept = np.polyfit(x[start_idx:end_idx+1], data[start_idx:end_idx+1], 1) # Fit a linear model to the selected region
linear_fit = slope * x[start_idx:end_idx+1] + intercept# Calculate the linear fit line for the selected region
# Print the start and end indices and the calculated slope
#print(f"Start Index: {start_idx}, End Index: {end_idx}")
print(f"Linear slope in selected region: {slope}")
line_values_spot1 = line_values
im=axs[2].plot(line_values,linewidth=2,label='Data', color='red')
axs[2].plot(clustered_data, label='Two-Value Data', color='black')# Plot the original data

phase_diff=np.abs(np.max(line_values))+np.abs(np.min(line_values))
axs[2].text(150, -0.2, 'Peak Phase Difference:'+ str(round(phase_diff, 2)), fontsize=15, color='Black')
# Plot the linear fit for the selected region
axs[2].plot(x[start_idx:end_idx+1], linear_fit, label='Linear Fit (Selected Region)', color='blue') # Highlight the selected region
axs[2].axvspan(start_idx, end_idx, color='yellow', alpha=0.3)

axs[2].set_ylabel("Phase", fontsize = fs)
axs[2].set_xlabel("Distance (pixels)", fontsize = fs)
#axs[2].set_ylim([-0.3, 0.3])
axs[2].grid()
plt.tight_layout()
axs[2].legend(fontsize = 12)
plt.show()

#%%
# second spot
image = np.copy(CTF_homo)
matrix = np.copy(image)
start_point = (572,740) # (y,x)
end_point = ( 603,606) #(y,x)
line_width = 21 # Specify the width of the line
# Call the function to draw parallel lines
line_points, x, y = draw_parallel_lines(matrix, start_point, end_point, line_width)
line_values= np.mean(image[y[:],x[:]],axis=0)

# Crosshair properties
line_length = 300          # Length of the crosshair lines
rotation_angle = -13       # Rotation angle in degrees
center_row_1 = 670           # Define the center row position of the crosshair
center_col_1 = 825           # Define the center column position of the crosshair


# Define the coordinates of the crosshair before rotation
horizontal_line = [(-line_length / 2, 0), (line_length / 2, 0)]
vertical_line = [(0, -line_length / 2), (0, line_length / 2)]
# Apply rotation to the crosshair lines
rotated_horizontal_line = [rotate(x, y, rotation_angle) for x, y in horizontal_line]
rotated_vertical_line = [rotate(x, y, rotation_angle) for x, y in vertical_line]



fs = 15
fig, axs = plt.subplots(1,3,figsize=(21, 6))
im=axs[0].imshow(image)#,cmap='gray',vmax=0.05,vmin=-0.05) # cmap='Greys',
axs[0].set_title(f'Reconstruction from {k+1} distances')
axs[0].plot(x,y, 'r.' )
axs[0].set_ylabel("Distance (pixels)", fontsize = fs)
axs[0].set_xlabel("Distance (pixels)", fontsize = fs)

divider = make_axes_locatable(axs[0])
cax = divider.append_axes("right", size="5%", pad=0.05)  # Adjust size and padding as needed
fig.colorbar(im, cax=cax) # Create the colorbar in the new axis


im=axs[1].imshow(image)#,cmap='gray',vmax=0.05,vmin=-0.05)
axs[1].set_ylim([900, 400])
axs[1].set_xlim([550, 1050])
x_spot2,y_spot2 = x,y
axs[1].plot(x,y, 'r.' )
axs[1].set_ylabel("Distance (pixels)", fontsize = fs)
axs[1].set_xlabel("Distance (pixels)", fontsize = fs)
# Plot the rotated horizontal line of the crosshair
axs[1].plot([center_col_1 + rotated_horizontal_line[0][0], center_col_1 + rotated_horizontal_line[1][0]],
         [center_row_1 + rotated_horizontal_line[0][1], center_row_1 + rotated_horizontal_line[1][1]],
         color='blue', linewidth=2)
# Plot the rotated vertical line of the crosshair
axs[1].plot([center_col_1 + rotated_vertical_line[0][0], center_col_1 + rotated_vertical_line[1][0]],
         [center_row_1 + rotated_vertical_line[0][1], center_row_1 + rotated_vertical_line[1][1]],
         color='blue', linewidth=2)
divider = make_axes_locatable(axs[1])
cax = divider.append_axes("right", size="5%", pad=0.05)  # Adjust size and padding as needed
fig.colorbar(im, cax=cax) # Create the colorbar in the new axis


data = line_values
data_reshaped = data.reshape(-1, 1)# Reshape the data for K-Means
#kmeans = KMeans(n_clusters=2, random_state=0)# Fit K-Means with 2 clusters
#kmeans.fit(data_reshaped)
#clustered_data = kmeans.cluster_centers_[kmeans.labels_].flatten()# Replace each data point with the center of its cluster
#x = np.arange(len(data)) # Generate the indices for the x-axis
#slope_data = np.diff(data) # Calculate the first derivative (slope) of the data
# Find the indices where the slope is consistently positive (e.g., upward trend)
start_idx = 242 # np.where(slope_data < 0)[0][0]  # Start of upward trend
end_idx = 260 #  np.where(slope_data > 0)[0][-1]   # End of upward trend
#slope, intercept = np.polyfit(x[start_idx:end_idx+1], data[start_idx:end_idx+1], 1) # Fit a linear model to the selected region
#linear_fit = slope * x[start_idx:end_idx+1] + intercept# Calculate the linear fit line for the selected region
# Print the start and end indices and the calculated slope
#print(f"Start Index: {start_idx}, End Index: {end_idx}")
#print(f"Linear slope in selected region: {slope}")
line_values_spot2 = line_values
im=axs[2].plot(line_values,linewidth=2,label='Data', color='red')
phase_diff=np.abs(np.max(line_values))+np.abs(np.min(line_values))
axs[2].text(150, -0.2, 'Peak Phase Difference:'+ str(round(phase_diff, 2)), fontsize=15, color='Black')

#axs[2].plot(clustered_data, label='Two-Value Data', color='black')# Plot the original data

# Plot the linear fit for the selected region
#axs[2].plot(x[start_idx:end_idx+1], linear_fit, label='Linear Fit (Selected Region)', color='blue') # Highlight the selected region
#axs[2].axvspan(start_idx, end_idx, color='yellow', alpha=0.3)

axs[2].set_ylabel("Phase", fontsize = fs)
axs[2].set_xlabel("Distance (pixels)", fontsize = fs)
#axs[2].set_ylim([-0.42, 0.12])
axs[2].grid()
plt.tight_layout()
axs[2].legend(fontsize = 12)
plt.show()

# Plot the original data
#axs[2].plot(clustered_data, label='Two-Value Data', color='black')
# Plot the linear fit for the selected region
#axs[2].plot(x[start_idx:end_idx+1], linear_fit, label='Linear Fit (Selected Region)', color='blue') # Highlight the selected region
#plt.axvspan(start_idx, end_idx, color='yellow', alpha=0.3)


#%%
# Inner circle
circle_info =[]
image = np.copy(CTF_homo)
matrix = np.copy(CTF_homo)
center_point = (393,1458 ) # y, x
num_circles = 1
radius = 115


fs = 15
fig, axs = plt.subplots(1, 2, figsize=(14, 6))
im=axs[0].imshow(image)
circle_info = draw_concentric_circles(matrix, center_point, radius, num_circles)
axs[0].plot(circle_info[:,0],circle_info[:,1],'r', linewidth=2)#,cmap='Greys')#,vmax=0.05,vmin=-0.05) # cmap='Greys',
axs[0].plot(x_spot1,y_spot1, 'r.' )
axs[0].plot(x_spot2,y_spot2, 'r.' )
axs[1].plot(circle_info[:,2],linewidth=2,label='Data', color='red')



#%%
# Create a figure and define a gridspec layout
fig = plt.figure(figsize=(10, 10))

# Create a grid with 1 column and 4 rows, where the first row takes up 1 column and spans 3 rows
gs = fig.add_gridspec(4, 2)

# First image on the left, spanning rows 0 to 2 in the first column
ax1 = fig.add_subplot(gs[0:3, 0])
ax1.imshow(CTF_homo)
ax1.plot(circle_info[:,0],circle_info[:,1],'r', linewidth=2)#,cmap='Greys')#,vmax=0.05,vmin=-0.05) # cmap='Greys',
ax1.plot(x_spot1,y_spot1, 'r.' )
ax1.plot(x_spot2,y_spot2, 'r.' )
ax1.text(335, 325, "Line 1", color='white', ha='center', va='center', fontsize=15)
ax1.text(683, 680, "Line 2", color='white', ha='center', va='center', fontsize=15)
ax1.text(1449, 589, "Ring", color='white', ha='center', va='center', fontsize=15)
#ax1.set_title('Image 1')
#ax1.axis('off')

# Second image (on the right side, top)
ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(circle_info[:,2],linewidth=2,label='Data', color='red')
ax2.set_ylabel(r"$\tilde{\phi}(f)$", fontsize = fs)
ax2.set_xlabel("Line distance (degree)", fontsize = fs)
ax2.text(275, 5.45, "Ring", color='black', ha='center', va='center', fontsize=15)

ax2.set_ylim(4.58, 5.6)
ax2.grid()
#ax2.set_title('Image 2')
#ax2.axis('off')

# Third image (on the right side, middle)
ax3 = fig.add_subplot(gs[1, 1])
ax3.plot(line_values_spot1[7:120],linewidth=2,label='Data', color='red')
ax3.set_ylabel(r"$\tilde{\phi}(f)$", fontsize = fs)
ax3.set_xlabel("Line distance (pixels)", fontsize = fs)
ax3.text(90, 4.6, "Line 1", color='black', ha='center', va='center', fontsize=15)
ax3.set_ylim(4.38, 5.65)


ax3.grid()
#ax3.set_title('Image 3')
#ax3.axis('off')

# Fourth image (on the right side, bottom)
ax4 = fig.add_subplot(gs[2, 1])
ax4.plot(line_values_spot2[:120-7],linewidth=2,label='Data', color='red')
ax4.set_ylabel(r"$\tilde{\phi}(f)$", fontsize = fs)
ax4.set_xlabel("Line distance (pixels)", fontsize = fs)
ax4.text(90, 4.7, "Line 2", color='black', ha='center', va='center', fontsize=15)
ax4.set_ylim(4.5, 5.65)

ax4.grid()
#ax4.set_title('Image 4')
#ax4.axis('off')

# Adjust layout to prevent overlap
plt.tight_layout()

# Show the plot
plt.show()
