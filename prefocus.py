import os
# Change to your local directory:
os.chdir(r"C:\Users\nige\OneDrive - Danmarks Tekniske Universitet\Dokumenter\Postdoc\Projects\Phase retrieval\GitHub")


from phase_retrieval_functions import *
from optical_constants import oc
os.chdir(r"siemens_star")



fnames_dist0 = ['scan-0657.h5','scan-0646.h5','scan-0635.h5','scan-0623.h5','scan-0612.h5'] # dist 1 for all CRL
ffs= ['scan-0656.h5','scan-0645.h5','scan-0634.h5','scan-0622.h5','scan-0611.h5'] # Flatfields
df = 'scan-0594.h5' # dark fields


n = 2156 # 
ndist = 5
ntheta = 1

rdata_dist10=[]
with h5py.File(df, 'r') as file:
    orca_data= file['entry']['measurement']['orca']
    df = np.mean(orca_data[:, 0:n,0:n],axis=0) # dark field, 50 fra,es 1 sec exposure
dark0 = df



all_ff=[]
for i in range(len(fnames_dist0)):
    scans = []
    # Get flat field data
    with h5py.File(ffs[i], 'r') as file:
        orca_data= file['entry']['measurement']['orca']
        ff = np.mean(orca_data[0:10, 0:n,0:n], axis=0)-dark0 #  flat field field, 20 frames 1 sec exposure, subtracted df
    
    all_ff.append(ff)
    rref = ff   
    
    with h5py.File(fnames_dist0[i], 'r') as file:
        orca_data= file['entry']['measurement']['orca']
        #scans.append(np.ndarray.sum(orca_data[:,0:n,0:n],axis=0))
        #scans.append(np.mean(orca_data[:,0:n,0:n],axis=0)-dark0) # mean of 10 frames, 1 sec exposure, subtracted darkfield
        scans.append(np.mean(orca_data[6:7,0:n,0:n],axis=0)-dark0) # mean of 10 frames, 1 sec exposure, subtracted darkfield

    rdata=np.array(scans) # Corrected projection initialisation
    rref[rref<0] = 0 # no negative values in flatfield
    rdata[rdata<0]=0 # no negative values in raw projection
    rdata/=(rref+1e-9) # subtract flatfield from rawprojection = corrected projection



    rdata_dist10.append(rdata[0])
#%%
# Summing rows and columns
fs=15
summed_rows = np.sum(all_ff, axis=2)  # Summing along the columns (axis 2)
summed_columns = np.sum(all_ff, axis=1)  # Summing along the rows (axis 1)

# Define the labels for the legend in the reversed order
labels = ['18 lenslets', '12 lenslets', '8 lenslets', '4 lenslets', '0 lenslets']

# Plot 1: Summed rows for all 5 flatfields
plt.figure(figsize=(7, 6))
for i in range(5):
    plt.plot(summed_rows[4-i], label=labels[i], marker='o', markersize=2)

#plt.title('Summed Rows of Flatfield')
plt.xlabel('Distance (pixels)', fontsize = fs)
plt.ylabel('Intensity', fontsize = fs)
plt.legend(fontsize =15)
plt.yscale('log')  # Set the y-axis to logarithmic scale
plt.grid(True, which="both", ls="--")  # Show grid for both linear and log scale
plt.tight_layout()
plt.show()

# Plot 2: Summed columns for all 5 flatfields
plt.figure(figsize=(7, 6))
for i in range(5):
    plt.plot(summed_columns[4-i], label=labels[i], marker='o', markersize=2)

#plt.title('Summed Columns of Flatfield')
plt.xlabel('Distance (pixels)', fontsize = fs)
plt.ylabel('Intensity', fontsize = fs)
plt.legend(fontsize =15)
plt.yscale('log')  # Set the y-axis to logarithmic scale
plt.grid(True, which="both", ls="--")  # Show grid for both linear and log scale
plt.tight_layout()
plt.show()

# Sum for 0 lenslets and 18 lenslets (indices 4 and 0 respectively in the reversed order)
summed_rows_18 = summed_rows[4]  # 0 lenslets
summed_rows_0 = summed_rows[0]  # 18 lenslets
summed_columns_18 = summed_columns[4]  # 0 lenslets
summed_columns_0 = summed_columns[0]  # 18 lenslets

# Calculate the average increase factor for rows and columns
avg_increase_rows = np.mean(summed_rows_18 / summed_rows_0)
avg_increase_columns = np.mean(summed_columns_18 / summed_columns_0)

# Print the results
print(f"Average factor increase in rows from 0 lenslets to 18 lenslets: {avg_increase_rows:.6f}")
print(f"Average factor increase in columns from 0 lenslets to 18 lenslets: {avg_increase_columns:.6f}")


#%%
radius = 4
threshold = 300
rdata_dist10[0] = remove_outliers(rdata_dist10[0], radius, threshold)
#rref = remove_outliers(rref, radius, threshold)  
#rdata_dist10[0] = rdata[0] 
    
#%%
fig, ax = plt.subplots(1, 1)
X = np.array(rdata_dist10)
tracker = IndexTracker(ax, X)
fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
plt.show()


#%%
norm_magnifications = np.array([1,1,1,1,1])
shifts_dist = np.zeros([ndist,ntheta,2],dtype='float32')
for k in range(ndist):
    shifts_dist[k] = registration_shift([rdata_dist10[k]],[rdata_dist10[0]],upsample_factor=10)
    print(f'{k}: {shifts_dist[k,0]}')
    
    
shifts = shifts_dist*norm_magnifications[:,np.newaxis,np.newaxis]



rdata_scaled = rdata_dist10.copy()
for k in range(ndist):    
    a = apply_shift(rdata_dist10[k:k+1],-shifts[k:k+1,0],n)[0]# note first shift then magnification
    a = ndimage.zoom(a,1/norm_magnifications[k])
    rdata_scaled[k] = a[a.shape[0]//2-n//2:a.shape[0]//2+n//2,a.shape[1]//2-n//2:a.shape[1]//2+n//2]        


# Plot the corrected projections that have been alligned. 

matrix = np.array(rdata_scaled)
rdata_reshaped = matrix[:,:n-200,200:n]
   
   
fig, ax = plt.subplots(1, 1)
X = np.array(rdata_reshaped)

tracker = IndexTracker(ax, X)


fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
plt.show()

#%%

all_line_values = []
# second spot on all CRL
for i in range(len(rdata_reshaped)):
    image = np.copy(rdata_reshaped[i])
    matrix = np.copy(image)
    start_point = (572,740) # (y,x)
    end_point = ( 656,416) #(y,x)
    line_width = 20 # Specify the width of the line
    #line_width = 91 # Specify the width of the line
    # Call the function to draw parallel lines
    line_points, x, y = draw_parallel_lines(matrix, start_point, end_point, line_width)
    line_values= np.mean(image[y[:],x[:]],axis=0)
    arr_len = np.arange(0, len(line_values) * 0.0378, 0.0378)
    
    all_line_values.append(line_values)
    
    fs = 15
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    im=axs[0].imshow(image)
    #axs[0].imshow(image,extent=[x_mu.min(), x_mu.max(), x_mu.min(), x_mu.max()])
    #axs[0].set_ylim([800,400])
    #axs[0].set_xlim([400-25,800-25])
    axs[0].plot(x,y, color=(1, 0, 0, 0.2)  )
    #axs[0].set_ylabel("Distance (pixels)", fontsize = fs)
    #axs[0].set_xlabel("Distance (pixels)", fontsize = fs)
    #axs[0].set_xticks([])  # Remove x-axis numbers
    #axs[0].set_yticks([])  # Remove y-axis numbers
    
    #axs[0].set_ylabel(r"Distance ($\mathrm{\mu}$m)", fontsize = fs)
    #axs[0].set_xlabel(r"Distance ($\mathrm{\mu}$m)", fontsize = fs)
    axs[1].plot(arr_len,line_values,'*-',linewidth=2,label='Line Values', color='red')
    #axs[1].plot(fitted_sine_wave,linewidth=2,label='Line Values', color='black')
    #axs[1].plot(corrected_data+np.mean(fitted_sine_wave), label='Corrected Data', color='red')
    axs[1].set_ylabel("Corrected intensity", fontsize = fs)
    #axs[1].set_xlabel("Line distance (pixels)", fontsize = fs)
    axs[1].set_xlabel("Spatial position ($\\mathrm{\\mu}$m)", fontsize = fs)
    axs[1].grid()

#%%
from scipy.optimize import curve_fit

all_fitted_sine_wave = []
all_corrected_data = []
all_line_values_corrected = np.copy(all_line_values)
for i in range(3): 
    #i = 2
    if i == 0:
        data = all_line_values [0]  # CHANGE HERE
        fit_range = slice(170, -1)#332)   for all_line_value_spot1 [0] # CHANGE HERE 174
    
    if i == 1:
        data = all_line_values [1] # CHANGE HERE
        fit_range = slice(190, -1)#332)   for all_line_value_spot1 [2]] # CHANGE HERE
          
        
    if i == 2:
        data = all_line_values [2] # CHANGE HERE
        #fit_range = slice(250, -1)#332)   for all_line_value_spot1 [2]] # CHANGE HERE
        fit_range = slice(150, -1)#332) 
    
    # Define the sine function model
    def sine_wave(xx, amplitude, frequency, phase, offset):
        return amplitude * np.sin(frequency * xx + phase) + offset
    
    # Generate x values corresponding to the data
    xx = np.arange(len(data))
    
    # Define the range for fitting the sine curve
    
    
    # Fit the sine curve to the relevant data segment
    p0 = [1, 0.01, 0, np.mean(data[fit_range])]  # initial guess for parameters
    params, _ = curve_fit(sine_wave, xx[fit_range], data[fit_range], p0=p0)
    fitted_sine_wave = sine_wave(xx, *params) # Generate the fitted sine wave for the entire array
    corrected_data = data - fitted_sine_wave # Subtract the fitted sine wave from the original data
    
    
    all_fitted_sine_wave.append(fitted_sine_wave)
    all_corrected_data.append(corrected_data)
    

    
    
    # Plot the results
    plt.figure(figsize=(14, 7))
    plt.plot(xx, data, label='Original Data', alpha=0.5)
    plt.plot(xx, fitted_sine_wave, label='Fitted Sine Wave', linestyle='--', color='orange')
    plt.plot(xx, corrected_data+np.mean(fitted_sine_wave), label='Corrected Data', color='green')
    plt.legend()
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.title('Sine Wave Removal from Data')
    plt.grid()
    plt.show()
    
    

    
    # Optional: Return the corrected data
    corrected_data
    if i == 0:
        all_line_values_corrected [0] = corrected_data+np.mean(fitted_sine_wave) # CHANGE HERE
    
    if i == 1:
        all_line_values_corrected [1] = corrected_data+np.mean(fitted_sine_wave) # CHANGE HERE
        
    if i == 2:
        all_line_values_corrected [2] = corrected_data+np.mean(fitted_sine_wave) # CHANGE HERE

#%%


# second spot on all CRL
for i in range(len(rdata_reshaped)):
    image = np.copy(rdata_reshaped[i])
    matrix = np.copy(image)
    start_point = (572,740) # (y,x)
    end_point = ( 656,416) #(y,x)
    line_width = 21 # Specify the width of the line
    #line_width = 91 # Specify the width of the line
    # Call the function to draw parallel lines
    line_points, x, y = draw_parallel_lines(matrix, start_point, end_point, line_width)
    line_values= np.mean(image[y[:],x[:]],axis=0)
    arr_len = np.arange(0, len(line_values) * 0.0378, 0.0378)
    
    all_line_values.append(line_values)
    
    fs = 15
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    im=axs[0].imshow(image)
    #axs[0].imshow(image,extent=[x_mu.min(), x_mu.max(), x_mu.min(), x_mu.max()])
    #axs[0].set_ylim([800,400])
    #axs[0].set_xlim([400-25,800-25])
    axs[0].plot(x,y, color=(1, 0, 0, 0.2)  )
    #axs[0].set_ylabel("Distance (pixels)", fontsize = fs)
    #axs[0].set_xlabel("Distance (pixels)", fontsize = fs)
    #axs[0].set_xticks([])  # Remove x-axis numbers
    #axs[0].set_yticks([])  # Remove y-axis numbers
    
    #axs[0].set_ylabel(r"Distance ($\mathrm{\mu}$m)", fontsize = fs)
    #axs[0].set_xlabel(r"Distance ($\mathrm{\mu}$m)", fontsize = fs)
    axs[1].plot(arr_len,all_line_values_corrected[i],'*-',linewidth=2,label='Line Values', color='red')
    #axs[1].plot(fitted_sine_wave,linewidth=2,label='Line Values', color='black')
    #axs[1].plot(corrected_data+np.mean(fitted_sine_wave), label='Corrected Data', color='red')
    axs[1].set_ylabel("Corrected intensity", fontsize = fs)
    #axs[1].set_xlabel("Line distance (pixels)", fontsize = fs)
    axs[1].set_xlabel("Spatial position ($\\mathrm{\\mu}$m)", fontsize = fs)
    axs[1].grid()
    
    
    
#%%

all_peaks_x = []
all_peaks_y = []
all_fits = []
all_slope = []
all_decay_fits = []
all_x_fits =[]
for i in range(len(all_line_values_corrected)):
    data = all_line_values_corrected[i][0:260]
    
    fig, (ax1) = plt.subplots(nrows=1, ncols=1,figsize=(12,5))
    fig.tight_layout(pad=3.0)
    ax1.plot(data,'-*',linewidth=2)#),label='0 CRL')

    # Find peaks with specified parameters to filter out noise
    distance = 20  # Minimum number of samples between peaks
    prominence = 0.005  # Minimum prominence of peaks
    peaks, _ = find_peaks(data, distance=distance, prominence=prominence)
    max_peak_index = np.argmax(data[peaks])
    peaks = peaks[max_peak_index:] 
    x_peaks = peaks
    y_peaks = data[peaks]
    
    #all_peaks_x.append(x_peaks)
    all_peaks_y.append(y_peaks)
    ax1.plot(x_peaks, y_peaks,'*', label='Peaks')
    
    
    
    x_fit = np.linspace(x_peaks[1], 260, 100) # Generate fitted values for the asymptotic decay model max(x_peaks)
    asymptotic_params, _ = curve_fit(asymptotic_decay, x_peaks, y_peaks, p0=(1, 0.01, 1)) # Fit the asymptotic decay model
    y_asymptotic_fit = asymptotic_decay(x_fit, *asymptotic_params)
    print(x_peaks[1])
    
    all_peaks_x.append(x_fit)
    all_fits.append(y_asymptotic_fit)

    # Calculate the slope of the asymptotic decay fit
    # Initial value from the fit at x=0
    initial_value = asymptotic_decay(0, *asymptotic_params)
    # Final stabilized value (asymptotic) which can be approximated as c
    stabilized_value = asymptotic_params[2]
    # Calculate the slope as the difference in values over the difference in x
    slope = (stabilized_value - initial_value) / (x_peaks[-1] - x_peaks[1])
    all_slope.append(slope)
    print("Slope of the Asymptotic Decay Fit:", slope)
    
    ax1.plot(x_fit, y_asymptotic_fit, label='Asymptotic Decay Fit', color='blue')
    ax1.legend(fontsize = 12)
    
    ax1.set_ylim(0.4, 1.45)
    ax1.set_xlim(0.0, 350)
    
    all_decay_fits.append(y_asymptotic_fit)
    all_x_fits.append(x_fit)
    
#%%

# second spot on all CRL
crl_string = ['0 lenslets','4 lenslets','8 lenslets','12 lenslets','18 lenslets']
for i in range(len(rdata_reshaped)):
    image = np.copy(rdata_reshaped[i])
    matrix = np.copy(image)
    start_point = (572,740) # (y,x)
    end_point = ( 656,416) #(y,x)
    line_width = 20 # Specify the width of the line
    #line_width = 91 # Specify the width of the line
    # Call the function to draw parallel lines
    line_points, x, y = draw_parallel_lines(matrix, start_point, end_point, line_width)
    line_values= np.mean(image[y[:],x[:]],axis=0)
    arr_len = np.arange(0, len(line_values) * 0.0378, 0.0378)
    
    x_decay_len = np.linspace(all_x_fits[i][0]* 0.0378, all_x_fits[i][-1] * 0.0378, 100)

    
    all_line_values.append(line_values)
    
    fs = 15
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    im=axs[0].imshow(image)#, vmin=0, vmax=3.0)
    cbar = fig.colorbar(im)#, ax=axs[0])
    #cbar.set_label('Intensity')  # You can customize this label if needed
    #axs[0].imshow(image,extent=[x_mu.min(), x_mu.max(), x_mu.min(), x_mu.max()])
    #axs[0].set_ylim([800,400])
    #axs[0].set_xlim([400-25,800-25])
    axs[0].plot(x,y, color=(1, 0, 0, 0.2)  )

    axs[0].set_xticks([])  # Remove x-axis numbers
    axs[0].set_yticks([])  # Remove y-axis numbers
    

    axs[1].plot(arr_len,all_line_values_corrected[i],'*-',linewidth=2,label='Line Values', color='red')
    axs[1].plot(x_decay_len, all_decay_fits[i], linewidth=3, label='Asymptotic Decay Fit', color='blue')


    axs[1].set_ylabel("Corrected intensity", fontsize = fs)
    axs[1].set_ylim(0.41, 1.5)
    axs[1].set_xlabel("Spatial position ($\\mathrm{\\mu}$m)", fontsize = fs)
    fig.text(0.81, 0.2, crl_string[i], ha='center', va='center', fontsize=20)
    axs[1].legend(fontsize = 12)
    axs[1].grid()
    
    
#%% 
# Calculate the theoretical number of fringes: 
    
energy = 19.9  # [keV] x-ray energy    
PLANCK_CONSTANT = 4.135667696e-18  # [keV*s]
SPEED_OF_LIGHT = 299792458  # [m/s]
wavelength = PLANCK_CONSTANT * SPEED_OF_LIGHT / energy # [m]
    
lenslet_radius = 100e-6  # Radius of the lenslet (in meters, 100 µm)
distance_to_optics = 32.1  # Distance from the source to the optical system (in meters)
beam_size = 1e-3  # Beam size at the optical system (in meters, 1 mm)

# Calculate spatial coherence length
coherence_length = (wavelength * distance_to_optics) / beam_size

# Assume diffraction angle θ is related to the lenslet radius, which can be estimated as the angular divergence
# For simplicity, assuming diffraction angle θ = lenslet_radius / distance_to_optics (in radians)
theta = lenslet_radius / distance_to_optics

# Calculate fringe spacing
fringe_spacing = wavelength / theta

# Calculate the theoretical number of fringes
num_fringes = (lenslet_radius * theta) / wavelength

# Output the results
print(f"Spatial coherence length: {coherence_length:.3e} m")
print(f"Fringe spacing: {fringe_spacing:.3e} m")
print(f"Theoretical number of fringes: {num_fringes:.2f}")

#%%

import math


energy = 19.9  # X-ray energy in keV    
PLANCK_CONSTANT = 4.135667696e-18  # Planck constant in keV*s
SPEED_OF_LIGHT = 299792458  # Speed of light in m/s
pixel_size = 0.55e-6  # Pixel size in meters. 0.55e-6 or 3.78e-8

# Calculate wavelength from energy
wavelength = PLANCK_CONSTANT * SPEED_OF_LIGHT / energy  # Wavelength in meters

# Parameters
lenslet_radius = 100e-6  # Radius of the lenslet in meters (100 µm)
distance_to_optics = 32.1  # Distance from the source to the optical system in meters
beam_size = 1e-3  # Beam size at the optical system in meters (1 mm) 
num_lenslets = 18  # Number of lenslets in the CRL device (example using 8 for now)

# Calculate total beam divergence (total diffraction angle)
theta_total = num_lenslets * (lenslet_radius / distance_to_optics)

# Calculate spatial coherence length
coherence_length = (wavelength * distance_to_optics) / beam_size

# Calculate fringe spacing
fringe_spacing = wavelength / theta_total

# Calculate the theoretical number of fringes
num_fringes = (lenslet_radius * theta_total) / wavelength

# Calculate number of pixels per fringe (fringe width in pixels)
fringe_width_in_pixels = fringe_spacing / pixel_size

# Output the results
print(f"Spatial coherence length: {coherence_length:.3e} m")
print(f"Fringe spacing: {fringe_spacing:.3e} m")
print(f"Theoretical number of fringes: {num_fringes:.2f}")
print(f"Fringe width in pixels: {fringe_width_in_pixels:.2f} pixels")

#%%


# Huygens-Frenel Principle
# Zone plat formula. 