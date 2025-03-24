import os
# Change to your local directory:
os.chdir(r"C:\Users\nige\OneDrive - Danmarks Tekniske Universitet\Dokumenter\Postdoc\Projects\Phase retrieval\GitHub")


from phase_retrieval_functions import *
from optical_constants import oc


os.chdir(r"Silica sphere analysis")


file_path = 'scan-0774_red_ds.tif'  # Change this to your file path
tiff_image = tifffile.imread(file_path)

# Step 2: Select a slice
# For 3D image data (Z, Y, X), you can select a slice along any axis. For example, if you want the first slice (index 0) along the first dimension (Z axis):
slice_data = tiff_image[665, :, :]  # First slice (can change the index to slice differently)

# Step 3: Plot the matrix
plt.imshow(slice_data, cmap='gray')
plt.title("Selected Slice of the TIFF Image")
plt.colorbar()
plt.show()

#%%

    
#%%

fig, ax = plt.subplots(1, 1)
X = np.array(tiff_image)
tracker = IndexTracker(ax, X)
fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
plt.show()

#%%

energy = 19.9  # [keV] x-ray energy    
PLANCK_CONSTANT = 4.135667696e-18  # [keV*s]
SPEED_OF_LIGHT = 299792458  # [m/s]
wavelength = PLANCK_CONSTANT * SPEED_OF_LIGHT / energy # [m]

detector_pixelsize = 0.55e-6 # m  
focusToDetectorDistance = 0.45086 # [m] 
S2D_distace = 0.40876 # [m]  Sample to detector distance

z1 = focusToDetectorDistance - S2D_distace ## m (focus and sample)
z2 = focusToDetectorDistance-z1 # propagation distance between the code and detector
magnification = (z1+z2)/z1 # magnification when propagating from the sample to the code
voxelsize = detector_pixelsize/magnification # object voxel size
effective_pixel_size = detector_pixelsize 

#%%




def draw_parallel_lines(matrix, start, end, num_lines, spacing=10, width=1):
    """
    Draws parallel lines to the central line defined by start and end points.

    Parameters:
    - matrix: The grid where the lines will be drawn (2D numpy array).
    - start: (x0, y0) starting point of the central line.
    - end: (x1, y1) ending point of the central line.
    - num_lines: The number of parallel lines to draw (including the central line).
    - spacing: The spacing between each parallel line in pixels.
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

    # Compute the vector of the central line
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    length = np.sqrt(dx ** 2 + dy ** 2)

    # Normalize the direction vector (unit vector)
    unit_dx = dx / length
    unit_dy = dy / length

    # Create lists to store all the lines' coordinates
    all_line_points = []
    all_xxx = []
    all_yyy = []

    # Loop to generate the parallel lines
    for i in range(-(num_lines // 2), (num_lines // 2) + 1):
        # Offset parallel lines by 'spacing' in both x and y directions
        offset_x = int(i * spacing * unit_dy)
        offset_y = int(i * spacing * unit_dx)

        # Calculate the start and end points for the parallel line
        parallel_start = (start[0] + offset_x, start[1] + offset_y)
        parallel_end = (end[0] + offset_x, end[1] + offset_y)

        # Call Bresenham's line algorithm to draw the line
        line_points, xxx, yyy = bresenham_line(matrix, parallel_start, parallel_end, width)

        # Store the points of the line
        all_line_points.append(line_points)
        all_xxx.append(xxx)
        all_yyy.append(yyy)

    return all_line_points, all_xxx, all_yyy




#%%


image = np.copy(tiff_image[725, :, :])
n = np.shape(image)
real_world_pixel_size = detector_pixelsize / magnification
x_mu = np.arange(0, n[0]) * (real_world_pixel_size*1e6) # micrometer
y_mu = np.arange(0, n[1]) * (real_world_pixel_size*1e6) # micrometer


fs = 15
fig, axs = plt.subplots(1,2,figsize=(21, 6))
im=axs[0].imshow(image,extent=[x_mu.min(), x_mu.max(), y_mu.min(), y_mu.max()])#,cmap='gray',vmax=0.05,vmin=-0.05) # cmap='Greys',
axs[0].set_xlabel(r"Distance ($\mathrm{\mu}$m)", fontsize = fs)
axs[0].set_ylabel(r"Distance ($\mathrm{\mu}$m)", fontsize = fs)
fig.colorbar(im)

# First spot
matrix = np.copy(image)
start_point = (20,12) # (y,x)  (354,356)
end_point = ( 20,40) #(y,x)
line_width = 1 # Specify the width of the line
line_points, x, y = draw_parallel_lines(matrix, start_point, end_point, line_width)
line_values= np.mean(image[y[:],x[:]],axis=0)
axs[0].plot(x,y, 'r.' )

start_point = (343,503) # (y,x)  (354,356)
end_point = ( 930,503) #(y,x)
line_points_2, x_2, y_2 = draw_parallel_lines(matrix, start_point, end_point, line_width)
line_values_2= np.mean(image[y_2[:],x_2[:]],axis=0)
#axs[0].plot(x_2,y_2, 'b.' )


axs[1].plot(line_values, '*-',color='red') # Highlight the selected region
#%%


# First spot
image = np.copy(tiff_image[725, :, :])
matrix = np.copy(image)
start_point = (661,200) # (y,x)  (354,356)
end_point = ( 661,850) #(y,x)
line_width = 1 # Specify the width of the line
#start_point = (430,252) # (y,x)
#end_point = ( 792,272) #(y,x)
#line_width = 91 # Specify the width of the line
# Call the function to draw parallel lines
line_points, x, y = draw_parallel_lines(matrix, start_point, end_point, line_width)
line_values= np.mean(image[y[:],x[:]],axis=0)

# Second spot
image = np.copy(tiff_image[725, :, :])
matrix = np.copy(image)
start_point = (343,503) # (y,x)  (354,356)
end_point = ( 930,503) #(y,x)
line_width = 1 # Specify the width of the line
#start_point = (430,252) # (y,x)
#end_point = ( 792,272) #(y,x)
#line_width = 91 # Specify the width of the line
# Call the function to draw parallel lines
line_points_2, x_2, y_2 = draw_parallel_lines(matrix, start_point, end_point, line_width)
line_values_2= np.mean(image[y_2[:],x_2[:]],axis=0)


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


arr_len = np.arange(0, np.shape(line_values)[0] * 0.05135, 0.05135)
arr_len_2 = np.arange(0, np.shape(line_values_2)[0] * 0.05135, 0.05135)
arr_len_3 = np.arange(0, 1048 * 0.05135, 0.05135)
arr_len_4 = np.arange(0, 1041 * 0.05135, 0.05135)

fs = 15
fig, axs = plt.subplots(1,2,figsize=(21, 6))
im=axs[0].imshow(image)#, extent=[0, 53.45535, 0, 53.45535])#,extent=[x_mu.min(), x_mu.max(), y_mu.min(), y_mu.max()])#,cmap='gray',vmax=0.05,vmin=-0.05) # cmap='Greys',
#axs[0].set_title(f'Reconstruction from {k+1} distances')
axs[0].plot(x,y, 'r.' )
axs[0].plot(x_2,y_2, 'b.' )
#axs[0].set_ylabel("Spatial position ($\\mathrm{\\mu}$m)", fontsize = fs)
#axs[0].set_xlabel("Spatial position ($\\mathrm{\\mu}$m)", fontsize = fs)
axs[0].set_xticks([])  # Remove x-axis numbers
axs[0].set_yticks([])  # Remove y-axis numbers



divider = make_axes_locatable(axs[0])
cax = divider.append_axes("right", size="5%", pad=0.05)  # Adjust size and padding as needed
fig.colorbar(im, cax=cax) # Create the colorbar in the new axis

axs[1].plot(arr_len,line_values, '*-',color='red') # Highlight the selected region
axs[1].plot(arr_len_2,line_values_2, '*-',color='blue') # Highlight the selected region

axs[1].set_ylabel(r"$\tilde{\phi}(f)$", fontsize = fs)
axs[1].set_xlabel("Spatial position ($\\mathrm{\\mu}$m)", fontsize = fs)
#axs[1].set_xlabel("Distance (pixels)", fontsize = fs)
#axs[2].set_ylim([-0.3, 0.3])
axs[1].grid()

#%%
resolutions = []
for i in range(1078):
    image = np.copy(tiff_image[i, :, :][200:800,200:800])
    result = calculate_resolution(image, detector_pixelsize, magnification)
    resolutions.append((((result[1]+result[3])/2)*10**9))
    #print('Resolution [nm]:'+str(((result[1]+result[3])/2)*10**9))

print(np.mean(resolutions))
#%%

os.chdir(r"C:\Users\nige\OneDrive - Danmarks Tekniske Universitet\Dokumenter\PhD\XRR_Fitting_tool")
from optical_constants import oc
from unit_conversion import wavelength2energy, energy2wavelength, thickness2energy, energy2thickness, convert_pixel_size_to_actual_distance
energy = 19.9  # [keV] x-ray energy   
density_Au = 19.3 # g/cm**3
density_SiN = 3.44 # g/cm**3
density_SiO2 = 2.334 # g/cm**3


elements_Au = ['Au'] # 
elements_SiN = ['Si','N'] # from CXRO transmission calculator
elements_SiO2 = ['Si','O'] # from CXRO transmission calculator


z_Au = 1600 # Thickness of gold [nm]    or 500 Å  = 0.05 mu, maybe 1.6 mu # plus minus 10 %
z_SiN = 1000 # Thickness of silicon [nm]
z_SiO2 = 9000 # Thickness of silicon [nm]

number_of_atoms_Au = [1]
number_of_atoms_SiN = [3,4]
number_of_atoms_SiO2 = [1,2]
oc_source = "oc_source/NIST/"

wavelength = energy2wavelength(energy*10**3) # nm

n_Au = oc(wavelength,density_Au,number_of_atoms_Au,elements_Au, oc_source)     
beta = n_Au.imag
delta = n_Au.real

n_SiN = oc(wavelength,density_SiN,number_of_atoms_SiN,elements_SiN, oc_source)    

phi_Au = (-2*np.pi*(1-n_Au.real)*(z_Au*1e-09))/(wavelength*1e-9) #- (n_Au.imag*z_Au) #  [m]
phi_SiN = (-2*np.pi*(1-n_SiN.real)*(z_SiN*1e-09))/(wavelength*1e-9) #- (n_SiN.imag*z_SiN)#   [m]
print("Phase difference for Au:"+str(phi_Au))


n_SiO2 = oc(wavelength,density_SiO2,number_of_atoms_SiO2,elements_SiO2, oc_source)     
beta = n_SiO2.imag
delta = n_SiO2.real
phi_SiO2 = (-2*np.pi*(1-n_SiO2.real)*(z_SiO2*1e-09))/(wavelength*1e-9) #- (n_Au.imag*z_Au) #  [m]
print("Phase difference for SiO2:"+str(phi_SiO2))


phi_SiO2 = (-2*np.pi*(1-1.000293)*(5000*1e-09))/(wavelength*1e-9) #- (n_Au.imag*z_Au) #  [m]
print("Phase difference for air:"+str(phi_SiO2))

#%%

from matplotlib import pyplot as plt
import matplotlib.patches as patches

# Define a function to add the scale bar
def add_scale_bar(ax, length_in_um, position=(0.15, 0.15), color='white', linewidth=2):
    # Length of the scale bar in pixels, assuming pixel size of 0.05135 µm/pixel
    pixel_length = length_in_um / 0.05135

    # Define the starting position of the scale bar in axis coordinates (fraction of figure size)
    x_pos, y_pos = position

    # Add a rectangle for the scale bar and a label below it
    ax.add_patch(patches.Rectangle((x_pos, y_pos), pixel_length, 0.02, linewidth=linewidth, edgecolor=color, facecolor=color))
    ax.text(x_pos + pixel_length / 2, y_pos + 15.52, f'{length_in_um} µm', ha='center', va='top', color=color, fontsize=12)

# Now include this function in your original code for the left subplot
fig, axs = plt.subplots(1, 2, figsize=(21, 6))

# Plot the first image
im = axs[0].imshow(image)  # Use your image here
axs[0].plot(x, y, 'r.')
axs[0].plot(x_2, y_2, 'b.')

# Remove axis numbers and ticks
axs[0].set_xticks([])
axs[0].set_yticks([])

## Add the scale bar to the left subplot
#add_scale_bar(axs[0], 50)  # Add scale bar of 50 µm
# Add the scale bar to the left subplot with a new position
add_scale_bar(axs[0], 25, position=(19.12, 1000.02))  # Adjust the position here

# Colorbar for the left subplot
divider = make_axes_locatable(axs[0])
cax = divider.append_axes("right", size="5%", pad=0.05)
fig.colorbar(im, cax=cax)

# Plot the second subplot
axs[1].plot(arr_len, line_values, '*-', color='red')
axs[1].plot(arr_len_2, line_values_2, '*-', color='blue')
axs[1].set_ylabel(r"$\tilde{\phi}(f)$", fontsize=fs)
axs[1].set_xlabel("Spatial position ($\\mathrm{\\mu}$m)", fontsize=fs)
axs[1].grid()

plt.show()
