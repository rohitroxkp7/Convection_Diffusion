import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from scipy.linalg import solve_banded

# Create directory for saving images
output_dir = "images_with_Source"
os.makedirs(output_dir, exist_ok=True)

# Parameters
Lx, Ly = 1.0, 1.0  # Domain size
Nx, Ny = 50, 50    # Number of grid points in x and y
dx = Lx / (Nx - 1)
dy = Ly / (Ny - 1)
dt = 0.01          # Time step
T = 1.0            # Total simulation time

a, b = 1.0, 1.0    # Convection velocities in x and y
nu = 0.01          # Diffusion coefficient
sigma_x = nu * dt / dx**2
sigma_y = nu * dt / dy**2

# CFL condition check for stability
assert sigma_x < 0.5 and sigma_y < 0.5, "Stability condition not satisfied!"

# Grid
x = np.linspace(0, Lx, Nx)
y = np.linspace(0, Ly, Ny)
X, Y = np.meshgrid(x, y)

# Initial condition: Gaussian distribution
u = np.exp(-((X - 0.5)**2 + (Y - 0.5)**2) / 0.05)  

# Define Source Term (Example: Heat generation in a localized region)
S = np.zeros((Nx, Ny))
S[Nx//4:Nx//2, Ny//4:Ny//2] = 5.0  # Localized heat source

# ADI Coefficients
alpha_x = 0.5 * (sigma_x + max(0, a * dt / dx))  
beta_x  = 0.5 * (sigma_x - min(0, a * dt / dx))  
alpha_y = 0.5 * (sigma_y + max(0, b * dt / dy))  
beta_y  = 0.5 * (sigma_y - min(0, b * dt / dy))  

# Tridiagonal matrices (for Thomas Algorithm)
Ax = np.zeros((3, Nx-2))  
Ay = np.zeros((3, Ny-2))

Ax[1, :] = 1 + 2 * sigma_x  # Main diagonal
Ax[0, 1:] = -alpha_x        # Upper diagonal
Ax[2, :-1] = -beta_x        # Lower diagonal

Ay[1, :] = 1 + 2 * sigma_y  # Main diagonal
Ay[0, 1:] = -alpha_y        # Upper diagonal
Ay[2, :-1] = -beta_y        # Lower diagonal

# Time-stepping loop
num_steps = int(T / dt)
for step in range(num_steps):

    # Step 1: Solve Implicit in x, Explicit in y
    u_star = np.copy(u)
    for j in range(1, Ny-1):
        rhs = (u[1:Nx-1, j] 
               + sigma_y * (u[1:Nx-1, j+1] - 2*u[1:Nx-1, j] + u[1:Nx-1, j-1])
               + dt * S[1:Nx-1, j])  # Adding Source Term

        # Upwind scheme in x-direction
        if a > 0:  
            rhs += (a * dt / dx) * (u[1:Nx-1, j] - u[0:Nx-2, j])  
        else:  
            rhs += (a * dt / dx) * (u[2:Nx, j] - u[1:Nx-1, j])  

        u_star[1:Nx-1, j] = solve_banded((1, 1), Ax, rhs)

    # Step 2: Solve Implicit in y, Explicit in x
    for i in range(1, Nx-1):
        rhs = (u_star[i, 1:Ny-1] 
               + sigma_x * (u_star[i+1, 1:Ny-1] - 2*u_star[i, 1:Ny-1] + u_star[i-1, 1:Ny-1])
               + dt * S[i, 1:Ny-1])  # Adding Source Term


        # Upwind scheme in y-direction
        if b > 0:  
            rhs += (b * dt / dy) * (u_star[i, 1:Ny-1] - u_star[i, 0:Ny-2])  
        else:  
            rhs += (b * dt / dy) * (u_star[i, 2:Ny] - u_star[i, 1:Ny-1])  

        u[i, 1:Ny-1] = solve_banded((1, 1), Ay, rhs)

    # Apply Boundary Conditions (Neumann for outflow, Dirichlet for inflow)
    u[:, 0] = u[:, 1]   
    u[0, :] = 0.5   
    u[-1, :] = 0.5  
    u[:, -1] = 1  

    # Plot and save at each time step
    plt.figure(figsize=(8, 6))
    plt.contourf(X, Y, u, 100, cmap="inferno", vmin=0, vmax=1)  # Consistent color scale
    plt.colorbar(label="u(x,y)")
    plt.title(f"2D Convection-Diffusion ADI Solution (Step {step})")
    plt.xlabel("x")
    plt.ylabel("y")

    # Add Green Borders
    for spine in plt.gca().spines.values():
        spine.set_color("green")
        spine.set_linewidth(2)

    # Save figure
    filename = os.path.join(output_dir, f"step_{step:04d}.png")
    plt.savefig(filename, dpi=150)
    plt.close()  # Close plot to free memory

print(f"Simulation complete. Images saved in {output_dir}/")

# === Generate Video from Images ===
video_filename = "convection_diffusion_simulation_with_source.mp4"
fps = 30  # Frames per second

# Get sorted list of images
images = sorted([img for img in os.listdir(output_dir) if img.endswith(".png")])

# Read first image to get dimensions
first_image = cv2.imread(os.path.join(output_dir, images[0]))
height, width, _ = first_image.shape

# Define video writer
fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec for MP4 format
video = cv2.VideoWriter(video_filename, fourcc, fps, (width, height))

# Write images to video
for image in images:
    img_path = os.path.join(output_dir, image)
    frame = cv2.imread(img_path)
    video.write(frame)

# Release video writer
video.release()

print(f"Video saved as {video_filename}")
