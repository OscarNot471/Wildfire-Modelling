import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.collections import RegularPolyCollection
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib import cm, colors

# ==============================================================================
# 1. SYSTEM PARAMETERS
# ==============================================================================

# Spatial grid
Nx, Ny = 75, 75
hex_dist = 0.1  # km, minimum distance between hexagonal cells
dx = hex_dist
dy = hex_dist * np.sqrt(3) / 2.0


# Time interval
dt = 0.005  # time step
t_max = 4.0 # maximum time of simulation
Nt = int(t_max / dt) # number of time steps

# Wind field
wind = np.array([0.8, 0.0])  # km/h

# ==============================================================================
# 2. HEXAGONAL GRID DEFINITION
# ==============================================================================

# Loops to construct the hexagonal grid
hex_grid = []

# Iterate row by row
for y in range(Ny):
    # Iterate over columns
    for x in range(Nx):
        # Horizontal shift for odd rows
        if y % 2 == 0:
            shift_x = 0.0
        else:
            shift_x = 0.5 * hex_dist

        # Cell centers
        cx = x * hex_dist + shift_x
        cy = y * hex_dist * np.sqrt(3) / 2.0

        hex_grid.append([cx, cy])

hex_grid = np.array(hex_grid)


# Matrices containing the coordinates of each hexagonal cell (on each axis)
X = np.zeros((Ny, Nx))
Y = np.zeros((Ny, Nx))

i = 0
for y in range(Ny):
    for x in range(Nx):
        X[y, x] = hex_grid[i, 0]
        Y[y, x] = hex_grid[i, 1]
        i += 1

# ==============================================================================
# 3. ELEVATION AND SLOPE FIELDS
# ==============================================================================

def compute_slope(H, dx, dy):
    dHdx = np.zeros_like(H) #slope in x
    dHdy = np.zeros_like(H) #slope in y

    dHdx[:, 1:-1] = (H[:, 2:] - H[:, :-2]) / (2 * dx)
    dHdy[1:-1, :] = (H[2:, :] - H[:-2, :]) / (2 * dy)

    dHdx[:, 0]  = (H[:, 1] - H[:, 0]) / dx
    dHdx[:, -1] = (H[:, -1] - H[:, -2]) / dx

    dHdy[0, :]  = (H[1, :] - H[0, :]) / dy
    dHdy[-1, :] = (H[-1, :] - H[-2, :]) / dy

    slope_magnitude = np.sqrt(dHdx**2 + dHdy**2)
    return dHdx, dHdy, slope_magnitude


# Define a mountain as a Gaussian surface
H_max = 1.2  # Maximum mountain height (Km)

# Compute the spatial domain midpoint to place the mountain peak
x_center = X.mean()
y_center = Y.mean()

# Define mountain width (Gaussian width)
sigma_x = 1.2
sigma_y = 0.9

# Elevation field (Gaussian profile along both axes)
Elevation = H_max * np.exp(
    -(((X - x_center) ** 2) / (2 * sigma_x ** 2) +
      ((Y - y_center) ** 2) / (2 * sigma_y ** 2))
)


# Compute slope field (slope_magnitude is |ΔH|)
Slope_x, Slope_y, slope_magnitude = compute_slope(Elevation, dx, dy)


# Uphill bias for positive elevation differences (ΔH > 0) (fire propagates faster uphill)
slope_coefficient = 1.2  # km/h per unit slope (ΔH / cell_distance)
# Controls how much the propagation speed increases when moving uphill

max_uphill_speed = 1.5  # km/h, maximum slope-induced speed enhancement
# Prevents non-physical values by limiting the additional speed contribution

# ==============================================================================
# 4. MOISTURE FIELD DEFINITION
# ==============================================================================

def normalize_field(field):
    max_value = np.max(field)
    min_value = np.min(field)

    if max_value == min_value:
        return field / max_value
    else:
        return (field - min_value) / (max_value - min_value)


def compute_moisture(Elevation_norm):
    """
    Elevation_norm: normalized elevation in [0, 1]
        0 == valley (minimum elevation)
        1 == peak (maximum elevation)
    """

    moisture_valley = 0.35   # Moisture at minimum elevation (maximum humidity)
    moisture_peak = 0.10     # Moisture at maximum elevation (minimum humidity)

    p = 1.5  # Controls moisture dependence on elevation (p = 1 -> linear)

    Moisture = moisture_peak + \
        (moisture_valley - moisture_peak) * (1.0 - Elevation_norm) ** p

    return Moisture


# Normalize elevation field to compute moisture
Elevation_norm = normalize_field(Elevation)

# Moisture field (higher at low elevation, lower at high elevation)
Moisture = compute_moisture(Elevation_norm)

# ==============================================================================
# 5. VEGETATION FIELD DEFINITION
# ==============================================================================

Vegetation = np.zeros((Ny, Nx))

vegetation_types = [1, 2, 3]

"""
1: Tree
   - Requires higher ignition temperature
   - Burns more slowly
   - Releases more energy

2: Shrub
   - Easier to ignite
   - Burns faster
   - Releases moderate energy

3: Dry branches
   - Ignites very easily
   - Burns very rapidly
   - Releases less energy
"""

grid_size = (Ny, Nx)
#probabilities to generate tree, shurb or dry branches at each height
prob_high_elevation = [0.1, 0.5, 0.4]
prob_mid_elevation  = [0.4, 0.3, 0.3]
prob_low_elevation  = [0.7, 0.2, 0.1]

# iteration among all the grid to generate the vegetation type of each cell
for y in range(Ny):
    for x in range(Nx):

        if Elevation_norm[y, x] <= 0.33:
            Vegetation[y, x] = np.random.choice(
                vegetation_types, p=prob_low_elevation
            )

        elif 0.33 < Elevation_norm[y, x] < 0.66:
            Vegetation[y, x] = np.random.choice(
                vegetation_types, p=prob_mid_elevation
            )

        else:
            Vegetation[y, x] = np.random.choice(
                vegetation_types, p=prob_high_elevation
            )

# Alternative uniform random vegetation distribution:
# Vegetation = np.random.choice([1, 2, 3], (Ny, Nx), p=[0.3, 0.4, 0.3])


# ==============================================================================
# 6. COMBUSTION AND IGNITION FIELDS AND PARAMETERS
# ==============================================================================

ignition_scale = 20.0  # K
# Controls how gradual the transition is between not burning and burning (in the sigmoid function)


alpha = np.zeros((Ny, Nx))                    # local thermal diffusivity
t_ign_base = np.zeros((Ny, Nx))               # base ignition temperature of the fuel
cooling_base = np.zeros((Ny, Nx))             # base cooling / thermal losses rate
heat_release_base = np.zeros((Ny, Nx))        # heat released during combustion
fuel_consumption_rate = np.zeros((Ny, Nx))    # fuel consumption rate

# Assign standard values for each vegetation type
# These values do not account for cell moisture; we include that afterwards
t_ign_tree,             t_ign_shrub,            t_ign_branches           = 480.0, 450.0, 410.0       # K
cooling_tree,           cooling_shrub,          cooling_branches         = 8.0, 10.0, 14.0           # h^-1
heat_release_tree,      heat_release_shrub,     heat_release_branches    = 90000.0, 25000.0, 6000.0  # effective "K/h"
consumption_tree,       consumption_shrub,      consumption_branches     = 4.0, 8.0, 16.0            # h^-1 (consumption)

alpha_base = 0.08  # base thermal diffusivity of terrain/vegetation
alpha[Vegetation == 1] = alpha_base * 0.8
alpha[Vegetation == 2] = alpha_base * 1.0
alpha[Vegetation == 3] = alpha_base * 1.3

t_ign_base[Vegetation == 1] = t_ign_tree
t_ign_base[Vegetation == 2] = t_ign_shrub
t_ign_base[Vegetation == 3] = t_ign_branches

cooling_base[Vegetation == 1] = cooling_tree
cooling_base[Vegetation == 2] = cooling_shrub
cooling_base[Vegetation == 3] = cooling_branches

heat_release_base[Vegetation == 1] = heat_release_tree
heat_release_base[Vegetation == 2] = heat_release_shrub
heat_release_base[Vegetation == 3] = heat_release_branches

fuel_consumption_rate[Vegetation == 1] = consumption_tree
fuel_consumption_rate[Vegetation == 2] = consumption_shrub
fuel_consumption_rate[Vegetation == 3] = consumption_branches


# Modify parameters as a function of cell moisture
def moisture_effect(t_ign_base, cooling_base, heat_release_base, moisture):
    ignition_increment = 180.0   # how much ignition temperature increases per unit moisture
    cooling_factor = 1.4         # how much thermal losses increase with moisture (evaporation)
    heat_reduction = 0.55        # how much effective heat release is reduced by moisture

    t_ign = t_ign_base + ignition_increment * moisture
    cooling = cooling_base * (1 + cooling_factor * moisture)
    heat_release = heat_release_base * np.clip(1 - heat_reduction * moisture, 0.05, 1.0)

    return t_ign, cooling, heat_release

"""
Moisture increases the ignition temperature.
Moisture increases thermal losses.
Moisture reduces the effective heat released.
"""

t_ign, cooling, heat_release = moisture_effect(t_ign_base, cooling_base, heat_release_base, Moisture)


# Ambient temperature
t_amb = np.zeros((Ny, Nx))  # K
t_amb_base = 300.0  # K

t_amb[Slope_y <= 0] = t_amb_base + 10
t_amb[Slope_y > 0] = t_amb_base - 10

t_amb[Elevation >= 0.8] = t_amb[Elevation >= 0.8] - 7


# Define the initial state
temperature = t_amb.copy()
fuel = np.ones((Ny, Nx))  # fuel matrix: each cell stores a fraction from 0 (min) to 1 (max)

# Define the ignition location
origin_y, origin_x = Ny // 2, Nx // 15
temperature[origin_y-2:origin_y+2, origin_x-2:origin_x+2] = 2000.0


# ==============================================================================
# 7. COMBUSTION/HEAT TRANSPORT MATRICES (Crank-Nicolson (CN) + wind + uphill bias per link ΔH>0)
# ==============================================================================

def flatten_index(y, x):
    #Convert a 2D cell index (y, x) into a 1D flattened index
    return y * Nx + x


def get_neighbors(y, x):
    """
    Return the valid neighbors of cell (y, x) on a hex grid.
    Each neighbor is returned as (ny, nx, dy_rel, dx_rel).
    """
    # Relative neighbor coordinates depend on whether the row is even or odd
    if y % 2 == 0:
        rel_neighbors = [(-1, -1), (-1, 0), (0, -1), (0, 1), (1, -1), (1, 0)]
    else:
        rel_neighbors = [(-1, 0), (-1, 1), (0, -1), (0, 1), (1, 0), (1, 1)]

    neighbors = []
    for dy_rel, dx_rel in rel_neighbors:
        ny_ = y + dy_rel
        nx_ = x + dx_rel
        if 0 <= ny_ < Ny and 0 <= nx_ < Nx:
            neighbors.append((ny_, nx_, dy_rel, dx_rel))
    return neighbors


print("Building matrices (CN + wind + uphill bias per link ΔH>0)...")
n_cells = Nx * Ny  # total number of cells (flattened grid size)

# Build matrices in LIL format (efficient for incremental construction)
LHS = lil_matrix((n_cells, n_cells))
RHS = lil_matrix((n_cells, n_cells))

# Fill matrices by looping over all cells and their neighbors
for y in range(Ny):
    for x in range(Nx):
        idx = flatten_index(y, x)

        # Diffusion weight (dimensionless)
        lam = (alpha[y, x] * dt) / (3.0 * hex_dist**2)
        # lam measures how much diffusion occurs in this cell during one time step:
        # - increases with local diffusivity (alpha) and with dt
        # - decreases with larger cell spacing (hex_dist)

        # Diagonal terms (start at 1 and get corrected by neighbor contributions)
        diag_LHS = 1.0
        diag_RHS = 1.0

        # 6 neighbors on the hex grid (filtered by boundaries)
        neighbors = get_neighbors(y, x)

        for ny_, nx_, dy_rel, dx_rel in neighbors:
            #ny_, nx_, are the index of the neighbour
            #dy_rel, dx_rel are the relative index of the neighbour from the point of view of the main cell
            idx_n = flatten_index(ny_, nx_)

            # Direction vector from current cell to neighbor (in physical coordinates)
            dir_vec = np.array([
                dx_rel * hex_dist,
                dy_rel * hex_dist * np.sqrt(3) / 2.0
            ])

            # Normalize direction vector
            norm = np.linalg.norm(dir_vec)
            if norm != 0:
                dir_vec = dir_vec / norm

            # Wind projection along the link direction (km/h)
            wind_projection = np.dot(wind, dir_vec)

            # Elevation difference between neighbor and current cell (km)
            dH = Elevation[ny_, nx_] - Elevation[y, x]

            # Uphill slope proxy along the link:
            # keep only positive slope (fire spreads faster uphill), ignore downhill
            slope_link = max(0.0, dH / hex_dist)  # ~ tan(phi)

            # Extra advection speed due to uphill slope
            v_slope = slope_coefficient * slope_link

            # Cap to avoid non-physical values
            if v_slope > max_uphill_speed:
                v_slope = max_uphill_speed

            # Effective advection speed along this link (wind + slope term)
            v_eff = wind_projection + v_slope

            # Advection weight (dimensionless)
            gamma = (dt * v_eff) / (6.0 * hex_dist)

            # Final link coefficient (advection-diffusion combined)
            coef = 0.5 * (lam - gamma)
            # If diffusion dominates: coef > 0
            # If advection dominates: coef can decrease or change sign

            # Off-diagonal terms
            LHS[idx, idx_n] = -coef
            RHS[idx, idx_n] =  coef

            # Update diagonal so each neighbor contribution is balanced at the origin cell
            diag_LHS += coef
            diag_RHS -= coef

        # Write diagonal terms
        LHS[idx, idx] = diag_LHS
        RHS[idx, idx] = diag_RHS


# Convert to CSC format (more efficient for solving linear systems)
LHS_csc = LHS.tocsc()
RHS_csc = RHS.tocsc()

# ==============================================================================
# 8. TIME EVOLUTION
# ==============================================================================

# Implement a function that computes one time step of the system evolution.
# Returns the updated temperature field and the remaining fuel field.
# Includes ignition, heat release, fuel consumption, diffusion+advection (matrices LHS/RHS),
# and cooling towards ambient temperature.
def time_step(temperature_current, fuel_current):
    # Ignition probability (sigmoid, returns values in [0, 1])
    # If T << T_ign: ignition ~ 0 (not burning)
    # If T >> T_ign: ignition ~ 1 (burning for sure)
    ignition = 1.0 / (1.0 + np.exp(-(temperature_current - t_ign) / ignition_scale))

    # Combustion heat source term (added to the temperature)
    # heat_release: effective energy release capacity of vegetation during combustion
    # fuel_current: remaining fuel fraction
    # ignition: how strongly it is actually burning
    # dt: time step
    source = heat_release * fuel_current * ignition * dt

    # Fuel consumption during this time step
    consumption_current = fuel_consumption_rate * fuel_current * ignition * dt

    # Remaining fuel after the time step
    fuel_next = np.clip(fuel_current - consumption_current, 0.0, 1.0)

    # Right-hand side of the linear system (explicit part + sources)
    rhs_vec = RHS_csc.dot(temperature_current.flatten()) + source.flatten()
    # flatten(): convert 2D temperature array into a 1D vector for sparse matrix ops
    # dot(): sparse matrix–vector multiplication (explicit part of the scheme)

    # Solve the implicit linear system to get next-step temperature
    temperature_next = spsolve(LHS_csc, rhs_vec).reshape((Ny, Nx))
    # spsolve(): efficient sparse linear system solver
    # reshape((ny, nx)): back to 2D grid shape

    # Cooling towards ambient temperature
    # If temperature_next > t_amb: cools down
    # If temperature_next < t_amb: warms up slightly
    # cooling controls the strength of this exchange
    temperature_next = temperature_next - cooling * (temperature_next - t_amb) * dt

    return temperature_next, fuel_next


# Store snapshots of temperature and fuel to build the temporal evolution
temperature_evolution = [temperature.copy()]
fuel_evolution = [fuel.copy()]

# Run the time evolution
print(f"Simulating {Nt} steps ({t_max} hours)...")
temperature_current, fuel_current = temperature, fuel

for i in range(Nt):
    temperature_current, fuel_current = time_step(temperature_current, fuel_current)

    if i % 5 == 0:
        temperature_evolution.append(temperature_current.copy())
        fuel_evolution.append(fuel_current.copy())


# ==============================================================================
# 9. 3D ANIMATION: z = Elevation; color = Temperature (SINGLE colormap); type = marker
# ==============================================================================

def wildfire_animation_3d_simple(
    hex_grid, X, Y, Elevation, vegetation,
    temperature_evolution, fuel_evolution, t_ign,
    t_amb, dt, hex_dist,
    steps_per_frame=5,
    elev=28, azim=-55,
    interval=50
):
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from matplotlib import cm, colors
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    # --- masks by vegetation type ---
    veg_flat = vegetation.astype(int).flatten()
    mask_tree     = (veg_flat == 1)
    mask_shrub    = (veg_flat == 2)
    mask_branches = (veg_flat == 3)

    # --- 1D coords and z (elevation) ---
    x_all = hex_grid[:, 0]
    y_all = hex_grid[:, 1]
    z_all = Elevation.flatten()

    # --- colormap ONLY for temperature ---
    tmin_plot = float(np.min(t_amb))
    tmax_plot = float(np.max([np.max(Ti) for Ti in temperature_evolution]))
    norm = colors.Normalize(vmin=tmin_plot, vmax=tmax_plot)
    cmap = cm.get_cmap("inferno")

    # --- base vegetation colors + burned ---
    color_tree_base     = np.array([0x1b/255, 0x5e/255, 0x20/255, 1.0])  # dark green
    color_shrub_base    = np.array([0x7f/255, 0xc9/255, 0x7f/255, 1.0])  # light green
    color_branches_base = np.array([0x8d/255, 0x6e/255, 0x63/255, 1.0])  # brown
    color_burned        = np.array([0.08, 0.08, 0.08, 1.0])              # black

    # --- figure ---
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlabel("East distance (km)")
    ax.set_ylabel("North distance (km)")
    ax.set_zlabel("Elevation (km)")

    # Neutral mountain surface (so it doesn't compete with the temperature colors)
    ax.plot_surface(
        X, Y, Elevation,
        color=(0.7, 0.7, 0.7, 1.0),
        alpha=0.25,
        linewidth=0,
        antialiased=True,
        shade=False
    )

    # Temperature colorbar
    mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array([])
    fig.colorbar(mappable, ax=ax, shrink=0.6, pad=0.08, label="Temperature (K)")

    # Scatter per type (depthshade False to keep "clean" colors)
    scat_tree = ax.scatter([], [], [], s=30, marker='^',
                           edgecolors='black', linewidth=0.5, depthshade=False)
    scat_shrub = ax.scatter([], [], [], s=14, marker='o',
                            edgecolors='black', linewidth=0.35, depthshade=False)
    scat_branches = ax.scatter([], [], [], s=16, marker='s',
                               edgecolors='black', linewidth=0.35, depthshade=False)

    # View and bounds
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlim(x_all.min(), x_all.max())
    ax.set_ylim(y_all.min(), y_all.max())
    ax.set_zlim(float(Elevation.min()), float(Elevation.max()))

    # Useful precomputations
    t_ign_flat = t_ign.flatten()
    n_tree   = int(np.sum(mask_tree))
    n_shrub  = int(np.sum(mask_shrub))
    n_branch = int(np.sum(mask_branches))

    def update(frame):
        T_flat = temperature_evolution[frame].flatten()
        F_flat = fuel_evolution[frame].flatten()

        # ---------- TREES ----------
        c_tree = np.tile(color_tree_base, (n_tree, 1))
        tree_burning = T_flat[mask_tree] > t_ign_flat[mask_tree]
        if np.any(tree_burning):
            c_tree[tree_burning] = cmap(norm(T_flat[mask_tree][tree_burning]))
        c_tree[F_flat[mask_tree] < 0.1] = color_burned

        # ---------- SHRUBS ----------
        c_shrub = np.tile(color_shrub_base, (n_shrub, 1))
        shrub_burning = T_flat[mask_shrub] > t_ign_flat[mask_shrub]
        if np.any(shrub_burning):
            c_shrub[shrub_burning] = cmap(norm(T_flat[mask_shrub][shrub_burning]))
        c_shrub[F_flat[mask_shrub] < 0.1] = color_burned

        # ---------- BRANCHES ----------
        c_branch = np.tile(color_branches_base, (n_branch, 1))
        branch_burning = T_flat[mask_branches] > t_ign_flat[mask_branches]
        if np.any(branch_burning):
            c_branch[branch_burning] = cmap(norm(T_flat[mask_branches][branch_burning]))
        c_branch[F_flat[mask_branches] < 0.1] = color_burned

        # Positions (z=elevation)
        scat_tree._offsets3d     = (x_all[mask_tree],     y_all[mask_tree],     z_all[mask_tree])
        scat_shrub._offsets3d    = (x_all[mask_shrub],    y_all[mask_shrub],    z_all[mask_shrub])
        scat_branches._offsets3d = (x_all[mask_branches], y_all[mask_branches], z_all[mask_branches])

        # Colors
        scat_tree.set_facecolors(c_tree)
        scat_shrub.set_facecolors(c_shrub)
        scat_branches.set_facecolors(c_branch)

        time_h = frame * steps_per_frame * dt
        ax.set_title(
            f"Time: {time_h:.2f} h"
        )

        return scat_tree, scat_shrub, scat_branches

    ani = FuncAnimation(fig, update, frames=len(temperature_evolution), interval=interval, blit=False)
    plt.show()
    return ani


ani3d = wildfire_animation_3d_simple(
    hex_grid, X, Y, Elevation, Vegetation,
    temperature_evolution, fuel_evolution, t_ign,
    t_amb, dt, hex_dist,
    steps_per_frame=5, elev=28, azim=-55, interval=50
)

ani3d.save(
    "wildfire_3d.gif",
    writer="pillow",
    fps=15,
    savefig_kwargs={"transparent": False}
)

# ==============================================================================
# 10. PLOTTING FUNCTIONS
# ==============================================================================

def plot_initial_vegetation(hex_grid, vegetation, hex_dist, Nx, Ny):
    """
    Visualize the initial vegetation distribution (t = 0)
    using a hexagonal grid and distinct colors per vegetation type.

    Types:
    1 -> Tree (dark green)
    2 -> Shrub (light green)
    3 -> Dry branches (brown)
    """
    veg_flat = vegetation.astype(int).flatten()

    mask_tree     = (veg_flat == 1)
    mask_shrub    = (veg_flat == 2)
    mask_branches = (veg_flat == 3)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_aspect('equal')
    ax.set_xlabel("East distance (km)")
    ax.set_ylabel("North distance (km)")
    ax.set_title("t = 0 | Initial vegetation distribution")

    # Hex background (animation-like style)
    hex_bg = RegularPolyCollection(
        numsides=6,
        sizes=(hex_dist * 10000,),
        offsets=hex_grid,
        transOffset=ax.transData,
        cmap='Greys',
        clim=(0, 1),
        alpha=0.25
    )
    ax.add_collection(hex_bg)
    hex_bg.set_array(np.zeros(Nx * Ny))

    # Colors
    color_tree     = '#1b5e20'  # dark green
    color_shrub    = '#7fc97f'  # light green
    color_branches = '#8d6e63'  # brown

    # Shrubs
    ax.scatter(hex_grid[mask_shrub, 0], hex_grid[mask_shrub, 1],
               s=12, marker='o',
               c=color_shrub,
               edgecolors='black', linewidth=0.3,
               label='Shrub')

    # Trees
    ax.scatter(hex_grid[mask_tree, 0], hex_grid[mask_tree, 1],
               s=28, marker='^',
               c=color_tree,
               edgecolors='black', linewidth=0.5,
               label='Tree')

    # Dry branches
    ax.scatter(hex_grid[mask_branches, 0], hex_grid[mask_branches, 1],
               s=14, marker='s',
               c=color_branches,
               edgecolors='black', linewidth=0.3,
               label='Dry branches')

    ax.legend(loc='upper right')
    ax.autoscale_view()
    plt.show()


def plot_wind_field(X, Y, wind_vector, Elevation, title="Wind field"):
    """
    Plot the wind velocity field on top of the elevation map.
    X, Y: 2D coordinate meshes.
    wind_vector: constant array [vx, vy] (or U, V matrices if spatially varying).
    Elevation: scalar field for the background.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_aspect('equal')
    ax.set_title(title)
    ax.set_xlabel("East distance (km)")
    ax.set_ylabel("North distance (km)")

    # Background: elevation contour map
    cf = ax.contourf(X, Y, Elevation, cmap='terrain', alpha=0.6, levels=20)
    plt.colorbar(cf, ax=ax, label="Elevation (km)")

    # Build constant wind component grids for quiver
    U = np.full_like(X, wind_vector[0])
    V = np.full_like(Y, wind_vector[1])

    # Quiver (arrows): skip points to avoid clutter
    skip = 4

    Q = ax.quiver(X[::skip, ::skip], Y[::skip, ::skip],
                  U[::skip, ::skip], V[::skip, ::skip],
                  color='black',
                  pivot='mid',
                  scale=15,
                  width=0.003,
                  headwidth=4)

    # Reference arrow key
    ref_speed = 1.0  # km/h
    ax.quiverkey(Q,
                 0.9, 1.03,
                 ref_speed,
                 f'{ref_speed} km/h',
                 labelpos='E', coordinates='axes')

    plt.show()


def plot_hex_field(coords, hex_dist, field_2d, title, cmap, vmin=None, vmax=None, cbar_label=""):
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_aspect('equal')
    ax.set_title(title)
    ax.set_xlabel("East distance (km)")
    ax.set_ylabel("North distance (km)")

    if vmin is None:
        vmin = float(np.min(field_2d))
    if vmax is None:
        vmax = float(np.max(field_2d))

    hexcol = RegularPolyCollection(
        numsides=6,
        sizes=(hex_dist * 10000,),
        offsets=coords,
        transOffset=ax.transData,
        cmap=cmap,
        clim=(vmin, vmax),
        alpha=0.95
    )
    ax.add_collection(hexcol)
    hexcol.set_array(field_2d.flatten())
    plt.colorbar(hexcol, ax=ax, label=cbar_label)
    ax.autoscale_view()
    return fig, ax


def plot_initial_ambient_temperature(hex_grid, t_amb, hex_dist, Nx, Ny):
    """
    Visualize the initial ambient temperature field (t = 0)
    on the hexagonal mesh.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_aspect('equal')
    ax.set_xlabel("East distance (km)")
    ax.set_ylabel("North distance (km)")
    ax.set_title("t = 0 | Initial ambient temperature")

    tmin = float(np.min(t_amb))
    tmax = float(np.max(t_amb))

    hexcol = RegularPolyCollection(
        numsides=6,
        sizes=(hex_dist * 10000,),
        offsets=hex_grid,
        transOffset=ax.transData,
        cmap='plasma',
        clim=(tmin, tmax),
        alpha=0.95
    )
    ax.add_collection(hexcol)
    hexcol.set_array(t_amb.flatten())

    plt.colorbar(hexcol, ax=ax, label="Temperature (K)")
    ax.autoscale_view()
    plt.show()


def plot_diagnostics(Elevation):
    # 1) Terrain and slope
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))

    im0 = ax[0].imshow(Elevation, origin="lower", cmap="terrain")
    ax[0].set_title("Elevation H [m]")
    plt.colorbar(im0, ax=ax[0], fraction=0.046, pad=0.04)

    im1 = ax[1].imshow(slope_magnitude, origin="lower", cmap="viridis")
    ax[1].set_title("Slope")
    plt.colorbar(im1, ax=ax[1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()


# Plot vegetation distribution
plot_initial_vegetation(hex_grid, Vegetation, hex_dist, Nx, Ny)

# Plot initial ambient temperature
plot_initial_ambient_temperature(hex_grid, t_amb, hex_dist, Nx, Ny)

# Plot fields
# plot_hex_field(red_hex, hex_dist, Elevation, "Elevation (km)", "terrain", cbar_label="km")
plot_hex_field(hex_grid, hex_dist, Moisture, "Moisture (0-1)", "Blues", vmin=0, vmax=1, cbar_label="M")
# plot_hex_field(red_hex, hex_dist, t_ign, "Ignition temperature (K) with moisture", "viridis", cbar_label="K")
# plot_hex_field(red_hex, hex_dist, cooling, "Cooling rate (h^-1) with moisture", "plasma", cbar_label="h^-1")
# plot_hex_field(red_hex, hex_dist, heat_release, "Heat release rate (effective K/h) with moisture", "inferno", cbar_label="K/h")

plot_wind_field(X, Y, wind, Elevation, title="Wind direction and topography")
plt.show(block=False)

plot_diagnostics(Elevation)


























