"added some preamble to make it work, such as the libary imports and the %matplotlib inline command"


import agama, numpy, matplotlib, matplotlib.pyplot as plt, numpy as np, time
import h5py
from tqdm import tqdm
from astropy.convolution import Gaussian2DKernel, convolve_fft
from scipy.ndimage import gaussian_filter


numpy.set_printoptions(precision=6, linewidth=999, suppress=True)
numpy.random.seed(111)
plt.rc('font', size=12)
plt.rc('figure', dpi=75)
plt.rc('mathtext', fontset='stix')
# all dimensional quantities are given in the units of 1 kpc, 1 km/s, 1 Msun;
agama.setUnits(length=1, velocity=1, mass=1)
timeUnitGyr = agama.getUnits()['time'].to_value('Gyr')  # time unit is 1 kpc / (1 km/s)
print('time unit: %.3f Gyr' % timeUnitGyr)




"now it seems I've got to add some of the prior code from the example, such as the function to create the stream particle spray"


"get_rj_rv_R"


def get_rj_vj_R(pot_host, orbit_sat, mass_sat):
    """
    Compute the Jacobi radius, associated velocity, and rotation matrix
    for generating streams using particle-spray methods.
    Arguments:
        pot_host:  an instance of agama.Potential for the host galaxy.
        orbit_sat: the orbit of the satellite, an array of shape (N, 6).
        mass_sat:  the satellite mass (a single number or an array of length N).
    Return:
        rj:  Jacobi radius at each point on the orbit (length: N).
        vj:  velocity offset from the satellite at each point on the orbit (length: N).
        R:   rotation matrix converting from host to satellite at each point on the orbit (shape: N,3,3)
    """
    N = len(orbit_sat)
    x, y, z, vx, vy, vz = orbit_sat.T
    Lx = y * vz - z * vy
    Ly = z * vx - x * vz
    Lz = x * vy - y * vx
    r = (x*x + y*y + z*z)**0.5
    L = (Lx*Lx + Ly*Ly + Lz*Lz)**0.5
    # rotation matrices transforming from the host to the satellite frame for each point on the trajectory
    R = numpy.zeros((N, 3, 3))
    R[:,0,0] = x/r
    R[:,0,1] = y/r
    R[:,0,2] = z/r
    R[:,2,0] = Lx/L
    R[:,2,1] = Ly/L
    R[:,2,2] = Lz/L
    R[:,1,0] = R[:,0,2] * R[:,2,1] - R[:,0,1] * R[:,2,2]
    R[:,1,1] = R[:,0,0] * R[:,2,2] - R[:,0,2] * R[:,2,0]
    R[:,1,2] = R[:,0,1] * R[:,2,0] - R[:,0,0] * R[:,2,1]
    # compute  the second derivative of potential by spherical radius
    der = pot_host.eval(orbit_sat[:,0:3], der=True)
    d2Phi_dr2 = -(x**2  * der[:,0] + y**2  * der[:,1] + z**2  * der[:,2] +
                  2*x*y * der[:,3] + 2*y*z * der[:,4] + 2*z*x * der[:,5]) / r**2
    # compute the Jacobi radius and the relative velocity at this radius for each point on the trajectory
    Omega = L / r**2
    rj = (agama.G * mass_sat / (Omega**2 - d2Phi_dr2))**(1./3)
    vj = Omega * rj
   
    return rj, vj, R


"create_ic_particle_spray"
def create_ic_particle_spray(orbit_sat, rj, vj, R, gala_modified=True):
    """
    Create initial conditions for particles escaping through Largange points,
    using the method of Fardal+2015
    Arguments:
        orbit_sat:  the orbit of the satellite, an array of shape (N, 6).
        rj:  Jacobi radius at each point on the orbit (length: N).
        vj:  velocity offset from the satellite at each point on the orbit (length: N).
        R:   rotation matrix converting from host to satellite at each point on the orbit (shape: N,3,3)
        gala_modified:  if True, use modified parameters as in Gala, otherwise the ones from the original paper.
    Return:
        initial conditions for stream particles, an array of shape (2*N, 6) -
        two points for each point on the original satellite trajectory.
    """
    N = len(rj)
    # assign positions and velocities (in the satellite reference frame) of particles
    # leaving the satellite at both lagrange points (interleaving positive and negative offsets).
    rj = numpy.repeat(rj, 2) * numpy.tile([1, -1], N)
    vj = numpy.repeat(vj, 2) * numpy.tile([1, -1], N)
    R  = numpy.repeat(R, 2, axis=0)
    mean_x  = 2.0
    disp_x  = 0.5 if gala_modified else 0.4
    disp_z  = 0.5
    mean_vy = 0.3
    disp_vy = 0.5 if gala_modified else 0.4
    disp_vz = 0.5
    rx  = numpy.random.normal(size=2*N) * disp_x + mean_x
    rz  = numpy.random.normal(size=2*N) * disp_z * rj
    rvy =(numpy.random.normal(size=2*N) * disp_vy + mean_vy) * vj * (rx if gala_modified else 1)
    rvz = numpy.random.normal(size=2*N) * disp_vz * vj
    rx *= rj
    offset_pos = numpy.column_stack([rx,  rx*0, rz ])  # position and velocity of particles in the reference frame
    offset_vel = numpy.column_stack([rx*0, rvy, rvz])  # centered on the progenitor and aligned with its orbit
    ic_stream = numpy.tile(orbit_sat, 2).reshape(2*N, 6)   # same but in the host-centered frame
    ic_stream[:,0:3] += numpy.einsum('ni,nij->nj', offset_pos, R)
    ic_stream[:,3:6] += numpy.einsum('ni,nij->nj', offset_vel, R)
    return ic_stream


"create_stream_particle_spray"


def create_stream_particle_spray(time_total, num_particles, pot_host, posvel_sat, mass_sat, gala_modified=True):
    """
    Construct a stream using the particle-spray method.
    Arguments:
        time_total:  duration of time for stream generation
            (positive; orbit of the progenitor integrated from present day (t=0) back to time -time_total).
        num_particles:  number of points in the stream (even; divided equally between leading and trailing arms).
        pot_host:    an instance of agama.Potential for the host galaxy.
        posvel_sat:  present-day position and velocity of the satellite (array of length 6).
        mass_sat:    the satellite mass (a single number or an array of length num_particles//2).
        gala_modified:  if True, use modified parameters as in Gala, otherwise the ones from the original paper.
    Return:
        xv_stream: position and velocity of stream particles at present time (shape: num_particles, 6).
    """
    # number of points on the orbit: each point produces two stream particles (leading and trailing arms)
    N = num_particles//2


    # integrate the orbit of the progenitor from its present-day posvel (at time t=0)
    # back in time for an interval time_total, storing the trajectory at N points
    time_sat, orbit_sat = agama.orbit(
        potential=pot_host, ic=posvel_sat, time=-time_total, trajsize=N+1)
    # remove the 0th point (the present-day posvel) and reverse the arrays to make them increasing in time
    time_sat  = time_sat [1:][::-1]
    orbit_sat = orbit_sat[1:][::-1]


    # at each point on the trajectory, create a pair of seed initial conditions
    # for particles released at both Lagrange points
    rj, vj, R = get_rj_vj_R(pot_host, orbit_sat, mass_sat)
    ic_stream = create_ic_particle_spray(orbit_sat, rj, vj, R, gala_modified)
    time_seed = numpy.repeat(time_sat, 2)
    xv_stream = agama.orbit(
        potential=pot_host, ic=ic_stream,
        timestart=time_seed,  # starting time for each orbit (negative)
        time=-time_seed,      # integration time for each orbit: time_start + time = 0 (end time, i.e. now)
        trajsize=1,           # each orbit produces just one point (at the end of the integration)
        separateTime=True,    # report times (not needed) and trajectories as separate arrays
    )[1].reshape(len(ic_stream), 6)  # the array of endpoints of each trajectory
    return xv_stream


"okay i've added them! that's all the preliminary functions defined "



# sky coordinates in degrees, distance in kpc, proper motion in mas/yr, heliocentric line-of-sight velocity in km/s
ra0, dec0 = 229.0, -0.124
dist0 = 22.9
pmra0, pmdec0 = -2.296, -2.257
vlos0 = -58.7


# use built-in coordinate conversion routines from Agama instead of Astropy: faster, no external dependencies,
# but need to do some extra steps, e.g., convert from degrees to radians and
# from mas/yr to km/s/kpc, the base units used throughout this example (the factor 4.74 below)
l0, b0, pml0, pmb0 = agama.transformCelestialCoords(agama.fromICRStoGalactic,
    ra0*numpy.pi/180, dec0*numpy.pi/180, pmra0, pmdec0)
posvel_sat = agama.getGalactocentricFromGalactic(l0, b0, dist0, pml0*4.74, pmb0*4.74, vlos0)


# Rotation matrix for conversion from ICRS to the Pal5 stream coordinate system (taken from Gala)
fromICRStoPal5 = numpy.array([
    [-0.65582036, -0.75491389, -0.00216422],
    [-0.62865701,  0.54771927, -0.55208149],
    [ 0.41795937, -0.36070573, -0.83378735],
])


# fiducial potential of the Milky Way, consisting of a disk and a spherical halo


"drawing priors etc. "


rng = np.random.default_rng()  # optionally pass a seed, e.g. default_rng(42)


def sample_priors(rng):
    """Draw all host, stream, and DECaLS observing parameters in one dictionary."""
    mn_mass = float(rng.uniform(2e10, 1.2e11))         # M_sun
    mn_scale_radius = float(rng.uniform(1.5, 6.0))     # kpc
    mn_scale_height = float(rng.uniform(0.1, 0.8))     # kpc
    while mn_scale_height >= mn_scale_radius:
        mn_scale_height = float(rng.uniform(0.1, 0.8))


    mass_sat = float(10.0 ** rng.uniform(6.8, 8.7))    # M_sun (or 6.0, 7.7 for ~1:1000 mass ratio)
    radius_sat = float(rng.uniform(0.003, 0.005))      # kpc
    time_total = float(rng.uniform(1.5, 6.0))          # ~1.5 to 5.9 Gyr
    num_particles = int(1e4)                           # number of particles in the stream
    r_current = float(rng.uniform(10.0, 40.0))         # present-day galactocentric radius (kpc)


    nfw_mass = float(rng.uniform(3e11, 3e12))          # M_sun
    nfw_scale_radius = float(rng.uniform(10.0, 40.0))  # kpc


    host_dist_mpc = float(rng.uniform(15.0, 35.0))     # Mpc
    viewing_angles = (
        float(rng.uniform(0.0, np.pi / 2)),
        float(rng.uniform(0.0, 2 * np.pi)),
    )
    seeing_fwhm_arcsec = tuple(float(x) for x in rng.uniform([1.0, 0.9, 0.8], [1.7, 1.6, 1.5]))
    sky_sigma_grz = tuple(float(x) for x in rng.uniform([0.0025, 0.004, 0.008], [0.006, 0.009, 0.018]))
    stream_mtl_grz = tuple(float(x) for x in rng.uniform([1.5, 1.2, 1.0], [2.6, 2.1, 1.7]))
    host_mtl_grz = tuple(float(x) for x in rng.uniform([2.5, 2.0, 1.6], [4.5, 3.6, 2.9]))
    center_offset_arcsec = tuple(float(x) for x in rng.uniform(-2.5, 2.5, size=2))

    # describing these variables within a dictionary to recall later.

    return {
        "mn_mass": mn_mass,
        "mn_scale_radius": mn_scale_radius,
        "mn_scale_height": mn_scale_height,
        "nfw_mass": nfw_mass,
        "nfw_scale_radius": nfw_scale_radius,
        "mass_sat": mass_sat,
        "radius_sat": radius_sat,
        "time_total": time_total,
        "num_particles": num_particles,
        "r_current": r_current,
        "host_dist_mpc": host_dist_mpc,
        "viewing_angles": viewing_angles,
        "seeing_fwhm_arcsec": seeing_fwhm_arcsec,
        "sky_sigma_grz": sky_sigma_grz,
        "stream_mtl_grz": stream_mtl_grz,
        "host_mtl_grz": host_mtl_grz,
        "center_offset_arcsec": center_offset_arcsec,
    }




def generate_sbi_pair(rng):
    params = sample_priors(rng)


# same coordinate conversion steps in reverse; also convert to stream coordinates (phi1 along, phi2 across)
def get_observed_coords(xv):
    l, b, dist, pml, pmb, vlos = agama.getGalacticFromGalactocentric(*xv.T)
    ra, dec = agama.transformCelestialCoords(agama.fromGalactictoICRS, l, b)
    phi1, phi2 = agama.transformCelestialCoords(fromICRStoPal5, ra, dec)
    return ra * 180/numpy.pi + 360, dec * 180/numpy.pi, vlos, phi1 * 180/numpy.pi, phi2 * 180/numpy.pi


"okay i'm going to add the galaxy coordinate conversion here"


def _project_particles_to_sky(xyz_kpc, host_dist_mpc, viewing_angles,center_offset_arcsec ):
    """Rotate 3D host-centered coordinates (kpc) and project to sky offsets (arcsec)."""
    inc, pa = viewing_angles
    x, y, z = xyz_kpc[:, 0], xyz_kpc[:, 1], xyz_kpc[:, 2]


    # Rotate around x by inclination, then around line-of-sight z by position angle
    y_inc = y * np.cos(inc) - z * np.sin(inc)
    x_sky = x * np.cos(pa) - y_inc * np.sin(pa)
    y_sky = x * np.sin(pa) + y_inc * np.cos(pa)


    dist_kpc = host_dist_mpc * 1000.0
    rad_to_arcsec = 180.0 * 3600.0 / np.pi
    dx_arcsec = (x_sky / dist_kpc) * rad_to_arcsec + center_offset_arcsec[0]
    dy_arcsec = (y_sky / dist_kpc) * rad_to_arcsec + center_offset_arcsec[1]
    return dx_arcsec, dy_arcsec


def _mass_to_nanomaggies(mass_msun, m_sun_band, mtl_band, dist_modulus):
    """Convert stellar particle masses (M_sun) to flux in nanomaggies."""
    lum_solar = mass_msun / mtl_band
    M_abs = m_sun_band - 2.5 * np.log10(lum_solar)
    m_app = M_abs + dist_modulus
    return 10.0 ** ((22.5 - m_app) / 2.5)


"adding the galaxy image_cube here"


def render_decals_cutout(
    xv_stream,
    stream_particle_mass_msun,
    pot_host,
    host_dist_mpc,
    seeing_fwhm_arcsec,
    stream_mtl_grz,
    host_mtl_grz,
    viewing_angles,
    center_offset_arcsec,
    sky_sigma_grz,
    n_disk_particles=600_000,
    asinh_softening=0.005,
    rng=None,


):
    """
    Project both the host stellar disk (pot_host[0], Miyamoto-Nagai) and
    stream particles into a DECaLs-style (3, 512, 512) cube in nanomaggies.
    """
    if rng is None:
        rng = np.random.default_rng()
    stream_mass = np.broadcast_to(stream_particle_mass_msun, len(xv_stream))


    # Sample 3D stellar tracer positions (kpc) and masses (M_sun) from the MN disk
    # pot_host[0] is the Miyamoto-Nagai component; pot_host[1] (NFW) is dark matter
    xyz_disk, disk_mass = pot_host[0].sample(n_disk_particles)


    npix = 512
    pix_scale = 0.524  # arcsec / pixel
    half_fov_arcsec = 0.5 * npix * pix_scale  # 134.144 arcsec
    edges = np.linspace(-half_fov_arcsec, half_fov_arcsec, npix + 1)


    # Project both components using the exact same observer geometry
    dx_disk, dy_disk = _project_particles_to_sky(
        xyz_disk, host_dist_mpc, viewing_angles, center_offset_arcsec
    )


    if len(xv_stream) > 0:
        dx_stream, dy_stream = _project_particles_to_sky(xv_stream[:, :3], host_dist_mpc, viewing_angles, center_offset_arcsec)
        stream_mass = np.broadcast_to(stream_particle_mass_msun, len(xv_stream))
    else:
        # Empty arrays for H0 (No stream)
        dx_stream, dy_stream, stream_mass = np.array([]), np.array([]), np.array([])
   


    M_sun_grz = np.array([5.11, 4.65, 4.53])
    dist_modulus = 5.0 * np.log10(host_dist_mpc) + 25.0
    noisy_cube = np.zeros((3, npix, npix), dtype=np.float32)


    for b in range(3):
        flux_stream = _mass_to_nanomaggies(
            stream_mass, M_sun_grz[b], stream_mtl_grz[b], dist_modulus
        )
        flux_disk = _mass_to_nanomaggies(
            disk_mass, M_sun_grz[b], host_mtl_grz[b], dist_modulus
        )


        # Bin stream and host disk onto the same 512x512 grid
        raw_stream, _, _ = np.histogram2d(
            dy_stream, dx_stream, bins=[edges, edges], weights=flux_stream
        )
        raw_disk, _, _ = np.histogram2d(
            dy_disk, dx_disk, bins=[edges, edges], weights=flux_disk
        )


        # Sum fluxes before PSF convolution (convolution is linear, saving 3 FFTs)
        raw_total = raw_stream + raw_disk


        psf_sigma_pix = (seeing_fwhm_arcsec[b] / 2.3548) / pix_scale
        convolved = gaussian_filter(raw_total, sigma=psf_sigma_pix, mode='constant', cval=0.0)


        if sky_sigma_grz is not None:
            sky_noise = rng.normal(0.0, sky_sigma_grz[b], size=(npix, npix))
            convolved = convolved + sky_noise


        noisy_cube[b] = convolved.astype(np.float32)


        s0 = np.float32(asinh_softening)
    nn_cube = np.arcsinh(noisy_cube / s0).astype(np.float32)


    return nn_cube, noisy_cube, xyz_disk




# SBI forwarding model: (ra, dec, vlos) -> (phi1, phi2)


def generate_sbi_pair(rng):
    """
    Brings all 4 models together to create the H0 and H1 hypotheses for the network.
    """
    # 1 & 2: Sample parameters and generate physics (Host + Stream)
    params = sample_priors(rng)
    pot_host, xv_stream, particle_mass = generate_physics_models(params)
   
    # Common jitter applied to both images to simulate telescope pointing error    
    # 3 & 4: Render Hypothesis 0 (Host Only - pass empty stream)
    nn_cube_H0,_,_ = render_decals_cutout(
        xv_stream=np.zeros((0, 6)),
        stream_particle_mass_msun=0.0,
        pot_host=pot_host,
        host_dist_mpc=params["host_dist_mpc"],
        seeing_fwhm_arcsec=params["seeing_fwhm_arcsec"],
        stream_mtl_grz=params["stream_mtl_grz"],
        host_mtl_grz=params["host_mtl_grz"],
        viewing_angles=params["viewing_angles"],
        center_offset_arcsec=params["center_offset_arcsec"],
        sky_sigma_grz=params["sky_sigma_grz"],
        rng=rng,
    )
   
    # 3 & 4: Render Hypothesis 1 (Host + Stream)
    nn_cube_H1,_,_ = render_decals_cutout(
        xv_stream=xv_stream,         # Full stream array
        stream_particle_mass_msun=particle_mass,
        pot_host=pot_host,
        host_dist_mpc=params["host_dist_mpc"],
        seeing_fwhm_arcsec=params["seeing_fwhm_arcsec"],
        stream_mtl_grz=params["stream_mtl_grz"],
        host_mtl_grz=params["host_mtl_grz"],
        viewing_angles=params["viewing_angles"],
        center_offset_arcsec=params["center_offset_arcsec"],
        sky_sigma_grz=params["sky_sigma_grz"],
        rng=rng,
    )


   # Build the 3-element host context vector expected by dset_meta
    h_meta = np.array(
        [
            np.log10(params["host_dist_mpc"]),
            np.log10(params["mn_mass"]),
            params["mn_scale_radius"] / params["mn_scale_height"],
        ],
        dtype=np.float32,
    )


    # Return the paired tensors and the host metadata context vector
    return {
        "H0_image": nn_cube_H0,  # Label 0
        "H1_image": nn_cube_H1,  # Label 1
        "h_meta": h_meta,
    }


"adding the generate_physics_models function to create the host potential and stream particles"


def generate_physics_models(params):
    """
    Build the AGAMA gravitational potentials and generate the 6D stream track
    using the provided Lagrangian particle-spray method.
    """
    pot_host = agama.Potential(
        dict(
            type="MiyamotoNagai",
            mass=params["mn_mass"],
            scaleRadius=params["mn_scale_radius"],
            scaleHeight=params["mn_scale_height"],
        ),
        dict(
            type="NFW",
            mass=params["nfw_mass"],
            scaleRadius=params["nfw_scale_radius"],
        ),
    )


    # 2. Present-Day Satellite Phase-Space (posvel_sat)
    # Calculate local circular velocity to set a bound, moderately eccentric present-day orbit
    v_circ = numpy.sqrt(pot_host.force([params["r_current"], 0, 0])[0] * -params["r_current"])
    posvel_sat = numpy.array([
        params["r_current"], 0.0, 0.0,          # Present-day x, y, z
        0.0, v_circ * 0.8, v_circ * 0.3         # Present-day vx, vy, vz
    ])
   
    # 3. Stellar Stream Model (Particle-Spray)
    # Uses your exact create_stream_particle_spray function
    xv_stream = create_stream_particle_spray(
        time_total=params["time_total"],
        num_particles=params["num_particles"],
        pot_host=pot_host,
        posvel_sat=posvel_sat,
        mass_sat=params["mass_sat"],
        gala_modified=True,
    )


    # Calculate mass per individual stream particle to pass to the renderer
    stream_particle_mass = params["mass_sat"] / params["num_particles"]
   
    return pot_host, xv_stream, stream_particle_mass




def build_sbi_dataset(output_path="D:/sbi_stream_dataset.h5", num_pairs=10_000, seed=42):
    rng = np.random.default_rng(seed)
    total_images = num_pairs * 2  # 20,000 total images


    with h5py.File(output_path, "w") as f:
        dset_images = f.create_dataset(
            "images", shape=(total_images, 3, 512, 512), dtype="float16"
        )
        dset_labels = f.create_dataset(
            "labels", shape=(total_images,), dtype="int64"
        )
        dset_meta = f.create_dataset(
            "h_meta", shape=(total_images, 3), dtype="float32"
        )


        for i in tqdm(range(num_pairs), desc="Generating SBI pairs"):
            pair = generate_sbi_pair(rng)


            idx_H0 = 2 * i
            idx_H1 = 2 * i + 1


            # Save H0 (Host Only -> label 0) directly to D: drive
            dset_images[idx_H0] = pair["H0_image"].astype(np.float16)
            dset_labels[idx_H0] = 0
            dset_meta[idx_H0] = pair["h_meta"]


            # Save H1 (Host + Stream -> label 1) directly to D: drive
            dset_images[idx_H1] = pair["H1_image"].astype(np.float16)
            dset_labels[idx_H1] = 1
            dset_meta[idx_H1] = pair["h_meta"]


            if (i + 1) % 100 == 0:
                f.flush()


    print(f"Saved {total_images} images to {output_path}")




if __name__ == "__main__":
    build_sbi_dataset(output_path="D:/sbi_stream_dataset.h5", num_pairs=10_000, seed=42)



