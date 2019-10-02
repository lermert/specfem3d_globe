import numpy as np
from noisi_v1.util.plot import plot_grid
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from smoothing_routines import smooth_weights_CAP_vardegree

output_file = 'simute_2017_model'
crust1_layers = 'crust1.0/crust1.bnds'
crust1_layers = np.loadtxt(crust1_layers)
cap_degree = 0.5
n_theta = 20
n_phi = 4
lat_min, lat_max = (16, 60)
lon_min, lon_max = (104, 166)  # there are a few samples at lon < 115:
# substitute them by using the ones just inside the model, i.e. 114 -> 115 deg
interval = 0.25
depth_interval = 1.0  # in km
model_depth_interval = 5.0
max_depth = 55.
depth_to_plot = 40.0
max_depth = 55.
min_radius = 6371. - max_depth
cm = plt.cm.gist_rainbow
# tasks:
# - read in model parameters
# - get moho from crust1
# - extrapolate values beyond the model domain

# some model quantities:
modeldir = 'simute_model'
colat_min_model = 30.
colat_max_model = 75.
lon_min_model = 115.
lon_max_model = 160.
r_min_model = 5771.
r_max_model = 6371.
dz = 5.
dx = 0.25
dy = 0.25
nx = 180
ny = 180
nz = 120
radii_model = np.arange(r_max_model - max_depth, r_max_model + dz, dz)
print(radii_model)


def get_smoothed_moho_crust1(lat, lon):
    sm_lats, sm_lons, sm_wghts = \
        smooth_weights_CAP_vardegree(lon, lat, cap_degree, n_theta, n_phi)
    sm_moho = 0.
    sm_topo = 0.

    for i in range(len(sm_lats)):
        sm_lat = sm_lats[i]
        sm_lon = sm_lons[i]
        int_lon_below_point = int(sm_lon)
        int_lat_below_point = int(sm_lat)
        # lat indices: 0 is 89.5; 180 is -89.5
        ix_lat = 90 - int_lat_below_point
        # lon indices: 0 is -179.5, 360 is 179.5
        ix_lon = 180 + int_lon_below_point
        ix_crust = (ix_lat - 1) * 360 + ix_lon

        sm_moho += crust1_layers[ix_crust, -1] * sm_wghts[i]
        sm_topo += crust1_layers[ix_crust, 0] * sm_wghts[i]
    return(sm_moho, sm_topo)


def get_indices(lat, lon):
    colat = 90. - lat
    ix_colat = int((colat - colat_min_model) // dx)
    if ix_colat < 0:
        ix_colat = 0
    if ix_colat > nx - 1:
        ix_colat = nx - 1
    ix_lon = int((lon - lon_min_model) // dy)
    if ix_lon < 0:
        ix_lon = 0
    if ix_lon > ny - 1:
        ix_lon = ny - 1
    return(ix_colat, ix_lon)


def interp_profile(radii_model, param, radii):
    f = interp1d(radii_model, param, kind='cubic', fill_value="extrapolate",
                 bounds_error=False)
    return(f(radii))


def get_smoothed_parameter(lat, lon, param_array, radii):

    sm_lats, sm_lons, sm_wghts = \
        smooth_weights_CAP_vardegree(lon, lat, cap_degree, n_theta, n_phi)
    sm_param = np.zeros(radii.shape)
    for i in range(len(sm_lats)):
        sm_lat = sm_lats[i]
        sm_lon = sm_lons[i]
        ix_colat, ix_lon = get_indices(sm_lat, sm_lon)
        sm_param += sm_wghts[i] *\
                    interp_profile(radii_model,
                                   param_array[ix_colat, ix_lon, :], radii)
    return(sm_param)


# open model
r = np.loadtxt(modeldir + '/block_z')[2:]
ix_min_r = np.argmin(np.abs(r - min_radius))
if min_radius % model_depth_interval != 0:
    ix_min_r -= 1
ix_plot = np.argmin(np.abs(r - (6371 - depth_to_plot)))

vsv = np.loadtxt(modeldir + '/dvsv_ORIG')[2:]
vsv = vsv.reshape(nx, ny, nz)
vsv = vsv[:, :, -len(radii_model):]

vsh = np.loadtxt(modeldir + '/dvsh_ORIG')[2:]
vsh = vsh.reshape(nx, ny, nz)
vsh = vsh[:, :, -len(radii_model):]

vp = np.loadtxt(modeldir + '/dvp_ORIG')[2:]
vp = vp.reshape(nx, ny, nz)
vp = vp[:, :, -len(radii_model):]

rho = np.loadtxt(modeldir + '/drho_ORIG')[2:]
rho = rho.reshape(nx, ny, nz)
rho = rho[:, :, -len(radii_model):]

out_file = open(output_file + '.txt', 'w')
out_file.write("LON\tLAT\tDEP\tVP\tVS\tRHO\tMOHO\n")

lons = np.arange(lon_min, lon_max + interval, interval)
lats = np.arange(lat_min, lat_max + interval, interval)
deps = np.arange(0, max_depth + depth_interval, depth_interval)

plot_lats = []
plot_lons = []
plot_mohos = []
vs_plot = []

# loop over lat, lon
for lon in lons:
    print(lon)
    for lat in lats:
        deps = np.arange(0, max_depth + depth_interval, depth_interval)
        # get topography / bathymetry
        # get moho depth
        moho, topo = get_smoothed_moho_crust1(lat, lon)
        moho = abs(moho)
        deps -= topo
        ix_plot_depth = np.argmin(np.abs(deps - depth_to_plot))
        radii = 6371. - deps

        # interpolate depth samples
        vsv_sm = get_smoothed_parameter(lat, lon, vsv, radii)
        vsh_sm = get_smoothed_parameter(lat, lon, vsh, radii)
        vp_sm = get_smoothed_parameter(lat, lon, vp, radii)
        rho_sm = get_smoothed_parameter(lat, lon, rho, radii)
        mu = 1. / 15. * (10. * rho_sm * vsv_sm ** 2 +
                         5 * rho_sm * vsh_sm ** 2)
        vs_sm = np.sqrt(mu / rho_sm)
        # hopefully correct Voigt average
        for i in range(len(deps)):
            out_file.write("%5.2f\t%4.2f\t%4.1f\t%8.6f\t%8.6f\t%8.6f\t%4.1f\n"
                           % (lon, lat, deps[i], vp_sm[i],
                              vs_sm[i], rho_sm[i], moho))

        plot_lats.append(lat)
        plot_lons.append(lon)
        plot_mohos.append(moho)
        vs_plot.append(vs_sm[ix_plot_depth])

plot_lats = np.asarray(plot_lats)
plot_lons = np.asarray(plot_lons)
plot_mohos = np.asarray(plot_mohos)
vs_plot = np.asarray(vs_plot)
plot_grid(plot_lons, plot_lats, plot_mohos, sequential=True,
          v_min=plot_mohos.min(), v=plot_mohos.max(), size=30,
          outfile=output_file + '_moho.png')
plot_grid(plot_lons, plot_lats, vs_plot, sequential=True,
          v_min=vs_plot.min(), v=vs_plot.max(), size=30,
          outfile=output_file + '_vs.png')
