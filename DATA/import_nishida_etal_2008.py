# transcribe Nishida et al. (2008) model into ascii format
import numpy as np
from noisi_v1.util.plot import plot_grid
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from noisi_v1.util.geo import geograph_to_geocent
from smoothing_routines import smooth_weights_CAP_vardegree
import time
from netCDF4 import Dataset
topo_file = 'ETOPO2v2g_f4.nc'
topo = Dataset(topo_file)
topo_lat = topo.variables['y'][:]
topo_lon = topo.variables['x'][:]

path_to_model = '/home/lermert/Dropbox/Japan/Models/velocity_models/\
Nishida_model/vel_iso100km_0.02_0.2Hz_0_60km-3'

crust1_layers = 'crust1.0/crust1.bnds'
crust1_vp = 'crust1.0/crust1.vp'
crust1_vs = 'crust1.0/crust1.vs'
crust1_rho = 'crust1.0/crust1.rho'
crust1_layers = np.loadtxt(crust1_layers)
crust1_moho = np.abs(crust1_layers[:, -1])
# start at the bottom of ice
crust1_vp = np.loadtxt(crust1_vp)[:, :]
crust1_vs = np.loadtxt(crust1_vs)[:, :]
crust1_rho = np.loadtxt(crust1_rho)[:, :]
# cap_degree = 1.
# n_theta = 4
# n_phi = 20
cap_degree = 0.25
cap_degree_crust1 = 1.0
n_theta = 4
n_phi = 20

output_file = 'nishida_2008_model_new.txt'
lat_min, lat_max = (15.99, 60.1)
lon_min, lon_max = (99.99, 170.1)
lat_min_model, lat_max_model = (30.9, 45.2)
lon_min_model, lon_max_model = (129.4, 145.1)
dx = 0.1
dy = 0.1
dz = 1.0
nx = 144  # lat
ny = 158  # lon
nz = 60
interval = 1.0
min_moho, max_moho = (7., 50.)
depth_interval = 1.0  # in km
max_depth = 59.0
depth_to_plot = 40.0
min_thick_sedi = 0.0  # in km, for crust1.0
cm = plt.cm.gist_rainbow
outfilename = 'nishida_model_new'
vp_vs_rho = np.loadtxt(path_to_model)[:, 4:]

# model_params = np.loadtxt(path_to_model)[:, 4:]
deps_km = np.arange(0.5, 60.0, depth_interval)
smash_topo = True  # compress upper layer to account for thicker crust due to
# topography during meshing


def get_moho_from_crust1(lat, lon):

    int_lon_below_point = int(lon)
    int_lat_below_point = int(lat)
    # lat indices: 0 is 89.5; 180 is -89.5
    ix_lat = 90 - int_lat_below_point
    # lon indices: 0 is -179.5, 360 is 179.5
    ix_lon = 180 + int_lon_below_point
    ix_crust = (ix_lat - 1) * 360 + ix_lon

    moho = abs(crust1_layers[ix_crust, -1])
    return(moho)


def get_parameter_profile(depth, parameters, new_depth):
    s = min(len(depth), len(parameters))
    f = interp1d(depth[:s], parameters[:s],
                 kind='cubic', fill_value="extrapolate",
                 bounds_error=False)
    return(f(new_depth))


# def get_profile_from_file(lat, lon):
#     params = []
#     ix_lat = int((lat - lat_min_model) // dx)
#     ix_lon = int((lon - lon_min_model) // dy)
#     if ix_lat > nx or ix_lon > ny:
#         return [[], [], []]
#     if ix_lat < 0 or ix_lon < 0:
#         return [[], [], []]
#     ix_model = ix_lon * nx * nz + ix_lat * nz
#     params.append(np.array(model_params[ix_model: ix_model + nz, 1][::-1]))
#     params.append(np.array(model_params[ix_model: ix_model + nz, 0][::-1]))
#     params.append(np.array(model_params[ix_model: ix_model + nz, 2][::-1]))
#     # return
#     return(params)
def brocher_vs_to_vp(vs_kmps):
    a = 0.9409
    b = 2.0947
    c = -0.8206
    d = 0.2683
    e = -0.0251
    return(a + b * vs_kmps + c * vs_kmps ** 2 +
           d * vs_kmps ** 3 + e * vs_kmps ** 4)


def brocher_vp_to_rho(vp_kmps):
    vp_kmps = np.clip(vp_kmps, vp_kmps.min(), 8.5)
    b = 1.6612
    c = -0.4721
    d = 0.0671
    e = -0.0043
    f = 0.000106
    return(b * vp_kmps + c * vp_kmps ** 2 + d * vp_kmps ** 3 +
           e * vp_kmps ** 4 + f * vp_kmps ** 5)


# def get_profile_from_gmt(lat, lon):
#     os.system('csh Smodel/Smodel.csh {} {} > temp.txt'.format(lon, lat))
#     try:
#         with open('temp.txt', 'r') as fh:
#             topo = float(fh.read().split('\n')[0].split('\t')[-1])
#     except (IndexError, ValueError):
#         return([[np.nan], [np.nan], [np.nan], np.nan])

#     profile = np.loadtxt('temp.txt', skiprows=1)
#     if True in np.isnan(profile[:, 0]):
#         return([[np.nan], [np.nan], [np.nan], np.nan])
#     else:
#         vs = profile[:, 0]
#         vp = brocher_vs_to_vp(vs)
#         rho = brocher_vp_to_rho(vp)
#         return(vp, vs, rho, topo)


def get_profile_from_file(lat, lon):

    lat = geograph_to_geocent(lat)
    if lat > lat_max_model or lat < lat_min_model or\
        lon > lon_max_model or lon < lon_min_model:
            return([[np.nan], [np.nan], [np.nan], np.nan])

    ix_lats = []
    ix_lons = []
    ix_topo_lats = []
    ix_topo_lons = []
    if lat % dx == 0 and lon % dy == 0:
        ix_lats.append(int(round((lat - lat_min_model) / dx, 1)))
        ix_lats.append(max(int(round((lat - lat_min_model) / dx, 1)) - 1, 0))
        ix_lons.append(int(round((lon - lon_min_model) / dy, 1)))
        ix_lons.append(max(int(round((lon - lon_min_model) / dy, 1)) - 1, 0))
        ix_topo_lats.append(np.argmin(abs(topo_lat - lat)) - 1)
        ix_topo_lats.append(np.argmin(abs(topo_lat - lat)))
        ix_topo_lons.append(np.argmin(abs(topo_lon - lon)) - 1)
        ix_topo_lons.append(np.argmin(abs(topo_lon - lon)))
    elif lon % dy == 0:
        ix_lats.append(int(round((lat - lat_min_model) / dx, 1)))
        ix_lons.append(int(round((lon - lon_min_model) / dy, 1)))
        ix_lons.append(max(int(round((lon - lon_min_model) / dy, 1)) - 1, 0))
        ix_topo_lats.append(np.argmin(abs(topo_lat - lat)))
        ix_topo_lons.append(np.argmin(abs(topo_lon - lon)) - 1)
        ix_topo_lons.append(np.argmin(abs(topo_lon - lon)))
    elif lat % dx == 0:
        ix_lats.append(int(round((lat - lat_min_model) / dx, 1)))
        ix_lats.append(max(int(round((lat - lat_min_model) / dx, 1)) - 1, 0))
        ix_lons.append(int(round((lon - lon_min_model) / dy, 1)))
        ix_topo_lats.append(np.argmin(abs(topo_lat - lat)))
        ix_topo_lats.append(np.argmin(abs(topo_lat - lat)) - 1)
        ix_topo_lons.append(np.argmin(abs(topo_lon - lon)))
    else:
        ix_lats.append(int(round((lat - lat_min_model) / dx, 1)))
        ix_lons.append(int(round((lon - lon_min_model) / dy, 1)))
        ix_topo_lats.append(np.argmin(abs(topo_lat - lat)))
        ix_topo_lons.append(np.argmin(abs(topo_lon - lon)))

    topography = 0
    vss = np.zeros(nz)
    for i in range(len(ix_lats)):
        for j in range(len(ix_lons)):
            ix_model = ix_lons[j] * nx * nz + ix_lats[i] * nz
            vss += vp_vs_rho[ix_model: ix_model + nz, 0]
            ix_topo_lat = ix_topo_lats[i]
            ix_topo_lon = ix_topo_lons[j]
            topography += topo.variables['z'][ix_topo_lat, ix_topo_lon] / 1000.

    vss /= (len(ix_lats) * len(ix_lons))
    topography /= (len(ix_topo_lats) * len(ix_topo_lons))
    print(vss[0])
    if True in np.isnan(vss):
        return([np.nan], [np.nan], [np.nan], np.nan)
    vps = brocher_vs_to_vp(vss)
    rhos = brocher_vp_to_rho(vps)
    return(vps, vss, rhos, topography)


def get_smoothed_parameters_crust1(lat, lon):

    sm_lats, sm_lons, sm_wghts = \
        smooth_weights_CAP_vardegree(lon, lat, cap_degree_crust1,
                                     n_theta, n_phi)
    sm_depths = np.zeros(9)
    sm_vps = np.zeros(9)
    sm_vss = np.zeros(9)
    sm_rhos = np.zeros(9)

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

        # Take out shallow sediments
        depths_crust1 = crust1_layers[ix_crust, :]
        crust_thickness = - (depths_crust1[1:] - depths_crust1[:-1])
        thick_sedi = crust_thickness[2:5].sum()
        if i == 0:
            print('Sediment thickness, km ', thick_sedi)
            print('vs, km/s ', crust1_vs[ix_crust, 2])
        if thick_sedi < min_thick_sedi:
            crust1_vp[ix_crust, 2:5] = crust1_vp[ix_crust, 5]
            crust1_vs[ix_crust, 2:5] = crust1_vs[ix_crust, 5]
            crust1_rho[ix_crust, 2:5] = crust1_rho[ix_crust, 5]

        sm_depths += crust1_layers[ix_crust, :] * sm_wghts[i]
        sm_vps += crust1_vp[ix_crust, :] * sm_wghts[i]
        sm_vss += crust1_vs[ix_crust, :] * sm_wghts[i]
        sm_rhos += crust1_rho[ix_crust, :] * sm_wghts[i]
    return(sm_depths, sm_vps, sm_vss, sm_rhos)


def get_parameters_from_crust1(lat, lon, depth_samples):

    # latitude and longitude indices

    vss = []
    vps = []
    rhos = []

    crust_depth_sm, crust1_vp_sm, crust1_vs_sm, crust1_rho_sm =\
        get_smoothed_parameters_crust1(lat, lon)
    crust_thickness = - (crust_depth_sm[1:] - crust_depth_sm[:-1])
    
    # this is what specfem is doing and I copied it for consistency, however: the water layer is not taken into account.
    # this causes slight issues if topography is corrected for, so I will introduce accounting for the water depth
    if not smash_topo:
        z_sedi_1 = crust_thickness[2]
        z_sedi_2 = crust_thickness[2:4].sum()
        z_sedi_3 = crust_thickness[2:5].sum()
        z_cryst_1 = crust_thickness[2:6].sum()
        z_cryst_2 = crust_thickness[2:7].sum()
        z_cryst_3 = crust_thickness[2:].sum()
    else:
        z_sedi_1 = crust_thickness[0:3].sum()
        z_sedi_2 = crust_thickness[0:4].sum()
        z_sedi_3 = crust_thickness[0:5].sum()
        z_cryst_1 = crust_thickness[0:6].sum()
        z_cryst_2 = crust_thickness[0:7].sum()
        z_cryst_3 = crust_thickness.sum()
    # moho = abs(crust_depth_sm[-1])
    topo = -crust_depth_sm[0]

    for dep in depth_samples:
        if dep < z_sedi_1:
            # the sediment velocity is replaced during the smoothing routine
            # by upper crust velocity, if sediments are thinner than min_thick
            vp = crust1_vp_sm[2]
            vs = crust1_vs_sm[2]
            rho = crust1_rho_sm[2]
        elif dep < z_sedi_2:
            vp = crust1_vp_sm[3]
            vs = crust1_vs_sm[3]
            rho = crust1_rho_sm[3]
        elif dep < z_sedi_3:
            vp = crust1_vp_sm[4]
            vs = crust1_vs_sm[4]
            rho = crust1_rho_sm[4]
        elif dep < z_cryst_1:
            vp = crust1_vp_sm[5]
            vs = crust1_vs_sm[5]
            rho = crust1_rho_sm[5]
        elif dep < z_cryst_2:
            vp = crust1_vp_sm[6]
            vs = crust1_vs_sm[6]
            rho = crust1_rho_sm[6]
        elif dep < z_cryst_3:
            vp = crust1_vp_sm[7]
            vs = crust1_vs_sm[7]
            rho = crust1_rho_sm[7]
        else:
            vp = crust1_vp_sm[8]
            vs = crust1_vs_sm[8]
            rho = crust1_rho_sm[8]
        vps.append(vp)
        vss.append(vs)
        rhos.append(rho)
    vps = np.asarray(vps)
    vss = np.asarray(vss)
    rhos = np.asarray(rhos)
    return(vps, vss, rhos, topo)


def get_profiles_from_model_or_crust1(x_lat, x_lon, depth_samples):
    from_crust1 = False
    moho = get_moho_from_crust1(x_lat, x_lon)

    # check if this profile is already in memory
    # if '{}_{}'.format(x_lat, x_lon) in profiles_in_mem:
    #   [vps, vss, rhos, topo] = profiles_in_mem['{}_{}'.format(x_lat, x_lon)]
    # ... or if it is NaN (meaning outside model domain of tomography)
    # else:
    [vps, vss, rhos, topo] = get_profile_from_file(x_lat, x_lon)
    if True in np.isnan(vps):
        vps, vss, rhos, topo =\
            get_parameters_from_crust1(x_lat, x_lon, depth_samples)
        from_crust1 = True
    else:
        # get the values from crust1
        if smash_topo:
            depth_samples = np.linspace(-topo, max_depth +
                                           depth_interval, nz)
            vss = get_parameter_profile(deps_km, vss, depth_samples)
            vps = get_parameter_profile(deps_km, vps, depth_samples)
            rhos = get_parameter_profile(deps_km, rhos, depth_samples)
        else:
            vss = get_parameter_profile(deps_km, vss, depth_samples)
            vps = get_parameter_profile(deps_km, vps, depth_samples)
            rhos = get_parameter_profile(deps_km, rhos, depth_samples)
        # profiles_in_mem['{}_{}'.format(x_lat, x_lon)] = [vps, vss, rhos, topo]

    return(vps, vss, rhos, topo, moho, from_crust1, depth_samples)


def get_smooth_parameters(lat, lon):

    depth_samples = np.linspace(0., max_depth + depth_interval, nz)
    sm_lats, sm_lons, sm_wghts = \
        smooth_weights_CAP_vardegree(lon, lat, cap_degree, n_theta, n_phi)
    sm_vps = np.zeros(nz)
    sm_vss = np.zeros(nz)
    sm_rhos = np.zeros(nz)
    sm_moho = 0
    nan_cnt = 0

    for ix_sm in range(n_theta * n_phi):
        x_lat = sm_lats[ix_sm]
        x_lon = sm_lons[ix_sm]
        weight = sm_wghts[ix_sm]
        vps, vss, rhos, topo, moho, from_crust1, depths =\
            get_profiles_from_model_or_crust1(x_lat, x_lon, depth_samples)
        if from_crust1:
            nan_cnt += 1
        # if many profiles are from crust 1, we have reached the edge of the
        # model, and stronger smoothing is safer
        if nan_cnt > 0.75 * n_phi * n_theta:
            break
        sm_vps += vps * weight
        sm_vss += vss * weight
        sm_rhos += rhos * weight
        sm_moho += moho * weight

    if nan_cnt > 0.75 * n_phi * n_theta:
        print('Edge of model')
        # Run again! But now with more smoothing.
        sm_lats, sm_lons, sm_wghts = \
            smooth_weights_CAP_vardegree(lon, lat, cap_degree_crust1,
                                         n_theta, n_phi)
        sm_vps = np.zeros(depth_samples.shape)
        sm_vss = np.zeros(depth_samples.shape)
        sm_rhos = np.zeros(depth_samples.shape)
        sm_moho = 0

        for ix_sm in range(n_theta * n_phi):
            x_lat = sm_lats[ix_sm]
            x_lon = sm_lons[ix_sm]
            weight = sm_wghts[ix_sm]
            vps, vss, rhos, topo, moho, from_crust1, depths =\
                get_profiles_from_model_or_crust1(x_lat, x_lon, depth_samples)
            sm_vps += vps * weight
            sm_vss += vss * weight
            sm_rhos += rhos * weight
            sm_moho += moho * weight

    return(sm_vps, sm_vss, sm_rhos, sm_moho, depth_samples)


if __name__ == '__main__':
    lats = []
    lons = []
    mohos = []
    vs_plot = []
    vp_plot = []
    rho_plot = []

    out_file = open(output_file, 'w')
    out_file.write("LON\tLAT\tDEP\tVP\tVS\tRHO\tMOHO\n")
    profiles_in_mem = {}
    for lon in np.arange(lon_min, lon_max + interval, interval):
        print(lon)
        print(time.strftime("%H:%M"))
        for lat in np.arange(lat_min, lat_max + interval, interval):
            deps = np.arange(0, max_depth + depth_interval, depth_interval)
            # get the depth index for plotting
            ix_plot = np.argmin(np.abs(deps - depth_to_plot))
            vp, vs, rho, moho, depths = get_smooth_parameters(lat, lon)
            lats.append(lat)
            lons.append(lon)
            mohos.append(moho)
            vs_plot.append(vs[ix_plot])
            vp_plot.append(vp[ix_plot])
            rho_plot.append(rho[ix_plot])
            if moho < min_moho:
                moho = min_moho
            elif moho > max_moho:
                moho = max_moho
            for i in range(len(depths)):
                out_file.write("%4.1f\t%3.1f\t%4.1f\t%8.6f\
                               \t%8.6f\t%8.6f\t%4.1f\n"
                               % (lon, lat, depths[i], vp[i],
                                  vs[i], rho[i], moho))
    lats = np.asarray(lats)
    lons = np.asarray(lons)
    print(np.arange(lon_min, lon_max + interval, interval).shape)
    print(np.arange(lat_min, lat_max + interval, interval).shape)
    mohos = np.asarray(mohos)
    vs_plot = np.asarray(vs_plot)
    vsrange = [vs.min() * 0.8, vs.max() * 1.1]
    vprange = [vp.min() * 0.8, vp.max() * 1.1]
    rhorange = [rho.min() * 0.8, rho.max() * 1.1]

    plot_grid(np.asarray(lons), np.asarray(lats), np.asarray(mohos),
              sequential=True, v_min=15, v=48., size=50, cmap=plt.cm.gist_rainbow,
              quant_unit='Moho depth (km)', outfile=outfilename + '_moho.png',
              axislabelpad=-0.1)

    plot_grid(np.asarray(lons), np.asarray(lats), np.asarray(vs_plot),
              sequential=True, v_min=vsrange[0], v=vsrange[1], size=50,
              quant_unit='vs (km/s)', cmap=cm, outfile=outfilename + '_vs.png',
              axislabelpad=-0.1)

    plot_grid(np.asarray(lons), np.asarray(lats), np.asarray(vp_plot),
              sequential=True, v_min=vprange[0], v=vprange[1], size=50,
              quant_unit='vp (km/s)', cmap=cm, outfile=outfilename + '_vp.png',
              axislabelpad=-0.1)

    plot_grid(np.asarray(lons), np.asarray(lats), np.asarray(rho_plot),
              sequential=True, v_min=rhorange[0], v=rhorange[1], size=50,
              quant_unit='rho (g/m^3)', cmap=cm, outfile=outfilename + '_rho.png',
              axislabelpad=-0.1)
