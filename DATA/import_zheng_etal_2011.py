# transcribe Zheng et al. (2011) model into ascii format
import os
import numpy as np
from noisi_v1.util.plot import plot_grid
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from netCDF4 import Dataset
from smoothing_routines import smooth_weights_CAP_vardegree
import time

path_to_model = '/home/lermert/Dropbox/Japan/Models/velocity_models/\
MODEL_Zheng_NEChina_2011/Zheng_NEChina_2011/'
topo_file = 'ETOPO2v2g_f4.nc'  # 'ETOPO1_Bed_g_gmt4.grd'
topo = Dataset(topo_file)
topo_lat = topo.variables['y'][:]
topo_lon = topo.variables['x'][:]

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
cap_degree = 1
n_theta = 4
n_phi = 20
smash_topo = True

output_file = 'zheng_2011_model_sed1km.txt'
lat_min, lat_max = (16, 60)
lon_min, lon_max = (100, 170)
lat_min_model, lat_max_model = (16, 60)   # (30, 58)
lon_min_model, lon_max_model = (100, 170)  # (114, 144)
interval = 0.5
min_moho, max_moho = (7., 50.)
depth_interval = 0.05  # in km
max_depth = 51.
depth_to_plot = 40.0
min_thick_sedi = 1.  # in km
subst_crust1 = 1
subst_miss = 0
vsrange = [3.8, 4.6]
missing_locations = [[36.0, 132.5], [36.0, 133.], [36.0, 133.5], [36., 134.],
                     [36.5, 133.], [36.5, 133.5],
                     [37., 133.], [37., 133.5],
                     [37.5, 133.], [37., 136.],
                     [38., 137.], [42., 139.5],
                     [41.5, 140.], [42., 140.], [42.5, 140.],
                     [45., 140.]]
cm = plt.cm.gist_rainbow
outfilename = 'zheng_model_sed1km_new'
# tasks:
# - read in model parameters
# - discard those at extra sampling in depth
# - get vp, rho from vs by Brocher
# - guess moho depth
# - extrapolate values beyond the model domain


def brocher_vs_to_vp(vs_kmps):
    a = 0.9409
    b = 2.0947
    c = -0.8206
    d = 0.2683
    e = -0.0251
    return(a + b * vs_kmps + c * vs_kmps ** 2 +
           d * vs_kmps ** 3 + e * vs_kmps ** 4)


def brocher_vp_to_rho(vp_kmps):
    b = 1.6612
    c = -0.4721
    d = 0.0671
    e = -0.0043
    f = 0.000106
    return(b * vp_kmps + c * vp_kmps ** 2 + d * vp_kmps ** 3 +
           e * vp_kmps ** 4 + f * vp_kmps ** 5)


def suggest_moho(deps, vs):
    dd = np.gradient(vs)
    ixs = np.where(deps < max_moho)
    ixs = np.where(deps[ixs] > min_moho)
    ix = np.argmax(dd[ixs])

    return(deps[ixs][ix])


def get_parameter_profile(depth, parameters, new_depth):

    f = interp1d(depth, parameters, kind='linear', fill_value="extrapolate",
                 bounds_error=False)
    return(f(new_depth))


# def get_smoothed_parameters_zheng(lat, lon, depth_samples):

#     sm_lats, sm_lons, sm_wghts = \
#         smooth_weights_CAP_vardegree(lon, lat, cap_degree, n_theta, n_phi)

#     vp_sm = np.zeros(depth_samples.shape)
#     vs_sm = np.zeros(depth_samples.shape)
#     rho_sm = np.zeros(depth_samples.shape)

#     for i in range(len(sm_lats)):
#         x_lat = sm_lats[i]
#         x_lon = sm_lons[i]
#         wght = sm_wghts[i]

#         vps, vss, rhos, deps_m = read_model_from_file(x_lat, x_lon)

#         vss = get_parameter_profile(deps_m, vss, depth_samples)
#         vps = get_parameter_profile(deps_m, vps, depth_samples)
#         rhos = get_parameter_profile(deps_m, rhos, depth_samples)
#         vs_sm += vss * wght
#         vp_sm += vps * wght
#         rho_sm += rhos * wght

#     return vp_sm, vs_sm, rho_sm


def read_model_from_file(lat, lon):
    if lat - int(lat) != 0 and lon - int(lon) != 0:
        f = os.path.join(path_to_model, '%4.1f_%3.1f_model' % (lon,
                                                               lat))
    elif lat - int(lat) != 0:
        f = os.path.join(path_to_model, '%g_%3.1f_model' % (lon,
                                                            lat))
    else:
        f = os.path.join(path_to_model, '%g_%g_model' % (lon, lat))
    mod = np.loadtxt(f)
    # print(f)
    deps_m = []
    vss = []
    vps = []
    rhos = []

    for row in mod:
        dep = row[0]
        vs = row[1]
        vss.append(vs)
        deps_m.append(dep)
    vss = np.asarray(vss)
    deps_m = np.asarray(deps_m)

    # remove the water layer
    if vss[0] == 0:
        ix_bathy = np.where(vss > 0)[0][0]
        print('water layer of %g km' % deps_m[ix_bathy])
        vss[0: ix_bathy] = vss[ix_bathy]

    # make more similar to treatment of crust1
    vss[deps_m < min_thick_sedi] = vss[np.where(deps_m > min_thick_sedi)[0][0]]

    for i in range(len(vss)):
        # if vss[i] < 0.1:
        #    vss[i] = vss[vss > 0.0].min()
        vp = brocher_vs_to_vp(vss[i])
        if vp > 8.5:
            vp = 8.5
        rho = brocher_vp_to_rho(vp)
        vps.append(vp)
        rhos.append(rho)
    vps = np.asarray(vps)
    rhos = np.asarray(rhos)

    return vps, vss, rhos, deps_m


def get_parameters_from_model(lat, lon, depth_samples):

    ix_topo_lat = np.argmin(abs(topo_lat - lat))
    ix_topo_lon = np.argmin(abs(topo_lon - lon))
    topography = topo.variables['z'][ix_topo_lat, ix_topo_lon] / 1000.

    sm_lats, sm_lons, sm_wghts = \
        smooth_weights_CAP_vardegree(lon, lat, cap_degree, n_theta, n_phi)
    sm_vps = np.zeros(depth_samples.shape)
    sm_vss = np.zeros(depth_samples.shape)
    sm_rhos = np.zeros(depth_samples.shape)
    sm_moho = 0

    for ix_sm in range(n_theta * n_phi):
        x_lat = sm_lats[ix_sm]
        x_lon = sm_lons[ix_sm]
        weight = sm_wghts[ix_sm]

        if x_lat % 1 not in [0.5, 0.0]:
            if x_lat % 1 < 0.25:
                x_lat = int(x_lat)
            elif x_lat % 1 < 0.75:
                x_lat = int(x_lat) + 0.5
            else:
                x_lat = int(x_lat) + 1
        if x_lon % 1 not in [0.5, 0.0]:
            if x_lon % 1 < 0.25:
                x_lon = int(x_lon)
            elif x_lon % 1 < 0.75:
                x_lon = int(x_lon) + 0.5
            else:
                x_lon = int(x_lon) + 1
        try:
            vps, vss, rhos, deps_m = read_model_from_file(x_lat, x_lon)
            moho = suggest_moho(deps_m, vss)
            # print('smoothing, ', x_lat, x_lon, ' Zheng model')
            if smash_topo:
                depth_samples = np.linspace(-topography, max_depth,
                                            len(depth_samples))
            vss = get_parameter_profile(deps_m, vss, depth_samples)
            vps = get_parameter_profile(deps_m, vps, depth_samples)
            rhos = get_parameter_profile(deps_m, rhos, depth_samples)
        except OSError:
            vps, vss, rhos, moho, topography =\
                get_parameters_from_crust1(x_lat, x_lon, depth_samples)
            # print('smoothing, ', x_lat, x_lon, ' Crust1')
        sm_vps += vps * weight
        sm_vss += vss * weight
        sm_rhos += rhos * weight
        sm_moho += moho * weight

    return(sm_vps, sm_vss, sm_rhos, sm_moho, topography)


# def get_parameters_interp(lat, lon, depth_samples):

#     for lat_i in np.arange(lat - interval, lat + interval + 0.5 * interval,
#                            interval):
#         for lon_j in np.arange(lon - interval, lon + 1.5 * interval,
#                                interval):
#             try:
#                 vp, vs, rho, moho, top =\
#                     get_parameters_from_model(lat_i, lon_j, depth_samples)
#             except OSError:
#                 continue
#             if 'vss' in locals():
#                 vps += vp
#                 vss += vs
#                 rhos += rho
#                 moho += moho
#                 top += top
#                 cnt += 1
#             else:
#                 vps = vp
#                 vss = vs
#                 rhos = rho
#                 moho = moho
#                 top = top
#                 cnt = 1
#     vss /= cnt
#     vps /= cnt
#     rhos /= cnt
#     moho /= cnt
#     top /= cnt
#     return(vps, vss, rhos, moho, top)


def get_smoothed_parameters_crust1(lat, lon):

    sm_lats, sm_lons, sm_wghts = \
        smooth_weights_CAP_vardegree(lon, lat, cap_degree, n_theta, n_phi)
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
    # crust 1:
    # lat0 / lon 0 .....point......lat1 / lon1
    # find the integer latitude / longitude below the point
    # find the index that belongs to the value integer_latitude + 0.5
    # same for longitude
    int_lon_below_point = int(lon)
    int_lat_below_point = int(lat)
    # lat indices: 0 is 89.5; 180 is -89.5
    ix_lat = 90 - int_lat_below_point
    # lon indices: 0 is -179.5, 360 is 179.5
    ix_lon = 180 + int_lon_below_point
    ix_crust = (ix_lat - 1) * 360 + ix_lon

    vss = []
    vps = []
    rhos = []
    ix_topo_lat = np.argmin(abs(topo_lat - lat))
    ix_topo_lon = np.argmin(abs(topo_lon - lon))
    topography = topo.variables['z'][ix_topo_lat, ix_topo_lon] / 1000.

    crust_depth_sm, crust1_vp_sm, crust1_vs_sm, crust1_rho_sm =\
        get_smoothed_parameters_crust1(lat, lon)
    crust_thickness = - (crust_depth_sm[1:] - crust_depth_sm[:-1])

    z_sedi_1 = crust_thickness[2]
    z_sedi_2 = crust_thickness[2:4].sum()
    z_sedi_3 = crust_thickness[2:5].sum()
    z_cryst_1 = crust_thickness[2:6].sum()
    z_cryst_2 = crust_thickness[2:7].sum()
    z_cryst_3 = crust_thickness[2:].sum()
    moho = abs(crust_depth_sm[-1])

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
    return(vps, vss, rhos, moho, topography)


if __name__ == '__main__':
    lats = []
    lons = []
    mohos = []
    vs_plot = []
    vp_plot = []
    rho_plot = []
    topo_plot = []

    out_file = open(output_file, 'w')
    out_file.write("LON\tLAT\tDEP\tVP\tVS\tRHO\tMOHO\n")
    for lon in np.arange(lon_min, lon_max + interval, interval):
        print(lon)
        print(time.strftime("%H:%M"))

        for lat in np.arange(lat_min, lat_max + interval, interval):
            deps = np.arange(0, max_depth + depth_interval, depth_interval)
            # get the depth index for plotting
            ix_plot = np.argmin(np.abs(deps - depth_to_plot))
            if lat >= lat_min_model and lat <= lat_max_model and\
                lon >= lon_min_model and lon <= lon_max_model:
                vp, vs, rho, moho, top = get_parameters_from_model(lat, lon, deps)
            else:
                if [lat, lon] in missing_locations and subst_miss:
                    vp, vs, rho, moho, top = get_parameters_interp(lat, lon, deps)
                elif subst_crust1:
                    # substitute missing values from crust1
                    vp, vs, rho, moho, top =\
                        get_parameters_from_crust1(lat, lon, deps)
                else:
                    lats.append(lat)
                    lons.append(lon)
                    mohos.append(np.nan)
                    vs_plot.append(np.nan)
                    vp_plot.append(np.nan)
                    rho_plot.append(np.nan)
                    topo_plot.append(0.0)
                    continue
            lats.append(lat)
            lons.append(lon)
            # if lon == 143.5 and lat in [41., 41.5, 42.]:
            #     moho = 35
            # if lat == 45. and lon == 140.:
            #     moho = 28.
            # if lat == 33.5 and lon == 136.:
            #     moho = 33

            mohos.append(moho)

            vs_plot.append(vs[ix_plot])
            vp_plot.append(vp[ix_plot])
            rho_plot.append(rho[ix_plot])
            topo_plot.append(top)
            for i in range(len(deps)):
                out_file.write("%4.1f\t%3.1f\t%4.1f\t%8.6f\t%8.6f\t%8.6f\t%4.1f\n"
                               % (lon, lat, deps[i], vp[i], vs[i], rho[i], moho))
    print(np.arange(lon_min, lon_max + interval, interval).shape)
    print(np.arange(lat_min, lat_max + interval, interval).shape)

    lats = np.asarray(lats)
    lons = np.asarray(lons)
    mohos = np.asarray(mohos)
    vs_plot = np.asarray(vs_plot)
    topo_plot = np.asarray(topo_plot)
    # print('topo / bathy min max', topo_plot.min(), topo_plot.max())
    # print('Moho min / max', mohos.min(), mohos.max())
    vprange = [brocher_vs_to_vp(vsi) for vsi in vsrange]
    rhorange = [brocher_vp_to_rho(vpi) for vpi in vprange]
    # print('topo', topo_plot.shape)
    # print('lat, lon shapes', lats.shape, lons.shape)
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

    plot_grid(np.asarray(lons), np.asarray(lats), np.asarray(topo_plot),
              sequential=False, size=50,
              quant_unit='Topo / Bathy (km)', outfile=outfilename + '_topo.png',
              axislabelpad=-0.1, v_min=topo_plot.min(), v=topo_plot.max(),
              cmap=plt.cm.gist_earth, title='ETOPO2 smooth')

    plot_grid(np.asarray(lons), np.asarray(lats), np.asarray(rho_plot),
              sequential=True, v_min=rhorange[0], v=rhorange[1], size=50,
              quant_unit='rho (g/m^3)', cmap=cm, outfile=outfilename + '_rho.png',
              axislabelpad=-0.1)
