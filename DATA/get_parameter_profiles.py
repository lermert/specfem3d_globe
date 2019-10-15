import numpy as np
from smoothing_routines import smooth_weights_CAP_vardegree, interp_profile
import os
from noisi_v1.util.geo import geograph_to_geocent


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


def parameter_profile_nishida_etal_2008(model, lat, lon, depth_samples):

    if geograph_to_geocent(lat) > model.lat_max or\
            geograph_to_geocent(lat) < model.lat_min or\
            lon > model.lon_max or lon < model.lon_min:
            return([np.nan])
    ix_lats, ix_lons = model.lat_lon_indices(lat, lon)

    vss = np.zeros(model.nz)
    for i in range(len(ix_lats)):
        for j in range(len(ix_lons)):
            ix_model = ix_lons[j] * model.nx * model.nz +\
                ix_lats[i] * model.nz
            vss += model.vp_vs_rho[ix_model: ix_model + model.nz, 0]
    if True in np.isnan(vss):
        return([np.nan])
    vss /= (len(ix_lats) * len(ix_lons))
    original_depths = model.depths
    vss = interp_profile(original_depths, vss, depth_samples,
                         kind=model.interp)
    return(vss)


def parameter_profile_zheng_etal_2010(model, lat, lon, depth_samples):
    if lat % 1 not in [0.5, 0.0]:
        if lat % 1 < 0.25:
            lat = int(lat)
        elif lat % 1 < 0.75:
            lat = int(lat) + 0.5
        else:
            lat = int(lat) + 1
    if lon % 1 not in [0.5, 0.0]:
        if lon % 1 < 0.25:
            lon = int(lon)
        elif lon % 1 < 0.75:
            lon = int(lon) + 0.5
        else:
            lon = int(lon) + 1
    if lat - int(lat) != 0 and lon - int(lon) != 0:
        f = os.path.join(model.modeldir, '%4.1f_%3.1f_model' % (lon,
                                                                lat))
    elif lat - int(lat) != 0:
        f = os.path.join(model.modeldir, '%g_%3.1f_model' % (lon,
                                                             lat))
    else:
        f = os.path.join(model.modeldir, '%g_%g_model' % (lon, lat))
    try:
        mod = np.loadtxt(f)
    except OSError:
        return([np.nan])

    original_depths = []
    vss = []

    for row in mod:
        dep = row[0]
        vs = row[1]
        vss.append(vs)
        original_depths.append(dep)
    vss = np.asarray(vss)
    original_depths = np.asarray(original_depths)

    # remove the water layer (is included in the model as vs = 0)
    if vss[0] == 0:
        ix_bathy = np.where(vss > 0)[0][0]
        # print('water layer of %g km' % original_depths[ix_bathy])
        vss[0: ix_bathy] = vss[ix_bathy]

    # make more similar to treatment of crust1
    vss[original_depths < model.min_thick_sedi] = \
        vss[np.where(original_depths > model.min_thick_sedi)[0][0]]

    vss = interp_profile(original_depths, vss, depth_samples,
                         kind=model.interp)

    return(vss)


def get_smooth_parameters_crust1(model, lat, lon):
    sm_lats, sm_lons, sm_wghts = \
        smooth_weights_CAP_vardegree(lon, lat, model.cap_degree,
                                     model.n_theta, model.n_phi)
    sm_depths = np.zeros(9)
    sm_vps = np.zeros(9)
    sm_vss = np.zeros(9)
    sm_rhos = np.zeros(9)

    for i in range(len(sm_lats)):
        sm_lat = sm_lats[i]
        sm_lon = sm_lons[i]
        ix_lats, ix_lons = model.lat_lon_indices(sm_lat, sm_lon)

        for ix_lat in ix_lats:
            for ix_lon in ix_lons:
                ix_crust = (ix_lat - 1) * 360 + ix_lon

                # Take out shallow sediments
                depths_crust1 = model.bnds[ix_crust, :]
                crust_thickness = - (depths_crust1[1:] - depths_crust1[:-1])
                thick_sedi = crust_thickness[2:5].sum()
                # if i == 0:
                #   print('Sediment thickness, km ', thick_sedi)
                #   print('vs, km/s ', model.vs[ix_crust, 2])
                if thick_sedi < model.min_thick_sedi:
                    model.vp[ix_crust, 2:5] = model.vp[ix_crust, 5]
                    model.vs[ix_crust, 2:5] = model.vs[ix_crust, 5]
                    model.rho[ix_crust, 2:5] = model.rho[ix_crust, 5]

                sm_depths += model.bnds[ix_crust, :] * sm_wghts[i]
                sm_vps += model.vp[ix_crust, :] * sm_wghts[i]
                sm_vss += model.vs[ix_crust, :] * sm_wghts[i]
                sm_rhos += model.rho[ix_crust, :] * sm_wghts[i]
        sm_depths /= len(ix_lats) * len(ix_lons)
        sm_vps /= len(ix_lats) * len(ix_lons)
        sm_vss /= len(ix_lats) * len(ix_lons)
        sm_rhos /= len(ix_lats) * len(ix_lons)

    return(sm_depths, sm_vps, sm_vss, sm_rhos)


def get_parameters_crust1(model, lat, lon):

    ix_lats, ix_lons = model.lat_lon_indices(lat, lon)

    for ix_lat in ix_lats:
        for ix_lon in ix_lons:
            ix_crust = (ix_lat - 1) * 360 + ix_lon

            # Take out shallow sediments
            depths_crust1 = model.bnds[ix_crust, :]
            crust_thickness = - (depths_crust1[1:] - depths_crust1[:-1])
            thick_sedi = crust_thickness[2:5].sum()
            if thick_sedi < model.min_thick_sedi:
                model.vp[ix_crust, 2:5] = model.vp[ix_crust, 5]
                model.vs[ix_crust, 2:5] = model.vs[ix_crust, 5]
                model.rho[ix_crust, 2:5] = model.rho[ix_crust, 5]

            c1_depths = model.bnds[ix_crust, :]
            c1_vps = model.vp[ix_crust, :]
            c1_vss = model.vs[ix_crust, :]
            c1_rhos = model.rho[ix_crust, :]

    return(c1_depths, c1_vps, c1_vss, c1_rhos)


def parameter_profile_crust1(model, lat, lon, depth_samples):

    vss = []
    vps = []
    rhos = []

    if model.internal_smoothing:
        crust_depth_sm, crust1_vp_sm, crust1_vs_sm, crust1_rho_sm =\
            get_smooth_parameters_crust1(model, lat, lon)
    else:
        crust_depth_sm, crust1_vp_sm, crust1_vs_sm, crust1_rho_sm =\
            get_parameters_crust1(model, lat, lon)
    crust_thickness = - (crust_depth_sm[1:] - crust_depth_sm[:-1])

    # the option "without smash_topo" is what specfem is doing and I copied it
    # for consistency, however: the water layer is not taken into account.
    # this causes slight issues if topography is corrected for, so I will
    # introduce accounting for the water depth
    if not model.smash_topo:
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
    return(vps, vss, rhos)
