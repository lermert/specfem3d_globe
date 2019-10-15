# transcribe Nishida et al. (2008) model into ascii format
import numpy as np
from noisi_v1.util.plot import plot_grid
from smoothing_routines import smooth_weights_CAP_vardegree
import time
from netCDF4 import Dataset
from get_parameter_profiles import parameter_profile_nishida_etal_2008
from get_parameter_profiles import parameter_profile_zheng_etal_2010
from get_parameter_profiles import parameter_profile_crust1
from get_parameter_profiles import brocher_vp_to_rho, brocher_vs_to_vp
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from noisi_v1.util.geo import geograph_to_geocent

topo_file = 'ETOPO2v2g_f4.nc'
topo = Dataset(topo_file)
topo_lat = topo.variables['y'][:]
topo_lon = topo.variables['x'][:]
cap_degree_nishida = 0.25
cap_degree_zheng = 0.75
cap_degree_crust1 = 1.0
n_theta = 4
n_phi = 20
output_file = 'combined_model_new.txt'
lat_min, lat_max = (16, 60.)
lon_min, lon_max = (100, 170.)
interval = 0.2
min_moho, max_moho = (7., 50.)
depth_interval = 0.1  # in km
max_depth = 50
smash_topo = True  # compress upper layer to account for thicker crust due to
# topography during meshing
interpolation_method = 'linear'
common_depth = np.arange(0., max_depth + depth_interval, depth_interval)
pal = sns.color_palette('hls', 3)
cm = ListedColormap(pal)
# depth_to_plot = 40.0
# vsrange = [3.8, 4.6]
depth_to_plot = 0
plotdots = 100
vsrange = [1.0, 4.0]
internally_smooth_crust1 = True


def get_topography(lat, lon):
    ix_topo_lat = np.argmin(abs(topo_lat - lat))
    ix_topo_lon = np.argmin(abs(topo_lon - lon))
    topography = topo.variables['z'][ix_topo_lat, ix_topo_lon] / 1000.
    return(topography)


def suggest_moho(deps, vs):
    dd = np.gradient(vs)
    ixs = np.where(deps < max_moho)
    ixs = np.where(deps[ixs] > min_moho)
    ix = np.argmax(dd[ixs])

    return(deps[ixs][ix])


def get_moho_from_crust1(model, lat, lon):

    int_lon_below_point = int(lon)
    int_lat_below_point = int(lat)
    # lat indices: 0 is 89.5; 180 is -89.5
    ix_lat = 90 - int_lat_below_point
    # lon indices: 0 is -179.5, 360 is 179.5
    ix_lon = 180 + int_lon_below_point
    ix_crust = (ix_lat - 1) * 360 + ix_lon

    moho = abs(model.bnds[ix_crust, -1])
    if moho < min_moho:
        moho = min_moho
    if moho > max_moho:
        moho = max_moho
    return(moho)


class model(object):
    """docstring for model"""

    def __init__(self, model_name):
        if model_name == 'nishida':
            self.model_name = 'nishida'
            self.modeldir = 'nishida_model'
            self.lat_min = 30.9
            self.lat_max = 45.2
            self.lon_min = 129.4
            self.lon_max = 145.1
            self.dx = 0.1
            self.dy = 0.1
            self.dz = 1.0
            self.nx = 144  # lat
            self.ny = 158  # lon
            self.nz = 60
            self.vp_vs_rho = np.loadtxt(self.modeldir +
                '/vel_iso100km_0.02_0.2Hz_0_60km-3')[:, 4:]
            self.depths = np.arange(0., 60., 1.)
            self.interp = 'linear'
            self.cap_degree = cap_degree_nishida
            self.n_theta = n_theta
            self.n_phi = n_phi
            self.smash_topo = smash_topo
            self.create_model_cover_map()
            # only hold in memory vs, vp, rho

        if model_name == 'zheng':
            self.model_name = 'zheng'
            self.modeldir = '/home/lermert/Dropbox/Japan/Models/\
velocity_models/MODEL_Zheng_NEChina_2011/Zheng_NEChina_2011/'
            self.lat_min = 30
            self.lat_max = 58
            self.lon_min = 114
            self.lon_max = 144
            self.dx = 0.5
            self.dy = 0.5
            self.nx = 57
            self.ny = 61
            self.interp = 'linear'
            self.min_thick_sedi = 1.0
            self.cap_degree = cap_degree_zheng
            self.n_theta = n_theta
            self.n_phi = n_phi
            self.smash_topo = smash_topo
            self.create_model_cover_map()

        if model_name == 'crust1':
            self.lat_min = -90
            self.lat_max = 90
            self.lon_min = -180
            self.lon_max = 180
            self.model_name = 'crust1'
            self.vp = np.loadtxt('crust1.0/crust1.vp')
            self.vs = np.loadtxt('crust1.0/crust1.vs')
            self.rho = np.loadtxt('crust1.0/crust1.rho')
            self.bnds = np.loadtxt('crust1.0/crust1.bnds')
            self.nx = 90.
            self.ny = 180.
            self.nz = 8
            self.moho = np.abs(self.bnds[:, -1])
            self.min_thick_sedi = 2.0  # in km, for crust1.0
            self.cap_degree = cap_degree_crust1
            self.n_theta = n_theta
            self.n_phi = n_phi
            self.smash_topo = smash_topo
            self.internal_smoothing = internally_smooth_crust1

    def create_model_cover_map(self):
        self.coverage = np.zeros((self.nx, self.ny))
        lats = np.arange(self.lat_min, self.lat_max + self.dx / 2, self.dx)
        for i, lat in enumerate(lats):
            if lat < self.lat_min or lat > lat_max:
                self.coverage[i, :] = 0
                continue

            for j, lon in enumerate(np.arange(self.lon_min, self.lon_max +
                                    self.dy / 2, self.dy)):

                if lon < self.lon_min or lon > lon_max:
                    self.coverage[:, j] = 0
                    continue

                vps, vss, rhos = self.get_profile_from_file(lat, lon)

                if True in np.isnan(vss):
                    self.coverage[i, j] = 0
                else:
                    self.coverage[i, j] = 1

    def lat_lon_indices(self, lat, lon):
        ix_lats = []
        ix_lons = []
        if lat < self.lat_min or lat > self.lat_max or lon < self.lon_min or\
                lon > self.lon_max:
            return(ix_lats, ix_lons)

        if self.model_name == 'nishida':
            lat = geograph_to_geocent(lat)
            # the following is built upon the assumtion of grid spacing
            # being 0.1 degree in either direction
            ix_lats.append(int(round((lat - self.lat_min) / self.dx, 1)))
            ix_lons.append(int(round((lon - self.lon_min) / self.dy, 1)))

            if lon % self.dy == 0:
                ix_lons.append(max(int(round((lon - self.lon_min) /
                                             self.dy, 1)) - 1, 0))
            if lat % self.dx == 0:
                ix_lats.append(max(int(round((lat - self.lat_min) /
                                             self.dx, 1)) - 1, 0))

        elif self.model_name == 'crust1':
            int_lon_below_point = int(lon)
            # lat indices: 0 is 89.5; 180 is -89.5
            # more like colat indices
            ix_lat = int(90 - lat)
            # lon indices: 0 is -179.5, 360 is 179.5
            ix_lon = 180 + int_lon_below_point
            if lat == int(lat):
                ix_lats.append(max(ix_lat - 1, 0))
            ix_lats.append(ix_lat)
            if lon == int(lon):
                ix_lons.append(max(ix_lon - 1, 0))
            ix_lons.append(ix_lon)

        elif self.model_name == 'zheng':
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
            ix_lats.append(int((lat - self.lat_min) / self.dx))
            ix_lons.append(int((lon - self.lon_min) / self.dy))
        # end if
        return(ix_lats, ix_lons)

    def get_profile_from_file(self, lat, lon):

        if self.model_name == 'nishida':
            if smash_topo:
                topography = get_topography(lat, lon)
                depth_samples = np.linspace(-topography, max_depth,
                                            len(common_depth))
            else:
                depth_samples = common_depth
            vss = parameter_profile_nishida_etal_2008(self, lat, lon,
                                                      depth_samples)

        elif self.model_name == 'zheng':
            if smash_topo:
                topography = get_topography(lat, lon)
                depth_samples = np.linspace(-topography, max_depth,
                                            len(common_depth))
            else:
                depth_samples = common_depth
            vss = parameter_profile_zheng_etal_2010(self, lat, lon,
                                                    depth_samples)

        elif self.model_name == 'crust1':
            if smash_topo:
                topography = get_topography(lat, lon)
                depth_samples = np.linspace(-topography, max_depth,
                                            len(common_depth))
            else:
                depth_samples = common_depth
            vps, vss, rhos = parameter_profile_crust1(self, lat, lon,
                                                      depth_samples)

        if self.model_name in ['nishida', 'zheng']:
            if True in np.isnan(vss):
                vps = vss
                rhos = vps
            else:
                vps = brocher_vs_to_vp(vss)
                vps = np.clip(vps, vps.min(), 8.5)
                rhos = brocher_vp_to_rho(vps)
        return(vps, vss, rhos)


def determine_smoothing(models, lat, lon):

    mod1, mod2, mod3 = models
    # first, determine how much smoothing we need
    sm_lats, sm_lons, sm_wghts = \
        smooth_weights_CAP_vardegree(lon, lat, mod1.cap_degree,
                                     n_theta, n_phi)
    cover_cnt = 0
    for sm_lat in sm_lats:
        for sm_lon in sm_lons:
            ix_lats, ix_lons = mod1.lat_lon_indices(sm_lat, sm_lon)

            # available?
            loc_cov = 0
            for i, ix_lat in enumerate(ix_lats):
                ix_lon = ix_lons[i]
                loc_cov += mod1.coverage[ix_lat, ix_lon]
            if loc_cov == len(ix_lats) and loc_cov > 0:
                cover_cnt += 1
    if cover_cnt == mod1.n_theta * mod1.n_phi:
        # print('Using Nishida Model for lat, lon ', lat, lon)
        return(mod1.cap_degree, mod1.n_theta, mod1.n_phi)

    sm_lats, sm_lons, sm_wghts = \
        smooth_weights_CAP_vardegree(lon, lat, mod2.cap_degree,
                                     n_theta, n_phi)
    cover_cnt = 0
    for sm_lat in sm_lats:
        for sm_lon in sm_lons:
            ix_lats, ix_lons = mod2.lat_lon_indices(sm_lat, sm_lon)

            # available?
            loc_cov = 0
            for i, ix_lat in enumerate(ix_lats):
                ix_lon = ix_lons[i]
                loc_cov += mod2.coverage[ix_lat, ix_lon]
            if loc_cov == len(ix_lats) and loc_cov > 0:
                cover_cnt += 1
    if cover_cnt == mod2.n_theta * mod2.n_phi:
        # print('Using Zheng Model for lat, lon ', lat, lon)
        return(mod2.cap_degree, mod2.n_theta, mod2.n_phi)
    else:
        # print('Using Crust1 Model for lat, lon ', lat, lon)
        return(mod3.cap_degree, mod3.n_theta, mod3.n_phi)


def get_smooth_parameters(models, lat, lon):
    mod1, mod2, mod3 = models
    cap_degree, n_theta, n_phi = determine_smoothing(models, lat, lon)
    sm_lats, sm_lons, sm_wghts = \
        smooth_weights_CAP_vardegree(lon, lat, cap_degree,
                                     n_theta, n_phi)
    sm_vps = np.zeros(common_depth.shape)
    sm_vss = np.zeros(common_depth.shape)
    sm_rhos = np.zeros(common_depth.shape)
    sm_moho = 0
    ms = [mod1, mod2, mod3]
    for ix_sm in range(n_theta * n_phi):
        x_lat = sm_lats[ix_sm]
        x_lon = sm_lons[ix_sm]
        weight = sm_wghts[ix_sm]

        for mod in ms:
            vps, vss, rhos = mod.get_profile_from_file(x_lat, x_lon)
            if True in np.isnan(vps):
                continue
            sm_vps += vps * weight
            sm_vss += vss * weight
            sm_rhos += rhos * weight
            if mod == mod2:
                moho = suggest_moho(common_depth, vss)
            else:
                moho = get_moho_from_crust1(mod3, x_lat, x_lon)
            sm_moho += moho * weight
            print('Using model {} at lat, lon {}, {}'.format(mod.model_name, x_lat, x_lon))
            break

    return(sm_vps, sm_vss, sm_rhos, sm_moho)


if __name__ == '__main__':
    lats = np.arange(lat_min, lat_max + interval, interval)
    lons = np.arange(lon_min, lon_max + interval, interval)
    lats_plot = np.zeros(len(lons) * len(lats))
    lons_plot = np.zeros(len(lons) * len(lats))
    cover_map = np.zeros(len(lons) * len(lats), dtype=np.int)
    vp_plot = []
    vs_plot = []
    moho_plot = []
    ix_plot = np.argmin(np.abs(common_depth - depth_to_plot))

    # initialize the models:
    m1 = model('nishida')
    m2 = model('zheng')
    m3 = model('crust1')

    for j, lon in enumerate(lons):
        for i, lat in enumerate(lats):
            lons_plot[j * len(lats) + i] = lon
            lats_plot[j * len(lats) + i] = lat
            ix_lats, ix_lons = m1.lat_lon_indices(lat, lon)
            scf = len(ix_lats) * len(ix_lons)

            for ix_lat in ix_lats:
                for ix_lon in ix_lons:
                    cover_map[j * len(lats) + i] +=\
                        m1.coverage[ix_lat, ix_lon] // scf

            if cover_map[j * len(lats) + i] == 0:
                ix_lats, ix_lons = m2.lat_lon_indices(lat, lon)
                scf = len(ix_lats) * len(ix_lons)
                for ix_lat in ix_lats:
                    for ix_lon in ix_lons:
                        cover_map[j * len(lats) + i] +=\
                            2 * m2.coverage[ix_lat, ix_lon] / scf

            if cover_map[j * len(lats) + i] == 0:
                cover_map[j * len(lats) + i] = 3

    plot_grid(lons_plot, lats_plot, cover_map - 1, cmap=cm,
              quant_unit='Model extent', size=5, sequential=True, v=3,
              colorbar_ticks=[[0.5, 1.5, 2.5], ['N08', 'Z11', 'L13']],
              outfile='model_extent.png')

    out_file = open(output_file, 'w')
    out_file.write("LON\tLAT\tDEP\tVP\tVS\tRHO\tMOHO\n")

    for j, lon in enumerate(lons):
        print(lon)
        print(time.strftime("%H:%M"))
        for i, lat in enumerate(lats):

            vp, vs, rho, moho = get_smooth_parameters([m1, m2, m3], lat, lon)
            for k in range(len(common_depth)):
                out_file.write("%4.1f\t%3.1f\t%4.1f\t%8.6f\
                               \t%8.6f\t%8.6f\t%4.1f\n"
                               % (lon, lat, common_depth[k], vp[k],
                                  vs[k], rho[k], moho))

            vp_plot.append(vp[ix_plot])
            vs_plot.append(vs[ix_plot])
            moho_plot.append(moho)
    plot_grid(lons_plot, lats_plot, np.asarray(moho_plot),
              sequential=True, v_min=min_moho, v=max_moho,
              size=plotdots, cmap=plt.cm.gist_rainbow,
              quant_unit='Moho depth (km)', outfile='moho.png',
              axislabelpad=-0.1)

    plot_grid(lons_plot, lats_plot, np.asarray(vs_plot),
              sequential=True, v_min=vsrange[0], v=vsrange[1],
              size=plotdots,
              quant_unit='vs (km/s)', cmap=plt.cm.gist_rainbow,
              outfile='vs.png',
              axislabelpad=-0.1)

    plot_grid(lons_plot, lats_plot, np.asarray(vp_plot),
              sequential=True, v_min=brocher_vs_to_vp(vsrange[0]),
              v=brocher_vs_to_vp(vsrange[1]), size=plotdots,
              quant_unit='vp (km/s)', cmap=plt.cm.gist_rainbow,
              outfile='vp.png',
              axislabelpad=-0.1)
    print(max(moho_plot))
    print(max(vs_plot))
    print(min(vs_plot))
