import numpy as np


def smooth_weights_CAP_vardegree(lon, lat, cap_degree,
                                 n_theta, n_phi):

    x_lon = np.zeros(n_theta * n_phi)
    x_lat = np.zeros(n_theta * n_phi)
    weight = np.zeros(n_theta * n_phi)
    rotation_matrix = np.zeros((3, 3))
    xc = np.zeros(3)
    x = np.zeros(3)

    if cap_degree < 1.e-6:
        raise ValueError("Smoothing CAP too small")

    cap = cap_degree * np.pi / 180.
    dtheta = 0.5 * cap / n_theta
    dphi = np.pi * 2. / n_phi

    cap_circ = 2. * np.pi * (1. - np.cos(cap))
    dweight = cap / n_theta * dphi / cap_circ
    pi_over_nphi = np.pi / n_phi

    theta = (90. - lat) * np.pi / 180.
    phi = lon * np.pi / 180.

    sint = np.sin(theta)
    cost = np.cos(theta)
    sinp = np.sin(phi)
    cosp = np.cos(phi)

    rotation_matrix[0, 0] = cosp * cost
    rotation_matrix[0, 1] = -sinp
    rotation_matrix[0, 2] = cosp * sint
    rotation_matrix[1, 0] = sinp * cost
    rotation_matrix[1, 1] = cosp
    rotation_matrix[1, 2] = sinp * sint
    rotation_matrix[2, 0] = -sint
    rotation_matrix[2, 1] = 0.
    rotation_matrix[2, 2] = cost

    total = 0.
    count = 0
    for itheta in range(n_theta):
        theta = (2. * itheta + 1) * dtheta
        cost = np.cos(theta)
        sint = np.sin(theta)
        wght = sint * dweight

        for iphi in range(n_phi):
            weight[count] = wght

            total += weight[count]

            phi = (2. * iphi + 1) * pi_over_nphi
            cosp = np.cos(phi)
            sinp = np.sin(phi)

            xc[0] = sint * cosp
            xc[1] = sint * sinp
            xc[2] = cost

            x = np.dot(rotation_matrix, xc)
            theta_r = np.arctan(np.sqrt(x[0]**2 + x[1]**2) / x[2])
            phi_r = np.arctan2(x[1], x[0])

            # !!!! This reduction of phi is not valid everywhere but
            # !!!! assumes we are around Japan
            redphi = abs(int(phi_r / np.pi))
            phi_r = phi_r + (redphi + 1.) * np.pi

            # r_r = np.sqrt(x[0]**2 + x[1]**2 + x[2]**2)

            x_lat[count] = (np.pi / 2. - theta_r) * 180. / np.pi
            x_lon[count] = phi_r * 180. / np.pi

            if x_lon[count] > 180.:
                x_lon[count] -= 180.
            count += 1
    if np.abs(total - 1.) > 1.e-4:
        raise ValueError('Something is weird here with smoothing weights')

    return(x_lat, x_lon, weight)
