"""Circle/triangle polynomial moments through degree four by Green's theorem."""

import math

import numpy as np

from traxis.config import positive


def triangle_powers(degree):
    if type(degree) is not int or not 0 <= degree <= 4:
        raise ValueError('surface polynomial degree must lie in [0,4]')
    return tuple((total-j, j) for total in range(degree+1) for j in range(total+1))


def _arc_integral(a, b, start, angle):
    # Expand cos(theta)^a sin(theta)^b in a finite Fourier series. The
    # midpoint sine form avoids subtracting nearly equal endpoint values.
    value = 0j
    for i in range(a+1):
        for j in range(b+1):
            frequency = a+b-2*(i+j)
            coefficient = math.comb(a, i)*math.comb(b, j)*(-1)**j / (2**a*(2*1j)**b)
            integral = angle if frequency == 0 else (
                np.exp(1j*frequency*(start+angle/2))*2*math.sin(frequency*angle/2)/frequency)
            value += coefficient*integral
    if abs(value.imag) > 1e-12:
        raise ArithmeticError('nonreal circular boundary integral')
    return value.real


def _unit_disk_monomials(vertices, degree):
    powers = triangle_powers(degree)
    result = np.zeros(len(powers))
    cross = lambda a, b: float(a[0]*b[1]-a[1]*b[0])
    orientation = cross(vertices[1]-vertices[0], vertices[2]-vertices[0])
    scale = max(float(np.linalg.norm(vertices[(i+1) % 3]-vertices[i])) for i in range(3))
    if abs(orientation) <= 1e-14*scale**2:
        raise ValueError('degenerate surface triangle')
    for a, b in zip(vertices, np.roll(vertices, -1, axis=0)):
        delta = b-a
        aa, bb, cc = float(delta @ delta), float(a @ delta), float(a @ a-1)
        discriminant = bb**2-aa*cc
        tolerance = 64*np.finfo(float).eps*max(bb**2, abs(aa*cc), 1.)
        cuts = [0., 1.]
        if discriminant >= -tolerance:
            root = math.sqrt(max(0., discriminant))
            cuts += [t for t in ((-bb-root)/aa, (-bb+root)/aa) if 0 < t < 1]
        cuts = sorted(set(cuts))
        for lo, hi in zip(cuts, cuts[1:]):
            p, q = a+lo*delta, a+hi*delta
            middle = (p+q)/2
            if float(middle @ middle) <= 1+64*np.finfo(float).eps:
                edge = q-p
                for k, (i, j) in enumerate(powers):
                    px = [math.comb(i, n)*p[0]**(i-n)*edge[0]**n for n in range(i+1)]
                    py = [math.comb(j, n)*p[1]**(j-n)*edge[1]**n for n in range(j+1)]
                    coefficients = np.convolve(px, py)
                    result[k] += cross(p, q)/(i+j+2)*sum(c/(n+1) for n, c in enumerate(coefficients))
            else:
                p, q = p/np.linalg.norm(p), q/np.linalg.norm(q)
                angle = math.atan2(cross(p, q), float(p @ q))
                start = math.atan2(p[1], p[0])
                for k, (i, j) in enumerate(powers):
                    result[k] += _arc_integral(i, j, start, angle)/(i+j+2)
    return result if orientation > 0 else -result


def disk_triangle_barycentric_moments(vertices_xy_m, center_xy_m, radius_m, degree):
    """Integrate lambda1**i lambda2**j over a triangle intersected by a disk.

    Entries follow ``triangle_powers(degree)`` and have units m^2. The
    barycentric coordinates refer to the second and third supplied vertices.
    Straight edges and circular arcs are integrated analytically. No element
    centroid selection or polygonal approximation of the disk is used.
    """
    powers = triangle_powers(degree)
    positive('patch radius', radius_m)
    vertices, center = np.asarray(vertices_xy_m), np.asarray(center_xy_m)
    if (np.iscomplexobj(vertices) or np.iscomplexobj(center) or vertices.shape != (3, 2)
            or center.shape != (2,) or not np.isfinite(vertices).all() or not np.isfinite(center).all()):
        raise ValueError('expected finite real triangle (3,2) and center (2,)')
    vertices = (vertices.astype(float)-center)/radius_m
    moments = _unit_disk_monomials(vertices, degree)
    if moments[0] < -1e-12:
        raise ArithmeticError('negative circle/triangle intersection area')
    if moments[0] <= 1e-14:
        return np.zeros(len(powers))
    affine = np.linalg.inv(np.column_stack([np.ones(3), vertices]))
    def multiply(left, right):
        result = {}
        for (i, j), a in left.items():
            for (k, l), b in right.items():
                key = (i+k, j+l)
                result[key] = result.get(key, 0.)+a*b
        return result
    linear = [{(0, 0): affine[0, k], (1, 0): affine[1, k], (0, 1): affine[2, k]} for k in (1, 2)]
    index = {power: k for k, power in enumerate(powers)}
    result = []
    for i, j in powers:
        polynomial = {(0, 0): 1.}
        for count, factor in ((i, linear[0]), (j, linear[1])):
            for _ in range(count):
                polynomial = multiply(polynomial, factor)
        result.append(sum(value*moments[index[power]] for power, value in polynomial.items()))
    result = np.array(result)
    if not np.isfinite(result).all() or result.min() < -1e-10*moments[0]-1e-13:
        raise ArithmeticError('invalid barycentric surface moments')
    return result*radius_m**2
