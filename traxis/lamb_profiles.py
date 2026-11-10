"""Fundamental traction-free Lamb displacement profiles and modal support.

Profiles are derived from scalar/SV potentials with exp(i*k*x-i*omega*t).
The coefficient nullspace is set by both surface traction conditions.
"""
from dataclasses import dataclass
import math
import numpy as np
from traxis.config import Material
from traxis.dispersion import wavenumber_rad_m


@dataclass(frozen=True)
class LambProfile:
    frequency_hz: float
    thickness_m: float
    mode: str
    material: Material
    wavenumber_rad_m: float
    alpha: complex
    beta: complex
    coefficients: np.ndarray
    normalization: float
    potential_null_relative_residual: float

    def fields(self,z_m):
        z = np.asarray(z_m)
        half = self.thickness_m/2
        if (z.ndim!=1 or np.iscomplexobj(z) or not np.isfinite(z).all()
                or np.any(abs(z)>half*(1+1e-12))):
            raise ValueError('Lamb profile points must be real centered through-thickness coordinates')
        eta,kappa = z/half,self.wavenumber_rad_m*half
        a,b = self.coefficients
        if self.mode=='S0':
            phi = a*np.cos(self.alpha*eta)
            phi_d = -a*self.alpha*np.sin(self.alpha*eta)
            psi = b*np.sin(self.beta*eta)
            psi_d = b*self.beta*np.cos(self.beta*eta)
        else:
            phi = a*np.sin(self.alpha*eta)
            phi_d = a*self.alpha*np.cos(self.alpha*eta)
            psi = b*np.cos(self.beta*eta)
            psi_d = -b*self.beta*np.sin(self.beta*eta)
        phi_dd,psi_dd = -self.alpha**2*phi,-self.beta**2*psi
        u = np.array([1j*kappa*phi-psi_d,np.zeros(len(z)),phi_d+1j*kappa*psi])/self.normalization
        du = np.array([1j*kappa*phi_d-psi_dd,np.zeros(len(z)),phi_dd+1j*kappa*psi_d])/(half*self.normalization)
        ddu = np.array([1j*kappa*phi_dd+self.beta**2*psi_d,np.zeros(len(z)),
                        -self.alpha**2*phi_d+1j*kappa*psi_dd])/(half**2*self.normalization)
        return u,du,ddu

    def traction(self,z_m):
        u,du,_ = self.fields(z_m)
        lam,mu = self.material.lame
        k = self.wavenumber_rad_m
        return np.array([mu*(du[0]+1j*k*u[2]),lam*1j*k*u[0]+(lam+2*mu)*du[2]])

    def navier_relative_residual(self,z_m):
        u,du,ddu = self.fields(z_m)
        lam,mu = self.material.lame
        k,w,rho = self.wavenumber_rad_m,2*math.pi*self.frequency_hz,self.material.density_kg_m3
        terms = np.array([mu*ddu[0]-(lam+2*mu)*k*k*u[0]+(lam+mu)*1j*k*du[2],
                          (lam+2*mu)*ddu[2]-mu*k*k*u[2]+(lam+mu)*1j*k*du[0]])
        residual = terms+rho*w*w*u[[0,2]]
        return float(np.linalg.norm(residual)/(rho*w*w*np.linalg.norm(u)))


def fundamental_profile(frequency_hz,thickness_m,mode,material=Material()):
    k = wavenumber_rad_m(frequency_hz,thickness_m,mode,material)
    half = thickness_m/2
    cl,ct = material.bulk_speeds_m_s
    w,kappa = 2*math.pi*frequency_hz,k*half
    alpha = np.sqrt(complex((w*half/cl)**2-kappa*kappa))
    beta = np.sqrt(complex((w*half/ct)**2-kappa*kappa))
    difference = beta*beta-kappa*kappa
    if mode=='S0':
        boundary = np.array([[-2j*kappa*alpha*np.sin(alpha),difference*np.sin(beta)],
                             [-difference*np.cos(alpha),2j*kappa*beta*np.cos(beta)]])
    else:
        boundary = np.array([[2j*kappa*alpha*np.cos(alpha),difference*np.cos(beta)],
                             [-difference*np.sin(alpha),-2j*kappa*beta*np.sin(beta)]])
    boundary /= np.linalg.norm(boundary)
    _,_,vh = np.linalg.svd(boundary)
    coefficients = vh[-1].conj()
    residual = float(np.linalg.norm(boundary@coefficients))
    if not math.isfinite(residual) or residual>1e-9:
        raise ArithmeticError('Lamb potential coefficients fail the surface-traction nullspace')
    coefficients.setflags(write=False)
    profile = LambProfile(frequency_hz,thickness_m,mode,material,k,alpha,beta,coefficients,1.,residual)
    normalization = float(np.max(abs(profile.fields(np.linspace(-half,half,65))[0])))
    if not math.isfinite(normalization) or normalization<=0:
        raise ArithmeticError('Lamb displacement profile is degenerate')
    return LambProfile(frequency_hz,thickness_m,mode,material,k,alpha,beta,coefficients,normalization,residual)


def spectral_support(eigenvectors,mass,reference,coverage=.99):
    """Select by profile mass overlap alone; no eigenfrequency is an input."""
    eigenvectors,mass,reference = (np.asarray(v) for v in (eigenvectors,mass,reference))
    n = len(reference)
    if (reference.ndim!=1 or eigenvectors.shape!=(n,n) or mass.shape!=(n,n) or not 0<coverage<=1
            or not all(np.isfinite(v).all() for v in (eigenvectors,mass,reference))):
        raise ValueError('profile support requires a complete finite generalized modal basis')
    norm = np.vdot(reference,mass@reference)
    if abs(norm.imag)>1e-12*abs(norm.real) or norm.real<=0:
        raise ValueError('reference profile must have positive real mass norm')
    overlaps = abs(eigenvectors.conj().T@(mass@reference))**2/float(norm.real)
    if abs(overlaps.sum()-1)>1e-9 or np.any(overlaps<0):
        raise ArithmeticError('modal basis does not account for the reference mass norm')
    order = np.argsort(-overlaps,kind='stable')
    count = int(np.searchsorted(np.cumsum(overlaps[order]),coverage,side='left')+1)
    count = min(count,n)
    selected = order[:count]
    if overlaps[selected].sum()<coverage-1e-12:
        raise ArithmeticError('modal profile coverage was not achieved')
    return selected,overlaps
