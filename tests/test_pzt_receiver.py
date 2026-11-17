"""Independent constitutive, electrode, spatial-integration and circuit checks."""
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import solve
from scipy.special import j1

from traxis.config import Plate
from traxis.geometry import structured_mesh
from traxis.gpu_tensor import make_layout
from traxis.piezoelectric import (EPSILON_0,PiezoMaterial,PlanarPiezo,Readout,
    assemble_piezoelectric,electrode_port,loaded_voltage,planar_from_profile,read_profile)
from traxis.pzt_receiver import DiskGeometry,disk_receiver

PROJECT=Path(__file__).resolve().parents[1]


def material():return PiezoMaterial.from_profile(read_profile(PROJECT/'configs/pzt/nce51-noliac-ver1404.json'))


class ConstitutiveTests(unittest.TestCase):
    def test_stress_and_strain_charge_forms_agree_for_random_3d_states(self):
        mat=material();random=np.random.default_rng(351)
        for _ in range(8):
            stress=random.normal(size=6)*1e5;field=random.normal(size=3)*100
            strain=mat.compliance_E_m2_n@stress+mat.d_c_n.T@field
            actual_stress,actual_d=mat.response(strain,field)
            np.testing.assert_allclose(actual_stress,stress,rtol=1e-12,atol=1e-8)
            np.testing.assert_allclose(actual_d,mat.d_c_n@stress+mat.permittivity_T_f_m@field,rtol=1e-12,atol=1e-18)
        self.assertFalse(mat.d_c_n.flags.writeable)

    def test_plane_stress_reduction_matches_independent_compliance_formula(self):
        mat=material();reduced=mat.planar();s=mat.compliance_E_m2_n;d=mat.d_c_n[2,0]
        self.assertAlmostEqual(reduced.coupling_c_m2[0]/(d/(s[0,0]+s[0,1])),1.,places=13)
        expected=mat.permittivity_T_f_m[2,2]-2*d*d/(s[0,0]+s[0,1])
        self.assertAlmostEqual(reduced.permittivity_f_m/expected,1.,places=13)
        self.assertAlmostEqual(mat.stress_charge[2][2,2]/EPSILON_0,822.50268997,places=6)

    def test_steminc_does_not_silently_inherit_a_full_nce51_tensor(self):
        data=read_profile(PROJECT/'configs/pzt/steminc-smd07t02s412.json');p=planar_from_profile(data)
        self.assertEqual(data['diameter_m'],.007);self.assertEqual(data['thickness_m'],.0002)
        self.assertLess(p.coupling_c_m2[0],0)
        with self.assertRaises(ValueError):PiezoMaterial.from_profile(data)
        free_cap=EPSILON_0*1850*np.pi*.0035**2/.0002
        self.assertLess(abs(free_cap/data['device_capacitance_f']-1),.15)

    def test_nonphysical_material_is_rejected(self):
        mat=material()
        with self.assertRaises(ValueError):PiezoMaterial('bad',mat.compliance_E_m2_n,mat.d_c_n*100,mat.permittivity_T_f_m,7850,{})
        with self.assertRaises(ValueError):Readout(0.,0.)
        with self.assertRaises(ValueError):Readout(None,-1.)


class SpatialReceiverTests(unittest.TestCase):
    def test_affine_fields_rigid_rotation_and_polarity(self):
        layout=make_layout(Plate(.04,.03,.003175),(8,6,1),4);xyz=layout.points();p=material().planar()
        fields=[np.zeros((3,len(xyz))) for _ in range(3)]
        fields[0][0]=2e-6*xyz[:,0];fields[0][1]=3e-6*xyz[:,1]
        fields[1][0]=xyz[:,0]**2;fields[1][1]=xyz[:,1]**2
        fields[2][0]=-xyz[:,1]+.001;fields[2][1]=xyz[:,0]-.002;fields[2][2]=.004
        for surface in ('top','bottom'):
            positive=disk_receiver(layout,p,DiskGeometry((.019,.014),.0035,.0002,surface,1))
            negative=disk_receiver(layout,p,DiskGeometry((.019,.014),.0035,.0002,surface,-1))
            area=np.pi*.0035**2
            expected=[area*(2e-6*p.coupling_c_m2[0]+3e-6*p.coupling_c_m2[1]),
                      area*(2*.019*p.coupling_c_m2[0]+2*.014*p.coupling_c_m2[1]),0.]
            for field,value in zip(fields,expected):
                observed=positive.strain_charge(field.ravel())
                self.assertAlmostEqual(observed,value,delta=max(abs(value)*2e-11,1e-16))
                self.assertAlmostEqual(negative.strain_charge(field.ravel()),-observed,delta=1e-16)

    def test_circle_plane_wave_cancellation_agrees_with_bessel_area_integral(self):
        layout=make_layout(Plate(.04,.03,.003175),(64,8,1),5);xyz=layout.points()
        p=PlanarPiezo('analytic',[1.,1.,0.],1e-8,7800,{})
        r=.0035;center=(.019,.014);receiver=disk_receiver(layout,p,DiskGeometry(center,r,.0002,'top',1))
        for kr in (1.,2.,3.8317059702,5.):
            k=kr/r;u=np.zeros((3,len(xyz)));u[0]=np.sin(k*(xyz[:,0]-center[0]))/k
            observed=receiver.strain_charge(u.ravel());expected=np.pi*r*r*2*j1(kr)/kr
            self.assertAlmostEqual(observed,expected,delta=np.pi*r*r*1e-6)

    def test_shear_and_rotation_use_both_in_plane_components(self):
        layout=make_layout(Plate(.04,.03,.003175),(8,6,1),4);xyz=layout.points()
        mat=PlanarPiezo('anisotropic',[2.,3.,4.],1e-8,7800,{})
        r=disk_receiver(layout,mat,DiskGeometry((.02,.015),.003,.0002,'top',1))
        u=np.zeros((3,len(xyz)));u[0]=1e-6*xyz[:,1];u[1]=2e-6*xyz[:,0]
        self.assertAlmostEqual(r.strain_charge(u.ravel())/(np.pi*.003**2),4.*3e-6,places=12)
        rotated=disk_receiver(layout,mat,DiskGeometry((.02,.015),.003,.0002,'top',1,np.pi/2))
        self.assertAlmostEqual(rotated.strain_charge(u.ravel())/r.strain_charge(u.ravel()),-1.,places=12)

    def test_geometry_and_allocation_guards(self):
        layout=make_layout(Plate(.04,.03,.003175),(8,6,1),4);mat=material().planar()
        with self.assertRaises(ValueError):DiskGeometry((.01,.01),.003,.0002,'side',1)
        with self.assertRaises(ValueError):disk_receiver(layout,mat,DiskGeometry((0.,.01),.003,.0002,'top',1))
        with self.assertRaises(MemoryError):disk_receiver(layout,mat,DiskGeometry((.02,.015),.003,.0002,'top',1),max_workspace_mib=.001)


class CircuitTests(unittest.TestCase):
    def test_open_circuit_conserves_total_free_charge_and_cable_loading(self):
        t=np.linspace(0,1e-4,200);q=1e-12*np.sin(2*np.pi*1e5*t)+2e-12
        result=loaded_voltage(t,q,2e-9,Readout(None,1e-9))
        np.testing.assert_allclose(result['voltage_v'],q/3e-9,rtol=1e-13)
        self.assertLess(np.max(abs(result['charge_balance_c'])),1e-25)

    def test_resistive_trace_agrees_with_independent_ode_and_harmonic_limit(self):
        c=2e-9;resistance=1000.;t=np.linspace(0,1e-4,1001);slope=2e-8;q=slope*t
        result=loaded_voltage(t,q,c,Readout(resistance,0.))
        exact=solve_ivp(lambda t,y:[(slope-y[0]/resistance)/c],(t[0],t[-1]),[0.],
                        t_eval=t,rtol=1e-11,atol=1e-15).y[0]
        np.testing.assert_allclose(result['voltage_v'],exact,rtol=1e-8,atol=1e-14)
        self.assertLess(np.max(abs(result['charge_balance_c'])),1e-24)
        f=1e5;t=np.linspace(0,50/f,50001);q=1e-12*np.sin(2*np.pi*f*t)
        v=loaded_voltage(t,q,c,Readout(resistance,0.))['voltage_v'];mask=t>40/f
        sine=np.sin(2*np.pi*f*t[mask]);cosine=np.cos(2*np.pi*f*t[mask])
        coefficients=np.linalg.lstsq(np.column_stack([sine,cosine]),v[mask],rcond=None)[0]
        expected=1j*2*np.pi*f/(1/resistance+1j*2*np.pi*f*c)*1e-12
        np.testing.assert_allclose(coefficients,[expected.real,expected.imag],rtol=5e-6)


class CoupledComponentTests(unittest.TestCase):
    def test_uniform_field_electrode_charge_mass_and_reciprocity(self):
        plate=Plate(.01,.008,.0005);mesh=structured_mesh(plate,(1,1,2));mat=material()
        system=assemble_piezoelectric(mesh,mat);n=len(mesh.points_m)
        bottom=np.flatnonzero(mesh.points_m[:,2]==0);top=np.flatnonzero(mesh.points_m[:,2]==plate.thickness_m)
        port=electrode_port(system,bottom,top)
        for component in range(3):self.assertAlmostEqual(system.mass[component*n:(component+1)*n,component*n:(component+1)*n].sum()/(7850*.01*.008*.0005),1.,places=12)
        u=np.zeros((3,n));strain=np.array([1e-6,2e-6,3e-6,0,0,0])
        for i in range(3):u[i]=strain[i]*mesh.points_m[:,i]
        voltage=.2;phi=port.potential_per_v*voltage+port.potential_per_displacement@u.ravel()
        charge=(system.dielectric@phi-system.coupling.T@u.ravel())[top].sum()
        _,electric_d=mat.response(strain,[0,0,-voltage/.0005])
        self.assertAlmostEqual(charge/(-electric_d[2]*.01*.008),1.,places=12)
        self.assertAlmostEqual(port.charge(u.ravel(),voltage)/charge,1.,places=12)
        np.testing.assert_allclose(system.coupling@port.potential_per_v,port.coupling_n_v,rtol=1e-13)
        self.assertGreater(np.linalg.norm(port.coupling_n_v),0.)

    def test_loaded_coupled_dynamics_obey_electrical_balance_and_passive_power(self):
        mesh=structured_mesh(Plate(.004,.003,.0005),(1,1,2));system=assemble_piezoelectric(mesh,material());n=len(mesh.points_m)
        bottom=np.flatnonzero(mesh.points_m[:,2]==0);top=np.flatnonzero(mesh.points_m[:,2]==.0005)
        port=electrode_port(system,bottom,top);force=np.zeros(3*n);force[2*n+top]=1/len(top)
        fixed=np.concatenate([bottom+i*n for i in range(3)]);readout=Readout(10000.,20e-12)
        for f in (50000,150000,300000):
            response=port.harmonic_response(f,force,readout,fixed_dofs=fixed)
            u,v=response['displacement_m'],response['voltage_v'];w=2*np.pi*f
            residual=1j*w*((port.capacitance_f+readout.capacitance_f)*v-port.coupling_n_v@u)+v/10000.
            self.assertLess(abs(residual),1e-12)
            supplied=.5*np.real(np.vdot(force,1j*w*u));loss=.5*abs(v)**2/10000.
            self.assertAlmostEqual(supplied/loss,1.,places=9)


if __name__=='__main__':unittest.main()
