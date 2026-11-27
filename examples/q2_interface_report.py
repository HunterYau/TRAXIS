"""Collect saved interface audit results; no numerical simulation."""
import csv,json,os
from pathlib import Path
import numpy as np


def main():
    root=Path(__file__).resolve().parents[1];p=root/'outputs/q2-interface-audit-002'
    models=[json.loads((p/f'bonded-{c}.json').read_text()) for c in (4,6,8)]
    assert all(m.get('status')=='completed' and len(m.get('electrical_port_frf',[]))==6 for m in models)
    rows=[]
    for index,m in enumerate(models):
        for i,r in enumerate(m['rows']):
            voltage=complex(r['voltage_real_v'],r['voltage_imag_v'])
            row=dict(cells=m['cells'],host_element_width_mm=12/m['cells'],frequency_khz=r['frequency_hz']//1000,mode=r['mode'],voltage_magnitude_mv=1000*abs(voltage),
                cancellation_percent=100*r['cancellation_fraction'],generated_charge_magnitude_pc=1e12*abs(complex(r['generated_charge_real_c'],r['generated_charge_imag_c'])),
                centroid_strain_rms=r['centroid_strain_rms'],runtime_seconds=m['seconds'])
            if index:
                old=models[index-1]['rows'][i];ref=complex(old['voltage_real_v'],old['voltage_imag_v'])
                row.update(previous_cells=models[index-1]['cells'],relative_voltage_change_percent=100*abs(voltage-ref)/abs(ref),absolute_voltage_change_mv=1000*abs(voltage-ref),phase_change_deg=float(np.degrees(np.angle(voltage/ref))))
            rows.append(row)
    (p/'bonded-comparisons.json').write_text(json.dumps(rows,indent=2))
    with (p/'bonded-results.csv').open('w',newline='') as f:
        fields=list(dict.fromkeys(k for r in rows for k in r));writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    interpolation=json.loads((p/'interpolation.json').read_text());receiver=json.loads((p/'receiver-interpolation.json').read_text())
    trace=json.loads((p/'surface-quadrature.json').read_text())
    free=json.loads((p/'free-pzt-resonances.json').read_text())
    lines=['# Eight-ring Q2 PZT / aluminum interface investigation','',
        'Branch: research/q2-plate-interface-audit. No production physics, solver or LTS code changed. No full plate simulations. All small solves use FP64.',
        '', '## Findings','',
        '1. Displacement interpolation reproduces constant, linear and quadratic fields to roundoff. Transpose force transfer preserves resultant force, first moments and mechanical work. These tests do not indicate a sign, scaling or energy-conservation bug.',
        '2. The production fifth-order aluminum mesh has 4 mm elements, not a low-order 4 mm nodal spacing. Its analytical A0 input is imperfectly interpolated. Refining host sampling improves the isolated receiver voltages while keeping the eight-ring PZT unchanged.',
        '3. The perfect tie uses nodal elimination and a transpose variational force map, not a penalty stiffness. There is no independent interface traction quadrature or duplicate force contribution. Equal displacements are imposed at the slave nodes. Between those nodes, the Q2 PZT trace and fifth-order host trace are not identical. This is a real discretization mismatch, though the sampled gaps are small.',
        '4. Strong tie restricts the PZT contact to the host interpolation space. That can stiffen unresolved interface deformations, but a roundoff patch test or exact work transfer alone cannot prove absence of locking. These tests do not establish excessive artificial stiffness in production.',
        '5. The small bonded patch has strongly mesh-dependent resonances and voltages. It demonstrates sensitivity to host resolution and loading, but its finite reflecting boundaries and second-order host are not the production plate. Its large differences cannot be assigned directly to the production interface.',
        '6. Elementwise generated-charge contributions strongly cancel for A0 near 150-200 kHz. This amplifies relative voltage sensitivity. It is not proof that all previously observed mesh differences are physical.',
        '', '## Production interpolation and isolated receiver','',
        'Same NCE51 profile, 10 mm diameter / 0.5 mm thickness disk, fixed Q2 8 rings / 64 polygon sides / 1 thickness cell, Gauss order 6 and full consistent mass. Host: E=71 GPa, nu=0.33, density=2700 kg/m3, 1 mm thickness. Analytical Lamb displacement amplitude scale is 1 nm. Both actual centers were checked.',
        '', '| Frequency kHz | S0 voltage difference % | A0 voltage difference % | Max A0 absolute difference mV |', '|---:|---:|---:|---:|']
    for f in (50000,100000,150000,200000,250000,300000):
        a=[r for r in receiver if r['cells']==250 and r['frequency_hz']==f]
        vals=[max(r['relative_voltage_difference_percent'] for r in a if r['mode']==m) for m in ('S0','A0')]
        lines.append(f"| {f//1000} | {vals[0]:.8f} | {vals[1]:.6f} | {max(r['absolute_voltage_difference_mv'] for r in a if r['mode']=='A0'):.8f} |")
    lines+=['','The refined 500/1000-cell host maps are coordinate/interpolation metadata only: no global plate matrices, vectors or simulation were built. Each voltage comparison uses direct surface displacement versus interpolation through the production shape functions.',
        '', '| Host element width mm | Max A0 receiver difference % | Max sampled work error |', '|---:|---:|---:|']
    for c in (250,500,1000):
        lines.append(f"| {1000/c:g} | {max(r['relative_voltage_difference_percent'] for r in receiver if r['cells']==c and r['mode']=='A0'):.8f} | {max(r['work_relative_error'] for r in interpolation if r['cells_per_axis']==c):.3g} |")
    lines+=['','At 300 kHz, the area-weighted A0 displacement trace gap between slave nodes is '+', '.join(f"{next(r['integrated_tie_trace_gap_percent'] for r in trace if r['cells']==c and r['frequency_hz']==300000 and r['mode']=='A0'):.5f}% ({1000/c:g} mm host)" for c in (250,500,1000))+'. The nonzero refined limit includes the unchanged Q2 face approximation. This trace check uses surface quadrature; force work conservation follows the discrete transpose map.',
        '', '## Bonded patch refinement and cancellation','',
        'A 12x12x1 mm aluminum patch is refined uniformly under and around the unchanged disk: 4x4, 6x6 and 8x8 Q2 host cells. GLL host mass matches the production mass treatment; the PZT keeps full consistent mass. Perimeter displacements follow the same analytical S0/A0 fields, with other exterior faces free. This is a bounded surrogate study, not true local refinement within the production plate. There is no damping, so nearby standing-wave resonances can exaggerate changes.',
        '', '| kHz | S0 4→6 % | S0 6→8 % | A0 4→6 % | A0 6→8 % | A0 cancellation at 8 cells % |', '|---:|---:|---:|---:|---:|---:|']
    for f in (50,100,150,200,250,300):
        def get(c,mode):return next(r for r in rows if r['cells']==c and r['frequency_khz']==f and r['mode']==mode)
        vals=[get(c,m)['relative_voltage_change_percent'] for m in ('S0','A0') for c in (6,8)]
        lines.append('| '+str(f)+' | '+' | '.join(f'{v:.3f}' for v in vals)+f" | {get(8,'A0')['cancellation_percent']:.3f} |")
    lines+=['','Absolute voltage differences, signed complex voltages, phase changes, generated charge, element-charge cancellation, and six centroid strain RMS components are recorded in the JSON/CSV. Cancellation is 1-|sum(q_element)|/sum(|q_element|), using the electrostatic electrode weighting field. This is generated strain charge; the net free electrode charge in an ideal open-circuit solution is zero. It is not a claim of measurable nonzero open-electrode free charge.',
        '', '## Free and bonded resonances / port frequency response','',
        'Four eigenpairs nearest 150 kHz were computed with full electromechanical open-electrode condensation and consistent PZT mass. These are not the four lowest modes. Eigenpair residuals are checked; mode shapes were not tracked between models, so adjacent list entries are not necessarily corresponding modes.', '', '| Model | Four resonances nearest 150 kHz (kHz) |', '|---|---|',
        '| Free PZT | '+', '.join(f'{v/1000:.3f}' for v in free['nearest_150khz_resonance_hz'])+' |']
    for m in models:lines.append(f"| Bonded {m['cells']}x{m['cells']} patch | "+', '.join(f'{v/1000:.3f}' for v in m['modes']['nearest_150khz_resonance_hz'])+' |')
    lines+=['','A separate identical 1-pC electrical port excitation measures dynamic capacitance and mechanical displacement per volt for free and bonded models at all six frequencies. Values are saved as electrical_port_frf. The bonded perimeter is fixed for this port test; it is analytically driven for the S0/A0 tests. Raw displacement norms are not comparable across differing DOF counts without spatial weighting.',
        '', '## Likely causes and possible follow-up solutions','',
        '- Confirmed numerical contribution: host interpolation of shorter-wavelength A0 input. Local host refinement or improved interface approximation could reduce it.',
        '- Confirmed approximation: nonmatching polynomial traces between slave tie nodes. A mortar or Nitsche comparison and a conforming reference patch would test whether a weak tie improves accuracy.',
        '- Confirmed sensitivity: element-charge cancellation and bonded resonances amplify small amplitude/strain changes. Compare absolute voltage/charge as well as relative percentages.',
        '- Unresolved: how much production error comes from artificial interface stiffness versus legitimate PZT mass/stiffness loading. A matched fifth-order local-patch refinement and tracked modes are needed; the low-order finite patch cannot settle this.',
        '- No evidence found in these checks of lost mechanical work, wrong force scaling, failed polynomial interpolation, or an altered LTS recurrence.',
        '', 'No fixes were implemented. Existing fixed LTS time steps were not assessed by these frequency-domain tests.',
        '', '## Primary references','',
        '[MOOSE node/face and mortar interface theory](https://mooseframework.inl.gov/modules/contact/) describes shape-function force transfer and the alternative mortar approach. [COMSOL continuity documentation](https://doc.comsol.com/6.3/doc/com.comsol.help.sme/sme_ug_solid.07.128.html) describes pointwise, weak and Nitsche ties for nonconforming meshes. These references motivate possible follow-up tests; they do not validate this particular implementation.',
        '', 'All results: outputs/q2-interface-audit-002/. Initial preliminary runs remain preserved in outputs/q2-interface-audit-001/. No RunPod activity occurred.']
    (root/'docs/q2-plate-interface-audit.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    os.environ['MPLCONFIGDIR']=str(p/'matplotlib-cache')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(1,2,figsize=(10,4))
    for ax,mode in zip(axs,('S0','A0')):
        for m in models:
            r=[v for v in rows if v['cells']==m['cells'] and v['mode']==mode]
            ax.plot([v['frequency_khz'] for v in r],[v['voltage_magnitude_mv'] for v in r],marker='o',label=str(m['cells'])+'x'+str(m['cells']))
        ax.set(title=mode+' small bonded patch',xlabel='Frequency kHz',ylabel='Voltage magnitude mV');ax.grid();ax.legend()
    fig.tight_layout();fig.savefig(p/'bonded-voltage.png',dpi=150);plt.close(fig)
    files=[p/f'strain-8-{f}-A0.npz' for f in (150,300)]
    if all(path.exists() for path in files):
        fig,axs=plt.subplots(1,2,figsize=(10,4))
        for ax,path,f in zip(axs,files,(150,300)):
            with np.load(path) as data:
                xy=data['centroid_xyz_m'][:,:2]*1000;charge=data['generated_element_charge_c']
                # Real excitation has a common complex phase; show the dominant component.
                field=charge.real if np.linalg.norm(charge.real)>np.linalg.norm(charge.imag) else charge.imag
                lim=max(abs(field));im=ax.scatter(xy[:,0],xy[:,1],c=field*1e12,cmap='coolwarm',vmin=-lim*1e12,vmax=lim*1e12,s=10)
            ax.set(title=f'A0 {f} kHz element generated charge',xlabel='Local x mm',ylabel='Local y mm',aspect='equal');fig.colorbar(im,ax=ax,label='pC')
        fig.tight_layout();fig.savefig(p/'charge-distributions.png',dpi=150);plt.close(fig)
    print('Report and complete per-frequency artifacts saved.')


if __name__=='__main__':main()
