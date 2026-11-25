"""Full-band numerical PZT mesh comparisons; no fitted shift, phase or gain."""
from pathlib import Path
import numpy as np
from scipy.signal import hilbert
from traxis.arrival import adaptive_packet
from traxis.config import Material
from traxis.reporting import _figure_api,_new_figure,_save_figure
from traxis.signal_comparison import waveform_metrics

CHANNELS=('top','bottom','S0-related','A0-related')


def channels(y):
    y=np.asarray(y)
    if y.ndim!=2 or y.shape[1]!=2 or not np.isfinite(y).all():raise ValueError('Finite paired voltage channels required')
    return dict(zip(CHANNELS,(y[:,0],y[:,1],y[:,0]+y[:,1],y[:,0]-y[:,1])))


def shared_grid(ta,ya,tb,yb):
    """Keep exact nested recording nodes; no interpolation of waveform wiggles."""
    ta,tb=np.asarray(ta),np.asarray(tb)
    for t,y in ((ta,ya),(tb,yb)):
        if (t.ndim!=1 or len(t)<4 or np.shape(y)!=(len(t),2) or not np.isfinite(t).all()
                or t[0]!=0 or np.any(np.diff(t)<=0)):raise ValueError('Invalid time grid')
    if len(ta)>len(tb):
        t,b,a=shared_grid(tb,yb,ta,ya);return t,a,b
    if (len(tb)-1)%(len(ta)-1):raise ValueError('Non-nested recordings')
    stride=(len(tb)-1)//(len(ta)-1)
    if not np.allclose(ta,tb[::stride],rtol=0,atol=1e-15):raise ValueError('Recording times differ')
    return ta,np.asarray(ya),np.asarray(yb)[::stride]


def _relative(a,b):
    norm=np.linalg.norm(a)
    return float(100*np.linalg.norm(b-a)/norm) if norm>0 else None


def _pearson(a,b):
    return float(np.clip(np.corrcoef(a,b)[0,1],-1,1)) if min(np.std(a),np.std(b))>0 else None


def threshold_results(metrics):
    fields=('raw_l2_percent','envelope_l2_percent','wiggle_l2_percent')
    return {str(limit):{key:(metrics[key] is not None and metrics[key]<limit) for key in fields}
            for limit in (1.,.1)}


def compare(ta,ya,tb,yb,frequency,config,*,arrivals=True):
    t,a,b=shared_grid(ta,ya,tb,yb);answer={}
    source=np.array(config['actuator_center_xy_m']);receiver=np.array(config['receiver_center_xy_m'])
    width,height,thickness=config['plate_m']
    images=[[-source[0],source[1]],[2*width-source[0],source[1]],
            [source[0],-source[1]],[source[0],2*height-source[1]]]
    geometry=dict(distance_m=float(np.linalg.norm(source-receiver)),thickness_m=thickness,
        material=Material(**config['host_material']),cycles=5,
        reflection_distance_m=min(float(np.linalg.norm(receiver-i)) for i in images))
    for name,x in channels(a).items():
        y=channels(b)[name];m=waveform_metrics(x,y);ex,ey=abs(hilbert(x)),abs(hilbert(y))
        row=dict(raw_l2_percent=m['raw_voltage_difference_percent'],
            envelope_l2_percent=_relative(ex,ey),
            envelope_normalized_l2_percent=_relative(ex/np.linalg.norm(ex),ey/np.linalg.norm(ey))
                if min(np.linalg.norm(ex),np.linalg.norm(ey))>0 else None,
            wiggle_l2_percent=m['normalized_wiggle_difference_percent'],
            raw_pearson=m['correlation'],wiggle_pearson=m['correlation'],
            envelope_pearson=_pearson(ex,ey),rmse_v=m['rmse_v'],
            max_absolute_voltage_difference_v=float(np.max(abs(y-x))),arrivals={})
        if arrivals:
            for mode in ('S0','A0'):
                try:
                    r=adaptive_packet(t,x,frequency,mode,**geometry)
                    s=adaptive_packet(t,y,frequency,mode,**geometry)
                    delta=s['packet_peak_us']-r['packet_peak_us']
                    reliable=r['status']==s['status']=='localized_packet'
                    row['arrivals'][mode]=dict(reference=r,candidate=s,difference_us=delta,
                        absolute_difference_us=abs(delta),reliable=reliable,
                        error_percent=100*abs(delta)/abs(r['flight_us']) if r['flight_us']!=0 else None,
                        note='Candidate peak differences retained even when localization is unresolved; sum/difference are not pure modes.')
                except ValueError as exc:
                    row['arrivals'][mode]=dict(reliable=False,difference_us=None,error_percent=None,error=str(exc))
        row['thresholds']=threshold_results(row);answer[name]=row
    return dict(frequency_hz=frequency,comparison_dt_s=float(t[1]-t[0]),channels=answer,
        method='Full shared interval; exact nested nodes; no fitted time/phase/gain; unit-L2 normalization for wiggles only',
        arrival_note='Existing theory-guided adaptive envelope locator; 0.2 us locator grid. Both mode candidates on all channels are proxies.')


def temporal_pass(rows,target=.01):
    return bool(rows) and all(v[k] is not None and v[k]<target for row in rows
        for v in row['channels'].values() for k in ('raw_l2_percent','envelope_l2_percent','wiggle_l2_percent'))


def plots(folder,ta,ya,tb,yb,label):
    """Reuse existing headless figure/cache/export infrastructure for overlays."""
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    t,a,b=shared_grid(ta,ya,tb,yb)
    api=_figure_api(folder/'matplotlib-cache');figure=_new_figure(api,(13,10))
    axes=figure.subplots(4,2);index=np.unique(np.linspace(0,len(t)-1,min(6000,len(t))).astype(int))
    for number,(name,x) in enumerate(channels(a).items()):
        y=channels(b)[name]
        for axis,u,v,title in ((axes[number,0],x,y,'Voltage and wiggles'),
                                (axes[number,1],abs(hilbert(x)),abs(hilbert(y)),'Envelope')):
            axis.plot(t[index]*1e6,u[index],lw=.7,label='Reference')
            axis.plot(t[index]*1e6,v[index],lw=.6,alpha=.75,label='Candidate')
            axis.set(title=name+' - '+title,xlabel='Time (microseconds)',ylabel='Voltage (V)')
            axis.grid(alpha=.2);axis.legend(fontsize=7)
    figure.suptitle(label+'; no fitted alignment; metrics use every shared sample')
    # Existing exporter is atomic and refuses overwriting, preserving records.
    return _save_figure(figure,folder,'waveforms',None)
