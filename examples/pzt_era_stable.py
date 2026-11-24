"""Stable/unstable spectral separation; a candidate correction, not pole clipping.

Ordered real Schur decomposition and Sylvester block diagonalization implement
the standard additive split G = G_stable + G_unstable. Removing a component
changes the approximation and is acceptable only after independent raw waveform
validation. It does not preserve mechanical energy or guarantee accuracy.
Reference: https://www.mathworks.com/help/control/ref/dynamicsystem.stabsep.html
"""
import numpy as np
from scipy.linalg import schur, solve_sylvester


def split(model):
    a, b, c = [np.asarray(model[k]) for k in ('a', 'b', 'c')]
    if (any(np.iscomplexobj(x) or not np.isfinite(x).all() for x in (a, b, c))
            or a.ndim != 2 or a.shape[0] != a.shape[1] or b.shape != (len(a), 1)
            or c.ndim != 2 or c.shape[1] != len(a)):
        raise ValueError('Compatible finite real state-space matrices required')
    t, q, kept = schur(a, output='real', sort=lambda re, im: np.hypot(re, im) < 1.)
    bt, ct = q.T @ b, c @ q
    s, u = t[:kept, :kept], t[kept:, kept:]
    bs, cs, bu, cu = bt[:kept].copy(), ct[:, :kept].copy(), bt[kept:].copy(), ct[:, kept:].copy()
    residual = 0.
    if 0 < kept < len(a):
        cross = t[:kept, kept:]
        x = solve_sylvester(s, -u, -cross)
        denominator = np.linalg.norm(s)*np.linalg.norm(x) + np.linalg.norm(x)*np.linalg.norm(u) + np.linalg.norm(cross)
        residual = float(np.linalg.norm(s @ x - x @ u + cross) / max(denominator, np.finfo(float).tiny))
        if not np.isfinite(x).all() or not np.isfinite(residual) or residual > 1e-10:
            raise ArithmeticError('Stable/unstable separation failed its Sylvester residual check')
        # z_s=x_s-X*x_u gives B_s'=B_s-X*B_u; C_u'=C_s*X+C_u.
        bs -= x @ bu
        cu += cs @ x
    def realization(a, b, c):
        if not all(np.isfinite(x).all() for x in (a, b, c)):
            raise ArithmeticError('Nonfinite separated realization')
        return dict(a=a.copy(), b=b.copy(), c=c.copy(), dt_s=model.get('dt_s', 1.),
                    singular_values=np.empty(0))
    return (realization(s, bs, cs), realization(u, bu, cu),
            dict(original_states=len(a), retained_stable_states=kept,
                 removed_states=len(a)-kept, sylvester_relative_residual=residual,
                 poles_moved=False, amplitudes_or_phase_fitted=False,
                 mechanical_energy_preservation_claimed=False,
                 accuracy_requires_independent_waveform_validation=True))
