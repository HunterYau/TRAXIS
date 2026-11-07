"""Native H1 elastic operator applications without global sparse matrices."""

from dataclasses import dataclass
import math
import threading

import numpy as np
from scipy.sparse.linalg import LinearOperator

from traxis.highorder import _native_space, highorder_dofs, highorder_forms


@dataclass(frozen=True)
class MatrixFreeSettings:
    order: int = 3
    max_dofs: int = 2000000
    max_vertices: int = 250000
    max_tetrahedra: int = 1000000
    max_topology_mib: float = 256
    max_vector_mib: float = 256

    def __post_init__(self):
        for name, lo, hi in (('order',1,4), ('max_dofs',12,5000000),
                             ('max_vertices',4,1000000), ('max_tetrahedra',1,5000000)):
            value = getattr(self,name)
            if type(value) is not int or not lo <= value <= hi:
                raise ValueError(f'invalid matrix-free {name}')
        for name in ('max_topology_mib','max_vector_mib'):
            if not math.isfinite(getattr(self,name)) or not 0 < getattr(self,name) <= 2048:
                raise ValueError(f'invalid matrix-free {name}')


class NativeOperator(LinearOperator):
    """Owned native buffers; each application returns an independent array."""

    def __init__(self, form, space, guard=None):
        import ngsolve as ng
        self.form, self.guard = form, guard
        prototype = ng.GridFunction(space)
        self.input = prototype.vec.CreateVector()
        self.output = prototype.vec.CreateVector()
        self.applications = 0
        self.lock = threading.Lock()
        super().__init__(dtype=np.dtype('float64'), shape=(space.ndof,space.ndof))

    def _matvec(self, values):
        values = np.asarray(values)
        if (np.iscomplexobj(values) or values.shape not in ((self.shape[1],), (self.shape[1],1))
                or not np.isfinite(values).all()):
            raise ValueError('native operator needs a finite real coefficient vector')
        with self.lock:
            if self.guard:
                self.guard.check()
            self.input.FV().NumPy()[:] = values.reshape(-1)
            self.output.data = self.form.mat*self.input
            result = self.output.FV().NumPy().copy()
            self.applications += 1
        if not np.isfinite(result).all():
            raise ArithmeticError('native operator produced nonfinite values')
        return result

    def _rmatvec(self, values):
        # These operators come only from the symmetric elastic/mass forms.
        return self._matvec(values)


@dataclass
class MatrixFreeSystem:
    stiffness: NativeOperator
    mass: NativeOperator
    mass_diagonal: np.ndarray
    stiffness_diagonal: np.ndarray
    mesh: object
    scalar_space: object
    space: object
    screen: dict


def operator_diagonals(native_mesh, scalar, material, guard=None):
    """Accumulate exact scalar/component diagonals without global storage.

    Only one element matrix exists at a time. Component-major vector mass is
    three identical scalar blocks, so no global CSR or lumped operator is made.
    """
    import ngsolve as ng
    u, v = scalar.TnT()
    form = ng.BilinearForm(scalar, nonassemble=True)
    form += material.density_kg_m3*u*v*ng.dx(bonus_intorder=2)
    lam, mu = material.lame
    stiffness_forms = []
    for component in range(3):
        current = ng.BilinearForm(scalar, nonassemble=True)
        current += (mu*ng.InnerProduct(ng.grad(u),ng.grad(v))
            +(lam+mu)*ng.grad(u)[component]*ng.grad(v)[component])*ng.dx(bonus_intorder=2)
        stiffness_forms.append(current)
    diagonals = np.zeros((4,scalar.ndof))
    for element in native_mesh.Elements(ng.VOL):
        if guard:
            guard.check()
        eid = ng.ElementId(ng.VOL, element.nr)
        finite_element, transformation = scalar.GetFE(eid), native_mesh.GetTrafo(eid)
        dofs = np.asarray(scalar.GetDofNrs(eid), dtype=int)
        if (dofs < 0).any() or dofs.max() >= scalar.ndof or len(np.unique(dofs)) != len(dofs):
            raise ValueError('mass diagonal requires unconstrained distinct element coefficients')
        for component, current in enumerate((form,*stiffness_forms)):
            local = sum(np.asarray(integrator.CalcElementMatrix(finite_element, transformation)) for integrator in current.integrators)
            if local.shape != (len(dofs),len(dofs)) or not np.isfinite(local).all():
                raise ArithmeticError('invalid native element matrix for diagonal extraction')
            diagonals[component,dofs] += np.diag(local)
    if not np.isfinite(diagonals).all() or np.any(diagonals[0] <= 0) or np.any(diagonals[1:] < 0):
        raise ArithmeticError('native mass/stiffness has an invalid diagonal')
    return np.tile(diagonals[0],3),diagonals[1:].reshape(-1)


def build_matrix_free(mesh, material, settings=None, guard=None):
    settings = settings or MatrixFreeSettings()
    screen = highorder_dofs(mesh, settings.order, max_vertices=settings.max_vertices,
        max_tetrahedra=settings.max_tetrahedra, max_topology_mib=settings.max_topology_mib)
    vector_mib = screen['vector_dofs']*8*8/1024**2
    if screen['vector_dofs'] > settings.max_dofs or vector_mib > settings.max_vector_mib:
        raise MemoryError('matrix-free space/vector screen exceeds its declared limit')
    screen['native_vector_screen_mib'] = vector_mib
    native_mesh, scalar, space = _native_space(mesh, settings.order, screen, guard)
    stiffness, mass = highorder_forms(space, material, matrix_free=True)
    diagonal, stiffness_diagonal = operator_diagonals(native_mesh, scalar, material, guard)
    system = MatrixFreeSystem(NativeOperator(stiffness,space,guard), NativeOperator(mass,space,guard),
        diagonal, stiffness_diagonal, native_mesh, scalar, space, screen)
    if guard:
        guard.check(force=True)
    return system
