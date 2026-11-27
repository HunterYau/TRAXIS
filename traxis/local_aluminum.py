"""Separate conforming local h-refinement of an existing tensor aluminum mesh.

Whole parent cells are replaced, not overlapped. Fine side traces are constrained
by coarse-face interpolation; both K and M use the same transpose projection.
No changes to the existing element formulation, PZT material, or LTS equations.
"""
from types import SimpleNamespace
import math
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu
from traxis.tensor_hex import line_rule,element_matrices,local_gll_action
from traxis.gpu_tensor import mass_diagonal,streamed_action
from traxis.quadratic_pzt_sparse import SparseAttachment

DEFAULT=dict(enabled=True,patch_width_m=.012,cells_per_patch_axis=16)


def line_weights(nodes,x):
    nodes=np.asarray(nodes);hit=np.flatnonzero(abs(nodes-x)<2e-12)
    if len(hit):
        out=np.zeros(len(nodes));out[hit[0]]=1.;return out
    weights=np.array([1/np.prod(nodes[i]-np.delete(nodes,i)) for i in range(len(nodes))])
    value=weights/(x-nodes);return value/value.sum()


def point_weights(knots,order,xyz):
    """Barycentric interpolation on a nonuniform rectilinear tensor grid."""
    ids=[];w=[];gll=line_rule(order)[0]
    for edges,x in zip(knots,xyz):
        if x<edges[0]-1e-12 or x>edges[-1]+1e-12:raise ValueError('Point outside block')
        cell=min(max(int(np.searchsorted(edges,x,side='right')-1),0),len(edges)-2)
        local=2*(x-edges[cell])/(edges[cell+1]-edges[cell])-1
        ids.append(cell*order+np.arange(order+1));w.append(line_weights(gll,local))
    shape=tuple((len(e)-1)*order+1 for e in knots)
    index=((ids[0][:,None,None]*shape[1]+ids[1][None,:,None])*shape[2]+ids[2][None,None,:]).ravel()
    values=np.einsum('i,j,k->ijk',*w).ravel();keep=abs(values)>1e-14
    return index[keep],values[keep]


class Block:
    def __init__(self,knots,order):
        self.knots=tuple(np.asarray(k,dtype=float) for k in knots);self.order=order
        g=(line_rule(order)[0]+1)/2
        self.axes=tuple(np.r_[np.concatenate([lo+(hi-lo)*g[:-1] for lo,hi in zip(e[:-1],e[1:])]),e[-1]] for e in self.knots)
        self.shape=tuple(len(a) for a in self.axes);self.n=math.prod(self.shape)
        counts=tuple(len(e)-1 for e in self.knots)
        ijk=np.indices(counts).reshape(3,-1).T;i,j,k=np.indices((order+1,)*3).reshape(3,-1)
        self.elements=((ijk[:,0,None]*order+i)*self.shape[1]+ijk[:,1,None]*order+j)*self.shape[2]+ijk[:,2,None]*order+k
        widths=np.column_stack([np.diff(e)[ijk[:,a]] for a,e in enumerate(self.knots)])
        self.inverse=2/widths;self.determinant=np.prod(widths/2,axis=1)
    def points(self):return np.stack(np.meshgrid(*self.axes,indexing='ij'),axis=-1).reshape(-1,3)
    def mass(self,rho):
        factors=[];w=line_rule(self.order)[1]
        for e,axis in zip(self.knots,self.axes):
            v=np.zeros(len(axis))
            for j,h in enumerate(np.diff(e)):v[j*self.order:j*self.order+self.order+1]+=w*h/2
            factors.append(v)
        return (rho*np.einsum('i,j,k->ijk',*factors)).ravel()
    def action(self,x,material):
        state=np.asarray(x).reshape(3,self.n);out=np.zeros_like(state)
        for ids,inv in zip(self.elements,self.inverse):
            value=local_gll_action(state[:,ids].reshape((3,)+(self.order+1,)*3),self.order,np.diag(1/inv),material).reshape(3,-1)
            for c in range(3):np.add.at(out[c],ids,value[c])
        return out.ravel()


class LocalMesh:
    def __init__(self,layout,material,centers,*,patch_width_m=.012,cells_per_patch_axis=16):
        if not np.array_equal(layout.transform,np.eye(3)):raise ValueError('Aligned host required')
        if patch_width_m<=0 or type(cells_per_patch_axis) is not int or cells_per_patch_axis<2:raise ValueError('Invalid local refinement')
        self.base=layout;self.material=material;self.old_n=layout.screen['scalar_dofs'];p=layout.order
        self.coarse_knots=tuple(np.linspace(0,L,n+1) for L,n in zip((layout.plate.width_m,layout.plate.height_m,layout.plate.thickness_m),layout.cells))
        centers=sorted(set(tuple(map(float,c)) for c in centers));spec=[];remove=np.zeros(self.old_n,bool)
        zcount=len(layout.axes_m[2]);oldshape=tuple(len(a) for a in layout.axes_m)
        for center in centers:
            knots=[];ranges=[]
            for axis,c in enumerate(center):
                edges=self.coarse_knots[axis];lo,hi=c-patch_width_m/2,c+patch_width_m/2
                if lo<=0 or hi>=edges[-1]:raise ValueError('Refinement must be interior to plate')
                first=max(0,np.searchsorted(edges,lo+1e-12,side='right')-1)
                stop=min(len(edges)-1,np.searchsorted(edges,hi-1e-12,side='left'))
                # Include parent-face cuts: a fine face never spans a coarse trace kink.
                cuts=np.sort(np.r_[lo,hi,edges[first:stop+1]])
                cuts=cuts[np.r_[True,np.diff(cuts)>1e-12]]
                target=patch_width_m/cells_per_patch_axis
                # Redistribute within each parent interval instead of creating
                # arbitrarily short cells where two independent grids intersect.
                pieces=[]
                for left,right in zip(cuts[:-1],cuts[1:]):
                    count=max(1,math.ceil((right-left)/target-1e-10)) if lo-1e-12<=(left+right)/2<=hi+1e-12 else 1
                    pieces.extend(np.linspace(left,right,count+1)[:-1])
                raw=np.r_[pieces,cuts[-1]]
                knots.append(raw);ranges.append((first,stop))
            bounds=[(e[0],e[-1]) for e in knots]
            for previous in spec:
                if all(min(b[1],q[1])-max(b[0],q[0])>=-1e-12 for b,q in zip(bounds,previous['bounds'])):
                    raise ValueError('Refinement transition regions must be disjoint')
            ix=np.flatnonzero((layout.axes_m[0]>bounds[0][0]+1e-12)&(layout.axes_m[0]<bounds[0][1]-1e-12))
            iy=np.flatnonzero((layout.axes_m[1]>bounds[1][0]+1e-12)&(layout.axes_m[1]<bounds[1][1]-1e-12))
            remove[((ix[:,None,None]*oldshape[1]+iy[None,:,None])*zcount+np.arange(zcount)).ravel()]=True
            ids=[]
            for i in range(*ranges[0]):
                for j in range(*ranges[1]):
                    for k in range(layout.cells[2]):ids.append((i*layout.cells[1]+j)*layout.cells[2]+k)
            spec.append(dict(center=center,bounds=bounds,block=Block((*knots,self.coarse_knots[2]),p),coarse_ids=np.array(ids,dtype=np.int32)))
        self.keep=np.flatnonzero(~remove);self.old_to_new=np.full(self.old_n,-1,np.int64);self.old_to_new[self.keep]=np.arange(len(self.keep))
        next_node=len(self.keep);self.patches=[]
        for s in spec:
            block=s['block'];xyz=block.points();boundary=np.zeros(block.n,bool)
            for axis,(lo,hi) in enumerate(s['bounds']):boundary|=(abs(xyz[:,axis]-lo)<1e-12)|(abs(xyz[:,axis]-hi)<1e-12)
            interior=np.flatnonzero(~boundary);scalar_new=np.full(block.n,-1,np.int64);scalar_new[interior]=np.arange(next_node,next_node+len(interior));next_node+=len(interior)
            rr=[];cc=[];vv=[]
            for node in np.flatnonzero(boundary):
                ids,w=point_weights(self.coarse_knots,p,xyz[node]);mapped=self.old_to_new[ids]
                if np.any(mapped<0):raise ArithmeticError('Hanging constraint references eliminated node')
                rr.extend([node]*len(ids));cc.extend(mapped);vv.extend(w)
            rr.extend(interior);cc.extend(scalar_new[interior]);vv.extend(np.ones(len(interior)))
            support=np.unique(cc);T=sparse.coo_matrix((vv,(rr,np.searchsorted(support,cc))),shape=(block.n,len(support))).tocsr()
            old_ids=np.concatenate([layout.element_nodes(int(i),int(i)+1) for i in s['coarse_ids']]);unique=np.unique(old_ids)
            mapped=self.old_to_new[unique];retained=mapped>=0
            C=sparse.coo_matrix((np.ones(retained.sum()),(np.flatnonzero(retained),np.searchsorted(support,mapped[retained]))),shape=(len(unique),len(support))).tocsr()
            self.patches.append(SimpleNamespace(**s,support=support,t=T,ct=C,coarse_elements=np.searchsorted(unique,old_ids),
                interior=interior,interior_ids=scalar_new[interior],boundary=boundary,points=xyz,coarse_nodes=unique))
        self.n=next_node
        old_mass=mass_diagonal(layout,material)[:self.old_n].copy()
        _,_,removed_mass=element_matrices(p,layout.jacobian,material,'gll')
        for q in self.patches:
            np.add.at(old_mass,q.coarse_nodes[q.coarse_elements].ravel(),-np.tile(removed_mass,len(q.coarse_elements)))
        self.diagonal=np.r_[old_mass[self.keep],np.zeros(self.n-len(self.keep))];self.mass_blocks=[]
        for q in self.patches:
            mf=q.block.mass(material.density_kg_m3);q.fine_mass=mf
            projected=(q.t.T@sparse.diags(mf)@q.t).tocsr()
            self.diagonal[q.support]+=projected.diagonal()
            nb=np.flatnonzero(q.support<len(self.keep));global_ids=q.support[nb]
            block=projected[nb][:,nb]+sparse.diags(old_mass[self.keep[global_ids]])
            self.mass_blocks.append(SimpleNamespace(ids=global_ids,matrix=block.tocsc(),factor=splu(block.tocsc())))
        if np.any(self.diagonal<=0):raise ArithmeticError('Nonpositive hybrid mass')
        self.removed_elements=np.concatenate([q.coarse_ids for q in self.patches])
        self.report=dict(target_width_m=patch_width_m,target_cells_per_axis=cells_per_patch_axis,
            target_spacing_m=patch_width_m/cells_per_patch_axis,order=p,scalar_dofs=self.n,
            original_scalar_dofs=self.old_n,removed_parent_elements=len(self.removed_elements),
            added_elements=sum(len(q.block.elements) for q in self.patches),
            patches=[dict(center=q.center,outer_bounds_m=q.bounds,actual_cells=[len(e)-1 for e in q.block.knots],
                minimum_cell_width_m=min(np.diff(e).min() for e in q.block.knots[:2]),
                actual_max_inner_spacing_m=float(max(np.diff(e)[((e[:-1]+e[1:])/2>=c-patch_width_m/2-1e-12)&
                    ((e[:-1]+e[1:])/2<=c+patch_width_m/2+1e-12)].max() for e,c in zip(q.block.knots[:2],q.center)))) for q in self.patches],
            constraint='conforming coarse-face interpolation; identical P^T K P and P^T M P',
            mass='original GLL element mass, exactly constrained; non-diagonal boundary blocks retained',
            unchanged_outside_transition=True)
    def points(self):
        if self.n*3*8>256*2**20:raise MemoryError('Use streaming metadata instead of full coordinate export')
        old=self.base.points()[self.keep];new=[q.points[q.interior] for q in self.patches]
        return np.concatenate([old,*new])
    def weights(self,xyz):
        for q in self.patches:
            if all(lo-1e-12<=x<=hi+1e-12 for x,(lo,hi) in zip(xyz,q.bounds)):
                ids,w=point_weights(q.block.knots,self.base.order,xyz)
                row=sparse.csr_matrix((w,(np.zeros(len(w),int),ids)),shape=(1,q.block.n))@q.t
                return q.support[row.indices],row.data
        ids,w=point_weights(self.coarse_knots,self.base.order,xyz);ids=self.old_to_new[ids]
        if np.any(ids<0):raise ArithmeticError('Point mapping enters removed cell')
        return ids,w
    def stiffness(self,x):
        u=np.asarray(x).reshape(3,self.n);expanded=np.zeros((3,self.old_n));expanded[:,self.keep]=u[:,:len(self.keep)]
        out=streamed_action(self.base,self.material,expanded.ravel()).reshape(3,self.old_n)[:,self.keep].copy()
        result=np.zeros((3,self.n));result[:,:len(self.keep)]=out
        ek=element_matrices(self.base.order,self.base.jacobian,self.material,'gll')[0]
        for q in self.patches:
            local=u[:,q.support];coarse=(q.ct@local.T).T;removed=np.zeros_like(coarse)
            for ids in q.coarse_elements:
                value=(ek@coarse[:,ids].ravel()).reshape(3,-1)
                for c in range(3):np.add.at(removed[c],ids,value[c])
            fine=q.block.action((q.t@local.T).T.ravel(),self.material).reshape(3,q.block.n)
            result[:,q.support]+=(q.t.T@fine.T-q.ct.T@removed.T).T
        return result.ravel()
    def mass(self,x,inverse=False):
        u=np.asarray(x).reshape(3,self.n);out=u/self.diagonal if inverse else u*self.diagonal
        for b in self.mass_blocks:out[:,b.ids]=(b.factor.solve(u[:,b.ids].T) if inverse else b.matrix@u[:,b.ids].T).T
        return out.ravel()


class LocalAttachment(SparseAttachment):
    """Unchanged PZT actions, with the host interpolation replaced by this mesh."""
    def __init__(self,data,mesh,geometry,*,first_dof,electrical='open',input_capacitance_f=0.):
        if electrical not in ('open','voltage') or input_capacitance_f<0:raise ValueError('Invalid port')
        self.data=data;self.geometry=geometry;self.electrical=electrical;ns=mesh.n;n=data.n
        contact=data.lower if geometry.surface=='top' else data.upper
        free=np.setdiff1d(np.arange(n),contact);self.free_nodes=free;self.nf=len(free)
        xyz=data.mesh.points_m.copy();xyz[:,:2]+=geometry.center_xy_m
        xyz[:,2]+=mesh.base.plate.thickness_m if geometry.surface=='top' else -geometry.thickness_m
        rr=[];cc=[];vv=[]
        for node in contact:
            ids,w=mesh.weights(xyz[node]);rr.extend([node]*len(ids));cc.extend(ids);vv.extend(w)
        host=np.unique(cc);self.nh=len(host);self.host_scalar_ids=host
        self.t=sparse.coo_matrix((np.r_[vv,np.ones(len(free))],(np.r_[rr,free],
            np.r_[np.searchsorted(host,cc),len(host)+np.arange(len(free))])),shape=(n,len(host)+len(free))).tocsr()
        ids=np.array([np.r_[host+c*ns,first_dof+c*len(free)+np.arange(len(free))] for c in range(3)])
        self.indices=np.unique(ids);self.positions=np.searchsorted(self.indices,ids).T;self.next_dof=first_dof+3*len(free)
        self.capacitance_f=data.capacitance_f;self.total_capacitance_f=data.capacitance_f+input_capacitance_f
        self.coupling=self.push(data.g)*geometry.poling_sign
        boundary=set(np.concatenate([b.ids for b in mesh.mass_blocks]).tolist())
        if boundary.intersection(host.tolist()):raise ValueError('PZT intersects hanging-node mass block; enlarge transition region')
        self.report=dict(data.report,center_xy_m=list(geometry.center_xy_m),surface=geometry.surface,
            bond='unchanged perfect tie on refined aluminum',body_mass_kg=float(data.scalar_mass.sum()),
            capacitance_f=data.capacitance_f,local_aluminum=mesh.report)
    def stiffness_action(self,x):
        if not hasattr(self.data,'tensor'):return super().stiffness_action(x)
        d=self.data;u=self.pull(x);k,charge=d.tensor.mechanical(u)
        phi=np.zeros(d.n);phi[d.inside]=d.electrical_lu.solve(charge[d.inside])
        y=self.push(k+d.tensor.electrical(phi)[0])
        if self.electrical=='open':y+=self.coupling*(self.coupling@x)/self.total_capacitance_f
        return y
