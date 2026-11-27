"""Q2 FP64 partial assembly: three 1D contractions per evaluation/transpose.

One CUDA block per PZT element, as in the plate's tensor kernel. Nonaffine PZT
geometry uses the ORIGINAL per-Gauss-point inverse Jacobian and six-point rule.
The original assembled CPU/GPU operators remain independent references.
"""
import numpy as np

KERNEL = r'''
#define P 3
#define Q 6
#define NP 27
#define NQ 216
__device__ void eval(const double* u,const double* A,const double* B,const double* C,
                     double* a,double* b,double* v) {
    int t=threadIdx.x;
    if(t<Q*P*P){int x=t/(P*P),j=(t/P)%P,k=t%P;double s=0.;
        for(int i=0;i<P;++i)s+=A[x*P+i]*u[(i*P+j)*P+k];a[t]=s;}
    __syncthreads();
    if(t<Q*Q*P){int x=t/(Q*P),y=(t/P)%Q,k=t%P;double s=0.;
        for(int j=0;j<P;++j)s+=B[y*P+j]*a[(x*P+j)*P+k];b[t]=s;}
    __syncthreads();
    if(t<NQ){int x=t/(Q*Q),y=(t/Q)%Q,z=t%Q;double s=0.;
        for(int k=0;k<P;++k)s+=C[z*P+k]*b[(x*Q+y)*P+k];v[t]=s;}
    __syncthreads();
}
__device__ void back(const double* v,const double* A,const double* B,const double* C,
                     double* a,double* b,double* out,const int* ids,int n,int axis) {
    int t=threadIdx.x;
    if(t<Q*Q*P){int x=t/(Q*P),y=(t/P)%Q,k=t%P;double s=0.;
        for(int z=0;z<Q;++z)s+=C[z*P+k]*v[(x*Q+y)*Q+z];b[t]=s;}
    __syncthreads();
    if(t<Q*P*P){int x=t/(P*P),j=(t/P)%P,k=t%P;double s=0.;
        for(int y=0;y<Q;++y)s+=B[y*P+j]*b[(x*Q+y)*P+k];a[t]=s;}
    __syncthreads();
    if(t<NP){int i=t/(P*P),j=(t/P)%P,k=t%P;double s=0.;
        for(int x=0;x<Q;++x)s+=A[x*P+i]*a[(x*P+j)*P+k];atomicAdd(out+axis*n+ids[t],s);}
    __syncthreads();
}
extern "C" __global__ void apply(const int* all_ids,const double* N,const double* G,
 const double* inv,const double* w,const double* C,const double* E,const double* eps,
 const double* u,double* mechanical,double* scalar,int n,int mode,double density) {
    int e=blockIdx.x,t=threadIdx.x;const int* ids=all_ids+e*NP;
    __shared__ double nodes[NP],a[Q*P*P],b[Q*Q*P],r[9*NQ],f[3*NQ],val[NQ];
    if(mode==2){
        if(t<NP)nodes[t]=u[ids[t]];__syncthreads();
        eval(nodes,N,N,N,a,b,val);
        if(t<NQ)val[t]*=density*w[e*NQ+t];__syncthreads();
        back(val,N,N,N,a,b,scalar,ids,n,0);return;
    }
    int fields=mode==0 ? 3 : 1;
    for(int c=0;c<fields;++c){
        if(t<NP)nodes[t]=u[c*n+ids[t]];__syncthreads();
        eval(nodes,G,N,N,a,b,r+(3*c)*NQ);
        eval(nodes,N,G,N,a,b,r+(3*c+1)*NQ);
        eval(nodes,N,N,G,a,b,r+(3*c+2)*NQ);
    }
    if(t<NQ){
        const double* J=inv+(e*NQ+t)*9;double stress[6],electric[3];
        if(mode==0){
            double g[9];for(int c=0;c<3;++c)for(int i=0;i<3;++i){double s=0.;
                for(int j=0;j<3;++j)s+=J[i*3+j]*r[(3*c+j)*NQ+t];g[c*3+i]=s;}
            double strain[6]={g[0],g[4],g[8],g[5]+g[7],g[2]+g[6],g[1]+g[3]};
            for(int s=0;s<6;++s){stress[s]=0.;for(int j=0;j<6;++j)stress[s]+=C[s*6+j]*strain[j];}
            for(int i=0;i<3;++i){electric[i]=0.;for(int s=0;s<6;++s)electric[i]+=E[i*6+s]*strain[s];}
        }else{
            double grad[3];for(int i=0;i<3;++i){grad[i]=0.;for(int j=0;j<3;++j)grad[i]+=J[i*3+j]*r[j*NQ+t];}
            for(int s=0;s<6;++s){stress[s]=0.;for(int i=0;i<3;++i)stress[s]+=E[i*6+s]*grad[i];}
            if(mode!=3)for(int i=0;i<3;++i){electric[i]=0.;for(int j=0;j<3;++j)electric[i]+=eps[i*3+j]*grad[j];}
        }
        int map[9]={0,5,4,5,1,3,4,3,2};double weight=w[e*NQ+t];
        for(int c=0;c<3;++c)for(int j=0;j<3;++j){double s=0.;
            for(int i=0;i<3;++i)s+=J[i*3+j]*stress[map[c*3+i]];r[(3*c+j)*NQ+t]=s*weight;}
        if(mode!=3)for(int j=0;j<3;++j){double s=0.;for(int i=0;i<3;++i)s+=J[i*3+j]*electric[i];f[j*NQ+t]=s*weight;}
    }
    __syncthreads();
    for(int c=0;c<3;++c){
        back(r+(3*c)*NQ,G,N,N,a,b,mechanical,ids,n,c);
        back(r+(3*c+1)*NQ,N,G,N,a,b,mechanical,ids,n,c);
        back(r+(3*c+2)*NQ,N,N,G,a,b,mechanical,ids,n,c);
    }
    if(mode!=3){
        back(f,G,N,N,a,b,scalar,ids,n,0);
        back(f+NQ,N,G,N,a,b,scalar,ids,n,0);
        back(f+2*NQ,N,N,G,a,b,scalar,ids,n,0);
    }
}
'''


class DeviceTensor:
    def __init__(self,tensor):
        import cupy as cp
        self.cp=cp;self.n=tensor.n;self.ne=len(tensor.mesh.hexahedra)
        if tensor.mesh.order!=2:raise ValueError('Q2 only')
        self.ids=cp.asarray(tensor.mesh.hexahedra,dtype=cp.int32)
        self.N=cp.asarray(tensor.N,order='C');self.G=cp.asarray(tensor.G,order='C')
        self.inverse=cp.asarray(tensor.inverse,order='C');self.weight=cp.asarray(tensor.weight,order='C')
        self.C,self.E,self.eps=[cp.asarray(v,order='C') for v in tensor.material.stress_charge]
        self.density=tensor.material.density_kg_m3
        self.kernel=cp.RawKernel(KERNEL,'apply',options=('--std=c++11','--fmad=false'))

    def apply(self,u,mechanical,scalar,mode=0):
        if u.dtype!=np.dtype('float64') or mechanical.dtype!=u.dtype or scalar.dtype!=u.dtype:
            raise TypeError('FP64 buffers required')
        if mode not in (0,1,2,3):raise ValueError('Unknown tensor action')
        if u.size!=(3*self.n if mode==0 else self.n) or mechanical.size!=3*self.n or scalar.size!=self.n:
            raise ValueError('Wrong fixed tensor buffer shape')
        if not all(v.flags.c_contiguous for v in (u,mechanical,scalar)):
            raise ValueError('Contiguous tensor buffers required')
        arrays=(u,mechanical,scalar)
        for i,x in enumerate(arrays):
            for y in arrays[i+1:]:
                if max(x.data.ptr,y.data.ptr)<min(x.data.ptr+x.nbytes,y.data.ptr+y.nbytes):
                    raise ValueError('Tensor buffers must not overlap')
        mechanical.fill(0.);scalar.fill(0.)
        self.kernel((self.ne,),(256,),(self.ids,self.N,self.G,self.inverse,self.weight,
            self.C,self.E,self.eps,u,mechanical,scalar,np.int32(self.n),np.int32(mode),np.float64(self.density)))

    def bytes(self):
        return sum(v.nbytes for v in (self.ids,self.N,self.G,self.inverse,self.weight,self.C,self.E,self.eps))
