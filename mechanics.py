"""Rigid-body equilibrium in SI units. See README for signed interface conventions."""
import numpy as np

NAMES = [f'{q}{a}' for q in ('F_D12', 'M_D12', 'F_B23', 'F_C23') for a in 'XYZ'] + ['M_C2EX'] + [f'{q}{a}' for q in ('F_G3E', 'M_G3E') for a in 'XYZ']
UNITS = ['N']*3 + ['N m']*3 + ['N']*6 + ['N m'] + ['N']*3 + ['N m']*3
INPUTS = ['F_A1EX', 'F_A1EY', 'F_A1EZ', 'R_dx', 'R_cx', 'R_az', 'R_gz']

def validate(x):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != 7 or not np.isfinite(x).all():
        raise ValueError('Expected finite array [samples,7]: Fx,Fy,Fz,d,L,r,gz')
    if np.any((x[:,3] <= 0) | (x[:,4] <= x[:,3]) | (x[:,5] <= 0)):
        raise ValueError('Require 0 < d < L and gear radius r > 0')
    return x

def points(x):
    x = validate(x)
    a = np.zeros((len(x),3)); a[:,0]=x[:,3]; a[:,2]=x[:,5]
    d=a.copy(); d[:,2]=0
    c=np.zeros_like(a); c[:,0]=x[:,4]
    g=np.zeros_like(a); g[:,2]=x[:,6]
    return a,d,c,g

def solve(x):
    x=validate(x); f=x[:,:3]; a,d,c,g=points(x)
    fd=-f
    md=-np.cross(a-d,f)  # shaft acting on tooth wheel
    applied_m=np.cross(a,f)
    fc=np.zeros_like(f)
    fc[:,1]=-applied_m[:,2]/x[:,4]
    fc[:,2]=applied_m[:,1]/x[:,4]
    fb=-f-fc
    torque=-applied_m[:,0:1]  # external drive/brake, not bearing moment
    fg=fb+fc
    mg=np.cross(c,fc)-np.cross(g,fg)
    return np.column_stack([fd,md,fb,fc,torque,fg,mg])

def residuals(x,y):
    """Independent six-component force/moment balance for each of three bodies."""
    a,d,c,g=points(x); f=x[:,:3]
    fd,md,fb,fc=y[:,:3],y[:,3:6],y[:,6:9],y[:,9:12]
    t=np.zeros_like(f); t[:,0]=y[:,12]
    fg,mg=y[:,13:16],y[:,16:19]
    return np.stack([np.column_stack([f+fd,np.cross(a,f)+np.cross(d,fd)+md]),
                     np.column_stack([fb+fc-fd,np.cross(c,fc)-np.cross(d,fd)-md+t]),
                     np.column_stack([fg-fb-fc,np.cross(g,fg)+mg-np.cross(c,fc)])],axis=1)

def generate(n=30000,seed=42,ood=False):
    rng=np.random.default_rng(seed)
    f=rng.uniform(-100,100,(n,3))
    L=rng.uniform(.12,.24,n)
    d=L*rng.uniform(.2,.8,n)
    r=rng.uniform(.025,.09,n)
    gz=rng.uniform(-.09,-.04,n)
    if ood:
        f*=2; L*=1.4; d*=1.4; r*=1.4; gz*=1.4
    x=np.column_stack([f,d,L,r,gz])
    return x,solve(x)
