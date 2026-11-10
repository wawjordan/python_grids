import numpy as np
from pathlib import Path
from copy import deepcopy
import shutil
class compressible_vortex:
  theta_0:    float = 90.0
  theta_1:    float = 0.0
  r_inner:    float = 2.0
  r_outer:    float = 3.0
  eta:        float = 1.5
  rho_inner:  float = 1.0
  p_inner:    float = 0.8611583247416177E+05
  M_inner:    float = 2.0
  v_inner:    float = 1.0
  gamma:      float = 1.4
  gm1x2:      float = 0.2
  rho_ref:    float = 1.0
  a_ref:      float = 340.0
  l_ref:      float = 1.0
  l_ref_grid: float = 1.0

  def __init__( this, r_inner=2.0, r_outer=3.0, theta_0=90.0, theta_1=0.0, 
                M_inner=2.0, rho_inner=1.0, p_inner=None, gamma=1.4,
                rho_ref=1.0, a_ref=340.0, l_ref=1.0, l_ref_grid=1.0 ):
    this.l_ref = l_ref
    this.l_ref_grid = l_ref_grid
    L = this.l_ref / this.l_ref_grid
    this.theta_0    = np.deg2rad( theta_0 )
    this.theta_1    = np.deg2rad( theta_1 )
    this.r_inner    = r_inner / L
    this.r_outer    = r_outer / L
    this.eta        = this.r_outer / this.r_inner
    this.rho_ref    = rho_ref
    this.a_ref      = a_ref
    this.gamma      = gamma
    this.gm1x2      = (this.gamma - 1.)/2.
    this.rho_inner  = rho_inner / this.rho_ref
    if p_inner is None:
      this.p_inner = 1./this.gamma
    else:
      this.p_inner = p_inner / (this.rho_ref * this.a_ref**2)
    this.M_inner   = M_inner
    this.v_inner   =  this.r_inner * this.M_inner * this.rho_inner**this.gm1x2

  def density(this,x,y):
    r2 = x**2 + y**2
    rho = this.r_inner * ( 1. + this.gm1x2*this.M_inner**2
                              *( 1. - (this.r_inner^2 / r2) )
                         )**(1./(this.gamma-1.))
    return rho
  def x_velocity(this,x,y):
    r2 = x**2 + y**2
    u  = this.v_inner*y / r2
    return u
  def y_velocity(this,x,y):
    r2 = x**2 + y**2
    v  = -this.v_inner*x / r2
    return v
  def pressure(this,x,y):
    p = this.density(x,y)**this.gamma / this.gamma
    return p
  
  @staticmethod
  def calc_AR_simple( r_0, r_1, theta_0, theta_1, N_r, N_theta ):
    eta = r_1 / r_0
    deta = eta**( 1. / (N_r-1) )
    dtheta = abs(theta_1 - theta_0)/(N_theta - 1)
    AR = np.tan( dtheta / 2.) * ( deta + 1.) / ( deta - 1.)
    return AR

  @staticmethod
  def r_distribution(r_0,r_1,tau):
    eta = r_1 / r_0
    r = ( r_1 - r_0 )*( eta ** tau - 1. )/( eta - 1. )  + r_0
    return r

  def make_grid(this,N_theta,N_r):
    grid = {}
    grid['imax'] = N_theta
    grid['jmax'] = N_r
    theta = np.linspace( this.theta_0, this.theta_1, N_theta )
    r     = compressible_vortex.r_distribution( this.r_inner, this.r_outer,
                                                np.linspace(0.,1.,num=N_r) )
    THETA, R = np.meshgrid(theta, r, indexing='ij')
    grid['x'] = R * np.cos(THETA)
    grid['y'] = R * np.sin(THETA)
    return grid

def compressible_vortex_grid_family_output(cv, imax, jmax, levels, folder, skip1=1, skip2=1, 
                                           prefix='svf', write_grids=True, write_bcs=True, 
                                           plot_grid=False ):
  import matplotlib.pyplot as plt
  out_folder = ( Path(folder) / 'grids' ).resolve()
  GRID = cv.make_grid( imax, jmax )
  Nfine = [imax,jmax]
  GRID = grid_subset_2D( GRID, [ np.arange(0,Nfine[0],skip1),
                                 np.arange(0,Nfine[1],skip2) ] )
  if plot_grid:
    s1 = levels[-1]
    s2 = levels[-1]
    off = 0 # use this to double check handedness
    x = GRID['x'][0:Nfine[0]-off:2**s1, 0::2**s2]
    y = GRID['y'][0:Nfine[0]-off:2**s1, 0::2**s2]
    ax = plt.gca()
    ax.plot( x, y, color='k' )
    ax.plot( np.transpose(x), np.transpose(y), color='k' )
    plt.show(block=False)
    plt.axis('equal')
    dumb = 1

  bc_id_list  = [201,201,-200,-200]

  n_levels = len(levels)
  if write_grids or write_bcs:
    for j in range(n_levels):
      s = 2**levels[j]
      print(f"writing level {j+1:d} of {n_levels:d}")
      GRID2 = grid_subset_2D( GRID, [ np.arange(0,Nfine[0],s),
                                      np.arange(0,Nfine[1],s) ] )
      jobfmt     = "_" + prefix + f"{GRID2['imax']:04d}x{GRID2['jmax']:04d}"
      foldername = Path(out_folder / jobfmt )
      foldername.mkdir(parents=True, exist_ok=True)
      filename = (Path(foldername) / (prefix + '.grd')).resolve()
      if write_grids:
        p2d_out(GRID2,filename)
      Ni = [ 1, int((GRID['imax'] - 1)/s + 1) ]
      Nj = [ 1, int((GRID['jmax'] - 1)/s + 1) ]
      N  = [ GRID2['imax'], GRID2['jmax'] ]
      idx_list = []
      idx_list.append([ Ni[0], Ni[1],  Nj[0], Nj[0] ])
      idx_list.append([ Ni[0], Ni[1],  Nj[1], Nj[1] ])
      idx_list.append([ Ni[0], Ni[0],  Nj[0], Nj[1] ])
      idx_list.append([ Ni[1], Ni[1],  Nj[0], Nj[1] ])
      additional_inputs = []
      additional_inputs.append([])
      additional_inputs.append([])
      additional_inputs.append([])
      additional_inputs.append([])
      filename = ( Path(foldername) / (prefix + '.bc')).resolve()
      if write_bcs:
        SENSEI_BC_write( N, idx_list, bc_id_list, additional_inputs, filename)


def SENSEI_BC_write( N, idx_list, bc_id_list, additional_inputs, filename ):
  N_bcs = len(bc_id_list)
  # intfmt = ' %-5d';{i:5d}
  # fltfmt = ' %-.16g';{i:.18e}
  with open(filename,'w', encoding="utf-8") as f:
    f.write("1\n")
    f.write("1\n")
    for n in N:
      f.write( f" {n:<5d}")
    f.write("\n")
    f.write("A\n")
    f.write(f"{N_bcs:<5d}\n")
    for n in range(N_bcs):
      f.write(f"bc{n+1:d}  ")
      for idx in idx_list[n]:
        f.write(f" {idx:<5d}")
      f.write(f" {bc_id_list[n]:<5d}")
      for input in additional_inputs[n]:
        f.write(f" {input:<.18e}")
      f.write("\n")


def grid_subset_2D(GRID,small_ind):
  GRID2 = deepcopy(GRID)
  GRID2['x'] = GRID2['x'][np.ix_(*small_ind)]
  GRID2['y'] = GRID2['y'][np.ix_(*small_ind)]
  GRID2['imax'] = len(small_ind[0])
  GRID2['jmax'] = len(small_ind[1])
  return GRID2

def p2d_out(GRID,filename):
  np.ravel(GRID['x'],order='F')
  with open(filename, "w", encoding="utf-8") as f:
    f.write(f"{GRID['imax']} {GRID['jmax']}\n")
    np.savetxt(f, np.ravel(GRID['x'],order='F'), delimiter='\n' )
    np.savetxt(f, np.ravel(GRID['y'],order='F'), delimiter='\n' )

def do_something():
  print("Hello World!")

if __name__ == "__main__":
  # xv, yv = np.meshgrid(np.linspace(0.,1.,num=11), np.linspace(0.,1.,num=11) )
  # c1 = [np.arange(0,9),np.arange(0,5)]
  # xv[np.ix_(*c1)]
  cv = compressible_vortex()
  # levels = np.arange(0,6)+1
  # imax   = 4097
  # jmax   = 1025
  levels = np.arange(0,6)
  imax   = 4097
  jmax   = 1025
  folder = '/home/wajordan/python_grids'
  compressible_vortex_grid_family_output( cv, imax, jmax, levels, folder, 
                                          skip1=1, skip2=1, prefix='svf', 
                                          write_grids=False,
                                          write_bcs=True,
                                          plot_grid=False   )
  # GRID = cv.make_grid(5,5)
  # p2d_out(GRID,'test.grd')
  # do_something()