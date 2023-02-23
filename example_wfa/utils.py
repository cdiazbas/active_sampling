import numpy as np

# ====================================================================
class line:
    """
    Class line is used to store the atomic data of spectral lines. We use this
    class as input for the WFA routines below.
    Usage: 
        lin = line(8542)
    """
    def __init__(self, cw=8542):

        self.larm = 4.668645048281451e-13
        
        if(cw == 8542):
            self.j1 = 2.5; self.j2 = 1.5; self.g1 = 1.2; self.g2 = 1.33; self.cw = 8542.091
        elif(cw == 6301):
            self.j1 = 2.0; self.j2 = 2.0; self.g1 = 1.84; self.g2 = 1.50; self.cw = 6301.4995
        elif(cw == 6302):
            self.j1 = 1.0; self.j2 = 0.0; self.g1 = 2.49; self.g2 = 0.0; self.cw = 6302.4931
        elif(cw == 8468):
            self.j1 = 1.0; self.j2 = 1.0; self.g1 = 2.50; self.g2 = 2.49; self.cw = 8468.4059
        elif(cw == 6173):
            self.j1 = 1.0; self.j2 = 0.0; self.g1 = 2.50; self.g2 = 0.0; self.cw = 6173.3340
        elif(cw == 5173):
            self.j1 = 1.0; self.j2 = 1.0; self.g1 = 1.50; self.g2 = 2.0; self.cw = 5172.6843
        elif(cw == 5896):
            self.j1 = 0.5; self.j2 = 0.5; self.g1 = 2.00; self.g2 = 2.0/3.0; self.cw = 5895.9242
        else:
            print("line::init: ERROR, line not implemented")
            self.j1 = 0.0; self.j2 = 0.0; self.g1 = 0.0; self.g2 = 0.0; self.cw = 0.0
            return

        j1 = self.j1; j2 = self.j2; g1 = self.g1; g2 = self.g2
        
        d = j1 * (j1 + 1.0) - j2 * (j2 + 1.0);
        self.geff = 0.5 * (g1 + g2) + 0.25 * (g1 - g2) * d;
        ss = j1 * (j1 + 1.0) + j2 * (j2 + 1.0);
        dd = j1 * (j1 + 1.0) - j2 * (j2 + 1.0);
        gd = g1 - g2;
        self.Gg = (self.geff * self.geff) - (0.0125  * gd * gd * (16.0 * ss - 7.0 * dd * dd - 4.0));

        print("line::init: cw={0}, geff={1}, Gg={2}".format(self.cw, self.geff, self.Gg))


# ====================================================================
def cder(x, y):
    """
    function cder computes the derivatives of Stokes I (y)
    
    Input: 
            x: 1D wavelength array
            y: 4D data array (ny, nx, nStokes, nw)
            Use the usual centered derivatives formula for non-equidistant grids.
    """
    ny, nx, nstokes, nlam = y.shape[:]
    yp = np.zeros((ny, nx, nlam), dtype='float32')
    
    odx = x[1]-x[0]; ody = (y[:,:,0,1] - y[:,:,0,0]) / odx
    yp[:,:,0] = ody
    
    for ii in range(1,nlam-1):
        dx = x[ii+1] - x[ii]
        dy = (y[:,:,0,ii+1] - y[:,:,0,ii]) / dx
        
        yp[:,:,ii] = (odx * dy + dx * ody) / (dx + odx)
        
        odx = dx; ody = dy
    
    yp[:,:,-1] = ody    
    return yp


# ====================================================================
def cder2(x, y):
    """
    function cder computes the derivatives of Stokes I (y)
    
    Input: 
            x: 1D wavelength array
            y: 4D data array (ny, nx, nStokes, nw)
            Use the usual centered derivatives formula for non-equidistant grids.
    """
    ny, nx, nstokes, nlam = y.shape[:]
    yp = np.zeros((ny, nx, nlam), dtype='float32')
    
    odx = x[1]-x[0]; ody = (y[:,:,0,1] - y[:,:,0,0]) / odx
    yp[:,:,0] = ody
    
    for ii in range(1,nlam-1):
        dx = x[ii+1] - x[ii]
        dy = (y[:,:,0,ii+1] - y[:,:,0,ii]) / dx
        
        # yp[:,:,ii] = (odx * dy + dx * ody) / (dx + odx)
        yp[:,:,ii] = (dx * dy + dx * dy) / (dx + odx)
        
        odx = dx; ody = dy
    
    yp[:,:,-1] = ody    
    return yp
