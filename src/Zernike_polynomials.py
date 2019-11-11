'''
Created on 4 mars 2016

@author: Marija
'''

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
#from mpl_toolkits.mplot3d import Axes3D
from scipy.misc import factorial as fac

#example 1
def radialPoly(rho, n,m):
    rp = 0
    m=np.abs(m)
    if is_odd(n-m):
        rp = 0
    else:
        for k in range(0,(n-m)/2):
            rp += (-1.0)**k * fac(n-k) * rho**(n-2.0*k) / (np.math.factorial(k) * fac((n+m)/2.0 - k) * fac((n-m)/2.0 - k)) 
    return rp

def ZernikePolyRad(rho,phi,n,m):
    if m>0:
        zp = radialPoly(rho, n, m) * np.cos(m*phi)
    else:
        zp = radialPoly(rho, n, m) * np.sin(-m*phi)         
    return zp

def ZernikePoly(x,y,n,m):
    rho = np.sqrt(x**2+y**2)
    phi = np.arctan2(x,y)
    if rho>1:
        zp = 0
    else:
        zp = ZernikePolyRad(rho, phi, n, m)
    return zp

def zernike_poly(x,y,n,m):
    rho = np.sqrt(x**2+y**2)
    phi = np.arctan2(x,y)
    if rho>1:
        zp = 0
    else:
        zp = zernike(rho, phi, n, m)
    return zp

def zernike_rad(rho, n,m):
    """
    Calculate the radial component of Zernike polynomial (m, n) 
    given a grid of radial coordinates rho.
    
    >>> zernike_rad(3, 3, 0.333)
    0.036926037000000009
    >>> zernike_rad(1, 3, 0.333)
    -0.55522188900000002
    >>> zernike_rad(3, 5, 0.12345)
    -0.007382104685237683
    """
    
    if (n < 0 or m < 0 or abs(m) > n):
        raise ValueError
    
    if ((n-m) % 2):
        return rho*0.0
    else:
        pre_fac = lambda k: (-1.0)**k * fac(n-k) / ( fac(k) * fac( (n+m)/2.0 - k ) * fac( (n-m)/2.0 - k ) )
        return sum(pre_fac(k) * rho**(n-2.0*k) for k in xrange((n-m)/2+1))

def zernike(rho, phi, n, m):
    """
    Calculate Zernike polynomial (m, n) given a grid of radial
    coordinates rho and azimuthal coordinates phi.
    
    >>> zernike(3,5, 0.12345, 1.0)
    0.0073082282475042991
    >>> zernike(1, 3, 0.333, 5.0)
    -0.15749545445076085
    """
    if (m > 0): return zernike_rad(m, n, rho) * np.cos(m * phi)
    if (m < 0): return zernike_rad(-m, n, rho) * np.sin(-m * phi)
    return zernike_rad(0, n, rho)
   
def is_odd(num):
    #print num % 2 != 0 
    return num % 2 != 0 
    
#plt.close('all')

n = 3
m = 1

print zernike(0.72345,0.5,n,m)
print ZernikePolyRad(0.72345,0.5,n,m)

x = y = np.arange(-1.0, 1.0, 0.01)
X, Y = np.meshgrid(x, y)
Z = np.array([ZernikePoly(x,y,n,m) for x,y in zip(np.ravel(X), np.ravel(Y))])
Z1 = np.array([zernike_poly(x,y,n,m) for x,y in zip(np.ravel(X), np.ravel(Y))])

#Z = np.array([radialPoly(np.sqrt(x**2.0+y**2.0), n, m) for x,y in zip(np.ravel(X), np.ravel(Y))])

Z = np.reshape(Z, (len(X), len(Y)))
Z1 = np.reshape(Z1, (len(X), len(Y)))

#===============================================================================
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.plot_surface(X, Y, Z, cmap=cm.coolwarm)
# plt.show()
#===============================================================================

plt.subplot(2, 1, 1)
plt.pcolor(X, Y, Z, cmap='RdBu')
plt.axis('scaled')
plt.ylabel('y')
plt.xlabel('x')
titleStr = 'Zernike polynomial n='+str(n)+', m='+str(m)
plt.title(titleStr)
#plt.show()


plt.subplot(2, 1, 2)
plt.pcolor(X, Y, Z1, cmap='RdBu')
plt.axis('scaled')
plt.ylabel('y')
plt.xlabel('x')
titleStr = 'Zernike polynomial n='+str(n)+', m='+str(m)
plt.title(titleStr)
plt.show()
#===============================================================================
# ax = plt.subplot(111, projection='polar')
#  
# ax.plot(theta, np.square(r), color='r', linewidth=3)
# ax.set_rmax(2.0)
# ax.grid(True)
#  
# ax.set_title("A line plot on a polar axis", va='bottom')
# plt.show()
#===============================================================================




