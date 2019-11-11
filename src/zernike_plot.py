'''
Created on 13 dec. 2016

@author: Marija
'''

from PyQt4 import QtGui, QtCore # (the example applies equally well to PySide)
import sys
import numpy as np
import threading
#from blaze.expr.expressions import symbol
import pyqtgraph as pg
from scipy import interpolate

class myWidgetTest(QtGui.QWidget):
    def __init__(self, parent=None):
        QtGui.QWidget.__init__(self, parent)
        
        self.x_points = 300
        self.y_points = 300
        #self.x, self.y = np.mgrid[-1:1:250j, -1:1:250j]
        self.x = np.linspace(-1.5,1.5,self.x_points)
        self.y = np.linspace(-1.5,1.5,self.y_points)
        self.zernike = np.zeros((self.x_points,self.y_points))
        self.grid0 = np.zeros((self.x_points,self.y_points))
        self.laserSpotSizeScale = 1
        
        #self.image = np.zeros((200,200))
        #print type(self.image)
        self.calc_voltages()
        self.myLayout()
        
    def func(self, x, y):
        return x*(1-x)*np.cos(4*np.pi*x) * np.sin(4*np.pi*y**2)**2
    
    def interp_membrane(self):
        #=======================================================================
        # points = np.random.rand(1000, 2)
        # values = self.func(points[:,0], points[:,1])
        # grid_x, grid_y = np.mgrid[0:1:100j, 0:1:200j]
        # grid_z0 = interpolate.griddata(points, values, (grid_x, grid_y), method='nearest')
        #=======================================================================
        
        points = self.electrodeCoordinates
        values = self.calc_zernike_one_point(points[:,0], points[:,1])
        grid_x, grid_y = np.mgrid[-1:1:self.x_points*1j, -1:1:self.x_points*1j]
        #grid_x, grid_y = np.mgrid[self.x,self.y]
        self.grid_z2 = interpolate.griddata(points, values, (grid_x, grid_y), method='nearest') 
        self.grid_z3 = interpolate.griddata(points, values, (grid_x, grid_y), method='linear') 
        self.grid_z4 = interpolate.griddata(points, values, (grid_x, grid_y), method='cubic') 
        self.grid_z3[np.isnan(self.grid_z3)] = 0
        self.grid_z4[np.isnan(self.grid_z4)] = 0

    def calc_voltages(self):
        self.a= 1/4.0
        self.h = self.a*np.sqrt(3)/2.0
        a = self.a
        h = self.h
        self.electrodeCoordinates = [[-3*a/2,3*h],[-a/2,3*h],[a/2,3*h],[3*a/2,3*h],[-2*a,2*h],[-a,2*h],[0,2*h],[a,2*h],[2*a,2*h],
[-5*a/2,h],[-3*a/2,h],[-a/2,h],[a/2,h],[3*a/2,h],[5*a/2,h],[-3*a,0],[-2*a,0],[-a,0],[0,0],[a,0],[2*a,0],[3*a,0],
[-5*a/2,-h],[-3*a/2,-h],[-a/2,-h],[a/2,-h],[3*a/2,-h],[5*a/2,-h],
[-2*a,-2*h],[-a,-2*h],[0,-2*h],[a,-2*h],[2*a,-2*h],
[-3*a/2,-3*h],[-a/2,-3*h],[a/2,-3*h],[3*a/2,-3*h]]
        self.electrodeCoordinates = np.array(self.electrodeCoordinates)
        
        
    def calc_zernike(self):
        n = self.n_button.value()
        m = self.m_button.value()
        for i in range(0,self.x_points):
            for j in range(0,self.y_points):
                rho = np.sqrt(self.x[i]**2 + self.y[j]**2)
                if rho > 1:
                    self.zernike[i,j] = 0
                elif m>=0:
                    phi = np.arctan2(self.y[j],self.x[i])
                    self.zernike[i,j] = self.radial(n,m,rho) * np.cos(phi*m)
                else:
                    phi = np.arctan2(self.y[j],self.x[i])
                    self.zernike[i,j] = self.radial(n,m,rho) * np.sin(phi*m)  

        
    def worker_calc_zernike(self):
        t1 = threading.Thread(name='calc_zernike', target=self.calc_zernike)
        t1.start()
        #t2 = threading.Thread(name='2d_interp', target=self.calc_zernike)

    
    def calc_zernike_one_point(self, x, y):
        n = self.n_button.value()
        m = self.m_button.value()
        rho = np.sqrt(x**2.0 + y**2.0)
        #if rho > 1:
        #    return 0
        if m>=0:
            phi = np.arctan2(y,x)
            return self.radial(n,m,rho) * np.cos(phi*m)
        else:
            phi = np.arctan2(y,x)
            return self.radial(n,m,rho) * np.sin(phi*m)        

    def radial(self,n,m,rho):
        if (n < 0 or abs(m) > n):
            raise ValueError
        if ((n-m) % 2):
            return rho*0.0        
        rad = 0
        m = np.abs(m)
        for k in range((n-m)/2+1):
            rad += (-1.0)**k*np.math.factorial(n-k) / (np.math.factorial(k)*np.math.factorial((n+m)/2.0-k)*np.math.factorial((n-m)/2.0-k)) *rho**(n-2.0*k)        
        return rad                

        
    def myLayout(self):
        self.layout = QtGui.QVBoxLayout(self) #the whole window, main layout
        self.inputLayout = QtGui.QGridLayout() 
        self.plotLayout = QtGui.QGridLayout()
        self.layout.addLayout(self.inputLayout)
        self.layout.addLayout(self.plotLayout)
        
        self.m_button = QtGui.QSpinBox()
        self.m_button.setRange(-50,50)
        self.m_button.setValue(0)
        self.inputLayout.addWidget(self.m_button,1,0)
        self.inputLayout.addWidget(QtGui.QLabel("Zernike m"), 0, 0)
        
        self.n_button = QtGui.QSpinBox()
        self.n_button.setValue(2)
        self.inputLayout.addWidget(self.n_button,1,1)
        self.inputLayout.addWidget(QtGui.QLabel("Zernike n"), 0, 1)

        
        self.m_button.editingFinished.connect(self.worker_calc_zernike)
        self.n_button.editingFinished.connect(self.worker_calc_zernike)

        self.scatterPlotItem1 = pg.ScatterPlotItem(size=30, pen=pg.mkPen(None), brush=pg.intColor(2))
        ah=0.5
        hh=ah*np.sqrt(3)/2.0
        hpoint1 = QtCore.QPointF( 0, -ah)
        hpoint2 = QtCore.QPointF( hh, -ah/2)
        hpoint3 = QtCore.QPointF(hh, ah/2)
        hpoint4 = QtCore.QPointF( 0,ah)
        hpoint5 = QtCore.QPointF( -hh, ah/2)
        hpoint6 = QtCore.QPointF(-hh, -ah/2)
        hexagonShape = QtGui.QPolygonF([hpoint1, hpoint2, hpoint3, hpoint4, hpoint5, hpoint6])
        spotShape = QtGui.QPainterPath()
        spotShape.addPolygon(hexagonShape)
        #self.scatterPlotItem1.Symbols['h'] = hexagonShape
        self.scatterPlotItem1.setData(symbol=spotShape)


        self.graphWindow1 = pg.GraphicsLayoutWidget()
        self.graphWindow2 = pg.GraphicsLayoutWidget()
        self.graphWindow3 = pg.GraphicsLayoutWidget()
        self.graphWindow4 = pg.GraphicsLayoutWidget()
        #self.graphWindow.setSizePolicy(QtGui.QSizePolicy.Expanding,QtGui.QSizePolicy.Expanding)
        self.plotLayout.addWidget(self.graphWindow1,0,0)
        self.plotLayout.addWidget(self.graphWindow2,0,1)
        self.plotLayout.addWidget(self.graphWindow3,1,0)
        self.plotLayout.addWidget(self.graphWindow4,1,1)
        
        view1 = self.graphWindow1.addViewBox()
        view1.setAspectLocked(True)
        self.img1 = pg.ImageItem(border='w')
        #self.scatterPlotItem1.addPoints(self.electrodeCoordinates[:,0]*self.x_points/2.0/self.laserSpotSizeScale+self.x_points/2, self.electrodeCoordinates[:,1]*self.y_points/2.0/self.laserSpotSizeScale+self.y_points/2)
        view1.addItem(self.img1)
        view1.addItem(self.scatterPlotItem1)
        
        view2 = self.graphWindow2.addViewBox()
        view2.setAspectLocked(True)
        self.img2 = pg.ImageItem(border='w')
        self.scatterPlotItem1.addPoints(self.electrodeCoordinates[:,0]*self.x_points/self.laserSpotSizeScale+self.x_points/2, self.electrodeCoordinates[:,1]*self.y_points/self.laserSpotSizeScale+self.y_points/2)
        view2.addItem(self.img2)
     
        view3 = self.graphWindow3.addViewBox()
        view3.setAspectLocked(True)
        self.img3 = pg.ImageItem(border='w')
        self.scatterPlotItem1.addPoints(self.electrodeCoordinates[:,0]*self.x_points/self.laserSpotSizeScale+self.x_points/2, self.electrodeCoordinates[:,1]*self.y_points/self.laserSpotSizeScale+self.y_points/2)
        view3.addItem(self.img3)
             
        view4 = self.graphWindow4.addViewBox()
        view4.setAspectLocked(True)
        self.img4 = pg.ImageItem(border='w')
        self.scatterPlotItem1.addPoints(self.electrodeCoordinates[:,0]*self.x_points/self.laserSpotSizeScale+self.x_points/2, self.electrodeCoordinates[:,1]*self.y_points/self.laserSpotSizeScale+self.y_points/2)
        view4.addItem(self.img4)
             
 
        # 
        #self.spots = QtGui.QGraphicsWidget()
        #self.spots.data(0)
        #self.graphWindow.addItem(self.spots,{'pos': self.electrodeCoordinates[1], 'data': 1, 'brush':pg.intColor(2), 'symbol': 'o', 'size': 50})
        
        
        #=======================================================================
        # spots = {'pos': self.electrodeCoordinates[1], 'data': 1, 'brush':pg.intColor(2), 'symbol': 'o', 'size': 50}
        # #for i in range(0,3):
        # #    spots.append({'pos': self.electrodeCoordinates[i], 'data': 1, 'brush':pg.intColor(2), 'symbol': 'o', 'size': 50})
        # 
        #=======================================================================

        
        self.calc_zernike()


        

if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)
    myapp = myWidgetTest()
    myapp.show()
    sys.exit(app.exec_())
