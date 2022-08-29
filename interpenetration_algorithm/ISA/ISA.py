import numpy as np
import mdtraj as md
import matplotlib as mpl
import matplotlib.pyplot as plt

import os
import glob
import subprocess
import sys
import itertools
import time

from math import ceil

from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable
# from mpl_toolkits.basemap import Basemap

from matplotlib.widgets import Lasso

from matplotlib.colors import colorConverter

from matplotlib.collections import RegularPolyCollection

from matplotlib import path

from multiprocessing import cpu_count, Pool, TimeoutError
from joblib import Parallel, delayed

from ipywidgets import FloatProgress
from IPython.display import display
from IPython.display import clear_output

# ISA
# Interpenetration and Scoring Algorithm

class ISA:
    """This docstring is just a placeholder
    
    Args:
        input_arr (np.ndarray): The atomic positions of the nucleosome.
        
    Methods:
        get_score:
        
    """
    
    init_bins = 25
    propagation_factor = 1.0
    threshold = 0.2
    
    def test(self):
        """placeholder function to check autoreload.
        
        It works.
        
        """
        print('Hello')
    
    def __init__(self, input_arr, weights=False):
        self.input_arr = input_arr
        self.weights = weights
        if not type(self.input_arr) is np.ndarray:
            raise Exception('Please give a N,3 numpy ndarray')
        if not type(self.weights) is bool:
            if not type(self.weights) is np.ndarray:
                raise Exception('Please give a N, numpy ndarray')
        self.point_no = self.input_arr.shape[0]
        print('creating ISA object with '+str(self.point_no)+' points.')
        self.dimnum = self.input_arr.shape[1]
        if self.dimnum != 3:
            raise Exception('Currently only supports 3D datasets')
        if not type(self.weights) is bool:
            if len(self.weights) != self.point_no:
                raise Exception('Weight array needs to be same length as input array')
        
        self.print_debug()
    
    def gen_histogram(self):
        # run numpy histogram
        if self.dimnum == 3:
            if not type(self.weights) is bool:
                self.H, self.edges = np.histogramdd(self.input_arr, weights=self.weights, bins=self.init_bins)
                # due to the weighing the lowest achievable value is below 0 and the background gets 
                # a lighter color
            else:
                self.H, self.edges = np.histogramdd(self.input_arr, bins=self.init_bins)
            # There has been a lot of confusion regarding the plotting of histograms.
            # For some unknown reason matplotlib expects for histograms the first dimension to be y.
            # So the calculation is correct, but for the plotting the histogram should be transposed
            # see: https://stackoverflow.com/questions/43568370/matplotlib-2d-histogram-seems-transposed
            # However this made the surface look not so good. I got a bad color gradient.
            # Maybe I need to reevaluate about this.
            # Pyemma does transpose the 2D histogram right after creation in its plot_free_energy function
            # If I would do this, the histogram needs to be transposed slice-wise
#             self.H_tmp = np.ones(shape=self.H.shape)
#             for slice_ in self.H:
#                 self.H_tmp[slice_] = self.H[slice_].T
#             self.H = self.H_tmp
        else:
            raise Exception('Currently only support 3d binning')
        # set histo_out
        self.histo_out = self.H
        
        # define some minor stuff
        self.x_bins = self.H.shape[0]
        self.y_bins = self.H.shape[1]
        self.z_bins = self.H.shape[2]
        self.n_bins = self.x_bins * self.y_bins * self.z_bins
        
    def gen_cumulative_histograms(self):
        # define corners
        self.corners = []
        for x in [0, self.x_bins - 1]:
            for y in [0, self.x_bins - 1]:
                for z in [0, self.x_bins - 1]:
                    append = [x, y, z]
                    self.corners.append(append)
        
        # initialize cumulative histogram search
        self.no_of_search_bubbles = 2 ** self.dimnum
        
        # initialize search bubbles
        self.iter_bubbles = []
        for i in range(self.no_of_search_bubbles):
            iteration = self.make_iter_bubble(i, self.corners[i])
            self.iter_bubbles.append(iteration)
        
        # get the values for each bubble
        self.values = []
        for i in range(self.no_of_search_bubbles):
            value = self.make_value_bubble(i, self.corners[i], self.iter_bubbles[i])
            self.values.append(value)
            
        # add the values according to my idea with 1D histograms
        self.histo_out = self.add_values()
        # I decided to skip the normalization to get integer values of the scores
        self.histo_out = self.histo_out / np.max(self.histo_out)
            
        # mean over all bubbles
        # self.histo_out = np.zeros((self.H.shape[0], self.H.shape[1], self.H.shape[2]))
        # mean = 0
        # for histo in self.values:
        #     self.histo_out = self.histo_out + histo
        #     mean += 1
        # self.histo_out = self.histo_out / mean
        
    def make_iter_bubble(self, number, corner):
        # print('generating iteration bubble for corner '+str(corner))
        H_iter = np.zeros((self.H.shape[0], self.H.shape[1], self.H.shape[2]), dtype=int)
        H_iter[corner[0],corner[1],corner[2]] = 1
        
        iteration = 1
        while 0 in H_iter:
            for (x, y, z), value in np.ndenumerate(self.H):
                # iterate over all bins
                if H_iter[x][y][z] != 0:
                    # only continue if bin is not finished
                    pass
                else:
                    # only calculate nearest neighbour values, if (x, y, z) aren't out of bounds
                    values = []
                    if x < self.init_bins - 1:
                        X_val = H_iter[x+1, y, z]
                        values.append(X_val)
                    if x > 0:
                        x_val = H_iter[x-1, y, z]
                        values.append(x_val)
                    if y < self.init_bins -1:
                        Y_val = H_iter[x, y+1, z]
                        values.append(Y_val)
                    if y > 0:
                        y_val = H_iter[x, y-1, z]
                        values.append(y_val)
                    if z < self.init_bins - 1:
                        Z_val = H_iter[x, y, z+1]
                        values.append(Z_val)
                    if z > 0:
                        z_val = H_iter[x, y, z-1]
                        values.append(z_val)
                    # if the iteration value occurs in neighbours. set new values
                    if iteration in values:
                        H_iter[x][y][z] = iteration + 1
            iteration += 1
        # globally subtract 1    
        H_iter = H_iter - 1
        return(H_iter)
    
    def make_value_bubble(self, number, corner, iter_bubble):
        H_out = np.zeros((self.H.shape[0], self.H.shape[1], self.H.shape[2])).astype(np.float)
        H_out[corner[0],corner[1],corner[2]] = 0.0
        
        for iteration in range(np.max(iter_bubble) + 1): # +1 is needed so the last bin also gets calculated
            if iteration == 0:
                pass
            else:
                iter_count = len(np.where(iter_bubble == iteration)[0])
                # print('assigning new values to '+str(iter_count)+' bins')
                # at what iteration are we
                where_x = np.where(iter_bubble == iteration)[0]
                where_y = np.where(iter_bubble == iteration)[1]
                where_z = np.where(iter_bubble == iteration)[2]
                append_values = []
                for i in range(len(where_x)):
                    x = where_x[i]
                    y = where_y[i]
                    z = where_z[i]
                    # print('searching for neighbours for bin ('+str(x)+', '+str(y)+', '+str(z)+')')
                    # define all the nearest neighbours of the bin
                    neighbours = []
                    # +1 neighbours
                    X_neighbour = [x+1, y, z]; neighbours.append(X_neighbour)
                    x_neighbour = [x-1, y, z]; neighbours.append(x_neighbour)
                    Y_neighbour = [x, y+1, z]; neighbours.append(Y_neighbour)
                    y_neighbour = [x, y-1, z]; neighbours.append(y_neighbour)
                    Z_neighbour = [x, y, z+1]; neighbours.append(Z_neighbour)
                    z_neighbour = [x, y, z-1]; neighbours.append(z_neighbour)
                    # +2 neighbours
#                     xx_neighbour = [x-2, y, z]; neighbours.append(xx_neighbour)
#                     XX_neighbour = [x+2, y, z]; neighbours.append(XX_neighbour)
#                     yy_neighbour = [x, y-2, z]; neighbours.append(yy_neighbour)
#                     YY_neighbour = [x, y+2, z]; neighbours.append(YY_neighbour)
#                     zz_neighbour = [x, y, z-2]; neighbours.append(zz_neighbour)
#                     ZZ_neighbour = [x, y, z+2]; neighbours.append(ZZ_neighbour)
#                     zx_neighbour = [x-1, y, z-1]; neighbours.append(zx_neighbour)
#                     zX_neighbour = [x+1, y, z-1]; neighbours.append(zX_neighbour)
#                     zy_neighbour = [x, y-1, z-1]; neighbours.append(zy_neighbour)
#                     zY_neighbour = [x, y+1, z-1]; neighbours.append(zY_neighbour)
#                     Zx_neighbour = [x-1, y, z+1]; neighbours.append(Zx_neighbour)
#                     ZX_neighbour = [x+1, y, z+1]; neighbours.append(ZX_neighbour)
#                     Zy_neighbour = [x, y-1, z+1]; neighbours.append(Zy_neighbour)
#                     ZY_neighbour = [x, y+1, z+1]; neighbours.append(ZY_neighbour)
                    # check, whether some neighbours are out of bounds
                    neighbours_tmp = []
                    for neighbour in neighbours:
                        if -1 in neighbour or self.init_bins in neighbour or -2 in neighbour or self.init_bins + 1 in neighbour:
                            pass # removes out of bound neighbours
                        else:
                            neighbours_tmp.append(neighbour)
                    # print('bin ('+str(x)+', '+str(y)+', '+str(z)+') has '+str(len(neighbours_out))+' neighbours inside boundaries')
                    # remove neighbours which are in front of the iter bubble
                    neighbours_out = []
                    for coords in neighbours_tmp:
                        x_n = coords[0]
                        y_n = coords[1]
                        z_n = coords[2]
                        if iter_bubble[x_n][y_n][z_n] == iteration + 1:
                            pass # removes neighbour in front of iter bubble
                        else:
                            neighbours_out.append(coords)
                    # count number of neighbours from previous iterations
                    neighbour_count = 0
                    for coords in neighbours_out:
                        x_n = coords[0]
                        y_n = coords[1]
                        z_n = coords[2]
                        if iter_bubble[x_n][y_n][z_n] == iteration - 1:
                            neighbour_count += 1
                    if neighbour_count != len(neighbours_out):
                        raise Exception('I have a mismatch between the list containing the past neighbours and the count of the neighbours to be considered')
                    # add the values up
                    val = 0.0
                    for coords in neighbours_out:
                        x_n = coords[0]
                        y_n = coords[1]
                        z_n = coords[2]
                        if iter_bubble[x_n][y_n][z_n] == iteration - 1:
                            # Option 1. The new bin only gets 1/nth of each neighbours count, where n
                            # is the number of neighbours
                            val = val + (1.0 / neighbour_count * H_out[x_n][y_n][z_n] * self.propagation_factor)
                            # Option 2. The new bin gets the full values of all of its neighbouring bins
                            # This makes the algorithm too quickly jump to too high numbers and leads to an integer overflow
                            # I have changed dataformat of H_out to np.float but this hasn't worked either, because
                            # The value is rising to quickly
                            # val = val + (H_out[x_n][y_n][z_n]) * self.propagation_factor
                    append_values.append(val + self.H[x][y][z])
                # 2nd for loop is needed, so the 2nd bin in iteration k doesn't already receive the value of it neighbour,
                # which has also to be calculated within iteration k
                for i in range(len(where_x)):
                    x = where_x[i]
                    y = where_y[i]
                    z = where_z[i]
                    new_val = append_values[i]
                    H_out[x][y][z] = new_val
        # normalize
        H_out = H_out / np.max(H_out)
        # don't normalize
        return(H_out)
            
    def remove_halo(self):
        self.histo_no_halo = self.histo_out
        
        for (x, y, z), value in np.ndenumerate(self.histo_no_halo):
            neighbours = []
            # +1 neighbours
            X_neighbour = [x+1, y, z]; neighbours.append(X_neighbour)
            x_neighbour = [x-1, y, z]; neighbours.append(x_neighbour)
            Y_neighbour = [x, y+1, z]; neighbours.append(Y_neighbour)
            y_neighbour = [x, y-1, z]; neighbours.append(y_neighbour)
            Z_neighbour = [x, y, z+1]; neighbours.append(Z_neighbour)
            z_neighbour = [x, y, z-1]; neighbours.append(z_neighbour)
            # +2 neighbours
            xx_neighbour = [x-2, y, z]; neighbours.append(xx_neighbour)
            XX_neighbour = [x+2, y, z]; neighbours.append(XX_neighbour)
            yy_neighbour = [x, y-2, z]; neighbours.append(yy_neighbour)
            YY_neighbour = [x, y+2, z]; neighbours.append(YY_neighbour)
            zz_neighbour = [x, y, z-2]; neighbours.append(zz_neighbour)
            ZZ_neighbour = [x, y, z+2]; neighbours.append(ZZ_neighbour)
            zx_neighbour = [x-1, y, z-1]; neighbours.append(zx_neighbour)
            zX_neighbour = [x+1, y, z-1]; neighbours.append(zX_neighbour)
            zy_neighbour = [x, y-1, z-1]; neighbours.append(zy_neighbour)
            zY_neighbour = [x, y+1, z-1]; neighbours.append(zY_neighbour)
            Zx_neighbour = [x-1, y, z+1]; neighbours.append(Zx_neighbour)
            ZX_neighbour = [x+1, y, z+1]; neighbours.append(ZX_neighbour)
            Zy_neighbour = [x, y-1, z+1]; neighbours.append(Zy_neighbour)
            ZY_neighbour = [x, y+1, z+1]; neighbours.append(ZY_neighbour)
            
            # remove out of bounds neighbours
            neighbours_out = []
            for neighbour in neighbours:
                if -1 in neighbour or self.init_bins in neighbour or -2 in neighbour or self.init_bins + 1 in neighbour:
                    pass
                else:
                    neighbours_out.append(neighbour)
            
            # count number of zero neighbours
            check = 0
            for coords in neighbours_out:
                x_n = coords[0]
                y_n = coords[1]
                z_n = coords[2]
                if self.H[x_n][y_n][z_n] == 0:
                    check += 1
                    
            if check >= len(neighbours_out):
                self.histo_no_halo[x][y][z] = 0
                
    def add_values(self):
        
        H_out = np.zeros((self.H.shape[0], self.H.shape[1], self.H.shape[2]))
        for (x, y, z), value in np.ndenumerate(self.H):
            if self.H[x][y][z] <= self.threshold:
                H_out[x][y][z] = self.H[x][y][z]
            else:
                # search for the smallest value in the 2 ** dimnum value bubbles
                test_values = []
                for i in range(len(self.values)):
                    test_values.append(self.values[i][x][y][z])
                min_pos = test_values.index(min(test_values))
                H_out[x][y][z] = self.values[min_pos][x][y][z]
                if min_pos > self.no_of_search_bubbles:
                    raise Exception('That did not work')
                # add the inverse of all greater values to the smallest value (see Test 1)
                # append_value = self.values[i][x][y][z]
                # for i in range(len(self.values)):
                #     if i != min_pos:
                #         append_value = append_value + ( 1.0 / self.values[i][x][y][z])
                # write value to out array
        return(H_out)
    
    def gen_net_charge(self):
        
        # H_tmp = np.zeros((self.H.shape[0], self.H.shape[1], self.H.shape[2]))
        self.H_net_charge = np.zeros((self.H.shape[0], self.H.shape[1], self.H.shape[2]))
        
        counter = 0
        for x, y, z in self.input_arr:
            try:
                x_b = (x <= self.edges[0]).tolist().index(True) - 1
                # this is an old line and not needed, because x_b can't possibly be 25
                # because edges has only a length of 26 (0..25), and 25 - 1  = 24
                # if x_b == 25: x_b = 24
            except ValueError: # means the x coordinate is outside the histogram boundaries
                if x > self.edges[0][25]: # right outside bounds
                    x_b = 24
                else: # left outside bounds
                    x_b = 0
            try:
                y_b = (y <= self.edges[1]).tolist().index(True) - 1
                # this is an old line and not needed, because y_b can't possibly be 25
                # because edges has only a length of 26 (0..25), and 25 - 1  = 24
                # if y_b == 25: y_b = 24
            except ValueError: # means the y coordinate is outside the histogram boundaries
                if y > self.edges[1][25]: # right outside bounds
                    y_b = 24
                else: # left outside bounds
                    y_b = 0
            try:
                z_b = (z <= self.edges[2]).tolist().index(True) - 1
                # this is an old line and not needed, because z_b can't possibly be 25
                # because edges has only a length of 26 (0..25), and 25 - 1  = 24
                # if z_b == 25: z_b = 24
            except ValueError: # means the z coordinate is outside the histogram boundaries
                if z > self.edges[2][25]: # right outside bounds
                    z_b = 24
                else: # left outside bounds
                    z_b = 0
            self.H_net_charge[x_b][y_b][z_b] = self.H_net_charge[x_b][y_b][z_b] + self.weights[counter]
            counter += 1
            
        for (x, y, z), value in np.ndenumerate(self.H_net_charge):
            if self.H_net_charge[x][y][z] > 0:
                self.H_net_charge[x][y][z] = 1
            else:
                self.H_net_charge[x][y][z] = 0
         
    def get_score(self, input_arr, halo=False):
        
        if len(input_arr[0]) != 3:
            raise ValueError('Currently only supports 3D objects')
        out = 0.0
        values = []
        for x, y, z in input_arr:
            # print('checking point x: '+str(x)+', y: '+str(y)+', z:'+str(z))
            try:
                x_b = (x <= self.edges[0]).tolist().index(True)
                if x_b == 25:
                    x_b = 24
            except ValueError: # means the x coordinate is outside the histogram boundaries
                if x > self.edges[0][25]: # right outside bounds
                    x_b = 24
                else: # left outside bounds
                    x_b = 0
            try:
                y_b = (y <= self.edges[1]).tolist().index(True)
                if y_b == 25:
                    y_b = 24
            except ValueError: # means the y coordinate is outside the histogram boundaries
                if y > self.edges[1][25]: # right outside bounds
                    y_b = 24
                else: # left outside bounds
                    y_b = 0
            try:
                z_b = (z <= self.edges[2]).tolist().index(True)
                if z_b == 25:
                    z_b = 24
            except ValueError: # means the z coordinate is outside the histogram boundaries
                if z > self.edges[2][25]: # right outside bounds
                    z_b = 24
                else: # left outside bounds
                    z_b = 0
            # print('This point is inside bin ('+str(x_b)+', '+str(y_b)+', '+str(z_b)+')')
            # print('The value of this bin is: '+str(self.histo_out[x_b][y_b][z_b]))
                
            if halo:
                values.append(self.histo_no_halo[x_b][y_b][z_b])
                out = out + self.histo_no_halo[x_b][y_b][z_b]
            else:
                values.append(self.histo_out[x_b][y_b][z_b])
                out = out + self.histo_out[x_b][y_b][z_b]
                
            # print('cumulative score currently at '+str(out))
            """This can to be changed so that it divides the total score
            by the number of atoms in non-zero bins. If this would be
            advantageous, needs to be discussed.
            
            """
        no_of_non_zero_bins = len(np.where(np.array(values) > 0.0)[0])
        if no_of_non_zero_bins == 0:
            return(out)
        else:
            return(out)
    
    def draw_histogram(self, net_charge=False):
               
        fig, axes = plt.subplots(nrows=3, ncols=3, sharex=True, sharey=True)

        X, Y = np.meshgrid(self.edges[0], self.edges[1])
        ct = np.linspace(0, self.init_bins - 1, 9, dtype=int)

        counter = 0
        for i in range(3):
            for j in range(3):
                if not net_charge:
                    axes[i,j].pcolormesh(X, Y, self.H[:,:,ct[counter]].T)
                else:
                    axes[i,j].pcolormesh(X, Y, self.H_net_charge[:,:,ct[counter]].T)

                axes[i,j].title.set_text('slice no. '+str(ct[counter]))
                axes[i,j].set_xlabel('x')
                axes[i,j].set_ylabel('y')
                counter += 1

        ticks = np.linspace(0, 1, 6)
        labels = np.linspace(self.H.min(), self.H.max(), 6).astype(int)
        sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis)
        sm.set_array([])
        cbaxes = fig.add_axes([0.91, 0.1, 0.03, 0.8]) 
        cbar1 = fig.colorbar(sm, cax=cbaxes, cmap=plt.cm.viridis, ticks=ticks)
        cbar1.ax.tick_params(length=0)
        cbar1.ax.set_yticklabels(labels)
        cbar1.set_label('Atom count')

        #plt.tight_layout()
        plt.show()
            
    def draw_histogram_3d(self, layers):
        self.layers = layers
        # Setup a 3D figure and plot points as well as a series of slices
        fig = plt.figure()
        ax1 = fig.add_subplot(111, projection='3d')
        ax1 = fig.add_subplot(111, projection='3d')
        ax1.plot(self.input_arr[:,0],self.input_arr[:,1],self.input_arr[:,2],'k.',alpha=0.1)

        # Use one less than bin edges to give rough bin location
        X, Y = np.meshgrid(self.edges[0][:-1],self.edges[1][:-1])

        #Loop over range of slice locations (default histogram uses 10 bins)
        norm = mpl.colors.Normalize(vmin=np.min(self.H), vmax=np.max(self.H))
        for ct in np.linspace(0, self.init_bins - 1, self.layers, dtype=int): 
            cs = ax1.contourf(X,Y,self.H[:,:,ct].T, zdir='z', offset=self.edges[2][ct], level=100, cmap=plt.cm.RdYlBu_r, alpha=0.5, antialiased=False, norm=norm)

        ax1.set_xlim(np.min(self.input_arr[:,0]) - 1, np.max(self.input_arr[:,0]) + 1)
        ax1.set_ylim(np.min(self.input_arr[:,1]) - 1, np.max(self.input_arr[:,1]) + 1)
        ax1.set_zlim(np.min(self.input_arr[:,2]) - 1, np.max(self.input_arr[:,2]) + 1)
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')

        # plt.colorbar(cs)
        plt.show()
        
    def draw_cumulative_histogram(self):
        fig, ax = plt.subplots(3,3, facecolor='w', edgecolor='k')

        ax = ax.ravel()

        X, Y = np.meshgrid(self.edges[0], self.edges[1])

        i = 0
        norm = mpl.colors.Normalize(vmin=np.min(self.histo_out), vmax=np.max(self.histo_out))
        for ct in np.linspace(0, self.init_bins - 1, 9, dtype=int):
            ax[i].pcolormesh(X, Y, self.histo_out[:,:,ct].T, norm=norm)
            ax[i].set_title('slice no. '+str(ct))
            i += 1
            
    def draw_single_bubble(self, number):
        if number >= self.no_of_search_bubbles:
            raise ValueError('Input must be less than '+str(self.no_of_search_bubbles))
            
        fig, axes = plt.subplots(nrows=3, ncols=3, sharex=True, sharey=True)

        X, Y = np.meshgrid(self.edges[0], self.edges[1])
        ct = np.linspace(0, self.init_bins - 1, 9, dtype=int)

        draw = self.values[number]
        
        counter = 0
        for i in range(3):
            for j in range(3):
                axes[i,j].pcolormesh(X, Y, draw[:,:,ct[counter]].T)

                axes[i,j].title.set_text('slice no. '+str(ct[counter]))
                axes[i,j].set_xlabel('X in nm')
                axes[i,j].set_ylabel('Y in nm')
                counter += 1

        ticks = np.linspace(0, 1, 6)
        labels = np.linspace(self.H.min(), self.H.max(), 6).astype(int)
        sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis)
        sm.set_array([])
        cbaxes = fig.add_axes([0.91, 0.1, 0.03, 0.8]) 
        cbar1 = fig.colorbar(sm, cax=cbaxes, cmap=plt.cm.viridis, ticks=ticks)
        cbar1.ax.tick_params(length=0)
        cbar1.ax.set_yticklabels(labels)
        cbar1.set_label('cumulative count')

        #plt.tight_layout()
        plt.show()
            
    def draw_iter_bubble(self, number):
        if number >= self.no_of_search_bubbles:
            raise ValueError('Input must be less than '+str(self.no_of_search_bubbles))
        fig, ax = plt.subplots(3,3, facecolor='w', edgecolor='k')

        ax = ax.ravel()

        X, Y = np.meshgrid(self.edges[0], self.edges[1])

        i = 0
        draw = self.iter_bubbles[number]
        norm = mpl.colors.Normalize(vmin=np.min(draw), vmax=np.max(draw))
        for ct in np.linspace(0, self.init_bins - 1, 9, dtype=int):
            ax[i].pcolormesh(X, Y, draw[:,:,ct], norm=norm)
            ax[i].set_title('slice no. '+str(ct))
            i += 1
            
    def draw_no_halo(self):
        fig, ax = plt.subplots(3,3, facecolor='w', edgecolor='k')

        ax = ax.ravel()

        X, Y = np.meshgrid(self.edges[0], self.edges[1])

        i = 0
        norm = mpl.colors.Normalize(vmin=np.min(self.histo_no_halo), vmax=np.max(self.histo_no_halo))
        for ct in np.linspace(0, self.init_bins - 1, 9, dtype=int):
            ax[i].pcolormesh(X, Y, self.histo_no_halo[:,:,ct], norm=norm)
            ax[i].set_title('slice no. '+str(ct))
            i += 1
        
    def print_debug(self):
        pass
        # print(self.bubbles)
        # print(self.corners)
            
    def get_bin_occupancy(self):
        self.max_val = np.max(self.H)
        self.min_val = np.min(self.H)
        print('Lowest occupied bin: '+str(self.min_val)+'. Highest occupied bin: '+str(self.max_val))
        
    def __repr__(self):
        return 'ISA (interpenetration and scoring algorithm) object with '+str(self.point_no)+' points.'
    
    def __str__(self):
        return 'ISA (interpenetration and scoring algorithm) object with '+str(self.point_no)+' points.'
        
    @classmethod
    def set_init_bins(cls, init_bins):
        cls.init_bins = init_bins
        
    @classmethod
    def set_propagation_factor(cls, propagation_factor):
        cls.propagation_factor = propagation_factor
        
    @classmethod
    def set_threshold(cls, threshold):
        cls.threshold = threshold