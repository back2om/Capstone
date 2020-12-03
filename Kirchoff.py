from prody import * 
from types import MethodType, FunctionType
from prody.utilities import *
import time
import numpy as np
import logging as LOGGER
import argparse
from matplotlib import pyplot as plt
from numpy import linalg as LA

parser = argparse.ArgumentParser()
parser.add_argument('_File', help='PDB file name')
parser.add_argument('sym', help=' C-?, D-?, tetra-4, octa-8, iso-20')
parser.add_argument('chain1', nargs=2, help='chain selected to be mirrored, format: chain A')
parser.add_argument('chain2', nargs='*', help='2nd chain which is in D-?, letter only, format: D')

args = parser.parse_args()

print args._File # 2GGT.pdb
print args.sym 
print args.chain1

S = args.sym.split('-')

def checkENMParameters(cutoff, gamma):
    """Check type and values of *cutoff* and *gamma*."""

    if not isinstance(cutoff, (float, int)):
        raise TypeError('cutoff must be a float or an integer')
    elif cutoff < 4:
        raise ValueError('cutoff must be greater or equal to 4')
    if isinstance(gamma, Gamma):
        gamma_func = gamma.gamma
    elif isinstance(gamma, FunctionType):
        gamma_func = gamma
    else:
        if not isinstance(gamma, (float, int)):
            raise TypeError('gamma must be a float, an integer, derived '
                            'from Gamma, or a function')
        elif gamma <= 0:
            raise ValueError('gamma must be greater than 0')
        gamma = float(gamma)
        gamma_func = lambda dist2, i, j: gamma
    return cutoff, gamma, gamma_func

def BK(self, coords, sym, num, num2 = 0, cutoff=5., gamma=1., **kwargs):

        try:
            coords = (coords._getCoords() if hasattr(coords, '_getCoords') else
                      coords.getCoords())
        except AttributeError:
            try:
                checkCoords(coords)
            except TypeError:
                raise TypeError('coords must be a Numpy array or an object '
                                'with `getCoords` method')

        cutoff, g, gamma = checkENMParameters(cutoff, gamma)
        self._reset()
        self._cutoff = cutoff
        self._gamma = g

        n_atoms = coords.shape[0]
        start = time.time()
        if kwargs.get('sparse', False):
            try:
                from scipy import sparse as scipy_sparse
            except ImportError:
                raise ImportError('failed to import scipy.sparse, which  is '
                                  'required for sparse matrix calculations')
            if num2 != 0:
                kirchhoff = scipy_sparse.lil_matrix((sym*num + sym*num2, sym*num + sym*num2))
            else:
                kirchhoff = scipy_sparse.lil_matrix((sym*num, sym*num))
        else:
            if num2 != 0:
                kirchhoff = np.empty((sym*num + sym*num2, sym*num + sym*num2), 'd')
            else:
                kirchhoff = np.empty((sym*num, sym*num), 'd')

        if kwargs.get('kdtree', True):
            kdtree = KDTree(coords)
            kdtree.search(cutoff)  # gets tree with only distances under 10 A
            dist2 = kdtree.getDistances() ** 2
            r = 0
            for i, j in kdtree.getIndices(): # iterates all j for i, then all j for i+1......
                g = gamma(dist2[r], i, j)  # returns force contant
                kirchhoff[i, j] = -g
                kirchhoff[j, i] = -g
                kirchhoff[i, i] = kirchhoff[i, i] + g
                kirchhoff[j, j] = kirchhoff[j, j] + g
                ii = i
                jj = j
                for x in range(1,11):
                    if num2 != 0:
                        if ii < (sym*num):
                            if ii + num < (sym*num):
                                    ii = ii + num
                            else:
                                ii = (ii+num) - (sym*num)
                        else: 
                            if ii + num2 < (sym*num2):
                                ii = ii + num2
                            else:
                                ii = (ii + num2) - (sym*num2)
                        if jj < (sym*num):
                            if jj + num < (sym*num):                                    
                                jj = jj + num
                            else:
                                jj = (jj+num) - (sym*num)
                        else:
                            if jj + num2 < (sym*num2):
                                jj = jj + num2
                            else:
                                jj = (jj + num2) - (sym*num2)
                    else:
                        if ii + num >= (sym*num) :
                            ii = (ii + num) - (sym*num)
                        else:
                            ii = ii + num
                        if jj + num >= (sym*num) :
                            jj = (jj + num) - (sym*num)  
                        else:
                            jj = jj + num
                    kirchhoff[ii, jj] = -g
                    kirchhoff[jj, ii] = -g
                    kirchhoff[ii, ii] = kirchhoff[ii, ii] + g
                    kirchhoff[jj, jj] = kirchhoff[jj, jj] + g
                r += 1
        else:
            LOGGER.info('Using slower method for building the Kirchhoff.')
            cutoff2 = cutoff * cutoff
            mul = np.multiply
            for i in range(n_atoms):
                xyz_i = coords[i, :]
                i_p1 = i+1
                i2j = coords[i_p1:, :] - xyz_i
                mul(i2j, i2j, i2j)
                for j, dist2 in enumerate(i2j.sum(1)):
                    if dist2 > cutoff2:
                        continue
                    j += i_p1
                    g = gamma(dist2, i, j)
                    kirchhoff[i, j] = -g
                    kirchhoff[j, i] = -g
                    kirchhoff[i, i] = kirchhoff[i, i] + g
                    kirchhoff[j, j] = kirchhoff[j, j] + g
                    ii = i
                    jj = j
                    for x in range(1,11):
                        if num2 != 0:
                            if ii < (sym*num):
                                if ii + num < (sym*num):
                                    ii = ii + num
                                else:
                                    ii = (ii+num) - (sym*num)
                            else: 
                                if ii + num2 < (sym*num2):
                                    ii = ii + num2
                                else:
                                    ii = (ii + num2) - (sym*num2)
                            if jj < (sym*num):
                                if jj + num < (sym*num):
                                    jj = jj + num
                                else:
                                    jj = (jj+num) - (sym*num)
                            else:
                                if jj + num2 < (sym*num2):
                                    jj = jj + num2
                                else:
                                    jj = (jj + num2) - (sym*num2)
                        else:
                            if ii + num >= (sym*num):
                                ii = (ii + num) - (sym*num) 
                            else:
                                ii = ii + num
                            if jj + num >= (sym*num):
                                jj = (jj + num) - (sym*num)
                            else:
                                jj = jj + num
                        kirchhoff[ii, jj] = -g
                        kirchhoff[jj, ii] = -g
                        kirchhoff[ii, ii] = kirchhoff[ii, ii] + g
                        kirchhoff[jj, jj] = kirchhoff[jj, jj] + g

        LOGGER.debug('Kirchhoff was built in {0:.2f}s.'
                     .format(time.time()-start))
        self._kirchhoff = kirchhoff
        self._n_atoms = n_atoms
        self._dof = n_atoms

def mergeSort(alist, blist):
    if len(alist)>1:
        mid = len(alist)//2

        lefthalf = alist[:mid]
        l = blist[:mid]
        righthalf = alist[mid:]
        r = blist[mid:]

        mergeSort(lefthalf, l)
        mergeSort(righthalf, r)

        i=0
        j=0
        k=0
        while i<len(lefthalf) and j<len(righthalf):
            if lefthalf[i]<righthalf[j]:
                alist[k]=lefthalf[i]
                blist[k]=l[i]
                i=i+1
            else:
                alist[k]=righthalf[j]
                blist[k]=r[j]
                j=j+1
            k=k+1

        while i<len(lefthalf):
            alist[k]=lefthalf[i]
            blist[k]=l[i]
            i=i+1
            k=k+1

        while j<len(righthalf):
            alist[k]=righthalf[j]
            blist[k]=r[j]
            j=j+1
            k=k+1
    return alist, blist

GNM.BK = MethodType(BK, None, GNM)
p = parsePDB(args._File)



if S[0] == "D":
    a = p.select(args.chain1[0] + " " + args.chain1[1] + ' or ' + args.chain1[0] + " " + args.chain2[0])
    b = p.select('within 5 of ' + args.chain1[0] + " " + args.chain1[1] + ' or ' + args.chain1[0] + " " + args.chain2[0])
    c = p.select('within 10 of ' + args.chain1[0] + " " + args.chain1[1] + ' or ' + args.chain1[0] + " " + args.chain2[0])
    
    num1 = p.select(args.chain1[0] + " " + args.chain1[1]).select('calpha').numAtoms()
    N2 = p.select(args.chain1[0] + " " + args.chain2[0]).select('calpha').numAtoms()

elif S[0] == "C":
    a = p.select(args.chain1[0] + " " + args.chain1[1])
    b = p.select('within 5 of ' + args.chain1[0] + " " + args.chain1[1])
    c = p.select('within 10 of ' + args.chain1[0] + " " + args.chain1[1])
   
    num = p.select(args.chain1[0] + " " + args.chain1[1]).numAtoms()
else:                                                               # for proteins with multiple chainse for sym
    strr = args.chain1[0] + " " + args.chain1[1]
    n = args.chain1[0] + " " + args.chain1[1]
    for x in range(0, len(args.chain2)):
        strr = strr + ' or ' + args.chain1[0] + " " + args.chain2[x]
        n = n + ' or ' + args.chain1[0] + " " + args.chain2[x]
    print strr
    print n
    a = p.select(strr)
    b = p.select('within 5 of ' + strr)
    c = p.select('within 10 of ' + strr)
    num = p.select(n).numAtoms() 



a = a.select('calpha')
b = b.select('calpha')
c = c.select('calpha')
num = a.numAtoms()

sym = int(S[1])
t = S[0]

indiA = a.getIndices()
indiB = b.getIndices()
indiC = c.getIndices()

difAC = 0
while indiA[0] != indiC[difAC]:
    difAC = difAC + 1

difBC = 0
while indiB[0] != indiC[difBC]:
    difBC = difBC + 1


gnm = GNM(a)                      # creates an instance of the network
if S[0] == "D":
    gnm.BK(a, sym, num1, num2 = N2) 
else:
    gnm.BK(a, sym, num)               # (data, sym)
k = gnm.getKirchhoff()            #k[k==0] = np.nan

gnm = GNM(b) 
if S[0] == "D":
    gnm.BK(b, sym, num1, num2 = N2) 
else:                     
    gnm.BK(b, sym, num)               
k1 = gnm.getKirchhoff()

gnm = GNM(c) 
if S[0] == "D":
    gnm.BK(c, sym, num1, num2 = N2) 
else:                     
    gnm.BK(c, sym, num)               
k2 = gnm.getKirchhoff()

plt.imshow(k)
plt.colorbar()
plt.show() 

plt.imshow(k1)
plt.colorbar()
plt.show() 

plt.imshow(k2)
plt.colorbar()
plt.show() 


w, v = LA.eig(k)     # w is array, v is matrix

w_sort = np.sort(w) # sorts array of eigenvalues
w_sort_indicies = np.argsort(w) # creats array for sorted indicies of original array w


v1 = v[:,w_sort_indicies[-1]]
print v1
v2 = v[:,w_sort_indicies[-2]]
v3 = v[:,w_sort_indicies[-3]]
v4 = v[:,w_sort_indicies[-4]]
v5 = v[:,w_sort_indicies[-5]]
v6 = v[:,w_sort_indicies[-6]]
v7 = v[:,w_sort_indicies[-7]]
v8 = v[:,w_sort_indicies[-8]]


arrA = np.zeros(len(v1))
if difAC != 0:
    for x in range(0,len(v1)):
        arrA[x]= x+difAC
else:
    for x in range(0,len(v1)):
        arrA[x]= x


f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex='col', sharey='row')
ax1.plot(arrA, v1, 'g')
ax1.set_title('0 Angstrom Radius, Plots 1-4')
ax2.plot(arrA, v2, 'g')
ax3.plot(arrA, v3, 'g')
ax4.plot(arrA, v4, 'g')
plt.show()
f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex='col', sharey='row')
ax1.plot(arrA, v5, 'g')
ax1.set_title('0 Angstrom Radius, Plots 5-8')
ax2.plot(arrA, v6, 'g')
ax3.plot(arrA, v7, 'g')
ax4.plot(arrA, v8, 'g')
plt.show()



wb, vb = LA.eig(k1)     # w is array, v is matrix

wB_sort = np.sort(wb) # sorts array of eigenvalues
wB_sort_indicies = np.argsort(wb) # creats array for sorted indicies of original array w

vb1 = v[:,wB_sort_indicies[-1]]
vb2 = v[:,wB_sort_indicies[-2]]
vb3 = v[:,wB_sort_indicies[-3]]
vb4 = v[:,wB_sort_indicies[-4]]
vb5 = v[:,wB_sort_indicies[-5]]
vb6 = v[:,wB_sort_indicies[-6]]
vb7 = v[:,wB_sort_indicies[-7]]
vb8 = v[:,wB_sort_indicies[-8]]

arrB = np.zeros(len(vb1))
if difBC != 0:
    for x in range(0,len(vb1)):
        arrB[x]= x+difBC
else:
    for x in range(0,len(vb1)):
        arrB[x]= x

f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex='col', sharey='row')
ax1.plot(arrB, vb1, 'r')
ax1.set_title('5 Angstrom Radius, Plots 1-4')
ax2.plot(arrB, vb2, 'r')
ax3.plot(arrB, vb3, 'r')
ax4.plot(arrB, vb4, 'r')
plt.show()
f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex='col', sharey='row')
ax1.plot(arrB, vb5, 'r')
ax1.set_title('5 Angstrom Radius, Plots 5-8')
ax2.plot(arrB, vb6, 'r')
ax3.plot(arrB, vb7, 'r')
ax4.plot(arrB, vb8, 'r')
plt.show()



wc, vc = LA.eig(k2)     # w is array, v is matrix

wC_sort = np.sort(wc) # sorts array of eigenvalues
wC_sort_indicies = np.argsort(wc) # creats array for sorted indicies of original array w

vc1 = v[:,wC_sort_indicies[-1]]
vc2 = v[:,wC_sort_indicies[-2]]
vc3 = v[:,wC_sort_indicies[-3]]
vc4 = v[:,wC_sort_indicies[-4]]
vc5 = v[:,wC_sort_indicies[-5]]
vc6 = v[:,wC_sort_indicies[-6]]
vc7 = v[:,wC_sort_indicies[-7]]
vc8 = v[:,wC_sort_indicies[-8]]

arrC= np.zeros(len(vc1))
for x in range(0,len(vc1)):
        arrC[x]= x 

f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex='col', sharey='row')
ax1.plot(arrC, vc1, 'b')
ax1.set_title('10 Angstrom Radius, Plots 1-4')
ax2.plot(arrC, vc2, 'b')
ax3.plot(arrC, vc3, 'b')
ax4.plot(arrC, vc4, 'b')
plt.show()
f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex='col', sharey='row')
ax1.plot(arrC, vc5, 'b')
ax1.set_title('10 Angstrom Radius, Plots 5-8')
ax2.plot(arrC, vc6, 'b')
ax3.plot(arrC, vc7, 'b')
ax4.plot(arrC, vc8, 'b')
plt.show()
