"""
Script for fast image reconstruction from gradients.
Based on Ramesh Raskar's Matlab script, available here:
http://web.media.mit.edu/~raskar/photo/code.pdf

Adapted slightly for doing "mixed" Poisson Image Editing [Perez et al.]
Paper: http://www.cs.jhu.edu/~misha/Fall07/Papers/Perez03.pdf
"""
from __future__ import division
import numpy as np 
import scipy.fftpack
import scipy.ndimage
import cv2
import matplotlib.pyplot as plt 
#sns.set(style="darkgrid")


def DST(x):
    """
    Converts Scipy's DST output to Matlab's DST (scaling).
    """
    X = scipy.fftpack.dst(x,type=1,axis=0)
    return X/2.0

def IDST(X):
    """
    Inverse DST. Python -> Matlab
    """
    n = X.shape[0]
    x = np.real(scipy.fftpack.idst(X,type=1,axis=0))
    return x/(n+1.0)

def get_grads(im):
    """
    return the x and y gradients.
    """
    [H,W] = im.shape
    Dx,Dy = np.zeros((H,W),'float32'), np.zeros((H,W),'float32')
    j,k = np.atleast_2d(np.arange(0,H-1)).T, np.arange(0,W-1)
    Dx[j,k] = im[j,k+1] - im[j,k]
    Dy[j,k] = im[j+1,k] - im[j,k]
    return Dx,Dy

def get_laplacian(Dx,Dy):
    """
    return the laplacian
    """
    [H,W] = Dx.shape
    Dxx, Dyy = np.zeros((H,W)), np.zeros((H,W))
    j,k = np.atleast_2d(np.arange(0,H-1)).T, np.arange(0,W-1)
    Dxx[j,k+1] = Dx[j,k+1] - Dx[j,k] 
    Dyy[j+1,k] = Dy[j+1,k] - Dy[j,k]
    return Dxx+Dyy

def poisson_solve(gx,gy,bnd):
    # convert to double:
    gx = gx.astype('float32')
    gy = gy.astype('float32')
    bnd = bnd.astype('float32')
 
    H,W = bnd.shape
    L = get_laplacian(gx,gy)

    # set the interior of the boundary-image to 0:
    bnd[1:-1,1:-1] = 0
    # get the boundary laplacian:
    L_bp = np.zeros_like(L)
    L_bp[1:-1,1:-1] = -4*bnd[1:-1,1:-1] \
                      + bnd[1:-1,2:] + bnd[1:-1,0:-2] \
                      + bnd[2:,1:-1] + bnd[0:-2,1:-1] # delta-x
    L = L - L_bp
    L = L[1:-1,1:-1]

    # compute the 2D DST:
    L_dst = DST(DST(L).T).T #first along columns, then along rows

    # normalize:
    [xx,yy] = np.meshgrid(np.arange(1,W-1),np.arange(1,H-1))
    D = (2*np.cos(np.pi*xx/(W-1))-2) + (2*np.cos(np.pi*yy/(H-1))-2)
    L_dst = L_dst/D

    img_interior = IDST(IDST(L_dst).T).T # inverse DST for rows and columns

    img = bnd.copy()

    img[1:-1,1:-1] = img_interior

    return img

def blit_images(im_top,im_back,scale_grad=1.0,mode='max'):
    """
    combine images using poission editing.
    IM_TOP and IM_BACK should be of the same size.
    """
    assert np.all(im_top.shape==im_back.shape)

    im_top = im_top.copy().astype('float32')
    im_back = im_back.copy().astype('float32')
    im_res = np.zeros_like(im_top)

    # frac of gradients which come from source:
    for ch in range(im_top.shape[2]):
        ims = im_top[:,:,ch]
        imd = im_back[:,:,ch]

        [gxs,gys] = get_grads(ims)
        [gxd,gyd] = get_grads(imd)

        gxs *= scale_grad
        gys *= scale_grad

        gxs_idx = gxs!=0
        gys_idx = gys!=0
        # mix the source and target gradients:
        if mode=='max':
            gx = gxs.copy()
            gxm = (np.abs(gxd))>np.abs(gxs)
            gx[gxm] = gxd[gxm]

            gy = gys.copy()
            gym = np.abs(gyd)>np.abs(gys)
            gy[gym] = gyd[gym]

            # get gradient mixture statistics:
            f_gx = np.sum((gx[gxs_idx]==gxs[gxs_idx]).flat) / (np.sum(gxs_idx.flat)+1e-6)
            f_gy = np.sum((gy[gys_idx]==gys[gys_idx]).flat) / (np.sum(gys_idx.flat)+1e-6)
            if min(f_gx, f_gy) <= 0.35:
                m = 'max'
                if scale_grad > 1:
                    m = 'blend'
                return blit_images(im_top, im_back, scale_grad=1.5, mode=m)

        elif mode=='src':
            gx,gy = gxd.copy(), gyd.copy()
            gx[gxs_idx] = gxs[gxs_idx]
            gy[gys_idx] = gys[gys_idx]

        elif mode=='blend': # from recursive call:
            # just do an alpha blend
            gx = gxs+gxd
            gy = gys+gyd

        im_res[:,:,ch] = np.clip(poisson_solve(gx,gy,imd),0,255)

    return im_res.astype('uint8')


def contiguous_regions(mask):
    """
    return a list of (ind0, ind1) such that mask[ind0:ind1].all() is
    True and we cover all such regions
    """
    in_region = None
    boundaries = []
    for i, val in enumerate(mask):
        if in_region is None and val:
            in_region = i
        elif in_region is not None and not val:
            boundaries.append((in_region, i))
            in_region = None

    if in_region is not None:
        boundaries.append((in_region, i+1))
    return boundaries


if __name__=='__main__':
    """
    example usage:
    """
    import seaborn as sns

    im_src = cv2.imread('i2.jpg').astype('float32')

    im_dst = cv2.imread('gg.jpg').astype('float32')

    mu = np.mean(np.reshape(im_src,[im_src.shape[0]*im_src.shape[1],3]),axis=0)
    # print mu
    sz = (700,700)
    im_src = cv2.resize(im_src,sz)
    im_dst = cv2.resize(im_dst,sz)
    
    im0 = im_dst[:,:,0] > 100
    im_dst[im0,:] = im_src[im0,:]
    im_dst[~im0,:] = 50
    im_dst = cv2.GaussianBlur(im_dst,(5,5),5)
    
    im_alpha = 0.8*im_dst + 0.2*im_src

    # plt.imshow(im_dst)
    # plt.show()

    im_res = blit_images(im_src,im_dst)

    import scipy
    scipy.misc.imsave('orig.png',im_src[:,:,::-1].astype('uint8'))
    scipy.misc.imsave('alpha.png',im_alpha[:,:,::-1].astype('uint8'))
    scipy.misc.imsave('poisson.png',im_res[:,:,::-1].astype('uint8'))

    im_actual_L = cv2.cvtColor(im_src.astype('uint8'),cv2.cv.CV_BGR2Lab)[:,:,0]
    im_alpha_L = cv2.cvtColor(im_alpha.astype('uint8'),cv2.cv.CV_BGR2Lab)[:,:,0]
    im_poisson_L = cv2.cvtColor(im_res.astype('uint8'),cv2.cv.CV_BGR2Lab)[:,:,0]

    # plt.imshow(im_alpha_L)
    # plt.show()
    for i in range(500,im_alpha_L.shape[1],5):
        l_actual = im_actual_L[i,:]#-im_actual_L[i,:-1]
        l_alpha = im_alpha_L[i,:]#-im_alpha_L[i,:-1]
        l_poisson = im_poisson_L[i,:]#-im_poisson_L[i,:-1]


        with sns.axes_style("darkgrid"):
            plt.subplot(2,1,2)
            plt.plot(l_alpha,label='alpha')
            plt.plot(l_poisson,label='poisson')
            plt.plot(l_actual,label='actual')
            plt.legend()

            # find "text regions":
            is_txt = ~im0[i,:]
            t_loc = contiguous_regions(is_txt)
            ax = plt.gca()
            for b0,b1 in t_loc:
                ax.axvspan(b0, b1, facecolor='red', alpha=0.1)
        
        with sns.axes_style("white"):
            plt.subplot(2,1,1)
            plt.imshow(im_alpha[:,:,::-1].astype('uint8'))
            plt.plot([0,im_alpha_L.shape[0]-1],[i,i],'r')
            plt.axis('image')
            plt.show()


    plt.subplot(1,3,1)
    plt.imshow(im_src[:,:,::-1].astype('uint8'))
    plt.subplot(1,3,2)
    plt.imshow(im_alpha[:,:,::-1].astype('uint8'))
    plt.subplot(1,3,3)    
    plt.imshow(im_res[:,:,::-1]) #cv2 reads in BGR
    plt.show()


