from __future__ import division
import random
import numpy as np


def fit_plane(xyz,z_pos=None):
    """
    if z_pos is not None, the sign
    of the normal is flipped to make 
    the dot product with z_pos (+).
    """
    mean = np.mean(xyz,axis=0)
    xyz_c = xyz - mean[None,:]
    l,v = np.linalg.eig(xyz_c.T.dot(xyz_c))
    abc = v[:,np.argmin(l)]
    d = -np.sum(abc*mean)
    # unit-norm the plane-normal:
    abcd =  np.r_[abc,d]/np.linalg.norm(abc)
    # flip the normal direction:
    if z_pos is not None:
        if np.sum(abcd[:3]*z_pos) < 0.0:
            abcd *= -1
    return abcd

def fit_plane_ransac(pts, neighbors=None,z_pos=None, dist_inlier=0.05, 
                     min_inlier_frac=0.60, nsample=3, max_iter=100):
    """
    Fits a 3D plane model using RANSAC. 
    pts : (nx3 array) of point coordinates   
    """
    n,_ = pts.shape
    ninlier,models = [],[]
    for i in range(max_iter):
        if neighbors is None:
            p = pts[np.random.choice(pts.shape[0],nsample,replace=False),:]
        else:
            p = pts[neighbors[:,i],:]
        m = fit_plane(p,z_pos)
        ds = np.abs(pts.dot(m[:3])+m[3])
        nin = np.sum(ds < dist_inlier)
        if nin/pts.shape[0] >= min_inlier_frac:
            ninlier.append(nin)
            models.append(m)

    if models == []:
        print ("RANSAC plane fitting failed!")
        return #None
    else: #refit the model to inliers:
        ninlier = np.array(ninlier)
        best_model_idx = np.argsort(-ninlier)
        n_refit, m_refit, inliers = [],[],[]
        for idx in best_model_idx[:min(10,len(best_model_idx))]:
            # re-estimate the model based on inliers:
            dists = np.abs(pts.dot(models[idx][:3])+models[idx][3])
            inlier = dists < dist_inlier
            m = fit_plane(pts[inlier,:],z_pos)
            # compute new inliers:
            d = np.abs(pts.dot(m[:3])+m[3])
            inlier = d < dist_inlier/2 # heuristic
            n_refit.append(np.sum(inlier))
            m_refit.append(m)
            inliers.append(inlier)
        best_plane = np.argmax(n_refit)
        return m_refit[best_plane],inliers[best_plane]




if __name__ == '__main__':
    from matplotlib import pylab
    from mpl_toolkits import mplot3d
    fig = pylab.figure()
    ax = mplot3d.Axes3D(fig)
    
    def plot_plane(a, b, c, d):
        xx, yy = np.mgrid[10:20, 10:20]
        return xx, yy, (-d - a * xx - b * yy) / c
    
    n = 100
    max_iterations = 100
    goal_inliers = n * 0.3
    
    # test data
    xyzs = np.random.random((n, 3)) * 10 + 10
    xyzs[:90, 2:] = xyzs[:90, :1]
    
    ax.scatter3D(xyzs.T[0], xyzs.T[1], xyzs.T[2])
    
    # RANSAC
    m, b = run_ransac(xyzs, estimate, lambda x, y: is_inlier(x, y, 0.01), 3, goal_inliers, max_iterations)
    a, b, c, d = m
    xx, yy, zz = plot_plane(a, b, c, d)
    ax.plot_surface(xx, yy, zz, color=(0, 1, 0, 0.5))
    plt.show()
