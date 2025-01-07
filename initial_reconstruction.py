from mpl_toolkits.mplot3d import Axes3D
from scipy import optimize
from scipy import linalg  # Important: on utilise scipy.linalg pour logm et expm

import matplotlib.pyplot as plt
import numpy as np
import cv2
import scipy.linalg as scAlg
import csv
import scipy as sc
import scipy.optimize as scOptim
import scipy.io as sio
from scipy.spatial.transform import Rotation as R
from scipy.optimize import least_squares
from configs import files_path as fp

def P(T,K):
    canonical_mat= np.identity(3)
    canonical_mat= np.hstack((canonical_mat,np.array([0,0,0]).reshape(3,1)))
    return K@canonical_mat@T

def T_w_c2(E, K1, K2, T_C1_W, x1, x2):
    '''Inputs:
       E: Essential matrix
       K1: Camera calibration matrix of camera 1
       K2: Camera calibration matrix of camera 2
       T_C1_W: Transformation matrix from camera 1 to world frame
       x1: points in camera 1
       x2: points in camera 2

       Returns: point in 3D format array [4,n] and the pose of the camera 2 in the 
       world frame'''
    Best_vote = 0
    u, s, vh = np.linalg.svd(E)    
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    
    # Get possible rotations
    R_90 = u @ W @ vh  
    if np.linalg.det(R_90) < 0:
        R_90 = -R_90
        
    R_n90 = u @ W.T @ vh
    if np.linalg.det(R_n90) < 0:    
        R_n90 = -R_n90
      
    # Get translations 
    t = u[:, -1]
    t = t / np.linalg.norm(t)  # Normalize translation
    A = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])  
    P1 = K1 @ A @ T_C1_W 
    
    # Choose the correct answer and calculate P1 and P2
    for i in range(4):
        Pose = np.eye(4, 4)  
        if i % 2 == 0:
            t_current = -t  # Flip translation
        else:
            t_current = t
            
        if i > 1:
            Rot = R_n90  # Use alternative rotation
        else:
            Rot = R_90
            
        Pose[:3, :3] = Rot
        Pose[:3, 3] = t_current                            
        Pose = Pose @ T_C1_W
        P2 = K2 @ A @ Pose  
        
        # Triangulate points
        p = np.empty([x1.shape[1], 4])        
        for k in range(x1.shape[1]):
            p[k, :] = triangulation(P1, P2, x1[:, k], x2[:, k])
        p = p.T 
        p = T_C1_W @ p
        p = normalize3Dpoints(p)        
        Pose = T_C1_W @ Pose
        
        # Count votes for points in front of both cameras
        vote = 0
        for k in range(x1.shape[1]):   
            if p[2, k] > 0 and Pose[2, 3] > 0:  # Ensure points are in front of both cameras
                vote += 1
                
        if vote > Best_vote:                 
            Best_vote = vote
            Bestp = np.linalg.inv(T_C1_W) @ p
            BestPose = np.linalg.inv(T_C1_W) @ Pose
            BestPose = np.linalg.inv(BestPose)

        # print(f"Pose {i}: Votes = {vote}")

    return Bestp, BestPose



# Usage:
# Replace calls to T_w_c2 with:
# pts3D, T_wc2 = T_w_c2_opencv(E, K1, K2, T_C1_W, x1Data, x2Data)

def normalize3Dpoints(A):
    for i in range(4):
        A[i] = A[i]/A[3]
        
    return A

def triangulation(P1,P2,x1,x2):
    Ec = np.empty([4,4])
    for i in range(4):
        Ec[i][0] = P1[2][i]*x1[0]-P1[0][i]
        Ec[i][1] = P1[2][i]*x1[1]-P1[1][i]
        Ec[i][2] = P2[2][i]*x2[0]-P2[0][i]
        Ec[i][3] = P2[2][i]*x2[1]-P2[1][i]
    u, s, vh = np.linalg.svd(Ec.T)
    p = vh[-1, :]   
    p = normalize3Dpoints(p)     
    return p

def resLineFitting(Op, xData):
    """
    Residual function for least squares method.
    
    - Input:
      Op: vector containing the model parameters to optimize (in this case the line).
      xData: list of (x, y) measurements for which we want to calculate residuals.
    
    - Output:
      res: vector of residuals to compute the loss.
    """
    res = []
    a, b = Op
    
    for x, y in xData:
        res.append(y - (a * x + b))
    
    return np.array(res)

 
# Ensamble T matrix
def ensamble_T(R_w_c, t_w_c) -> np.array:
    """
    Ensamble the a SE(3) matrix with the rotation matrix and translation vector.
    """
    T_w_c = np.zeros((4, 4))
    T_w_c[0:3, 0:3] = R_w_c
    T_w_c[0:3, 3] = t_w_c.flatten()
    T_w_c[3, 3] = 1
    return T_w_c


def Parametrice_Pose(T):
    R_mat = T[:3, :3]
    t = T[:3, 3]
    rot = R.from_matrix(R_mat)
    theta_rot = rot.as_euler('zyx')
    norm_t = np.linalg.norm(t)
    if norm_t > 0:
        theta_tras = np.arccos(t[2] / norm_t)
        phi_tras = np.arctan2(t[1], t[0])
    else:
        theta_tras = 0.0
        phi_tras = 0.0
    return theta_rot, np.array([theta_tras, phi_tras])

def ObtainPose(theta_rot, theta_tras, phi_tras):
    rot = R.from_euler('zyx', theta_rot)
    R_mat = rot.as_matrix()
    t = np.array([
        np.sin(theta_tras) * np.cos(phi_tras),
        np.sin(theta_tras) * np.sin(phi_tras),
        np.cos(theta_tras)
    ])
    T = np.eye(4)
    T[:3, :3] = R_mat
    T[:3, 3] = t
    return T

def matchesListToIndexMatrix(dMatchesList):
    """
    Convert a list of DMatch OpenCv matches into a numpy matrix of index.

     -input:
         dMatchesList: list of n DMatch object
     -output:
        matchesList: nMatches x 3 --> [[indexDesc1,indexDesc2,descriptorDistance],...]]
     """
    matchesList = []
    for k in range(len(dMatchesList)):
        matchesList.append([np.int(dMatchesList[k].queryIdx), np.int(dMatchesList[k].trainIdx), dMatchesList[k].distance])
    return matchesList

def plotResidual(x,xProjected,strStyle):
    """
        Plot the residual between an image point and an estimation based on a projection model.
         -input:
             x: Image points.
             xProjected: Projected points.
             strStyle: Line style.
         -output: None
         """

    for k in range(x.shape[1]):
        plt.plot([x[0, k], xProjected[0, k]], [x[1, k], xProjected[1, k]], strStyle)

def plotNumberedImagePoints(x,strColor,offset):
    """
        Plot indexes of points on a 2D image.
         -input:
             x: Points coordinates.
             strColor: Color of the text.
             offset: Offset from the point to the text.
         -output: None
         """
    for k in range(x.shape[1]):
        plt.text(x[0, k]+offset, x[1, k]+offset, str(k), color=strColor)

def plotNumbered3DPoints(ax, X,strColor, offset):
    """
        Plot indexes of points on a 3D plot.
         -input:
             ax: axis handle
             X: Points coordinates.
             strColor: Color of the text.
             offset: Offset from the point to the text.
         -output: None
         """
    for k in range(X.shape[1]):
        ax.text(X[0, k]+offset, X[1, k]+offset, X[2,k]+offset, str(k), color=strColor)

def draw3DLine(ax, xIni, xEnd, strStyle, lColor, lWidth):
    """
    Draw a segment in a 3D plot
    -input:
        ax: axis handle
        xIni: Initial 3D point.
        xEnd: Final 3D point.
        strStyle: Line style.
        lColor: Line color.
        lWidth: Line width.
    """
    ax.plot([np.squeeze(xIni[0]), np.squeeze(xEnd[0])], [np.squeeze(xIni[1]), np.squeeze(xEnd[1])], [np.squeeze(xIni[2]), np.squeeze(xEnd[2])],
            strStyle, color=lColor, linewidth=lWidth)

def drawRefSystem(ax, T_w_c, strStyle, nameStr):
    """
        Draw a reference system in a 3D plot: Red for X axis, Green for Y axis, and Blue for Z axis
    -input:
        ax: axis handle
        T_w_c: (4x4 matrix) Reference system C seen from W.
        strStyle: lines style.
        nameStr: Name of the reference system.
    """
    draw3DLine(ax, T_w_c[0:3, 3:4], T_w_c[0:3, 3:4] + T_w_c[0:3, 0:1], strStyle, 'r', 1)
    draw3DLine(ax, T_w_c[0:3, 3:4], T_w_c[0:3, 3:4] + T_w_c[0:3, 1:2], strStyle, 'g', 1)
    draw3DLine(ax, T_w_c[0:3, 3:4], T_w_c[0:3, 3:4] + T_w_c[0:3, 2:3], strStyle, 'b', 1)
    ax.text(np.squeeze( T_w_c[0, 3]+0.1), np.squeeze( T_w_c[1, 3]+0.1), np.squeeze( T_w_c[2, 3]+0.1), nameStr)



def visualize_residuals(image, observed_points, projected_points, title="Résidus de projection"):
    plt.figure()
    plt.imshow(image, cmap='gray')
    plt.scatter(observed_points[0, :], observed_points[1, :], color='red', label="Points observés", marker='x')
    plt.scatter(projected_points[0, :], projected_points[1, :], color='blue', label="Points projetés", marker='o')
    for obs, proj in zip(observed_points.T, projected_points.T):
        plt.plot([obs[0], proj[0]], [obs[1], proj[1]], 'k--')  # Ligne pointillée noire pour le résidu

    # plotting
    plt.legend()
    plt.xlabel("u (pixels)")
    plt.ylabel("v (pixels)")
    plt.title(title)
    plt.show()



def unpackParameters(Op, n_points):
    rot = Op[0:3]
    theta_tras = Op[3]
    phi_tras = Op[4]
    X = Op[5:].reshape(3, n_points)
    return rot, theta_tras, phi_tras, X

def unpackParameters3cameras(Op, n_points):
    rot = Op[0:3]
    theta_tras = Op[3]
    phi_tras = Op[4]
    X = Op[5:5+3*n_points].reshape(3, n_points)
    rot2 = Op[ -5:-2]
    theta_tras2 = Op[-2]
    phi_tras2 = Op[-1]
    return rot, theta_tras, phi_tras, rot2, theta_tras2, phi_tras2, X

def crossMatrixInv(M):
    x = [M[2, 1], M[0, 2], M[1, 0]]
    return x

def resBundleProjection(Op, x1Data, x2Data, K_c, n_points):
    """
    Compute residuals between observed and projected 2D points in two views.
    """
    # Unpack parameters
    rot = Op[0:3]
    theta_tras = Op[3]
    phi_tras = Op[4]
    X = Op[5:].reshape(3, n_points)
    
    # Compute relative pose
    T_c1_c2 = ObtainPose(rot, theta_tras, phi_tras)
    T_wc2 = T_wc1 @ T_c1_c2  # Transform to world frame
    
    # Project points in both views
    X_h = np.vstack([X, np.ones((1, n_points))])  # Homogeneous coordinates
    x1_proj = K_c @ np.eye(3, 4) @ np.linalg.inv(T_wc1) @ X_h
    x1_proj /= x1_proj[2, :]  # Normalize
    
    x2_proj = K_c @ np.eye(3, 4) @ np.linalg.inv(T_wc2) @ X_h
    x2_proj /= x2_proj[2, :]  # Normalize
    
    # Compute residuals
    res = np.hstack([
        (x1Data[0, :] - x1_proj[0, :]),  # x residuals (view 1)
        (x1Data[1, :] - x1_proj[1, :]),  # y residuals (view 1)
        (x2Data[0, :] - x2_proj[0, :]),  # x residuals (view 2)
        (x2Data[1, :] - x2_proj[1, :])   # y residuals (view 2)
    ])
    
    return res


def resBundleProjection3cameras(Op, x1Data, x2Data, x3Data, K_c, n_points):
    """
    Compute residuals between observed and projected 2D points in three views.

    Parameters
    ----------
    Op : Array of parameters to optimize, including rotation, translation angles, 
        and 3D coordinates of points.
    x1Data : Observed 2D points in the first view, shape (2, n_points).
    x2Data : Observed 2D points in the second view, shape (2, n_points).
    x3Data : Observed 2D points in the  third  view, shape (2, n_points).

    K_c :  Intrinsic camera matrix, shape (3, 3).
    n_points : Number of 3D points.

    Returns
    -------
    res : Residuals for all points in both views, shape (4 * n_points,).
    """
    rot, theta_tras, phi_tras, rot2, theta_tras2, phi_tras2, X = unpackParameters3cameras(Op, n_points)
    T = np.eye(4)
    T_c1c = ObtainPose(rot,theta_tras,phi_tras)
    T_c2c = ObtainPose(rot2,theta_tras2,phi_tras2)
    res1 = []
    res2 = []
    res3= []
    
    X_h = np.vstack([X, np.ones((1, n_points))])
    x1_proj = K_c @ np.eye(3, 4) @ np.linalg.inv(T) @ X_h
    x1_proj /= x1_proj[2, :]
    
    x2_proj =  K_c @ np.eye(3, 4) @ np.linalg.inv(T_c1c) @ X_h
    x2_proj = x2_proj / x2_proj[2]
    
    x3_proj =  K_c @ np.eye(3, 4) @ np.linalg.inv(T_c2c) @ X_h
    x3_proj = x3_proj / x3_proj[2]

    for i in range(n_points):
        #View 1
        res1.append(x1Data[0,i] - x1_proj[0, i] )    # x
        res1.append(x1Data[1,i] - x1_proj[1, i])   # y
        # View 2
        res2.append(x2Data[0,i] - x2_proj[0,i]  ) # x
        res2.append(x2Data[1,i]- x2_proj[1,i]  ) # y
        
        #View3
        res3.append(x3Data[0,i] - x3_proj[0,i]  ) # x
        res3.append(x3Data[1,i]- x3_proj[1,i]  ) # y
        
        
    res = np.hstack([res1,res2, res3])
    res = np.array(res).flatten()
    return res

def crossMatrix(x):
    x = np.asarray(x).flatten() 
    return np.array([[0, -x[2], x[1]],
                    [x[2], 0, -x[0]],
                    [-x[1], x[0], 0]])


def filter_by_residuals(x1, x2, x3, residuals, residual_threshold=10):
    valid_indices = residuals < residual_threshold
    return x1[:, valid_indices], x2[:, valid_indices], x3[:, valid_indices]

def compute_projection_residuals(K_c, T_wc, X_3D, x_observed):

    x_projected = K_c @ np.eye(3, 4) @ np.linalg.inv(T_wc) @ X_3D
    x_projected /= x_projected[2, :]  # Normaliser les coordonnées
    residuals = np.linalg.norm(x_observed - x_projected[:2, :], axis=0)
    return residuals

def test_scales(T_wc1, T_c1_c2, x1Data, x2Data, K_c, scales):
    best_scale = None
    best_error = float('inf')
    best_points3D = None
    
    for scale in scales:
        # Appliquer l'échelle à la translation
        T_c1_c2_scaled = T_c1_c2.copy()
        T_c1_c2_scaled[:3, 3] *= scale
        T_wc2_scaled = T_wc1 @ T_c1_c2_scaled
        
        # Triangulate points using cv2.triangulatePoints
        P1 = K_c @ np.eye(3, 4) @ np.linalg.inv(T_wc1)
        P2 = K_c @ np.eye(3, 4) @ np.linalg.inv(T_wc2_scaled)
        points4D = cv2.triangulatePoints(P1, P2, x1Data, x2Data)
        points3D_scaled = points4D / points4D[3, :]  # Normalize homogeneous coordinates
        
        # Compute reprojection error
        x1_proj = K_c @ np.eye(3, 4) @ np.linalg.inv(T_wc1) @ points3D_scaled
        x1_proj /= x1_proj[2, :]
        x2_proj = K_c @ np.eye(3, 4) @ np.linalg.inv(T_wc2_scaled) @ points3D_scaled
        x2_proj /= x2_proj[2, :]
        
        error = np.mean(np.linalg.norm(x1Data - x1_proj[:2, :], axis=0) + np.linalg.norm(x2Data - x2_proj[:2, :], axis=0))
        
        # Mettre à jour la meilleure échelle
        if error < best_error:
            best_error = error
            best_scale = scale
            best_points3D = points3D_scaled
    
    return best_scale, best_points3D

# Exemple d'utilisation


if __name__ == '__main__':
    np.set_printoptions(precision=4,linewidth=1024,suppress=True)


    # Load ground truth
    T_wc1 = np.eye(4, dtype=np.float32)

    K_c = np.load(fp.K_PATH)['mtx']
    # X_w = np.loadtxt('X_w.txt')
    x1Data = np.loadtxt(fp.X1_DATA).T
    x2Data = np.loadtxt(fp.X2_DATA).T
    x3Data = np.loadtxt(fp.X3_DATA).T


    F_2_1 = np.loadtxt(fp.F12_PATH)
    F_3_1 = np.loadtxt(fp.F13_PATH)

  
    path_image_1 = fp.P_IMG_HOUSE_1
    path_image_2 = fp.P_IMG_HOUSE_2
    path_image_3 = fp.P_IMG_HOUSE_3
    image_pers_1 = cv2.imread(path_image_1)
    image_pers_2 = cv2.imread(path_image_2)
    image_pers_3 = cv2.imread(path_image_3)

    print("Nouvelle matrice intrinsèque :\n", K_c)
    

 
    ##############################
    ##### Question 2 #############
    #############################
    E = K_c.T @ F_2_1 @ K_c
    x1Data_cv = x1Data.T.astype(np.float32).reshape(-1, 2)
    x2Data_cv = x2Data.T.astype(np.float32).reshape(-1, 2)
    E, mask = cv2.findEssentialMat(x1Data_cv, x2Data_cv, K_c, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    pts3D,T_wc2= T_w_c2(E, K_c, K_c, T_wc1, x1Data, x2Data)

    _, R1, t, mask_pose = cv2.recoverPose(E, x1Data_cv, x2Data_cv, K_c)
    T_21 = np.eye(4, dtype=np.float32)
    T_21[:3, :3] = R1
    T_21[:3, 3] = t.flatten()
    T_wc1 = np.eye(4, dtype=np.float32)
    R1 = T_wc1[:3, :3]
    t1 = T_wc1[:3, 3]
    R2 = T_wc2[:3, :3]
    t2 = T_wc2[:3, 3]

    # Matrices de projection
    P1 = K_c @ np.hstack((R1, t1.reshape(-1, 1)))
    P2 = K_c @ np.hstack((R2, t2.reshape(-1, 1)))

    

    # Triangulation
    points4D = cv2.triangulatePoints(P1, P2, x1Data, x2Data)


    x1_p = K_c @ np.eye(3, 4) @ np.linalg.inv(T_wc1) @pts3D
    x2_p = K_c @ np.eye(3, 4) @ np.linalg.inv(T_wc2) @ pts3D
    x1_p /= x1_p[2, :]
    x2_p /= x2_p[2, :]

    visualize_residuals(image_pers_1, x1Data, x1_p, title="Résidus de projection dans Image 1")
    visualize_residuals(image_pers_2, x2Data, x2_p, title="Résidus de projection dans Image 2")

    
    T_c1_c2 = np.linalg.inv(T_wc1) @ T_wc2 
    print(T_c1_c2)
    # scales = [2, 2.5, 2.75, 3, 3.25, 3.5]  # Échelles à tester
    # best_scale, best_points3D = test_scales(T_wc1, T_c1_c2, x1Data, x2Data, K_c, scales)
    # print(f"Meilleure échelle : {best_scale}") 
    # T_c1_c2[:3, 3] *= best_scale
    theta_rot, tras = Parametrice_Pose(T_c1_c2) #relative pose parametrization
    n_points = np.shape(pts3D)[1] 
    
    fig3D = plt.figure(figsize=(12, 8))
    ax = plt.axes(projection='3d', adjustable='box')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    # Plotting the system
    # drawRefSystem(ax, np.eye(4, 4), '-', 'W')
    drawRefSystem(ax, T_wc1, '-', 'C1')
    drawRefSystem(ax, T_wc2, '-', 'C2 (GT)')
    X_3D = np.linalg.inv(T_wc1) @ pts3D

    ax.scatter(X_3D[0, :], X_3D[1, :], X_3D[2, :], 
               c='b', marker='.', label='not opt')
    ax.legend()
    plt.title('3D Reconstruction: Ground Truth vs Optimized')

    plt.show()
    #Op vector  :  rotation parameters, translation, and 3D points in a specific format
    Op = np.zeros(5 + 3*n_points)
    Op[0:3] = theta_rot 
    Op[3:5] = tras
    X_3D = np.linalg.inv(T_wc1) @ pts3D
    Op[5:] = X_3D[:3, :].flatten()  
    
    print("least squares 1")
    result = least_squares(resBundleProjection, Op, args=(x1Data, x2Data, K_c, n_points), verbose=2)
    T_c1_c2_optim = ObtainPose(result.x[0:3],result.x[3],result.x[4])
    T_wc2_optim = T_wc1 @ T_c1_c2_optim
    
    # Visualisation 3D
    fig3D = plt.figure(figsize=(12, 8))
    ax = plt.axes(projection='3d', adjustable='box')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    # Plotting the system
    # drawRefSystem(ax, np.eye(4, 4), '-', 'W')
    drawRefSystem(ax, T_wc1, '-', 'C1')
    drawRefSystem(ax, T_wc2, '-', 'C2 (GT)')
    drawRefSystem(ax, T_wc2_optim, '-', 'C2 (Optimized)')
    X_opt = result.x[5:].reshape((3, n_points))
    X_opt_h = np.vstack([X_opt, np.ones((1, n_points))])
    x_3d_optim = T_wc1 @ X_opt_h
    x_3d_optim /= x_3d_optim[3, :]  # Normalize the homogeneous coordinate
    
    ax.scatter(x_3d_optim[0, :], x_3d_optim[1, :], x_3d_optim[2, :], 
               c='g', marker='.', label='Optimized')    

    max_range = np.array([X_3D[0,:].max()-X_3D[0,:].min(),
                          X_3D[1,:].max()-X_3D[1,:].min(),
                          X_3D[2,:].max()-X_3D[2,:].min()]).max() 
    
    mid_x = (X_3D[0,:].max()+X_3D[0,:].min()) 
    mid_y = (X_3D[1,:].max()+X_3D[1,:].min()) 
    mid_z = (X_3D[2,:].max()+X_3D[2,:].min())
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    ax.legend()
    plt.title('3D Reconstruction: Ground Truth vs Optimized')

    plt.show()
    
    pos_C1 = T_wc1[:3, 3]
    pos_C2 = T_wc2_optim[:3, 3]
    distance_between_cameras = np.linalg.norm(pos_C1 - pos_C2)

    # Calculate the closest point distance from the two cameras
    distances_C1 = np.linalg.norm(pts3D[:3, :] - pos_C1[:, np.newaxis], axis=0)
    distances_C2 = np.linalg.norm(pts3D[:3, :] - pos_C2[:, np.newaxis], axis=0)
    closest_distance = min(np.min(distances_C1), np.min(distances_C2))

    print("Distance between cameras:", distance_between_cameras)
    print("Closest point distance from cameras:", closest_distance)

    
    ####################################################################
    ####### Pnp estimation of the 3rd camera pose ######################
    ####################################################################
    distCoeffs = np.zeros((5,1))

    # Reformater les points 3D et 2D pour solvePnP
    X_3D_pnp = X_3D[:3, :].T  # Points 3D (n x 3)
    X_3D_pnp = np.ascontiguousarray(X_3D_pnp, dtype=np.float32)

    x3Data_pnp = x3Data.T  # Points 2D (n x 2)
    x3Data_pnp = np.ascontiguousarray(x3Data_pnp, dtype=np.float32)

    # Vérifier qu'on a assez de points
    n_points = X_3D_pnp.shape[0]
    if n_points < 4:
        print(f"Pas assez de points pour solvePnP: {n_points} points (minimum 4 requis)")
    else:
        print(f"Nombre de points pour solvePnP: {n_points}")

        # Résoudre PnP
        retval, rvec, tvec = cv2.solvePnP(X_3D_pnp, 
                                         x3Data_pnp, 
                                         K_c, 
                                         distCoeffs,
                                         flags=cv2.SOLVEPNP_EPNP)

        # Convertir le résultat en matrice de transformation
        R2, _ = cv2.Rodrigues(rvec)
        t = tvec
        T_c3c1 = ensamble_T(R2, t.T)
        T_wc3_est = T_wc1 @ np.linalg.inv(T_c3c1)

    fig3D = plt.figure(figsize=(12, 8))
    ax = plt.axes(projection='3d', adjustable='box')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    #Plotting the system
    
    drawRefSystem(ax, np.eye(4, 4), '-', 'W')
    drawRefSystem(ax, T_wc1, '-', 'C1')

    drawRefSystem(ax, T_wc2_optim, '-', 'C2 (Optimized)')
    drawRefSystem(ax, T_wc3_est, '-', 'C3 (Optimized)')
    # drawRefSystem(ax, T_wc3, '-', 'C3 ')

    plt.show()
    # plt.waitforbuttonpress()



    ####################################################
    ################## Resbundle wih 3 cameras #########
    ####################################################
    x1_p = K_c @ np.eye(3, 4) @ np.linalg.inv(T_wc1) @pts3D
    x3_p = K_c @ np.eye(3, 4) @ np.linalg.inv(T_wc3_est) @ pts3D
    x1_p /= x1_p[2, :]
    x3_p /= x3_p[2, :]

    visualize_residuals(image_pers_3, x3Data, x3_p, title="Résidus de projection dans Image 2")

    T_c1_c3 = np.linalg.inv(T_wc1)@T_wc3_est
    theta_rot2, tras2 = Parametrice_Pose(T_c1_c3)
    
    Op = np.zeros(10 + 3*n_points)
    Op[0:3] = theta_rot 
    Op[3:5] = tras
    X_3D = (np.linalg.inv(T_wc1) @ pts3D)
    Op[5:-5] = X_3D[:3, :].flatten()  
    Op[-5:-2] = theta_rot2
    Op[-2:] = tras2
    
    
    print("least squares 2 : 3 cameras !!!!")
    result = least_squares(resBundleProjection3cameras, Op, args=(x1Data, x2Data, x3Data, K_c, n_points), verbose=2, method='trf', ftol=1e-8, xtol=1e-8, gtol=1e-8, max_nfev=300)
    
    trasl_c1_c2 = T_c1_c2[0:3, 3]
    scale_c2 = np.linalg.norm(trasl_c1_c2)# Norm as the scale factor
    
    T_c1_c2_optim3 = ObtainPose(result.x[0:3],result.x[3],result.x[4])
    T_c1_c3_optim3 = ObtainPose(result.x[-5:-2],result.x[-2],result.x[-1])
    
    #Apply the scale factor to the transformations
    T_c1_c2_optim3[0:3, 3] *= scale_c2
    T_c1_c3_optim3[0:3, 3] *= scale_c2

    T_wc2_optim3 = T_wc1 @ T_c1_c2_optim3
    T_wc3_optim3 = T_wc1 @ T_c1_c3_optim3


    X_opt_3 = result.x[5:-5].reshape((3, n_points)) * scale_c2
    X_opt_3 = np.vstack([X_opt_3, np.ones((1, n_points))])
    x_3d_optim3 = T_wc1 @ X_opt_3
    
    X_opt *=scale_c2
    X_opt_h = np.vstack([X_opt, np.ones((1, n_points))])
    x_3d_optim = T_wc1 @ X_opt_h
    
    # Visualisation 3D
    fig3D = plt.figure(figsize=(12, 8))
    ax = plt.axes(projection='3d', adjustable='box')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    #Plotting the system
    drawRefSystem(ax, np.eye(4, 4), "-", "W")
    drawRefSystem(ax, T_wc1, "-", "C1")
    drawRefSystem(ax, T_wc2, "-", "C2")

    drawRefSystem(ax, T_wc2_optim3, "-", "C2_optim3")

    drawRefSystem(ax, T_wc3_optim3, "-", "C3_optim3")


   
    ax.scatter(x_3d_optim3[0, :],x_3d_optim3[1, :], x_3d_optim3[2, :], 
              c='g', marker='.', label='Optimized : 3 cameras')
    ax.scatter(x_3d_optim[0, :],x_3d_optim[1, :], x_3d_optim[2, :], 
              c='b', marker='.', label='Optimized : 2 cameras')

    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    
    ax.legend()
    plt.title('3D Reconstruction with 3 cameras: Ground Truth vs Optimized')
    
    plt.show()
