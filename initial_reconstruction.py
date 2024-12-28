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
def T_w_c2(E,K1,K2,T_C1_W,x1,x2):
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
    W = np.array([[0, -1, 0, ], [1, 0, 0], [0, 0, 1]])
    
    #Get posible rotations
    R_90 = u@W@vh  
    if (np.linalg.det(R_90)<0):
        R_90 = -R_90
        
    R_n90 = u@W.T@vh
    if (np.linalg.det(R_n90)<0):    
        R_n90 = -R_n90
      
    #get translations 
    A = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])  
    P1 = K1@A@T_C1_W 
    #Choose the correct answer and calculate P1 and P2
    for i in range(4):
        Pose = np.eye(4,4)  
        t = u[:,-1] 
        Rot = R_90 
        vote = 0            
        if(i%2 == 0):
            t = -(t)            
        if(i>1):
            Rot = R_n90
        Pose[:3,:3] = Rot
        Pose[:3,3] = t                            
        Pose = Pose@T_C1_W
        P2 = K2@A@Pose  
        p = np.empty([x1.shape[1],4])        
        for k in range(x1.shape[1]):
            p[:][k] = (triangulation(P1,P2,x1[:,k],x2[:,k]))
        p = p.T 
        p = T_C1_W@p
        p = normalize3Dpoints(p)        
        Pose = T_C1_W@Pose
        for k in range(x1.shape[1]):   
            #if a point is in front both cameras votes +1                     
            if(p[2,k] > 0 and p[2,k] > Pose[2,3]):
                    vote = vote + 1
        if (vote>Best_vote):                 
            Best_vote = vote
            Bestp = np.linalg.inv(T_C1_W)@p
            BestPose = np.linalg.inv(T_C1_W)@Pose
            BestPose = np.linalg.inv(BestPose)
           
    return Bestp, BestPose

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
    T_w_c[0:3, 3] = t_w_c
    T_w_c[3, 3] = 1
    return T_w_c


def ObtainPose(theta_rot,theta_tras,phi_tras):
    """
    -input:theta_rot,theta_tras,phi_tras
    -output: Pose matrix
    """
    #Rodrigues
    R = scAlg.expm(crossMatrix(theta_rot))
    
    #Calculate the 3x1 translation vector in Cartesian coordinates
    Tras = np.array([np.sin(theta_tras)*np.cos(phi_tras),np.sin(theta_tras)*np.sin(phi_tras),
    np.cos(theta_tras)])
    T = np.hstack([R, Tras.reshape((3, 1))])
    T = np.vstack([T, [0, 0, 0, 1]])
    return T
    
def indexMatrixToMatchesList(matchesList):
    """
    Convert a numpy matrix of index in a list of DMatch OpenCv matches.
     -input:
         matchesList: nMatches x 3 --> [[indexDesc1,indexDesc2,descriptorDistance],...]]
     -output:
        dMatchesList: list of n DMatch object
     """
    dMatchesList = []
    for row in matchesList:
        dMatchesList.append(cv2.DMatch(_queryIdx=row[0].astype('int'), _trainIdx=row[1].astype('int'), _distance=row[2]))
    return dMatchesList


def Parametrice_Pose(T):
    """
    -input:
    T: Transformation matrix
    -output:List rot with the 3 angles and list tras with the 2 angles
    """
    R = T[0:3, 0:3]
    theta_rot = crossMatrixInv(scAlg.logm(R))

    theta_tras = np.arccos(T[2,2])
    phi_tras = np.arccos(T[2,0]/np.sin(theta_tras))   
    tras = [theta_tras, phi_tras]
    return theta_rot,tras

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

    Parameters
    ----------
    Op : Array of parameters to optimize, including rotation, translation angles, 
        and 3D coordinates of points.
    x1Data : Observed 2D points in the first view, shape (2, n_points).
    x2Data : Observed 2D points in the second view, shape (2, n_points).
    K_c :  Intrinsic camera matrix, shape (3, 3).
    n_points : Number of 3D points.

    Returns
    -------
    res : Residuals for all points in both views, shape (4 * n_points,).
    """
    rot, theta_tras, phi_tras, X = unpackParameters(Op, n_points)
    T = np.eye(4)
    T_c1c = ObtainPose(rot,theta_tras,phi_tras)
    res1 = []
    res2 = []
    X_h = np.vstack([X, np.ones((1, n_points))])
    x1_proj = K_c @ np.eye(3, 4) @ T @ X_h
    x1_proj /= x1_proj[2, :]
    
    x2_proj =  K_c @ np.eye(3, 4) @ np.linalg.inv(T_c1c) @ X_h
    x2_proj = x2_proj / x2_proj[2] 

    for i in range(n_points):
        #View 1
        res1.append(x1Data[0,i] - x1_proj[0, i] )    # x
        res1.append(x1Data[1,i] - x1_proj[1, i])   # y
        # View 2
        res2.append(x2Data[0,i] - x2_proj[0,i]  ) # x
        res2.append(x2Data[1,i]- x2_proj[1,i]  ) # y
    res = np.hstack([res1,res2])
    res = np.array(res).flatten()
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
    x1_proj = K_c @ np.eye(3, 4) @ T @ X_h
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

    return np.array([[0, -x[2], x[1]],
                    [x[2], 0, -x[0]],
                    [-x[1], x[0], 0]])

  
    
if __name__ == '__main__':
    np.set_printoptions(precision=4,linewidth=1024,suppress=True)


    # Load ground truth
    T_wc1 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, -1], [0, 0, 0, 1]])

    K_c = np.load(fp.K_PATH)['mtx']
    # X_w = np.loadtxt('X_w.txt')
    x1Data = np.loadtxt(fp.X1_DATA).T
    x2Data = np.loadtxt(fp.X2_DATA).T
    F_2_1 = np.loadtxt(fp.F_PATH)

    #x1 et x2 are in pixel coordinates, we need to convert into 3d points by using the decompositon of the essential matrix and 
    #the camera calibration matrix

    #Read the images
    path_image_1 = fp.P_IMG_HOUSE_1
    path_image_2 = fp.P_IMG_HOUSE_2
    image_pers_1 = cv2.imread(path_image_1)
    image_pers_2 = cv2.imread(path_image_2)



 
    ##############################
    ##### Question 2 #############
    #############################
    E = K_c.T @ F_2_1 @ K_c
    pts3D,T_wc2= T_w_c2(E,K_c,K_c,np.linalg.inv(T_wc1),x1Data,x2Data)
    print(pts3D[:, :])
    
    ##############################
    ##### Question 2.1 #############
    #############################
    
    # Residuals between each poiint
    T_c1c2 = np.linalg.inv(T_wc1)@T_wc2
    x1_p = K_c @ np.eye(3, 4) @ np.linalg.inv(T_wc1) @pts3D
    x2_p = K_c @ np.eye(3, 4) @ np.linalg.inv(T_wc2) @ pts3D
    x1_p /= x1_p[2, :]
    x2_p /= x2_p[2, :]
    # visualize_residuals(image_pers_1, x1Data, x1_p, title="Résidus de projection dans Image 1")
    # visualize_residuals(image_pers_2, x2Data, x2_p, title="Résidus de projection dans Image 2")
    
    #3D difference 
 #essential matrix
    T_c1_c2 = np.linalg.inv(T_wc1) @ T_wc2 #relative transformation between the 2 cameras
    theta_rot, tras = Parametrice_Pose(T_c1_c2) #relative pose parametrization
    n_points = np.shape(pts3D)[1] 
    
    #Op vector  :  rotation parameters, translation, and 3D points in a specific format
    Op = np.zeros(5 + 3*n_points)
    Op[0:3] = theta_rot 
    Op[3:5] = tras
    X_3D = (np.linalg.inv(T_wc1) @ pts3D)
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
    #Plotting the system
    drawRefSystem(ax, np.eye(4, 4), '-', 'W')
    drawRefSystem(ax, T_wc1, '-', 'C1')
    drawRefSystem(ax, T_wc2, '-', 'C2 (GT)')
    drawRefSystem(ax, T_wc2_optim, '-', 'C2 (Optimized)')
    X_opt = result.x[5:].reshape((3, 517))
    X_opt_h = np.vstack([X_opt, np.ones((1, n_points))])
    x_3d_optim = T_wc1 @ X_opt_h
    
    
    # ax.scatter(X_w[0, :], X_w[1, :], X_w[2, :], c='r', marker='.', label='Ground Truth')
    ax.scatter(x_3d_optim[0, :],x_3d_optim[1, :], x_3d_optim[2, :], 
              c='g', marker='.', label='Optimized')
    
    # max_range = np.array([X_w[0,:].max()-X_w[0,:].min(),
    #                       X_w[1,:].max()-X_w[1,:].min(),
    #                       X_w[2,:].max()-X_w[2,:].min()]).max() / 2.0
    
    # mid_x = (X_w[0,:].max()+X_w[0,:].min()) * 0.5
    # mid_y = (X_w[1,:].max()+X_w[1,:].min()) * 0.5
    # mid_z = (X_w[2,:].max()+X_w[2,:].min()) * 0.5
    
    # ax.set_xlim(mid_x - max_range, mid_x + max_range)
    # ax.set_ylim(mid_y - max_range, mid_y + max_range)
    # ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    ax.legend()
    plt.title('3D Reconstruction: Ground Truth vs Optimized')
    
    plt.show()
    
    
    ####################################################
    ################## Exercise 3 ######################
    ####################################################
    # distCoeffs = np.zeros((5,1))
    # x_3d_optim=x_3d_optim[:3,:]
    # x_3d_optim=x_3d_optim.T
    # imagePoints = np.ascontiguousarray(x3Data, dtype=np.float32).reshape((x3Data.shape[1], 1, 2))
    # retval, rvec, tvec  = cv2.solvePnP(x_3d_optim, imagePoints, K_c,            
    # distCoeffs,flags=cv2.SOLVEPNP_EPNP) 
    # R = scAlg.expm(crossMatrix(rvec))
    # t= tvec
    # T_c3c1= ensamble_T(R,t.T)
    # T_wc3_est = T_wc1 @ np.linalg.inv(T_c3c1)

    # fig3D = plt.figure(figsize=(12, 8))
    # ax = plt.axes(projection='3d', adjustable='box')
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    # #Plotting the system
    # drawRefSystem(ax, np.eye(4, 4), '-', 'W')
    # drawRefSystem(ax, T_wc1, '-', 'C1')
    # drawRefSystem(ax, T_wc2, '-', 'C2 (GT)')
    # drawRefSystem(ax, T_wc2_optim, '-', 'C2 (Optimized)')
    # drawRefSystem(ax, T_wc3_est, '-', 'C3 (Optimized)')
    # plt.waitforbuttonpress()



    ####################################################
    ################## Exercise 4 ######################
    ####################################################
    
    # T_c1c3 = np.linalg.inv(T_wc1)@T_wc3
    # theta_rot2, tras2 = Parametrice_Pose(T_c1_c2)
    
    # Op = np.zeros(10 + 3*n_points)
    # Op[0:3] = theta_rot 
    # Op[3:5] = tras
    # X_3D = (np.linalg.inv(T_wc1) @ pts3D)
    # Op[5:-5] = X_3D[:3, :].flatten()  
    # Op[-5:-2] = theta_rot2
    # Op[-2:] = tras2
    
    
    # print("least squares 2 : 3 cameras !!!!")
    # result = least_squares(resBundleProjection3cameras, Op, args=(x1Data, x2Data, x3Data, K_c, n_points), verbose=2)
    
    # trasl_c1_c2 = T_c1_c2[0:3, 3]
    # scale_c2 = np.linalg.norm(trasl_c1_c2)# Norm as the scale factor
    
    # T_c1_c2_optim3 = ObtainPose(result.x[0:3],result.x[3],result.x[4])
    # T_c1_c3_optim3 = ObtainPose(result.x[-5:-2],result.x[-2],result.x[-1])
    
    # #Apply the scale factor to the transformations
    # T_c1_c2_optim3[0:3, 3] *= scale_c2
    # T_c1_c3_optim3[0:3, 3] *= scale_c2

    # T_wc2_optim3 = T_wc1 @ T_c1_c2_optim3
    # T_wc3_optim3 = T_wc1 @ T_c1_c3_optim3


    # X_opt_3 = result.x[5:-5].reshape((3, 103)) * scale_c2
    # X_opt_3 = np.vstack([X_opt_3, np.ones((1, n_points))])
    # x_3d_optim3 = T_wc1 @ X_opt_3
    
    # X_opt *=scale_c2
    # X_opt_h = np.vstack([X_opt, np.ones((1, n_points))])
    # x_3d_optim = T_wc1 @ X_opt_h
    
    # # Visualisation 3D
    # fig3D = plt.figure(figsize=(12, 8))
    # ax = plt.axes(projection='3d', adjustable='box')
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    # #Plotting the system
    # drawRefSystem(ax, np.eye(4, 4), "-", "W")
    # drawRefSystem(ax, T_wc1, "-", "C1")
    # drawRefSystem(ax, T__wc2, "-", "C2")

    # drawRefSystem(ax, T_wc2_optim3, "-", "C2_optim3")

    # drawRefSystem(ax, T_wc3_optim3, "-", "C3_optim3")


    # ax.scatter(X_w[0, :], X_w[1, :], X_w[2, :], c='r', marker='.', label='Ground Truth')
    # ax.scatter(x_3d_optim3[0, :],x_3d_optim3[1, :], x_3d_optim3[2, :], 
    #           c='g', marker='.', label='Optimized : 3 cameras')
    # ax.scatter(x_3d_optim[0, :],x_3d_optim[1, :], x_3d_optim[2, :], 
    #           c='b', marker='.', label='Optimized : 2 cameras')

    
    # max_range = np.array([X_w[0,:].max()-X_w[0,:].min(),
    #                      X_w[1,:].max()-X_w[1,:].min(),
    #                      X_w[2,:].max()-X_w[2,:].min()]).max() / 2.0
    
    # mid_x = (X_w[0,:].max()+X_w[0,:].min()) * 0.5
    # mid_y = (X_w[1,:].max()+X_w[1,:].min()) * 0.5
    # mid_z = (X_w[2,:].max()+X_w[2,:].min()) * 0.5
    
    # ax.set_xlim(mid_x - max_range, mid_x + max_range)
    # ax.set_ylim(mid_y - max_range, mid_y + max_range)
    # ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    # ax.legend()
    # plt.title('3D Reconstruction with 3 cameras: Ground Truth vs Optimized')
    
    # plt.waitforbuttonpress()
        
