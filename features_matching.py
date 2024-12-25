import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from configs import files_path as fp    
from tqdm import tqdm

def compute_epipolar_lines(F_matrix, inlier_matches):
    epipolar_lines_2_1 = []
    for match in inlier_matches:
        x0 = np.array([match[0][0], match[0][1], 1])
        x1 = np.array([match[1][0], match[1][1], 1])
        epipolar_line = np.dot(F_matrix, x1)
        epipolar_lines_2_1.append(epipolar_line)
    epipolar_lines_1_2 = []
    for match in inlier_matches:
        x0 = np.array([match[0][0], match[0][1], 1])
        x1 = np.array([match[1][0], match[1][1], 1])
        epipolar_line = np.dot(F_matrix.T, x0)
        epipolar_lines_1_2.append(epipolar_line)
    return epipolar_lines_2_1, epipolar_lines_1_2

def index_matrix_to_matches_list(matches_list):
    d_matches_list = []
    for row in matches_list:
        d_matches_list.append(cv.DMatch(_queryIdx=row[0], _trainIdx=row[1], _distance=row[2]))
    return d_matches_list

def matches_list_to_index_matrix(d_matches_list):
    matches_list = []
    for k in range(len(d_matches_list)):
        matches_list.append([int(d_matches_list[k].queryIdx), int(d_matches_list[k].trainIdx), d_matches_list[k].distance])
    return matches_list

def plot_epipolar_lines(image_1, image_2, epipolar_lines_2_1, epipolar_lines_1_2):    
    # now plot the lines
    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(image_1)
    axs[1].imshow(image_2)
    for i in range(len(epipolar_lines_2_1)):
        line = epipolar_lines_2_1[i]
        # get the x and y coordinates for the line
        x = np.array([0, image_1.shape[1]])
        y = (-line[2] - line[0] * x) / line[1]
        axs[0].plot(x, y)
    for i in range(len(epipolar_lines_1_2)):
        line = epipolar_lines_1_2[i]
        # get the x and y coordinates for the line
        x = np.array([0, image_2.shape[1]])
        y = (-line[2] - line[0] * x) / line[1]
        axs[1].plot(x, y)
    plt.show()

def epipolar_line(image, l):
    # Load the image
    img = cv.imread(image)
    # Plot the image
    plt.figure(3)
    plt.clf()
    plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    # Plot the epipolar lines
    for i in range(len(l[0])):
        x = np.linspace(0, 750, 100)
        y = -(l[0][i]*x+l[2][i])/l[1][i]
        plt.plot(x, y)
    plt.draw()
    plt.waitforbuttonpress()
    return

def match_with_2nd_rr(desc1, desc2, dist_ratio, min_dist):
    matches = []
    n_desc1 = desc1.shape[0]

    for k_desc1 in tqdm(range(n_desc1), desc="Matching descriptors"):
        dist = np.sqrt(np.sum((desc2 - desc1[k_desc1, :]) ** 2, axis=1))
        index_sort = np.argsort(dist)
        dist1 = dist[index_sort[0]]
        dist2 = dist[index_sort[1]]
        if (dist1 < dist_ratio * dist2) and (dist1 < min_dist):
            matches.append([k_desc1, index_sort[0], dist1])
    return matches

def normalization_matrix(nx, ny):
    Nv = np.array([[1/nx, 0, -1/2], [0, 1/ny, -1/2], [0, 0, 1]])
    return Nv


def compute_f(matches, N1, N2):
    x0 = [match[0] for match in matches]
    x1 = [match[1] for match in matches]
    # Add 1 to the last row of x0 and x1
    x0 = np.asarray(x0).T
    x0 = np.append(x0, np.ones((1, len(x0[0]))), axis=0)
    x1 = np.asarray(x1).T
    x1 = np.append(x1, np.ones((1, len(x1[0]))), axis=0)
    x0_norm = np.empty([3, len(x0[0])])
    x1_norm = np.empty([3, len(x1[0])])
    # Multiply x0 with N1 to normalize the points
    for i in range(len(x0[0])):
        x0_norm[:, i] = N1 @ x0[:, i]
    # Multiply x1 with N2 to normalize the points
    for i in range(len(x1[0])):
        x1_norm[:, i] = N2 @ x1[:, i]
    # Eliminate the last row of x0 and x1
    x0 = x0[:-1, :]
    x1 = x1[:-1, :]
    # Given this N matches, get the F
    Ec = np.empty([9, len(x0[0])])
    for i in range(len(x0[0])):
        Ec[:, i] = [x0_norm[0][i] * x1_norm[0][i], x0_norm[1][i] * x1_norm[0][i], x1_norm[0][i],
                    x0_norm[0][i] * x1_norm[1][i], x0_norm[1][i] * x1_norm[1][i], x1_norm[1][i],
                    x0_norm[0][i], x0_norm[1][i], 1]
    u, s, vh = np.linalg.svd(Ec.T)
    Fn = vh[-1, :]
    Fn = Fn.reshape(3, 3)
    # Make F rank 2
    u, s, vh = np.linalg.svd(Fn)
    S = np.zeros([3, 3])
    S[0, 0] = s[0]
    S[1, 1] = s[1]
    Fn = u @ S @ vh
    # Unnormalize F
    F = N2.T @ Fn @ N1
    return F, x0, x1


def compute_f_ransac(image1, image2, match_pairs, ransac_pixel_threshold, ransac_confidence, ransac_inlier_ratio):
    number_of_iterations = np.log(1 - ransac_confidence) / np.log(1 - ransac_inlier_ratio**8)
    number_of_iterations = int(number_of_iterations)
    print(f'The number of iterations is {number_of_iterations}')
    print(f'The number of matches is {len(match_pairs)}')
    best_F = None
    best_inlier_count = 0
    best_matches = None

    N1 = normalization_matrix(image1.shape[1], image1.shape[0])
    N2 = normalization_matrix(image2.shape[1], image2.shape[0])

    for j in range(number_of_iterations):
        if j % 500 == 0:
            print(f'Iteration {j}')
        # Get a sample of 8 matches
        random_numbers = np.random.randint(0, len(match_pairs), 8)
        matches_sample = [match_pairs[random_number] for random_number in random_numbers]
        # compute F with the sample
        F, x0, x1 = compute_f(matches_sample, N1, N2)
        # Let the pixels vote for F: the more inliers, the better is this solution
        inlier_count = 0
        inlier_matches = []
        x0_m = np.empty([0, 2])
        x1_m = np.empty([0, 2])
        # iterate over all the matches to vote for F if they are inliers
        for match in match_pairs:
            # Make array x0_m and x1_m empty
            x0p = match[0]
            x1p = match[1]
            x0p = np.append(x0p, 1)
            x1p = np.append(x1p, 1)
            l1 = F @ x0p
            l2 = F.T @ x1p
            # Calculate the distance
            distance1 = np.abs(x1p @ l1 / np.sqrt((l1[0] * l1[0] + l1[1] * l1[1])))
            distance2 = np.abs(x0p @ l2 / np.sqrt((l2[0] * l2[0] + l2[1] * l2[1])))
            # Check if the pixel is in the inlier set
            if (distance1 < ransac_pixel_threshold) and (distance2 < ransac_pixel_threshold):
                # i want the distance threshold to be respected in both images
                inlier_count += 1
                inlier_matches.append(match)
                # Add the match to the inlier set
                x0_m = np.append(x0_m, [match[0]], axis=0)
                x1_m = np.append(x1_m, [match[1]], axis=0)
        if inlier_count > best_inlier_count:    # if the current F is better than the best F
            print(f'New best F, inlier count is {inlier_count}')
            best_inlier_count = inlier_count
            best_inlier_matches = inlier_matches
            best_F = F
            # Make x0_m and x1_m homogeneous
            x0_m = x0_m.T
            x0_m = np.append(x0_m, np.ones((1, len(x0_m[0]))), axis=0)
            x1_m = x1_m.T
            x1_m = np.append(x1_m, np.ones((1, len(x1_m[0]))), axis=0)
            # Copy the matches from x0_m and x1_m to x0_m_F and x1_m_F
            x0_m_F = x0_m.copy()
            x1_m_F = x1_m.copy()
            # Parse x0 and x1 to cv.KeyPoint
            x0_F = x0
            x1_F = x1
            x0k_F = [cv.KeyPoint(x0[0, i], x0[1, i], 1) for i in range(8)]
            x1k_F = [cv.KeyPoint(x1[0, i], x1[1, i], 1) for i in range(8)]
            # Make the matches list with x0 and x1 using index_matrix_to_matches_list and index_matrix_to_matches_list
            matches_list = []
            for k in range(8):
                matches_list.append([int(k), int(k), 0])
            d_matches_list = index_matrix_to_matches_list(matches_list)
            # Plot the 8 random matches on the images with their inliers
            img_matched = cv.drawMatches(image_1, x0k_F, image_2, x1k_F, d_matches_list[:100],
                                 None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS and cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            # plt.figure(1)
            # plt.clf()
            # plt.imshow(img_matched, cmap='gray', vmin=0, vmax=255)
            # # Plot x0_m in the first image and x1_m in the second image
            # plt.plot(x0_m_F[0][:], x0_m_F[1][:], 'rx')
            # plt.plot(x1_m_F[0][:]+image2.shape[0], x1_m_F[1][:], 'bx')
            # plt.draw()
            # plt.waitforbuttonpress()

    print(f'The best F is\n{best_F}')
    print(f'The number of inliers is {best_inlier_count}')

    # now that we have the best F, we can compute it using all the inliers
    # get the inliers from the best matches
    F, x0, x1 = compute_f(best_inlier_matches, N1, N2)
    print(f'Computed new F from all the inliers:\n{F}')
    x0_F = x0
    x1_F = x1
    x0k_F = [cv.KeyPoint(x0[0, i], x0[1, i], 1) for i in range(8)]
    x1k_F = [cv.KeyPoint(x1[0, i], x1[1, i], 1) for i in range(8)]
    # Plot the 8 random matches on the images with their inliers
    img_matched = cv.drawMatches(image1, x0k_F, image2, x1k_F, d_matches_list[:100],
                            None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS and cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    plt.figure(2)
    plt.clf()
    plt.imshow(img_matched, cmap='gray', vmin=0, vmax=255)

    # Plot x0_m in the first image and x1_m in the second image
    plt.plot(x0_m_F[0][:], x0_m_F[1][:], 'rx')
    plt.plot(x1_m_F[0][:]+750, x1_m_F[1][:], 'bx')
    plt.draw
    plt.draw()
    plt.waitforbuttonpress()

    # calculate the epipolar lines
    l_2_1, l_1_2 = compute_epipolar_lines(F, best_inlier_matches)
    plot_epipolar_lines(image1, image2, l_2_1, l_1_2)


    return best_F

if __name__ == "__main__":
    np.set_printoptions(precision=4, linewidth=1024, suppress=True)
    # Load the images
    image_1 = cv.imread(fp.P_IMG_HOUSE_1)
    image_2 = cv.imread(fp.P_IMG_HOUSE_2)
    # Import the pairs .npz file
    matches_dict = np.load(fp.FEATURES_MATCHING_PATH)
    keypoints_0 = matches_dict['keypoints0']
    keypoints_1 = matches_dict['keypoints1']
    matches = matches_dict['matches']
    match_confidence = matches_dict['match_confidence']

    RANSAC_inlier_ratio = 0.4
    RANSAC_confidence = 0.999
    RANSAC_pixel_threshold = 2

    # SUPERGLUE
    # Get the match pairs and remove the -1
    # match_pairs = np.empty([len(matches)], dtype=object)
    # for i in range(len(matches)):
    #     if matches[i] == -1:
    #         match_pairs[i] = None
    #     else:
    #         match_pairs[i] = [keypoints_0[i], keypoints_1[matches[i]]]

    # match_pairs = [np.array(match_pairs[i]) for i in range(len(match_pairs)) if match_pairs[i] is not None]
    # F = compute_f_ransac(image_1, image_2, match_pairs, RANSAC_pixel_threshold, RANSAC_confidence, RANSAC_inlier_ratio)
    
    # SIFT
    print('SIFT')
    sift = cv.SIFT_create(nfeatures=0, nOctaveLayers=5, contrastThreshold=0.02, edgeThreshold=20, sigma=0.5)
    keypoints_sift_1, descriptors_1 = sift.detectAndCompute(image_1, None)
    keypoints_sift_2, descriptors_2 = sift.detectAndCompute(image_2, None)

    dist_ratio = 0.9
    min_dist = 200
    print('Matching with 2nd nearest neighbor')
    matches_list = match_with_2nd_rr(descriptors_1, descriptors_2, dist_ratio, min_dist)
    d_matches_list = index_matrix_to_matches_list(matches_list)
    d_matches_list = sorted(d_matches_list, key=lambda x: x.distance)

    # Conversion from DMatches to Python list
    matches_list = matches_list_to_index_matrix(d_matches_list)

    # Matched points in numpy from list of DMatches
    src_pts = np.float32([keypoints_sift_1[m.queryIdx].pt for m in d_matches_list]).reshape(len(d_matches_list), 2)
    dst_pts = np.float32([keypoints_sift_2[m.trainIdx].pt for m in d_matches_list]).reshape(len(d_matches_list), 2)

    # Make the list of match pairs 
    match_pairs_sift = np.empty([len(matches_list)], dtype=object)

    for i in range(len(matches_list)):
        match_pairs_sift[i] = [src_pts[i], dst_pts[i]]
    print('Computing F with RANSAC')
    F = compute_f_ransac(image_1, image_2, match_pairs_sift, RANSAC_pixel_threshold, RANSAC_confidence, RANSAC_inlier_ratio)