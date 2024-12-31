import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from configs import files_path as fp    
from tqdm import tqdm

def save_points_to_file(points, filename):
    points = points.T  # Transpose the points to shape (2, n)
    with open(filename, "w") as f:
        for point in points:
            formatted_line = f"{point[0]:.18e} {point[1]:.18e}\n"
            f.write(formatted_line)



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
    for i in range(len(l)):
        x = np.linspace(0, img.shape[1], 100)
        y = -(l[i][0] * x + l[i][2]) / l[i][1]
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
    number_of_iterations = min(number_of_iterations, 300)

    print(f'The number of iterations is {number_of_iterations}')
    print(f'The number of matches is {len(match_pairs)}')
    best_F = None
    best_inlier_count = 0
    best_matches = None
    pts_img1 = np.empty([0, 2])
    N1 = normalization_matrix(image1.shape[1], image1.shape[0])
    N2 = normalization_matrix(image2.shape[1], image2.shape[0])
    inliers = []    
    for j in range(number_of_iterations):
        if j % 500 == 0:
            print(f'Iteration {j}')
        # Get a sample of 8 matches
        random_numbers = np.random.randint(0, len(match_pairs), 10)
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
              # Sauvegarder src_pts et dst_pts
            # save_points_to_file(x0_m.T, fp.X1_DATA)
            # save_points_to_file(x1_m.T, fp.X2_DATA)
            # Make x0_m and x1_m homogeneous
            x0_m = x0_m.T
            x0_m = np.append(x0_m, np.ones((1, len(x0_m[0]))), axis=0)
            
            pts_img1 = x0_m
            x1_m = x1_m.T
            x1_m = np.append(x1_m, np.ones((1, len(x1_m[0]))), axis=0)

            inliers = [x0_m, x1_m]
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

    print(f'The best F is\n{best_F}')
    print(f'The number of inliers is {best_inlier_count}')


    return best_F,inliers

if __name__ == "__main__":
    np.set_printoptions(precision=4, linewidth=1024, suppress=True)
    # Load the images
    image_1 = cv.imread(fp.P_IMG_HOUSE_1)
    image_2 = cv.imread(fp.P_IMG_HOUSE_2)
    image_3 = cv.imread(fp.P_IMG_HOUSE_3)
    # Import the pairs .npz file
    matches_dict_12 = np.load(fp.FEATURES_MATCHING_PATH_1_2)
    keypoints_0_12 = matches_dict_12['keypoints0']
    keypoints_1_12 = matches_dict_12['keypoints1']
    
    matches_dict_13 = np.load(fp.FEATURES_MATCHING_PATH_1_3)
    keypoints_0_13 = matches_dict_13['keypoints0']
    keypoints_1_13 = matches_dict_13['keypoints1']

    matches_12 = matches_dict_12['matches']
    match_confidence_12 = matches_dict_12['match_confidence']

    matches_13 = matches_dict_13['matches']
    match_confidence_13 = matches_dict_13['match_confidence']

    RANSAC_inlier_ratio = 0.4
    RANSAC_confidence = 0.999
    RANSAC_pixel_threshold = 5

    # SUPERGLUE
    # Get the match pairs and remove the -1
    match_pairs_12 = np.empty([len(matches_12)], dtype=object)
    for i in range(len(matches_12)):
        if matches_12[i] == -1:
            match_pairs_12[i] = None
        else:
            match_pairs_12[i] = [keypoints_0_12[i], keypoints_1_12[matches_12[i]]]

    match_pairs_13 = np.empty([len(matches_13)], dtype=object)
    for i in range(len(matches_13)):
        if matches_13[i] == -1:
            match_pairs_13[i] = None
        else:
            match_pairs_13[i] = [keypoints_0_13[i], keypoints_1_13[matches_13[i]]]

    
    match_pairs_12 = [np.array(match_pairs_12[i]) for i in range(len(match_pairs_12)) if match_pairs_12[i] is not None]
    F, inliers_12 = compute_f_ransac(image_1, image_2, match_pairs_12, RANSAC_pixel_threshold, RANSAC_confidence, RANSAC_inlier_ratio)
    np.savetxt(fp.F_PATH, F)
    inlier_points_1 = inliers_12[0][:2].T  # Take only x,y coordinates
    inlier_points_2 = inliers_12[1][:2].T  # Take only x,y coordinates
    print(inlier_points_1.shape, inlier_points_2.shape)
    
    match_pairs_13_filtered = [pair for pair in match_pairs_13 if pair is not None]
points_1_from_13 = np.array([pair[0] for pair in match_pairs_13_filtered])
points_3 = np.array([pair[1] for pair in match_pairs_13_filtered])

# Initialize arrays for ordered matches
n_inliers = len(inlier_points_1)
ordered_matches = []

# For each inlier point, find the closest match in match_pairs_13
for inlier_idx, inlier_point in enumerate(inlier_points_1):
    # Calculate distances to all points in match_pairs_13
    distances = np.sqrt(np.sum((points_1_from_13 - inlier_point) ** 2, axis=1))
    
    # Find the closest point within threshold
    closest_idx = np.argmin(distances)
    min_distance = distances[closest_idx]
    
    if min_distance < 3:
        # Add the match using the closest_idx for points_1_from_13 and points_3
        ordered_matches.append([
            points_1_from_13[closest_idx],  # Point from image 1
            inlier_points_2[inlier_idx],    # Point from image 2 (keep original inlier order)
            points_3[closest_idx]           # Point from image 3 corresponding to closest match
        ])

# Filter out None values while maintaining relative order
filtered_matches = [match for match in ordered_matches if match is not None]
print(f"Number of filtered matches: {len(filtered_matches)}")

if filtered_matches:
    points_1 = np.array([m[0] for m in filtered_matches])
    points_2 = np.array([m[1] for m in filtered_matches])
    points_3 = np.array([m[2] for m in filtered_matches])
    
    save_points_to_file(points_1.T, fp.X1_DATA)
    save_points_to_file(points_2.T, fp.X2_DATA)
    save_points_to_file(points_3.T, fp.X3_DATA)
    
    print(f"Saved {len(filtered_matches)} corresponding points for all three images")