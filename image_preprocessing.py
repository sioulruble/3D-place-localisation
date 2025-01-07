import cv2
import numpy as np
from configs import files_path as fp

# Load the old image of the townhall and the last one I took
old_house = cv2.imread(fp.OLD_IMG_PATH)
new_house = cv2.imread(fp.P_IMG_HOUSE_1)

old_house = cv2.resize(old_house, (new_house.shape[1], new_house.shape[0]))
cv2.imwrite(fp.P_OLD_IMG_PATH , old_house)



