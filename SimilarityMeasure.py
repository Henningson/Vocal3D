import cv2
import numpy as np
import os


def getGlottalAreas(A, B, im):
    im_copy = im.copy()
    im_copy = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    cv2.circle(im_copy, np.flip(A), radius=4, color=(255, 255, 255))
    cv2.circle(im_copy, np.flip(B), radius=4, color=(255, 255, 255))
    cv2.line(im_copy, np.flip(A), np.flip(B), thickness=2, color=(255, 255, 255))

    #cv2.imshow("IM", im_copy)
    P = np.array(im.nonzero()).T

    # a = (x1, y1)
    # b = (x2, y2)
    # P = (x, y)
    # d = (p[0] - a[0])*(b[1] - a[1]) - (p[1] - a[1])(b[0] - a[0])

    d = (P[:, 0] - A[0])*(A[1] - B[1]) - (P[:, 1] - A[1])*(A[0] - B[0])

    for i in range(P.shape[0]):
        if d[i] < 0:
            im_copy[P[i, 0], P[i, 1]] = (255, 0, 0)
            # cv2.circle(im, np.flip(P[0]), radius=4, color=(255, 0, 0))
        else:
            im_copy[P[i, 0], P[i, 1]] = (0, 0, 255)
            # cv2.circle(im, np.flip(P[0]), radius=4, color=(0, 0, 255))

    #cv2.imshow("Crop", im_copy)
    #cv2.waitKey(0)
    return P[d >= 0].shape[0], P[d < 0].shape[0], im_copy


def loadImages(path):
    images = list()
    number_files = len(os.listdir(path))

    for i in range(1, number_files):
        images.append(cv2.imread(path + '{0:05d}'.format(i) + ".png"))

    return images


if __name__ == "__main__":
    names = [
            ("65_Kay", "Kay_65_M2_-15"),
            ("65_Kay", "Kay_65_M2_-10"),
            ("65_Kay", "Kay_65_M2_-5"),
            ("65_Kay", "Kay_65_M2_0"),
            ("65_Kay", "Kay_65_M2_5"),
            ("65_Kay", "Kay_65_M2_10"),
            ("65_Kay", "Kay_65_M2_15")]
    path_middle = "png/"

    paths = list()
    for _, name in names:
        path = name + "/"
        paths.append(path)

    areas = list()
    for i, path in enumerate(paths):
        #print(path)
        images = loadImages(path)
        print("Computing area for " + path)

        perVideoAreas = list()
        for image in images:
            og_image = image.copy()
            image = cv2.cvtColor(og_image, cv2.COLOR_BGR2GRAY)
            crop_size = 40
            crop = image[crop_size:-crop_size, crop_size:-crop_size]
            og_image = og_image[crop_size:-crop_size, crop_size:-crop_size]
            #cv2.imshow("Crop", crop)
            #cv2.waitKey(0)

            a = 1
            nonzeros = np.array(crop.nonzero()).T
            minX = nonzeros[:, 1].min()
            maxX = nonzeros[:, 1].max()
            minY = nonzeros[:, 0].min()
            maxY = nonzeros[:, 0].max()

            cropcrop = crop[minY:maxY, minX:maxX]
            og_image = og_image[minY:maxY, minX:maxX]
            cv2.imwrite("Cropcrop.png", cropcrop)
            #cv2.imshow("Crop", cropcrop)
            #cv2.waitKey(0)

            cropcropcrop = cropcrop[crop_size//4:-crop_size//4, crop_size//4:-crop_size//4]
            og_image = og_image[crop_size//4:-crop_size//4, crop_size//4:-crop_size//4]
            #cv2.imshow("Crop", cropcropcrop)
            #cv2.waitKey(0)

            thresh = ((cropcropcrop == 0)*255).astype(np.uint8)
            #cv2.imshow("Crop", thresh)
            #cv2.waitKey(0)

            numLabels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh)

            bounding_box = stats[1 + np.argmax(stats[1:, 4])]

            top = bounding_box[cv2.CC_STAT_TOP]
            height = bounding_box[cv2.CC_STAT_HEIGHT]

            left = bounding_box[cv2.CC_STAT_LEFT]
            width = bounding_box[cv2.CC_STAT_WIDTH]

            seg = np.zeros(thresh.shape, dtype=np.uint8)
            seg[top:top+height, left:left+width] = 1
            thresh *= seg

            #cv2.imshow("Crop", thresh)
            #cv2.waitKey(0)

            white_points = np.argwhere(thresh != 0)

            min_thresh = np.min(white_points[:, 0])
            min_points = white_points[np.where(white_points[:, 0] == min_thresh)]
            upper_point = (np.sum(min_points, axis=0) / min_points.shape[0]).astype(np.int64)

            max_thresh = np.max(white_points[:, 0])
            max_points = white_points[np.where(white_points[:, 0] == max_thresh)]
            lower_point = (np.sum(max_points, axis=0) / max_points.shape[0]).astype(np.int64)



            #A = np.vstack(np.vstack([x, np.ones(len(x))]).T)
            #m, c = np.linalg.lstsq(A, y, rcond=None)[0]

            #upperPoint = np.array([m*x.min() + c, x.min()])
            #lowerPoint = np.array([m*x.max() + c, x.max()])

            #cv2.imshow("Crop", thresh)
            #cv2.waitKey(0)

            thresh_test = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
            
            #cv2.imwrite("Segment.png", thresh_test)
            #cv2.imshow("Crop", thresh_test)
            #cv2.waitKey(0)
            #exit()

            area1, area2, im = getGlottalAreas(upper_point.astype(np.int), lower_point.astype(np.int), thresh)
            
            og_image += im
            #cv2.circle(og_image, np.flip(upper_point).astype(np.int), 2, color=(255, 255, 255))
            #cv2.circle(og_image, np.flip(lower_point).astype(np.int), 2, color=(255, 255, 255))
            #cv2.line(og_image, np.flip(upper_point).astype(np.int), np.flip(lower_point).astype(np.int), color=(255, 255, 255))
            #cv2.imshow("Crocropcrop", og_image)

            #print("Area 1 has {0} points. Area 2 has {1} points. Average: {2}".format(area1, area2, area1/area2))
            if area1 < 1 or area2 < 1:
                continue
            perVideoAreas.append((area1, area2, area1/area2))
        areas.append(perVideoAreas)

    for name, area in zip(names, areas):
        ar = np.array(area).reshape(-1, 3)
        np.savetxt(name[1] + ".csv", ar, delimiter=" ")
