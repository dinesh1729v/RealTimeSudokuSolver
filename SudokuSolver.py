import copy

import cv2
import numpy as np
from scipy import ndimage
import math
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import bestFirstSearch


def show_solution_on_image(image, grid, unsolved_grid):
    if grid == False or unsolved_grid == False:
        return image
    print("show solution function")
    print(grid)
    print(unsolved_grid)
    SIZE = 9
    width = image.shape[1] // 9
    height = image.shape[0] // 9
    for i in range(SIZE):
        for j in range(SIZE):
            if (unsolved_grid[i][j] != '0'):  # If user fill this cell
                continue  # Move on
            text = str(grid[i][j])
            off_set_x = width // 15
            off_set_y = height // 15
            font = cv2.FONT_HERSHEY_SIMPLEX
            (text_height, text_width), baseLine = cv2.getTextSize(text, font, fontScale=1, thickness=3)
            marginX = math.floor(width / 7)
            marginY = math.floor(height / 7)

            font_scale = 0.6 * min(width, height) / max(text_height, text_width)
            text_height *= font_scale
            text_width *= font_scale
            bottom_left_corner_x = width * j + math.floor((width - text_width) / 2) + off_set_x
            bottom_left_corner_y = height * (i + 1) - math.floor((height - text_height) / 2) + off_set_y
            image = cv2.putText(image, text, (bottom_left_corner_x, bottom_left_corner_y),
                                font, font_scale, (0, 255, 0), thickness=2, lineType=cv2.LINE_AA)

    # cv2.imshow("Hello", image)
    return image


def are_two_matrices_equal(mat1, mat2):
    if mat1 == False or mat2 == False:
        return False
    for i in range(9):
        for j in range(9):
            if mat1[i][j] != mat2[i][j]:
                return False
    return True


def prepare(img_array):
    new_array = img_array.reshape(-1, 28, 28, 1)
    new_array = new_array.astype('float32')
    new_array /= 255
    return new_array


def get_best_shift(img):
    cy, cx = ndimage.measurements.center_of_mass(img)
    rows, cols = img.shape
    shiftx = np.round(cols / 2.0 - cx).astype(int)
    shifty = np.round(rows / 2.0 - cy).astype(int)
    return shiftx, shifty


def shiftTransform(img, sx, sy):
    rows, cols = img.shape
    M = np.float32([[1, 0, sx], [0, 1, sy]])
    shifted = cv2.warpAffine(img, M, (cols, rows))
    return shifted


def extract_digit(image):
    image = image.astype('uint8')
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=8)
    sizes = stats[:, -1]

    if (len(sizes) <= 1):
        blank_image = np.zeros(image.shape)
        blank_image.fill(255)
        return blank_image

    max_label = 1
    # Start from component 1 (not 0) because we want to leave out the background
    max_size = sizes[1]

    for i in range(2, nb_components):
        if sizes[i] > max_size:
            max_label = i
            max_size = sizes[i]

    img2 = np.zeros(output.shape)
    img2.fill(255)
    img2[output == max_label] = 0
    return img2


def order(pts):
    sum = 2147483647
    index = 0
    orderedPts = np.zeros((4, 2), dtype="float32")
    for ind in range(4):
        if (pts[ind][0] + pts[ind][1] < sum):
            sum = pts[ind][0] + pts[ind][1]
            index = ind

    orderedPts[0] = pts[index]
    pts = np.delete(pts, index, 0)

    sum = -1
    for ind in range(3):
        if (pts[ind][0] + pts[ind][1] > sum):
            sum = pts[ind][0] + pts[ind][1]
            index = ind
    orderedPts[2] = pts[index]

    pts = np.delete(pts, index, 0)

    if (pts[0][0] > pts[1][0]):
        orderedPts[1] = pts[0]
        orderedPts[3] = pts[1]
    else:
        orderedPts[1] = pts[1]
        orderedPts[3] = pts[0]

    orderedPts.reshape(4, 2)
    return orderedPts


cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)


def load_model():
    input_shape = (28, 28, 1)
    num_classes = 9
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.load_weights("digitRecognition.h5")
    return model


prev_sudoku = None

while True:
    model = load_model()
    _, frame = cap.read()

    # hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    cv2.imshow("Frame", frame)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("Gray", gray)

    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # cv2.imshow("blur", blur)

    thresh = cv2.adaptiveThreshold(blur, 255, 1, 1, 11, 2)
    # cv2.imshow("thresh", thresh)
    max_area = 0
    c = 0
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # print(len(contours))
    best_cnt = None
    approx = []
    for i in contours:
        area = cv2.contourArea(i)
        # print(len(i))
        if area > max_area:
            max_area = area
            approx = cv2.approxPolyDP(i, 0.01 * cv2.arcLength(i, True), True)
            best_cnt = i
        c += 1
    if len(approx) == 4:
        mask = np.zeros((gray.shape), np.uint8)
        cv2.drawContours(mask, [best_cnt], 0, 255, -1)
        cv2.drawContours(mask, [best_cnt], 0, 0, 1)
        # cv2.imshow("mask", mask)
        out = np.zeros_like(gray)
        out[mask == 255] = gray[mask == 255]
        # cv2.imshow("Only Sudoku", out)
        screenCnt = approx
        pts = screenCnt.reshape(4, 2)
        sudoku = np.zeros((4, 2), dtype="float32")
        orderedPts = order(pts)
        # (tl, tr, bl, br)\
        # pts1 = np.float32([orderedPts[1], orderedPts[3], orderedPts[0], orderedPts[2]])
        pts2 = np.float32([[0, 0], [540, 0], [0, 540], [540, 540]])

        (tl, tr, br, bl) = orderedPts
        width_A = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        width_B = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))

        # the height of our Sudoku board
        height_A = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        height_B = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))

        # print(width_A, width_B, height_A, height_B)

        # take the maximum of the width and height values to reach
        # our final dimensions
        max_width = max(int(width_A), int(width_B))
        max_height = max(int(height_A), int(height_B))

        # construct our destination points which will be used to
        # map the screen to a top-down, "birds eye" view
        dst = np.array([
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1]], dtype="float32")

        M = cv2.getPerspectiveTransform(orderedPts, dst)
        warp = cv2.warpPerspective(frame, M, (max_width, max_height))
        # cv2.imshow("Cropped Sudoku", warp)
        prev_warp = np.copy(warp)

        warp = cv2.cvtColor(warp, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(warp, (5, 5), 0)
        # cv2.imshow("blur1", blur)
        thresh = cv2.adaptiveThreshold(blur, 255, 1, 1, 11, 2)
        thresh = cv2.bitwise_not(thresh)
        _, thresh = cv2.threshold(thresh, 150, 255, cv2.THRESH_BINARY)
        # cv2.imshow("thresh1", thresh)

        threshHeight = thresh.shape[0]
        threshWidth = thresh.shape[1]
        height = threshHeight // 9
        width = threshWidth // 9
        flag = 0
        sudoku_grid = []
        for i in range(9):
            sudoku_grid.append(['0', '0', '0', '0', '0', '0', '0', '0', '0'])

        height_offset = math.floor(height / 10)
        width_offset = math.floor(width / 10)

        for i in range(9):
            for j in range(9):
                single_cell = thresh[height * i + height_offset:height * (i + 1) - height_offset,
                              width * j + width_offset:width * (j + 1) - width_offset]
                # if i == 0 and j == 3:
                # cv2.imshow("before removing lines", single_cell)
                ratio = 0.6
                while np.sum(single_cell[0]) <= (1 - ratio) * single_cell.shape[1] * 255:
                    single_cell = single_cell[1:]
                while np.sum(single_cell[:, -1]) <= (1 - ratio) * single_cell.shape[1] * 255:
                    single_cell = np.delete(single_cell, -1, 1)
                while np.sum(single_cell[:, 0]) <= (1 - ratio) * single_cell.shape[0] * 255:
                    single_cell = np.delete(single_cell, 0, 1)
                while np.sum(single_cell[-1]) <= (1 - ratio) * single_cell.shape[0] * 255:
                    single_cell = single_cell[:-1]

                single_cell = cv2.bitwise_not(single_cell)
                # if i == 0 and j == 3:
                # cv2.imshow("digit_before", single_cell)
                single_cell = extract_digit(single_cell)
                # if i == 0 and j == 3:
                # cv2.imshow("digit_after", single_cell)
                training_image_size = 28
                single_cell = cv2.resize(single_cell, (training_image_size, training_image_size))
                if single_cell.sum() >= training_image_size ** 2 * 255 - training_image_size * 255:
                    sudoku_grid[i][j] = '0'
                    continue

                center_height = single_cell.shape[0] // 2
                center_width = single_cell.shape[1] // 2
                x_start = center_height // 2
                x_end = center_height // 2 + center_height
                y_start = center_width // 2
                y_end = center_width // 2 + center_width
                center_region = single_cell[x_start:x_end, y_start:y_end]

                if center_region.sum() >= center_width * center_height * 255 - 255:
                    sudoku_grid[i][j] = '0'
                    continue
                _, single_cell = cv2.threshold(single_cell, 200, 255, cv2.THRESH_BINARY)
                single_cell = single_cell.astype(np.uint8)
                single_cell = cv2.bitwise_not(single_cell)
                sx, sy = get_best_shift(single_cell)
                # print(sx, sy)
                single_cell = shiftTransform(single_cell, sx, sy)
                single_cell = cv2.bitwise_not(single_cell)
                # if i==0 and j==3:
                # cv2.imshow("digit", single_cell)

                # cv2.imshow(name, single_cell)
                single_cell = prepare(single_cell)

                prediction = model.predict([single_cell])
                sudoku_grid[i][j] = str(np.argmax(prediction[0]) + 1)

        # if (sudoku_grid[0] == ['0', '0', '0', '2', '6', '0', '7', '0', '1']):
        print(sudoku_grid)
        unsolved_sudoku_grid = copy.deepcopy(sudoku_grid)
        if (not prev_sudoku is None) and (are_two_matrices_equal(prev_sudoku, sudoku_grid)):
            if (bestFirstSearch.is_sudoku_filled(sudoku_grid)):
                print("case 1")
                prev_warp = show_solution_on_image(prev_warp, prev_sudoku, unsolved_sudoku_grid)
        else:
            solved_sudoku_grid = bestFirstSearch.solve(sudoku_grid)
            if (solved_sudoku_grid != False and bestFirstSearch.is_sudoku_filled(sudoku_grid)):
                print("case 2")
                prev_warp = show_solution_on_image(prev_warp, solved_sudoku_grid, unsolved_sudoku_grid)
                prev_sudoku = copy.deepcopy(solved_sudoku_grid)

        if solved_sudoku_grid!=False:
            # cv2.imshow("Before putting", prev_warp)
            solution_image = cv2.warpPerspective(prev_warp, M, (frame.shape[1], frame.shape[0])
                                                 , flags=cv2.WARP_INVERSE_MAP)

            a=solution_image.sum(axis=-1, keepdims=True)
            print(a.shape)
            print(solution_image.shape)
            print(frame.shape)
            temp = np.where(a!= 0, solution_image, frame)
            new_image = np.copy(temp)
            new_image = cv2.resize(new_image, (1280, 720))

            cv2.imshow("Frame", new_image)

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
