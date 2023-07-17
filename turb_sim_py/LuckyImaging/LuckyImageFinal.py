import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2
import os
from scipy.ndimage import sobel
from scipy.signal import correlate
from scipy.signal import fftconvolve
from PIL import Image
import time
import math
from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA


def allign_imgs(train_img, query_img1):
    # Input: Takes two greyscale images
    # the references image (what you are aligning to) is called train image
    # the query image is what you want to align to the reference image
    # Output: Returns the query image aligned to the training image and offset matrix for alignment
    height, width = train_img.shape

    ##########################
    # Find location of alignment
    ##########################
    # get rid of the averages, otherwise the results are not good
    # source: https://stackoverflow.com/questions/24768222/how-to-detect-a-shift-between-images
    query_img1_avg_sub = query_img1 - np.mean(query_img1)
    train_avg_sub = train_img - np.mean(train_img)

    # Do for image
    # Finds location of max correlation
    # Stores that location in index1
    corr_train_1 = correlate(train_avg_sub, query_img1_avg_sub, mode='same')
    max_cor1 = np.max(corr_train_1)
    index1 = np.where(corr_train_1 == max_cor1)

    # Finds center of image
    center_x, center_y = np.array(train_img.shape) // 2

    ##########################
    # Apply alignment
    ##########################
    # Check if the max value occurs at more than one space
    if len(index1[0]) > 1:
        print('allign_imgs')
        print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!WARNING!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        print('There are more than one maximum location. By default the code takes the first.')
        print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!WARNING!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        # print(index1)

    # Do for img
    # Compute offset with affine matrix
    offset_y1 = index1[0][0] - center_x
    offset_x1 = index1[1][0] - center_y
    # If the offset from center is more than 20 pixels, search for the next best offset that is less far away
    # 20 pixels was arbitrarilly decided, deppending on your data this could be more or less
    if np.abs(offset_y1) > 20 or np.abs(offset_x1) > 20:
        print('large offset, setting finding smaller offset...')
        sorted = np.sort(corr_train_1, axis=None)[::-1]
        print('offsets before:', offset_x1, offset_y1)
        while np.abs(offset_y1) > 20 or np.abs(offset_x1) > 20:
            next_max = sorted[1]
            sorted = sorted[1:]
            index1 = np.where(corr_train_1 == next_max)
            offset_y1 = index1[0][0] - center_x
            offset_x1 = index1[1][0] - center_y
            print('offsets after:', offset_x1, offset_y1)

    # Use offset values to make transformation matrix
    warp_matrix1 = np.float32([[1, 0, offset_x1], [0, 1, offset_y1]])
    # print(warp_matrix1)
    # Apply the matrix
    query_1_shifted = cv2.warpAffine(query_img1, warp_matrix1, (width, height))

    return query_1_shifted, warp_matrix1, corr_train_1


# def crop_imgs_old(x_array, y_array, image):
#     # takes an array of x and y offsets and finds biggest (absolute value) of the array and uses that to crop image
#     # Returns cropped image
#
#     # Find biggest offset
#     max_x = np.max(np.abs(x_array))
#     index = np.where(np.abs(x_array) == max_x)
#     # if len(index[0]) > 1:
#         # print('crop imgs x')
#         # print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!WARNING!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
#         # print('There are more than one maximum location. By default the code takes the first.')
#         # print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!WARNING!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
#         # print(x_array[index])
#         # print()
#     index_x = index[0][0]
#     biggest_y = int(x_array[index_x])
#     # print(biggest_y)
#
#     max_y = np.max(np.abs(y_array))
#     index = np.where(np.abs(y_array) == max_y)
#     # if len(index[0]) > 1:
#         # print('crop imgs y')
#         # print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!WARNING!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
#         # print('There are more than one maximum location. By default the code takes the first.')
#         # print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!WARNING!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
#         # print(y_array[index])
#         # print()
#     index_y = index[0][0]
#     biggest_x = int(y_array[index_y])
#     # print(biggest_x)
#
#     # Depending on offset, crop images
#     if biggest_x >= 0:
#         biggest_x = biggest_x + 1
#         if biggest_y >= 0:
#             biggest_y = biggest_y + 1
#             image_cropped = image[biggest_x:, biggest_y:]
#         else:
#             biggest_y = biggest_y - 1
#             image_cropped = image[biggest_x:, :biggest_y]
#     else:
#         biggest_x = biggest_x - 1
#         if biggest_y >= 0:
#             biggest_y = biggest_y + 1
#             image_cropped = image[:biggest_x, biggest_y:]
#         else:
#             biggest_y = biggest_y - 1
#             image_cropped = image[:biggest_x, :biggest_y]
#     return image_cropped


def crop_imgs(image_data):
    # takes a grey image
    # Returns cropped image (where image is not 0) and box used to make it
    # cropBox is used in crop_coords

    # Finds where image is not 0
    non_empty_columns = np.where(image_data.max(axis=0) > 0)[0]
    non_empty_rows = np.where(image_data.max(axis=1) > 0)[0]

    # Find edges of where it is not 0
    cropBox = (min(non_empty_rows), max(non_empty_rows), min(non_empty_columns), max(non_empty_columns))
    # Return cropped image
    image_cropped = image_data[cropBox[0]:cropBox[1] + 1, cropBox[2]:cropBox[3] + 1]
    return image_cropped, cropBox


def crop_coords(image_data, lr, rr, lc, rc):
    # Takes an image and a cropBox output from crop_imgs function
    # Outputs a cropped image
    return image_data[lr:rr+1, lc:rc + 1]


def workOnDirectory(directory, output, textfile):
    # Function created to run through every directory inside directory provided for image analysis
    # Input: directory (path to folder) to work on and ouput directory to save data to
    # Returns: Nothing but saves a lot of plots as part of function
    ###################################
    # FOR VICKY: This is the function that is most tooled to my system, you will probably have to
    # tweak this to fit with what you are using. However the code in here should help you align
    # your images. It calls the functions above to do work on the data sets.
    #
    # I have tried to comment everything as best as I can. Please let me know if you want to meet
    # on teams to go through anything. I'd rather you waste as little time trying to understand
    # my spaghetti code
    ###################################
    print()
    print('Working on:', directory)

    # Create dataframe for data storage
    df = pd.DataFrame(columns=['filename', 'sharpness', 'array_PIL', 'laplace_var', 'array_cv'])
    # iterate over files in directory to find sharpest images
    for filename in os.listdir(directory):
        # Add file to directory path
        f = os.path.join(directory, filename)

        # checking if it is a file, if it is do work
        if os.path.isfile(f):
            ########################
            # Find sharpest image
            ########################
            # Several ways to find sharpest image...
            # Method 1
            # Source: https://stackoverflow.com/questions/6646371/detect-which-image-is-sharper
            img_c = cv2.imread(f)
            img = cv2.cvtColor(img_c, cv2.COLOR_BGR2GRAY)
            img_mean = img - np.mean(img)
            img_unsub = img
            #######################
            # Testing to see if mean subtracting makes a difference
            #######################
            # Unsub and mean subtracted are NOT identical
            # print(img_mean)
            # print(img_unsub)
            img = img_mean

            gy, gx = np.gradient(img)
            gnorm = np.sqrt(gx**2 + gy**2)
            sharpness = np.average(gnorm)

            gy, gx = np.gradient(img_unsub)
            gnorm = np.sqrt(gx ** 2 + gy ** 2)
            sharpness_unsub = np.average(gnorm)
            #######################
            # Testing to see if mean subtracting makes a difference
            #######################
            # Unsub and mean subtracted are identical
            # print(sharpness)
            # print(sharpness_unsub)

            # Method 2
            # source: https://www.bogotobogo.com/python/OpenCV_Python/python_opencv3_Image_Gradient_Sobel_Laplacian_Derivatives_Edge_Detection.php
            # https://stackoverflow.com/questions/48319918/whats-the-theory-behind-computing-variance-of-an-image
            laplace = cv2.Laplacian(img, cv2.CV_64F)
            laplace_var = laplace.var()
            laplace_unsub = cv2.Laplacian(img_unsub, cv2.CV_64F)
            laplace_var_unsub = laplace_unsub.var()
            #######################
            # Testing to see if mean subtracting makes a difference
            #######################
            # Unsub and mean subtracted are identical
            # print(laplace)
            # print(laplace_unsub)
            # plt.imshow(cv2.Laplacian(img, cv2.CV_64F), cmap='gray')
            # plt.show()

            # Method 3 and 4
            fft = np.sum(np.abs(np.fft.fftshift(np.fft.fft2(img)))/(img.shape[0] * img.shape[1]))
            fft_laplace = np.sum(np.abs(np.fft.fftshift(np.fft.fft2(laplace)))/(laplace.shape[0] * laplace.shape[1]))
            fft_unsub = np.sum(np.abs(np.fft.fftshift(np.fft.fft2(img_unsub))) / (img_unsub.shape[0] * img_unsub.shape[1]))
            fft_laplace_unsub = np.sum(np.abs(np.fft.fftshift(np.fft.fft2(laplace_unsub))) / (laplace_unsub.shape[0] * laplace_unsub.shape[1]))
            #######################
            # Testing to see if mean subtracting makes a difference
            #######################
            # Unsub and mean subtracted are NOT identical
            # print(fft)
            # print(fft_unsub)

            # Unsub and mean subtracted are identical
            # print(fft_laplace)
            # print(fft_laplace_unsub)

            # Store image data in a dataframe
            df = df.append({'filename': filename,
                            'sharpness': sharpness,
                            'laplace_var': laplace_var,
                            'fft': fft,
                            'fft_laplace': fft_laplace,
                            'img_array': img,
                            'fft_unsub': fft_unsub},
                           ignore_index=True)

    ##########################
    # Set output directory for different output storage
    ##########################
    # If directory doesn't exist, make it
    old_output = output
    output_final = os.path.join(old_output, 'Final_Images')
    if not os.path.exists(output_final):
        os.mkdir(output_final)
    zoom_str = os.path.basename(os.path.dirname(directory))
    output_exam = os.path.join(os.path.dirname(os.path.dirname(directory)), zoom_str+'Examine_sharpness')
    if not os.path.exists(output_exam):
        os.mkdir(output_exam)

    ##########################
    # Derive Sharpness Score
    ##########################
    # Score is derived by ranking each image based off of its performance within each method 1-4 above.
    # Sharper images are sorted to be the first few lines in the dataframe. Each image then is assigned
    # a score based off of its location in the dataframe
    df_sorted_sharpness = df.sort_values(by='sharpness', ascending=False)
    df_sorted_laplace = df.sort_values(by='laplace_var', ascending=False)
    df_sorted_fft = df.sort_values(by='fft', ascending=False)
    df_sorted_fft_laplace = df.sort_values(by='fft_laplace', ascending=False)
    df_sorted_sharpness['sharpness_score'] = df.index
    df_sorted_laplace['laplace_score'] = df.index
    df_sorted_fft['fft_score'] = df.index
    df_sorted_fft_laplace['fft_laplace_score'] = df.index
    # Store final score in parent dataframe
    df['sharpness_score'] = df_sorted_sharpness.sort_index(axis=0)['sharpness_score']
    df['laplace_score'] = df_sorted_laplace.sort_index(axis=0)['laplace_score']
    df['fft_score'] = df_sorted_fft.sort_index(axis=0)['fft_score']
    df['fft_laplace_score'] = df_sorted_fft_laplace.sort_index(axis=0)['fft_laplace_score']

    # Lower score is better
    df['score'] = df['sharpness_score'] + df['laplace_score'] + df['fft_score'] + df['fft_laplace_score']
    # Sort by final score
    df_sorted_score = df.sort_values(by='score', ascending=True)
    # Retain image number
    df_sorted_score['img_num'] = df_sorted_score.index
    df_sorted_score = df_sorted_score.sort_values(by='score', ascending=True, ignore_index=True)
    # print(df[['filename', 'laplace_var', 'laplace_score', 'score']])

    ##########################
    # Examine different sharpness measures
    ##########################
    # This whole section generates plots to examine how score does in measureing sharpness
    # Set plot parameters
    range_str = os.path.basename(os.path.normpath(directory))
    alpha = 0.6

    ##########################
    # Examine difference between unsubtracted image and mean subtracted image for fft metric
    plt.close('all')
    fig, ax = plt.subplots(1, figsize=(10, 8))
    plt.plot(df.index, df.fft, '^', label='Sum of FFT Mean Subtracted', alpha=alpha, color='C2')
    plt.plot(df.index, df.fft_unsub, '*', label='Sum of FFT Unsubtracted', alpha=alpha, color='g')
    plt.xlabel('Img #', size=15)
    plt.ylabel('FFT of Image / Img Area', size=15)
    ax.tick_params(bottom=True, top=True, left=True, right=False, which='both', labelsize=12)
    ax.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False)
    # ax.xaxis.set_minor_locator(plt.MultipleLocator(1))
    ax.xaxis.set_major_locator(plt.MultipleLocator(1))
    ax.tick_params('both', length=6, width=1, which='major')
    ax.tick_params('both', length=3, width=1, which='minor')
    plt.legend(loc='best')
    plt.title('Range:' + range_str + ' FFTvFFTunsub Metric Comparison', size=15, y=1.04)
    # plt.show()
    plt.savefig(os.path.join(output_exam, range_str + 'examine_FFT.png'), bbox_inches='tight')

    plt.close('all')
    fig, ax = plt.subplots(1, figsize=(10, 8))
    plt.plot(df.index, df.fft/np.linalg.norm(df.fft), '^', label='Sum of FFT Mean Subtracted', alpha=alpha, color='C2')
    plt.plot(df.index, df.fft_unsub/np.linalg.norm(df.fft_unsub), '*', label='Sum of FFT Unsubtracted', alpha=alpha, color='g')
    plt.xlabel('Img #', size=15)
    plt.ylabel('Normalzed FFT of Image / Img Area', size=15)
    ax.tick_params(bottom=True, top=True, left=True, right=False, which='both', labelsize=12)
    ax.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False)
    # ax.xaxis.set_minor_locator(plt.MultipleLocator(1))
    ax.xaxis.set_major_locator(plt.MultipleLocator(1))
    ax.tick_params('both', length=6, width=1, which='major')
    ax.tick_params('both', length=3, width=1, which='minor')
    plt.legend(loc='best')
    plt.title('Range:' + range_str + 'Normalized FFTvFFTunsub Metric Comparison', size=15, y=1.04)
    # plt.show()
    plt.savefig(os.path.join(output_exam, range_str + 'examine_FFT_scaled.png'), bbox_inches='tight')

    ##########################
    # Examine differnce between different methods
    plt.close('all')

    fig, ax = plt.subplots(1, figsize=(10, 8))
    # print(df.fft / np.linalg.norm(df.fft))
    # print(df.fft_unsub / np.linalg.norm(df.fft_unsub))
    # print((df.fft / np.linalg.norm(df.fft)) - (df.fft_unsub / np.linalg.norm(df.fft_unsub)))
    # stopp

    plt.plot(df.index, df.sharpness/np.linalg.norm(df.sharpness), 'o', label='Avg of normalized gradiant', alpha=alpha)
    plt.plot(df.index, df.laplace_var/np.linalg.norm(df.laplace_var), 's', label='Variance of Laplacian', alpha=alpha)
    plt.plot(df.index, df.fft/np.linalg.norm(df.fft), '^', label='Sum of FFT', alpha=alpha)
    plt.plot(df.index, df.fft_laplace/np.linalg.norm(df.fft_laplace), 'v', label='Sum of FFT of Laplacian', alpha=alpha)
    plt.plot(df.index, df.score/1000 + 0.18, '-', label='Score/1000 + 0.18', alpha=alpha, color='black')

    col = 'sharpness'
    sort = df.sort_values(by=[col], ascending=False)[col].values
    first = sort[0]
    second = sort[1]
    third = sort[2]
    con = (df[col] == first)
    plt.plot(df.index[con], df[col][con]/np.linalg.norm(df[col]), '*', ms=14, markeredgecolor='black',
             markerfacecolor='None', label='First')
    con = (df[col] == second)
    plt.plot(df.index[con], df[col][con] / np.linalg.norm(df[col]), 's', ms=14, markeredgecolor='black',
             markerfacecolor='None', label='Second')
    con = (df[col] == third)
    plt.plot(df.index[con], df[col][con] / np.linalg.norm(df[col]), 'o', ms=14, markeredgecolor='black',
             markerfacecolor='None', label='Third')

    col = 'sharpness'
    sort = df.sort_values(by=[col], ascending=False)[col].values
    first = sort[0]
    second = sort[1]
    third = sort[2]
    con = (df[col] == first)
    plt.plot(df.index[con], df[col][con] / np.linalg.norm(df[col]), '*', ms=14, markeredgecolor='C0',
             markerfacecolor='None')
    con = (df[col] == second)
    plt.plot(df.index[con], df[col][con] / np.linalg.norm(df[col]), 's', ms=14, markeredgecolor='C0',
             markerfacecolor='None')
    con = (df[col] == third)
    plt.plot(df.index[con], df[col][con] / np.linalg.norm(df[col]), 'o', ms=14, markeredgecolor='C0',
             markerfacecolor='None')

    col = 'laplace_var'
    sort = df.sort_values(by=[col], ascending=False)[col].values
    first = sort[0]
    second = sort[1]
    third = sort[2]
    con = (df[col] == first)
    plt.plot(df.index[con], df[col][con] / np.linalg.norm(df[col]), '*', ms=14, markeredgecolor='C1',
             markerfacecolor='None')
    con = (df[col] == second)
    plt.plot(df.index[con], df[col][con] / np.linalg.norm(df[col]), 's', ms=14, markeredgecolor='C1',
             markerfacecolor='None')
    con = (df[col] == third)
    plt.plot(df.index[con], df[col][con] / np.linalg.norm(df[col]), 'o', ms=14, markeredgecolor='C1',
             markerfacecolor='None')

    col = 'fft'
    sort = df.sort_values(by=[col], ascending=False)[col].values
    first = sort[0]
    second = sort[1]
    third = sort[2]
    con = (df[col] == first)
    plt.plot(df.index[con], df[col][con] / np.linalg.norm(df[col]), '*', ms=14, markeredgecolor='C2',
             markerfacecolor='None')
    con = (df[col] == second)
    plt.plot(df.index[con], df[col][con] / np.linalg.norm(df[col]), 's', ms=14, markeredgecolor='C2',
             markerfacecolor='None')
    con = (df[col] == third)
    plt.plot(df.index[con], df[col][con] / np.linalg.norm(df[col]), 'o', ms=14, markeredgecolor='C2',
             markerfacecolor='None')

    col = 'fft_laplace'
    sort = df.sort_values(by=[col], ascending=False)[col].values
    first = sort[0]
    second = sort[1]
    third = sort[2]
    con = (df[col] == first)
    plt.plot(df.index[con], df[col][con] / np.linalg.norm(df[col]), '*', ms=14, markeredgecolor='C3',
             markerfacecolor='None')
    con = (df[col] == second)
    plt.plot(df.index[con], df[col][con] / np.linalg.norm(df[col]), 's', ms=14, markeredgecolor='C3',
             markerfacecolor='None')
    con = (df[col] == third)
    plt.plot(df.index[con], df[col][con] / np.linalg.norm(df[col]), 'o', ms=14, markeredgecolor='C3',
             markerfacecolor='None')

    plt.xlabel('Img #', size=15)
    plt.ylabel('Normalized Metric Score', size=15)
    ax.tick_params(bottom=True, top=True, left=True, right=False, which='both', labelsize=12)
    ax.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False)
    # ax.xaxis.set_minor_locator(plt.MultipleLocator(1))
    ax.xaxis.set_major_locator(plt.MultipleLocator(1))
    ax.tick_params('both', length=6, width=1, which='major')
    ax.tick_params('both', length=3, width=1, which='minor')
    plt.legend(loc='best')
    plt.title('Range:' + range_str + ' Normalized Metric Comparison', size=15, y=1.04)
    plt.savefig(os.path.join(output_exam, range_str + 'examine_sharpness_methods.png'), bbox_inches='tight')

    ##########################
    # Examine differnce between different methods but sorted by score
    plt.close('all')
    df_old = df
    df = df_sorted_score

    fig, ax = plt.subplots(1, figsize=(10, 8))
    # print(df.fft / np.linalg.norm(df.fft))
    # print(df.fft_unsub / np.linalg.norm(df.fft_unsub))
    # print((df.fft / np.linalg.norm(df.fft)) - (df.fft_unsub / np.linalg.norm(df.fft_unsub)))
    # stopp

    plt.plot(df.index, df.sharpness / np.linalg.norm(df.sharpness), 'o', label='Avg of normalized gradiant',
             alpha=alpha)
    plt.plot(df.index, df.laplace_var / np.linalg.norm(df.laplace_var), 's', label='Variance of Laplacian', alpha=alpha)
    plt.plot(df.index, df.fft / np.linalg.norm(df.fft), '^', label='Sum of FFT', alpha=alpha)
    plt.plot(df.index, df.fft_laplace / np.linalg.norm(df.fft_laplace), 'v', label='Sum of FFT of Laplacian',
             alpha=alpha)
    plt.plot(df.index, df.score/1000 + 0.18, '-', label='Score/1000 + 0.18', alpha=alpha, color='black')

    plt.xlabel('Img #', size=15)
    plt.ylabel('Normalized Metric Score', size=15)
    ax.tick_params(bottom=True, top=True, left=True, right=False, which='both', labelsize=12)
    ax.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False)
    # ax.xaxis.set_minor_locator(plt.MultipleLocator(1))
    ax.xaxis.set_major_locator(plt.MultipleLocator(1))
    ax.tick_params('both', length=6, width=1, which='major')
    ax.tick_params('both', length=3, width=1, which='minor')
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(df.img_num.values)
    plt.legend(loc='best')
    plt.title('Range:' + range_str + ' Normalized Metric Comparison Score Sorted', size=15, y=1.04)
    plt.savefig(os.path.join(output_exam, range_str + 'examine_sharpness_methods_score_sorted.png'), bbox_inches='tight')
    # stopp
    df = df_old
    ########################################################################################
    # End plot generation
    ########################################################################################

    # Find Best Image, set that as the training image
    df_sorted = df.sort_values(by='score', ascending=True)
    im1_name = df_sorted.iloc[0]['filename']
    print('Best Image:', im1_name)
    img1_color = cv2.imread(os.path.join(directory, im1_name))
    train_color = img1_color
    # Convert to grayscale.
    train_img = cv2.cvtColor(train_color, cv2.COLOR_BGR2GRAY)
    height, width = train_img.shape

    ########################################################################################
    # For X images, do work
    ########################################################################################
    # Set parameters
    X = int(20)  # number of images to use, 20 is all right now
    num_patch_comb = int(1)  # number of each image to use in each patch, 1 takes the best image and makes that the patch
    patchsize_pixels = 25  # pixel size of each patch
    overlap = 5  # pixel overlap of patches, 0 is no overlap

    # Get list of filenames
    files = df_sorted['filename'][:X]

    # Make additional output directories for later analysis
    shiftedoutput = os.path.join(old_output, str(X)+'_Shifted_Images')
    if not os.path.exists(shiftedoutput):
        os.mkdir(shiftedoutput)
    correlationoutput = os.path.join(old_output, str(X)+'_correlation_Images')
    if not os.path.exists(correlationoutput):
        os.mkdir(correlationoutput)

    # To build shifted images and box for cropping array need to know image total size
    leftrow = 0
    rightrow = width
    leftcolumn = 0
    rightcolumn = height
    # For each file in directory
    for filename in files:
        f = os.path.join(directory, filename)
        # if it is a file
        if os.path.isfile(f):
            # read in the file (image)
            img = cv2.imread(f)
            # convert to grayscale
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # print(f)

            # allign the images you are reading in to the best image according to score
            img_shifted, img_matrix, corr = allign_imgs(train_img, img_gray)

            # plot to correlation graph for future examination
            plt.close('all')
            plt.imshow(corr)
            plt.savefig(os.path.join(correlationoutput, "correlation_" + filename), bbox_inches='tight')
            cv2.imwrite(os.path.join(shiftedoutput, "shifted_" + filename), img_shifted)

            # Crop the image after alignment
            img_cropped, cropBox = crop_imgs(img_shifted)
            lr, rr, lc, rc = cropBox
            # If the cropBox is smaller than the previous smallest in any dimension, keep it as the new smallest
            if lr > leftrow:
                leftrow = lr
            if rr < rightrow:
                rightrow = rr
            if lc > leftcolumn:
                leftcolumn = lc
            if rc < rightcolumn:
                rightcolumn = rc

    # After going through every image, create an empty image array the size of the cropBox
    img_array = np.empty((0, rightrow + 1 - leftrow, rightcolumn + 1 - leftcolumn))
    # Make a directory for output
    croppedoutput = os.path.join(old_output, str(X)+'_Cropped_Images')
    if not os.path.exists(croppedoutput):
        os.mkdir(croppedoutput)
    # Build cropped images and array of images
    # For each file
    for filename in os.listdir(shiftedoutput):
        f = os.path.join(shiftedoutput, filename)
        # if it is a file
        if os.path.isfile(f):
            # create the cropped image from that file
            img_shifted = cv2.imread(f, cv2.COLOR_BGR2GRAY)
            img_cropped = crop_coords(img_shifted, leftrow, rightrow, leftcolumn, rightcolumn)
            # save it
            cv2.imwrite(os.path.join(croppedoutput, "cropped_" + filename), img_cropped)
            # and add it to an array of images for either mean or median combining
            img_array = np.append(img_array, [img_cropped], axis=0)

    # Median combine and compute sharpness for total combination
    med_comb = np.nanmedian(img_array, axis=0)
    mean_comb = np.nanmean(img_array, axis=0)
    # Save image for later examination
    cv2.imwrite(os.path.join(output_final, "med_comb_%i.png" % X), med_comb)
    cv2.imwrite(os.path.join(output_final, "mean_comb_%i.png" % X), mean_comb)
    # use previous methods to examine if image got sharper
    gy, gx = np.gradient(med_comb)
    gnorm = np.sqrt(gx ** 2 + gy ** 2)
    sharpness_med_20 = np.average(gnorm)
    laplace_var_med_20 = cv2.Laplacian(med_comb, cv2.CV_64F).var()

    ############################
    # Try to do patch based combination and make a frankenstein's image
    ############################
    # make directory for output
    patch_output = os.path.join(old_output, str(X) + '_Patched_Images')
    if not os.path.exists(patch_output):
        os.mkdir(patch_output)

    # set up frankenstein image as an array of zeros the same size as the images
    frank_img = np.zeros((img_array.shape[1], img_array.shape[2]))
    # experiment with multiple images per patch
    frank_img_multi = np.zeros((img_array.shape[1], img_array.shape[2]))
    crop_height, crop_width = img_array.shape[1], img_array.shape[2]

    # The next loop goes through image in a step by step way
    # It examines patch 1 in the top left, then moves to the right a number of pixels equal to the patch size
    # then moves back a number of pixels equal to the overlap provided above
    step = patchsize_pixels  # set step parameter from user provided input above
    rows = np.arange(0, crop_height, step-overlap)  # set the top row of each patch as an array that goes from 0 to the edge in steps of step - overlap
    rows = np.append(rows, crop_height)  # then set the last value as the edge of the image
    columns = np.arange(0, crop_width, step-overlap)  # do the same for columns
    columns = np.append(columns, crop_width)

    # Loop through each patch and create Frankenstien img
    i = 0
    # for each value in rows
    while i < (len(rows) - 1):
        j = 0
        # and each value in columns
        while j < (len(columns) - 1):
            # outline your patch, being careful not to go outside the bounds of your image
            leftrow = rows[i]
            rightrow = rows[i] + step
            if rightrow > crop_height:
                rightrow = rows[i + 1]
            leftcolumn = columns[j]
            rightcolumn = columns[j] + step
            if rightcolumn > crop_width:
                rightcolumn = columns[j + 1]
            patchs = img_array[:, leftrow:rightrow, leftcolumn:rightcolumn]

            # create empty arrays for the values of each patch as determined by the four methods
            laplace_var_patch_array = np.array([])
            sharpness_patch_array = np.array([])
            fft_patch_array = np.array([])
            fft_laplace_patch_array = np.array([])
            # for each patch
            for patch in patchs:
                # Mean subtract
                patch_med_sub = patch - np.mean(patch)
                # and compute the sharpness of each patch
                # Method 1
                gy, gx = np.gradient(patch_med_sub)
                gnorm = np.sqrt(gx ** 2 + gy ** 2)
                sharpness_patch = np.average(gnorm)
                sharpness_patch_array = np.append(sharpness_patch_array, sharpness_patch)

                # Method 2
                laplace_patch = cv2.Laplacian(patch_med_sub, cv2.CV_64F)
                laplace_var_patch = laplace_patch.var()
                laplace_var_patch_array = np.append(laplace_var_patch_array, laplace_var_patch)

                # Method 3
                fft_patch = np.sum(np.abs(np.fft.fftshift(np.fft.fft2(patch_med_sub))) /
                             (patch_med_sub.shape[0] * patch_med_sub.shape[1]))
                fft_patch_array = np.append(fft_patch_array, fft_patch)

                fft_laplace_patch = np.sum(np.abs(np.fft.fftshift(np.fft.fft2(laplace_patch))) /
                                     (laplace_patch.shape[0] * laplace_patch.shape[1]))
                fft_laplace_patch_array = np.append(fft_laplace_patch_array, fft_laplace_patch)

            # Make df of different patch scores
            d = {'sharpness': sharpness_patch_array,
                 'laplace_var': laplace_var_patch_array,
                 'fft': fft_patch_array,
                 'fft_laplace': fft_laplace_patch_array}
            df_patch = pd.DataFrame(data=d)

            ##########################
            # Derive Score of each patch
            ##########################
            df_sorted_sharpness = df_patch.sort_values(by='sharpness', ascending=False)
            df_sorted_laplace = df_patch.sort_values(by='laplace_var', ascending=False)
            df_sorted_fft = df_patch.sort_values(by='fft', ascending=False)
            df_sorted_fft_laplace = df_patch.sort_values(by='fft_laplace', ascending=False)
            df_sorted_sharpness['sharpness_score'] = df_patch.index
            df_sorted_laplace['laplace_score'] = df_patch.index
            df_sorted_fft['fft_score'] = df_patch.index
            df_sorted_fft_laplace['fft_laplace_score'] = df_patch.index
            df_patch['sharpness_score'] = df_sorted_sharpness.sort_index(axis=0)['sharpness_score']
            df_patch['laplace_score'] = df_sorted_laplace.sort_index(axis=0)['laplace_score']
            df_patch['fft_score'] = df_sorted_fft.sort_index(axis=0)['fft_score']
            df_patch['fft_laplace_score'] = df_sorted_fft_laplace.sort_index(axis=0)['fft_laplace_score']
            # Lower score is better
            df_patch['score'] = df_patch['sharpness_score'] + df_patch['laplace_score'] + df_patch['fft_score'] + df_patch['fft_laplace_score']

            # Find Best Patch
            df_patch_sorted = df_patch.sort_values(by='score', ascending=True)
            best_patch = patchs[df_patch_sorted.index[0], :, :]

            # sort the array of patches by best score
            patch_img_array = patchs[np.argsort(df_patch['score'])[:num_patch_comb], :, :]
            # median combine them for the multi image per patch experiment
            med_comb_patch = np.nanmedian(patch_img_array, axis=0)

            # Because there is sometimes overlap between patches we need to figure out how to determine what happens in the overlaping areaa
            sum_img = np.sum(frank_img[leftrow:rightrow, leftcolumn:rightcolumn])
            # If there is nothing in the array (no overlap) then the best patch is set to be the patch
            if sum_img == 0:
                frank_img[leftrow:rightrow, leftcolumn:rightcolumn] = best_patch
            # If there is something, then everywhere it isn't 0, average it with what is there already (aka: overlap is averaged)
            else:
                working_frank_img = frank_img[leftrow:rightrow, leftcolumn:rightcolumn]
                temp_i = 0
                # for each row in the frank_img
                for row in working_frank_img:
                    # find where it isn't 0
                    nonzero_frank_img = (row != 0)

                    # The best batch is an average of whats there and the best patch we are adding
                    best_patch[temp_i, :][nonzero_frank_img] = np.floor(np.mean(
                        np.array([row[nonzero_frank_img], best_patch[temp_i, :][nonzero_frank_img]]), axis=0))

                    temp_i = temp_i + 1
                # store the final patch is the frank image
                frank_img[leftrow:rightrow, leftcolumn:rightcolumn] = best_patch

            # store the final patch made of multiple patches in the frank multi patch image
            frank_img_multi[leftrow:rightrow, leftcolumn:rightcolumn] = med_comb_patch

            j = j + 1
        i = i + 1

    # cv2.imwrite(os.path.join(
    #     patch_output,
    #     'patch_'+str(num_patch_comb)+'_perpatch_' + str(X) + 'images_' + str(step) + 'sizesteps.png'),
    #             frank_img_multi)

    # Save final franken image
    cv2.imwrite(os.path.join(patch_output, "patch_1perpatch_" + str(X) + 'images_' + str(step) + 'sizesteps_overlap' + str(overlap) + '.png'), frank_img)


##########################
# User provides directory to work in
##########################
# directory = r'C:\Users\david.j.carr2\OneDrive - US Navy-flankspeed\Documents\JSSAP\sharpest_new\z5000'
directory = r'C:\Users\david.j.carr2\OneDrive - US Navy-flankspeed\Documents\JSSAP\sharpest_new\z5000_test'

# For each file in that directory set the output variable and send the path to the main function, workOnDirectory
for filename in os.listdir(directory):
    d_f = os.path.join(directory, filename)
    # if there is a directory in there
    if os.path.isdir(d_f):
        output = os.path.join(d_f, filename+'_output')
        if not os.path.exists(output):
            os.mkdir(output)

        ouputext = os.path.join(output, 'output.txt')
        try:
            os.remove(ouputext)
        except OSError:
            pass
        with open(ouputext, "a") as text:
            print(filename, file=text)
            workOnDirectory(d_f, output, text)

# # Examine some slices across some images for the z5000 r1000 data
# output = r'C:\Users\david.j.carr2\OneDrive - US Navy-flankspeed\Documents\JSSAP\sharpest_new\z5000Examine_sharpness'
# baseline_path = r'C:\Users\david.j.carr2\OneDrive - US Navy-flankspeed\Documents\JSSAP\sharpest_new\z5000\baseline_z5000_r1000.png'
# baseline_color = cv2.imread(baseline_path)
# baseline_gray = cv2.cvtColor(baseline_color, cv2.COLOR_BGR2GRAY)
# baseline_gray = baseline_gray - np.mean(baseline_gray)
# # Best image according to var in laplacian
# best_img_path = r'C:\Users\david.j.carr2\OneDrive - US Navy-flankspeed\Documents\JSSAP\sharpest_new\z5000_test\1000\1000_output\20_Cropped_Images\cropped_shifted_image_z05000_f48069_e02842_i09.png'
# best_img_color = cv2.imread(best_img_path)
# best_img_gray = cv2.cvtColor(best_img_color, cv2.COLOR_BGR2GRAY)
# best_img_gray = best_img_gray - np.mean(best_img_gray)
# # Patched image according to var in laplacian
# patch_img_path = r'C:\Users\david.j.carr2\OneDrive - US Navy-flankspeed\Documents\JSSAP\sharpest_new\z5000_test\1000\1000_output\20_Patched_Images\patch_1_perpatch_20images_25sizesteps.png'
# patch_img_color = cv2.imread(patch_img_path)
# patch_img_gray = cv2.cvtColor(patch_img_color, cv2.COLOR_BGR2GRAY)
# patch_img_gray = patch_img_gray - np.mean(patch_img_gray)
# # Patched image according to var in laplacian
# med_img_path = r'C:\Users\david.j.carr2\OneDrive - US Navy-flankspeed\Documents\JSSAP\sharpest_new\z5000_test\1000\1000_output\Final_Images\med_comb_3.png'
# med_img_color = cv2.imread(med_img_path)
# med_img_gray = cv2.cvtColor(med_img_color, cv2.COLOR_BGR2GRAY)
# med_img_gray = med_img_gray - np.mean(med_img_gray)
#
# height, width = patch_img_gray.shape
# bheight, bwidth = baseline_gray.shape
# baseline_gray_cropped = baseline_gray[0:(height-bheight), 0:(width - bwidth)]
#
# row = int(height/2)
#
# plt.close('all')
# fig, ax = plt.subplots()
# ax.plot(best_img_gray[row, :], '--', label='best img', zorder=20, color='blue')
# ax.plot(patch_img_gray[row, :], '-.', label='patched image', zorder=20, color='orange')
# # ax.axvline(x=25, color='red')
# # ax.axvline(x=50, color='red')
# # ax.axvline(x=75, color='red')
# # ax.axvline(x=100, color='red')
# # ax.axvline(x=125, color='red')
# # ax.axvline(x=150, color='red')
# # ax.axvline(x=175, color='red')
# # ax.axhline(y=0, color='grey')
# ax.imshow(patch_img_gray, cmap='gray', extent=[0, width, min(patch_img_gray[row, :]), max(patch_img_gray[row, :])])
#
# # ax.plot(med_img_gray[row, :], ':', label='med comb image', zorder=20)
# # ax.autoscale_view() # force auto-scale to update data limits based on scatter
# # ax.set_autoscale_on(False)
# # ax.plot(baseline_gray_cropped[row, :], color='grey', label='base line', zorder=0, alpha=0.6)
# plt.legend(loc='lower right')
# plt.savefig(os.path.join(output, 'best_v_patch_slice.png'), bbox_inches='tight')
#
# # plt.show()


