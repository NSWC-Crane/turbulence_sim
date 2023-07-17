import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2
import os
from scipy.signal import correlate


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
                            'img_array': img_unsub,
                            'fft_unsub': fft_unsub},
                           ignore_index=True)

    ##########################
    # Set output directory for different output storage
    ##########################
    # If directory doesn't exist, make it
    old_output = output
    # output_final = os.path.join(old_output, 'Final_Images')
    # if not os.path.exists(output_final):
    #     os.mkdir(output_final)
    zoom_str = os.path.basename(os.path.dirname(directory))
    # output_exam = os.path.join(os.path.dirname(os.path.dirname(directory)), zoom_str+'Examine_sharpness')
    # if not os.path.exists(output_exam):
    #     os.mkdir(output_exam)

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
    shiftedoutput = os.path.join(old_output, str(X) + '_Shifted_Images')
    if not os.path.exists(shiftedoutput):
        os.mkdir(shiftedoutput)
    correlationoutput = os.path.join(old_output, str(X) + '_correlation_Images')
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
    croppedoutput = os.path.join(old_output, str(X) + '_Cropped_Images')
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


def stitch(directory):
    height_min = np.inf
    width_min = np.inf
    df = pd.DataFrame(columns=['directory', 'range', 'img_num', 'filename', 'image', 'full_path'])
    # Stitch imgs together into one image.
    # First things first, find the cropped images in the directory,
    # this is very specifically keyed to directory structure.....
    dirname, basename = os.path.split(directory)
    stitchoutput = os.path.join(dirname, 'Stitched_Images_'+basename)
    if not os.path.exists(stitchoutput):
        os.mkdir(stitchoutput)
    num_dir = 0
    for filename in os.listdir(directory):
        d_f = os.path.join(directory, filename)
        # print(d_f)
        dirname1, basename1 = os.path.split(d_f)
        # if there is a directory in there
        if os.path.isdir(d_f):
            for filename2 in os.listdir(d_f):
                d_f_2 = os.path.join(d_f, filename2)
                # if there is a directory in there
                if os.path.isdir(d_f_2):
                    for filename3 in os.listdir(d_f_2):
                        d_f_3 = os.path.join(d_f_2, filename3)
                        # if there is a directory in there
                        if os.path.isdir(d_f_3):
                            dirname3, basename3 = os.path.split(d_f_3)
                            # If it is the cropped images directory
                            if basename3 == '20_Cropped_Images':
                                for filename in os.listdir(d_f_3):
                                    f = os.path.join(d_f_3, filename)
                                    # if it is a file
                                    if os.path.isfile(f):
                                        # read in the file (image)
                                        img = cv2.imread(f)
                                        # convert to grayscale
                                        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                                        height, width = img_gray.shape
                                        # Check to see what the smallest images are
                                        if height < height_min:
                                            height_min = height
                                        if width < width_min:
                                            width_min = width
                                        # print(height, width)
                                        # Store image data in a dataframe
                                        # print(filename)
                                        img_num = filename.split('_')[-1]
                                        img_num = img_num.split('.')[0]
                                        # print(img_num)
                                        # print(f)
                                        df = df.append({'directory': d_f_3,
                                                        'range': basename1,
                                                        'img_num': img_num,
                                                        'filename': filename,
                                                        'image': img_gray,
                                                       'full_path': f},
                                                       ignore_index=True)
    # print()
    # print(height_min, width_min)
    # print(df['range'])

    numbers = df['img_num'].unique()
    for number in numbers:
        print(number)
        condition = (df['img_num'] == number)
        stitch_time = 0
        for path in df[condition]['full_path']:
            print(path)
            # read in the file (image)
            img = cv2.imread(path)
            # convert to grayscale
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_cropped = img_gray[0:height_min,0:width_min]
            if stitch_time == 0:
                stitched_img = img_cropped
            else:
                stitched_img = cv2.hconcat([stitched_img, img_cropped])
            stitch_time = stitch_time + 1
        # plt.close('all')
        # plt.imshow(stitched_img)
        # plt.show()
        cv2.imwrite(os.path.join(stitchoutput, "stiched_" + number+'.png'), stitched_img)
        print()


##########################
# User provides directory to work in
##########################
directory = r'C:\Users\david.j.carr2\OneDrive - US Navy-flankspeed\Documents\JSSAP\IRSDS\TurbRecon_TCI-master\2fp_test\fp2'
# Toggle cropping images
generate = False

if generate:
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

stitch(directory)
