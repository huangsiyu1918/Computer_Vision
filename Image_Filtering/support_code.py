# Supporting code

import math

import numpy as np
from skimage import io


def load(img_path):
    """Loads an image from a file path.

    HINT: Look up `skimage.io.imread()` function.
    HINT: Converting all pixel values to a range between 0.0 and 1.0
    (i.e. divide by 255) will make your life easier later on!

    Inputs:
        image_path: file path to the image.

    Returns:
        out: numpy array of shape(image_height, image_width, n_channels).
    """
    img = io.imread(img_path)
    out = img/255

    return out


def print_stats(img):
    """ Prints the height, width and number of channels in an image.

    Inputs:
        image: numpy array of shape(image_height, image_width, n_channels).

    Returns: none

    """
    dimensions = img.shape

    height = img.shape[0]
    width = img.shape[1]

    if (dimensions.count(3)):
        channels = img.shape[2]
    else:
        channels = 0

    print('Image Dimension    : ', dimensions)
    print('Image Height       : ', height)
    print('Image Width        : ', width)
    print('Number of Channels : ', channels)

    return None


def crop(image, x1, y1, x2, y2):
    """Crop an image based on the specified bounds. Use array slicing.

    Inputs:
        image: numpy array of shape(image_height, image_width, 3).
        (x1, y1): the coordinator for the top-left point
        (x2, y2): the coordinator for the bottom-right point


    Returns:
        out: numpy array of shape(x2 - x1, y2 - y1, 3).
    """

    out = image[y1:y2, x1:x2]

    return out


def resize(input_image, fx, fy):
    """Resize an image using the nearest neighbor method.
    Not allowed to call the matural function.
    i.e. for each output pixel, use the value of the nearest input pixel after scaling

    Inputs:
        input_image: RGB image stored as an array, with shape
            `(image_height, image_width, 3)`.
        fx (float): the resize scale on the original width.
        fy (float): the resize scale on the original height.

    Returns:
        np.ndarray: Resized image, with shape `(image_height * fy, image_width * fx, 3)`.
    """
    h, w = input_image.shape[:2]
    x_new = int(w*fx)
    y_new = int(h*fy)
    newImage = np.zeros([y_new, x_new, 3])

    for i in range(y_new):
        for j in range(x_new):
            newImage[i, j] = input_image[int(i / fy),
                                         int(j / fx)]

    out = newImage

    return out


def change_contrast(image, factor):
    """Change the value of every pixel by following

                        x_n = factor * (x_p - 0.5) + 0.5

    where x_n is the new value and x_p is the original value.
    Assumes pixel values between 0.0 and 1.0 
    If you are using values 0-255, divided by 255.

    Inputs:
        image: numpy array of shape(image_height, image_width, 3).
        factor (float): contrast adjustment

    Returns:
        out: numpy array of shape(image_height, image_width, 3).
    """

    new_image = factor * (image - 0.5) + 0.5
    out = np.clip(new_image, 0, 1)

    return out


def greyscale(input_image):
    """Convert a RGB image to greyscale. 
    A simple method is to take the average of R, G, B at each pixel.
    Or you can look up more sophisticated methods online.

    Inputs:
        input_image: RGB image stored as an array, with shape
            `(image_height, image_width, 3)`.

    Returns:
        np.ndarray: Greyscale image, with shape `(image_height, image_width)`.
    """
    R, G, B = input_image[:, :, 0], input_image[:, :, 1], input_image[:, :, 2]
    out = 0.2989 * R + 0.587 * G + 0.114 * B

    return out


def binary(grey_img, th):
    """Convert a greyscale image to a binary mask with threshold.

                  x_n = 0, if x_p < th
                  x_n = 1, if x_p > th

    Inputs:
        input_image: Greyscale image stored as an array, with shape
            `(image_height, image_width)`.
        th (float): The threshold used for binarization, and the value range is 0 to 1
    Returns:
        np.ndarray: Binary mask, with shape `(image_height, image_width)`.
    """
    h, w = grey_img.shape[:2]
    newImage = np.zeros([h, w])

    for i in range(h):
        for j in range(w):
            if grey_img[i, j] < th:
                newImage[i, j] = 0
            else:
                newImage[i, j] = 1

    out = newImage

    return out


def conv2D(image, kernel):
    """ Convolution of a 2D image with a 2D kernel. 
    Convolution is applied to each pixel in the image.
    Assume values outside image bounds are 0.

    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk). Dimensions will be odd.

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    Hi, Wi = image.shape[:2]
    Hk, Wk = kernel.shape[:2]

    kernel = np.flip(kernel)    # Flip the kernel
    output = np.zeros_like(image)            # convolution output

    r_pad = math.floor(Hk/2)
    c_pad = math.floor(Wk/2)

    image_padded = np.pad(
        image, ((r_pad, r_pad), (c_pad, c_pad)), 'constant', constant_values=(0))

    for i in range(Hi):     # Loop over every pixel of the image
        for j in range(Wi):
            # element-wise multiplication of the kernel and the image
            output[i, j] = np.sum((kernel) * image_padded[i:i+Hk, j:j+Wk])

    if np.sum(kernel) != 0:
        output = output/np.sum(kernel)
    return output


def test_conv2D():
    """ A simple test for your 2D convolution function.
        You can modify it as you like to debug your function.

    Returns:
        None
    """

    # Test code written by
    # Simple convolution kernel.
    kernel = np.array(
        [
            [1, 0, 1],
            [0, 0, 0],
            [1, 0, 0]
        ])

    # Create a test image: a white square in the middle
    test_img = np.zeros((9, 9))
    test_img[3:6, 3:6] = 1

    # Run your conv_nested function on the test image
    test_output = conv2D(test_img, kernel)

    # Build the expected output
    expected_output = np.zeros((9, 9))
    expected_output[2:7, 2:7] = 1
    expected_output[5:, 5:] = 0
    expected_output[4, 2:5] = 2
    expected_output[2:5, 4] = 2
    expected_output[4, 4] = 3

    # Test if the output matches expected output
    assert np.max(
        test_output - expected_output) < 1e-10, "Your solution is not correct."


def conv(image, kernel):
    """Convolution of a RGB or grayscale image with a 2D kernel

    Args:
        image: numpy array of shape (Hi, Wi, 3) or (Hi, Wi)
        kernel: numpy array of shape (Hk, Wk). Dimensions will be odd.

    Returns:
        out: numpy array of shape (Hi, Wi, 3) or (Hi, Wi)
    """
    # YOUR CODE HERE
    Hi, Wi = image.shape[:2]
    Hk, Wk = kernel.shape[:2]

    colour = 0

    if image.ndim < 3:
        output = conv2D(image, kernel)
        return output
    else:
        colour = image.shape[2]
        kernel = np.flip(kernel)
        output = np.zeros_like(image)            # convolution output

        r_pad = math.floor(Hk/2)
        c_pad = math.floor(Wk/2)

        image_pad = np.pad(image, ((r_pad, r_pad), (c_pad, c_pad),
                           (0, 0)), 'constant', constant_values=(0))

        for i in range(Hi):     # Loop over every pixel of the image
            for j in range(Wi):
                for k in range(colour):
                    # element-wise multiplication of the kernel and the image
                    output[i, j, k] = np.sum(
                        (kernel) * image_pad[i:i+Hk, j:j+Wk, k])

        if np.sum(kernel) != 0:
            output = output/np.sum(kernel)
        return output


def gauss2D(size, sigma):
    """Function to mimic the 'fspecial' gaussian MATLAB function.
       You should not need to edit it.

    Args:
        size: filter height and width
        sigma: std deviation of Gaussian

    Returns:
        numpy array of shape (size, size) representing Gaussian filter
    """

    x, y = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
    g = np.exp(-((x**2 + y**2)/(2.0*sigma**2)))
    return g/g.sum()


def corr(image, kernel):
    """Cross correlation of a RGB image with a 2D kernel

    Args:
        image: numpy array of shape (Hi, Wi, 3) or (Hi, Wi)
        kernel: numpy array of shape (Hk, Wk). Dimensions will be odd.

    Returns:
        out: numpy array of shape (Hi, Wi, 3) or (Hi, Wi)
    """
    kernel = np.flip(kernel)
    if image.ndim < 3:
        output = conv2D(image, kernel)
        return output
    else:
        output = conv(image, kernel)
        return output
