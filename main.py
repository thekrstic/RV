import cv2
import numpy as np
import matplotlib.pyplot as plot

def fft(image):
    img_fft = np.fft.fft2(image)
    img_fft = np.fft.fftshift(img_fft)
    magnitude = 20 * np.log(np.abs(img_fft) + 1) # izbegavanje log(0) jer je nedefinisana

    return img_fft, magnitude

def remove_specific_noise(img_fft_shifted, coordinates):
    for (x, y) in coordinates:
        img_fft_shifted[x, y] = 0
        img_fft_shifted[-x, -y] = 0  # simetricne tacke

    return img_fft_shifted

def inverse_fft(img_fft_shifted):
    img_ifft_unshifted = np.fft.ifftshift(img_fft_shifted)
    img_filtered = np.abs(np.fft.ifft2(img_ifft_unshifted))

    return img_filtered

def low_pass_filter(img_fft, center, radius):
    for x in range(img_fft.shape[0]):
        for y in range(img_fft.shape[1]):
            if (x-center[0])*(x-center[0]) + (y-center[1])*(y-center[1]) > radius*radius: # > low, < high
                img_fft[x,y] *= 0.1

    return img_fft

if __name__ == '__main__':
    lena = cv2.imread("slika_2.png", cv2.IMREAD_GRAYSCALE)
    img_fft_shifted, magnitude = fft(lena)

    plot.subplot(3, 2, 1)
    plot.title("Original")
    plot.imshow(lena, cmap='gray')

    plot.subplot(3, 2, 2)
    plot.title("Magnitude")
    plot.imshow(magnitude, cmap='gray')

    coordinates = [(231, 251), (245, 261), (266, 251), (281, 261)]
    img_fft_denoised = remove_specific_noise(img_fft_shifted, coordinates)

    img_denoised = inverse_fft(img_fft_denoised)
    plot.subplot(3, 2, 3)
    plot.title("Denoised Image")
    plot.imshow(img_denoised, cmap='gray')

    img_denoised, mag = fft(img_denoised)
    plot.subplot(3, 2, 4)
    plot.title("Denoised Magnitude")
    plot.imshow(mag, cmap='gray')

    center = (256, 256)
    radius = 256

    img_low_fft = low_pass_filter(img_fft_shifted, center, radius) # nisko propusni filter
    img_low = inverse_fft(img_low_fft)
    plot.subplot(3, 2, 5)
    plot.title("Low pass")
    plot.imshow(img_low, cmap='gray')

    img_f, img_low_mag = fft(img_low)
    plot.subplot(3, 2, 6)
    plot.title("Low mag")
    plot.imshow(img_low_mag, cmap='gray')

    plot.show()
    cv2.imwrite("denoised_image.png", img_denoised)
