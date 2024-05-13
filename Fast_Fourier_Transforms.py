import numpy as np
import cv2

# Hàm FFT (Fast Fourier Transform) 1D
def fft(x):
    N = len(x)
    if N <= 1:
        return x
    even = fft(x[::2])
    odd = fft(x[1::2])
    factor = np.exp(-2j * np.pi * np.arange(N) / N)
    return np.concatenate([even + factor[:N//2] * odd, even + factor[N//2:] * odd])

# Hàm FFT (Fast Fourier Transform) 2D
def fft2d(image):
    # Chuyển đổi ảnh màu thành ảnh grayscale nếu cần
    if len(image.shape) == 3:  # Nếu ảnh có 3 kênh (ảnh màu)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:  # Nếu ảnh đã là ảnh grayscale
        gray_image = image
    # Làm cho kích thước của ảnh là lũy thừa của 2 để tối ưu tính toán FFT
    rows, cols = gray_image.shape
    padded_rows = int(2 ** np.ceil(np.log2(rows)))
    padded_cols = int(2 ** np.ceil(np.log2(cols)))
    padded_image = np.zeros((padded_rows, padded_cols))
    padded_image[:rows, :cols] = gray_image
    # Tính FFT 2D theo từng hàng
    fft_result = np.apply_along_axis(fft, 1, padded_image)
    # Tính FFT 2D theo từng cột
    fft_result = np.apply_along_axis(fft, 0, fft_result)
    return fft_result

# Hàm IFFT (Inverse Fast Fourier Transform) 1D
def ifft(x):
    N = len(x)
    if N <= 1:
        return x
    conjugate_x = np.conjugate(x)
    y = fft(conjugate_x) / N
    return np.conjugate(y)

# Hàm IFFT (Inverse Fast Fourier Transform) 2D
def ifft2d(fft_result):
    # Tính IFFT 2D theo từng hàng
    ifft_result = np.apply_along_axis(ifft, 1, fft_result)
    # Tính IFFT 2D theo từng cột
    ifft_result = np.apply_along_axis(ifft, 0, ifft_result)
    return ifft_result

# Bộ lọc low-pass
def low_pass_filter(fft_image, cutoff_freq):
    rows, cols = fft_image.shape
    crow, ccol = rows // 2, cols // 2
    # Tạo bộ lọc low-pass
    low_pass_mask = np.zeros((rows, cols), np.uint8)
    low_pass_mask[crow - cutoff_freq:crow + cutoff_freq, ccol - cutoff_freq:ccol + cutoff_freq] = 1
    # Áp dụng bộ lọc low-pass bằng cách nhân với bộ lọc trong miền tần số
    fft_filtered = fft_image * low_pass_mask
    return fft_filtered

# Bộ lọc high-pass
def high_pass_filter(fft_image, cutoff_freq):
    rows, cols = fft_image.shape
    crow, ccol = rows // 2, cols // 2
    # Tạo bộ lọc high-pass
    high_pass_mask = np.ones((rows, cols), np.uint8)
    high_pass_mask[crow - cutoff_freq:crow + cutoff_freq, ccol - cutoff_freq:ccol + cutoff_freq] = 0
    # Áp dụng bộ lọc high-pass bằng cách nhân với bộ lọc trong miền tần số
    fft_filtered = fft_image * high_pass_mask
    return fft_filtered

# Áp dụng bộ lọc low-pass và high-pass cho ảnh grayscale
def apply_filter_to_gray_image(image, cutoff_freq):
    fft_result = fft2d(image)
    low_pass_result = low_pass_filter(fft_result, cutoff_freq)
    high_pass_result = high_pass_filter(fft_result, cutoff_freq)
    return np.abs(ifft2d(low_pass_result)).astype(np.uint8), np.abs(ifft2d(high_pass_result)).astype(np.uint8)

# Áp dụng bộ lọc low-pass và high-pass cho ảnh màu
def apply_filter_to_rgb_image(image, cutoff_freq):
    b, g, r = cv2.split(image)
    fft_b = fft2d(b)
    fft_g = fft2d(g)
    fft_r = fft2d(r)
    low_pass_b = low_pass_filter(fft_b, cutoff_freq)
    low_pass_g = low_pass_filter(fft_g, cutoff_freq)
    low_pass_r = low_pass_filter(fft_r, cutoff_freq)
    high_pass_b = high_pass_filter(fft_b, cutoff_freq)
    high_pass_g = high_pass_filter(fft_g, cutoff_freq)
    high_pass_r = high_pass_filter(fft_r, cutoff_freq)
    low_pass_output = np.abs(ifft2d(low_pass_b)).astype(np.uint8), np.abs(ifft2d(low_pass_g)).astype(np.uint8), np.abs(ifft2d(low_pass_r)).astype(np.uint8)
    high_pass_output = np.abs(ifft2d(high_pass_b)).astype(np.uint8), np.abs(ifft2d(high_pass_g)).astype(np.uint8), np.abs(ifft2d(high_pass_r)).astype(np.uint8)
    low_pass_result = cv2.merge(low_pass_output)
    high_pass_result = cv2.merge(high_pass_output)
    return low_pass_result, high_pass_result

# Main function
if __name__ == "__main__":
    # Đọc ảnh
    image = cv2.imread('2.png')
    image = cv2.resize(image, (256, 256))  # Thay đổi kích thước ảnh thành 256x256
    # Áp dụng FFT 2D
    fft_result = fft2d(image)

    # Áp dụng bộ lọc low-pass và high-pass
    cutoff_frequency = 370  # Tần số cắt cho bộ lọc
    low_pass_result_gray, high_pass_result_gray = apply_filter_to_gray_image(image, cutoff_frequency)
    low_pass_result_color, high_pass_result_color = apply_filter_to_rgb_image(image, cutoff_frequency)

    # Hiển thị kết quả
    cv2.imshow("Original Image", image.astype(np.uint8))
    cv2.imshow("Low Pass Filtered Gray Image", low_pass_result_gray)
    cv2.imshow("High Pass Filtered Gray Image", high_pass_result_gray)
    cv2.imshow("Low Pass Filtered Color Image", low_pass_result_color)
    cv2.imshow("High Pass Filtered Color Image", high_pass_result_color)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
