import cv2
import numpy as np

# Функция для построения и сохранения гистограммы
def histogram_maker_function(image):
    image = np.array(image)
    channels = cv2.split(image)
    histograma = np.zeros((400, 512, 3), dtype=np.uint8)
    colors = {'b': (255, 0, 0), 'g': (0, 255, 0), 'r': (0, 0, 255)}

    for channel, (color_name, color) in zip(channels, colors.items()):
        hist = cv2.calcHist([channel], [0], None, [256], [0, 256])
        cv2.normalize(hist, hist, 0, 400, cv2.NORM_MINMAX)
        hist = hist.flatten()

        for i in range(1, 256):
            cv2.line(histograma, (2 * (i - 1), 400 - int(hist[i - 1])),
                     (2 * i, 400 - int(hist[i])), color, 2)

    return histograma

# 1. Арифметические операции (увеличение яркости)
def arithmetic_operations(image, value=-50):
    brightened_image = cv2.add(image, value)
    brightened_image = np.clip(brightened_image, 0, 255).astype(np.uint8)
    return brightened_image

# 2. Растяжение динамического диапазона
def stretch_dynamic_range(image):
    stretched_channels = []
    for channel in cv2.split(image):
        min_val = np.min(channel)
        max_val = np.max(channel)
        stretched_channel = ((channel - min_val) / (max_val - min_val) * 255).astype(np.uint8)
        stretched_channels.append(stretched_channel)
    stretched_image = cv2.merge(stretched_channels)
    return stretched_image

# 3. Равномерное преобразование (гистограммная эквализация)
def equalize_histogram(image):
    equalized_channels = []
    for channel in cv2.split(image):
        equalized_channel = cv2.equalizeHist(channel)
        equalized_channels.append(equalized_channel)
    equalized_image = cv2.merge(equalized_channels)
    return equalized_image

# 4. Экспоненциальное преобразование
def exponential_transformation(image, gamma=3.0):
    exponential_channels = []
    for channel in cv2.split(image):
        exponential_channel = np.uint8(np.clip((255 * (channel / 255) ** gamma), 0, 255))
        exponential_channels.append(exponential_channel)
    exponential_image = cv2.merge(exponential_channels)
    return exponential_image


# 5. Преобразование по закону Рэлея
def rayleigh_transformation(image, alpha=100):
    rayleigh_channels = []
    for channel in cv2.split(image):
        normalized_channel = channel / 255.0
        valid_mask = normalized_channel < 1.0
        rayleigh_channel = np.zeros_like(channel, dtype=np.float32)
        rayleigh_channel[valid_mask] = (
            np.sqrt(2 * alpha ** 2 * np.log(1 / (1 - normalized_channel[valid_mask])))
        )
        rayleigh_channel = np.clip(rayleigh_channel, 0, 255).astype(np.uint8)
        rayleigh_channels.append(rayleigh_channel)
    rayleigh_image = cv2.merge(rayleigh_channels)
    return rayleigh_image


# 6. Преобразование по закону степени 2/3
def power_law_transformation(image, power=2/3):
    power_law_channels = []
    for channel in cv2.split(image):
        power_law_channel = np.uint8(np.clip(255 * (channel / 255) ** power, 0, 255))
        power_law_channels.append(power_law_channel)
    power_law_image = cv2.merge(power_law_channels)
    return power_law_image

# 7. Гиперболическое преобразование
def hyperbolic_transformation(image):
    hyperbolic_channels = []
    for channel in cv2.split(image):
        hyperbolic_channel = np.uint8(np.clip(255 * (np.arcsinh(channel / 255.0)), 0, 255))
        hyperbolic_channels.append(hyperbolic_channel)
    hyperbolic_image = cv2.merge(hyperbolic_channels)
    return hyperbolic_image

# 8. Таблица поиска (LUT)
def apply_lut(image, gamma=2.0):
    lut = np.array([255 * (i / 255) ** gamma for i in range(256)], dtype=np.uint8)
    lut_image = cv2.LUT(image, lut)
    return lut_image

# Основная часть программы
if __name__ == "__main__":
    image = cv2.imread('image.png')
    cv2.imwrite('original_image.jpg', image)
    cv2.imwrite('original_histogram.png', histogram_maker_function(image))
    brightened_image = arithmetic_operations(image)

    cv2.imwrite('brightened_image.jpg', brightened_image)
    cv2.imwrite('brightened_histogram.png', histogram_maker_function(brightened_image))

    stretched_image = stretch_dynamic_range(image)
    cv2.imwrite('stretched_image.jpg', stretched_image)
    cv2.imwrite('stretched_histogram.png', histogram_maker_function(stretched_image))

    equalized_image = equalize_histogram(image)
    cv2.imwrite('equalized_image.jpg', equalized_image)
    cv2.imwrite('equalized_histogram.png', histogram_maker_function(equalized_image))

    exponential_image = exponential_transformation(image)
    cv2.imwrite('exponential_image.jpg', exponential_image)
    cv2.imwrite('exponential_histogram.png', histogram_maker_function(exponential_image))

    rayleigh_image = rayleigh_transformation(image)
    cv2.imwrite('rayleigh_image.jpg', rayleigh_image)
    cv2.imwrite('rayleigh_histogram.png', histogram_maker_function(rayleigh_image))

    power_law_image = power_law_transformation(image)
    cv2.imwrite('power_law_image.jpg', power_law_image)
    cv2.imwrite('power_law_histogram.png', histogram_maker_function(power_law_image))

    hyperbolic_image = hyperbolic_transformation(image)
    cv2.imwrite('hyperbolic_image.jpg', hyperbolic_image)
    cv2.imwrite('hyperbolic_histogram.png', histogram_maker_function(hyperbolic_image))

    lut_image = apply_lut(image)
    cv2.imwrite('lut_image.jpg', lut_image)
    cv2.imwrite('lut_histogram.png', histogram_maker_function(lut_image))
