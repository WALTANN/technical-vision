import cv2
import matplotlib.pyplot as plt
import numpy as np

def plotting(profile):
    plt.figure(figsize=(8, 4))
    plt.xlabel("По столбцам")
    plt.ylabel("Интенсивность элементов штрих-кода")
    plt.xticks([])
    x = np.arange(profile.shape[1])
    y = profile[0, :]
    plt.plot(x, y)
    plt.show()

if __name__ == "__main__":
    image_path = "image.png"
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)

    profile = image[image.shape[0] // 2, :, 1]  
    profile = np.expand_dims(profile, axis=0)  

    plotting(profile)
