import cv2
import matplotlib.pyplot as plt
import numpy as np

def plot_x(proection):
    plt.plot(range(len(proection)), proection)
    plt.ylabel("Интенсивность")
    plt.show()

def plot_y(proection):
    plt.plot(proection, range(len(proection)))
    plt.xlabel("Интенсивность")
    plt.show()

if __name__ == "__main__":
    image = cv2.imread("image.png", cv2.IMREAD_COLOR)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    proection_y = np.sum(image, axis=1) / (256 * image.shape[2])
    plot_y(proection_y)

    proection_x = np.sum(image, axis=0) / (256 * image.shape[2])
    plot_x(proection_x)
