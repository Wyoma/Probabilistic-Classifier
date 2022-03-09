import numpy as np
import matplotlib.pyplot as plt

train = np.array([plt.imread('Data/Train/'+str(i)+'.jpg').reshape(-1) for i in range(1,2401)])
test = np.array([plt.imread('Data/Test/'+str(i)+'.jpg').reshape(-1) for i in range(1,201)])
train_labels = np.loadtxt('Data/Train/Training Labels.txt')
test_labels = np.loadtxt('Data/Test/Test Labels.txt')

def test_fn(imgArray,test,levels):
    test_array = quantize(test, levels)
    imgArray = non_zero_probability(imgArray)
    result_array = np.zeros((200))
    for test_image in range(0,200):
        image_result_array = np.zeros((10))
        for digit_class in range(0,10):
            total_class_probability = 1
            for pixel in test_array[test_image]:
                pixel_intensity = test_array[test_image, pixel]
                total_class_probability = total_class_probability * imgArray[digit_class,pixel,pixel_intensity]
            image_result_array[digit_class] = total_class_probability
        result_array[test_image] = np.where(image_result_array == np.amax(image_result_array))[0]
    return result_array

def quantize(imgArray,levels):
    for i in range (0,imgArray.shape[0]):
        for j in range(0,imgArray.shape[1]):
            for k in range(0,levels):
                if k*(256/levels) <= imgArray[i,j]:
                    if imgArray[i,j] < (k+1)*(256/levels):
                        imgArray[i,j] = k*(256/levels)
    return imgArray

def train_fn(imgArray,levels):
    imgArray = quantize(imgArray, levels)
    imgMatrix = np.zeros((10,784,256))
    for classIndex in range(0,imgMatrix.shape[0]):
        for pixelIndex in range(0,imgMatrix.shape[1]):
            for trainIndex in range(classIndex * 240,(classIndex+1) * 240):
                imgMatrix[classIndex,pixelIndex,imgArray[trainIndex,pixelIndex]]+=(1/240)
    return imgMatrix

def non_zero_probability(img_matrix):
    for classes in range(0,10):
        for pixels in range(0,784):
            for intensity_levels in range(0,256):
                if img_matrix[classes,pixels,intensity_levels] == 0:
                    img_matrix[classes,pixels,intensity_levels] = 0.00001
    return img_matrix


quantizedMatrix = quantize(train,2)
trainedMatrix = train_fn(train,2)
non_zero_array = non_zero_probability(trainedMatrix,2)
result_array = test_fn(trainedMatrix,test,2)

print(result_array)
