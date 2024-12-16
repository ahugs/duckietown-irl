import numpy as np

img_queue = np.zeros((3, 3))
img = np.zeros((2, 2))
resized_img = cv2.resize(img, (3, 3))
queue = np.append(img_queue, resized_img)
print(queue)
