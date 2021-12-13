import cv2

images = open("/home/shubham/Documents/MTP/datasets/detection/hindi/output/gt1.txt").readlines()
for img in images:
	img = img[0]
	img = cv2.imread(img)
	cv2.imwrite("gt1_{}".format(i))
	