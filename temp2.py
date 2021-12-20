import cv2

"""images = open("/home/shubham/Documents/MTP/datasets/detection/hindi/output/gt1.txt").readlines()
for img in images:
	img = img[0]
	img = cv2.imread(img)
	cv2.imwrite("gt1_{}".format(i))
	"""
"""import lmdb

def writeCache(env, cache):
    # print("Writing to LMDB")
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            txn.put(k, v)



env = lmdb.open("/home/shubham/Documents/MTP/datasets/detection/hindi/validation/ST_valid", map_size=1099511627776)
with env.begin(write=True) as txn:
	cursor = txn.cursor()
	num= int(txn.get('num-samples'.encode()))
	cursor.replace('num-samples'.encode() , str(4257513).encode())"""



