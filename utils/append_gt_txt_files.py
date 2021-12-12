import glob

files =  glob.glob("/home/shubham/Documents/MTP/datasets/detection/hindi/output/*.txt")
print(files)

file = open("/home/shubham/Documents/MTP/datasets/detection/hindi/gt.txt", "w")
for f in files:
	txt = open(f).read()
	file.write(txt)
file.close()