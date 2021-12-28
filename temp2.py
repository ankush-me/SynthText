special_symbols = "०  १  २  ३  ४  ५  ६  ७  ८  ९ %  /  ?  :  ,  .  -"


def is_dev(text):
	for c in list(text):
		if c in special_symbols:
			continue
		
		else:
			if "ऀ" <= c <= "ॿ":
				continue
			else:
				return False
	return True


f = open("/home/shubham/Downloads/mr.vocabfreq.tsv").readlines()
out = open("newsgroup_mr.txt","w")
for i ,line in enumerate(f):
	
	line = line.split()[0]
	line.replace("\n" , "")
	if is_dev(line):
		out.write(line)
		
		if i % 100 == 0:
			out.write("\n")
		else:
			out.write(" ")

