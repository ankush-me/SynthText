from synthgen import *


import configuration
for l in ["TE","BN","AR", "GU", "GUR", "KN", "HI", "ML", "OR", "TA"]:
	configuration.lang=l
	configuration.text_soruce = "newsgroup/newsgroup_{}1.txt".format(configuration.lang)
	configuration.fontlist_file = "fonts/fontlist/fontlist_{}.txt".format(configuration.lang)
	configuration.char_freq_path = 'models/{}/char_freq.cp'.format(configuration.lang)
	configuration.font_px2pt = 'models/{}/font_px2pt.cp'.format(configuration.lang)
	
	DATA_PATH = 'data'
	RV3 = RendererV3(DATA_PATH, max_time=None)
	
	
	for i in range(400):
		for j in range(11):
			try:
				RV3.rendor_text(i, j)
			except Exception:
				print("")


"""lines = open("data/fonts/fontlist/fontlist_HI.txt").readlines()
fonts = ["Poppins",
"Rajdhani",
"RozhaOne",
"sarpanch",
"Teko",
"Hind",
"Sahadeva",
"samanata",
"varta",
"Amiko",
"Arya",
"Biryani",
"Dekko",
"Dinah",
"InknutAntiqua",
"khand",
"kurale",
"modak",
"Tilana"]
a = lines
for line in lines:
	for j in fonts:
		if j in line:
			a.remove(line)
			
print("".join(map(str, a)))"""
s= "ट्रांसपोर्ट"
s1='टं्रासपोर्ट'
print(list(s))
print(list(s1))
a=['थ ','ऺ ' ]


a=['ऀ '	,'ँ', 	'ं', 	'ः','ऺ', ]