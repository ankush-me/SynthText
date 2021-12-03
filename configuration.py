DATA_PATH="/data"

lang = "HI"
text_soruce = "newsgroup/newsgroup_{}.txt".format(lang)

"""   ********** font_list file path********"""
fontlist_file = "fonts/fontlist/fontlist_{}.txt".format(lang)

"""   ********** char freq file path********"""
char_freq_path = 'models/{}/char_freq.cp'.format(lang)

""" ******** font pixel to point configuration file"""
font_px2pt = 'models/{}/font_px2pt.cp'.format(lang)


"""" *** characters ranges for each language ***"""
special_char=["1" , "2", "3" , "4", "5" , "6", "7" , "8" , "9", "10" , "%" , "/", "?", "-", ":", ",","."]

range={}
range["HI"] = [
        {"from" :'ऀ' , "to" : "९"}
    ]

range["TA"] = [
        {"from" :'ஂ' , "to" : "௯"}
    ]

range["GU"] = [
        {"from" :'ઁ' , "to" : "૯"}
    ]

range["ML"] = [
        {"from" :'ഀ' , "to" : "൯"}
    ]

range["TE"] = [
        {"from" :'ఀ' , "to" : "౯"}
    ]
range["KN"] = [
    {"from": 'ಀ', "to": "೯"}
]

range["GUR"] = [
    {"from": 'ਁ', "to": "੯"}
]
range["OR"] = [
    {"from": 'ଁ', "to": "୯"}
]
range["BN"] = [
    {"from": 'ঀ', "to": "৯"}
]
range['AR'] = [
{"from": '؀', "to": "ۿ"}
]

range['ENG'] = [
{"from": '!', "to": "}"}
]