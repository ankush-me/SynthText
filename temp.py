# -*- coding: utf-8 -*-
from PIL import Image, ImageFont, ImageDraw

im = Image.new("RGB",(160, 160))
draw = ImageDraw.Draw(im)

font_hindi = ImageFont.truetype("data/fonts/ubuntuhindi/Akshar Unicode.ttf",50)
text_hindi = "संयुक्‍त"

draw.text((10, 90), text_hindi, font=font_hindi)
im.show()