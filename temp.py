"""from synthgen import *


import configuration
configuration.lang="HI"
configuration.text_soruce = "newsgroup/newsgroup_{}1.txt".format(configuration.lang)
configuration.fontlist_file = "fonts/fontlist/fontlist_{}.txt".format(configuration.lang)
configuration.char_freq_path = 'models/{}/char_freq.cp'.format(configuration.lang)
configuration.font_px2pt = 'models/{}/font_px2pt.cp'.format(configuration.lang)

DATA_PATH = 'data'
RV3 = RendererV3(DATA_PATH, max_time=None)


for i in range(400):
	RV3.rendor_text(i)"""

"""    #delete this function
    def rendor_text(self, i):
        font = self.text_renderer.font_state.sample(i)
        f = font
        font = self.text_renderer.font_state.init_font(font)
        collision_mask= np.zeros((1600,1596) , dtype=int)

        render_res = self.text_renderer.render_sample(font, collision_mask)
        if render_res is None:  # rendering not successful
            return  # None
        else:
            text_mask, loc, bb, text = render_res
        text_mask = 255*(text_mask>0).astype('uint8')
        im=Image.fromarray(text_mask , 'L')
        im.save("temp/{}_{}.jpg".format(f["font"], text[0:min(len(text),3)]))"""