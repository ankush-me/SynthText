import cv2 as cv
import numpy as np 
import matplotlib.pyplot as plt 
import scipy.interpolate as si
import scipy.ndimage as scim 
import scipy.ndimage.interpolation as sii
import os
import os.path as osp
import cPickle as cp
#import Image
from PIL import Image
from poisson_reconstruct import blit_images


def sample_weighted(p_dict):
    ps = p_dict.keys()
    return ps[np.random.choice(len(ps),p=p_dict.values())]

class Layer(object):

    def __init__(self,alpha,color):

        # alpha for the whole image:
        assert alpha.ndim==2
        self.alpha = alpha
        [n,m] = alpha.shape[:2]

        color=np.atleast_1d(np.array(color)).astype('uint8')
        # color for the image:
        if color.ndim==1: # constant color for whole layer
            ncol = color.size
            if ncol == 1 : #grayscale layer
                self.color = color * np.ones((n,m,3),'uint8')
            if ncol == 3 : 
                self.color = np.ones((n,m,3),'uint8') * color[None,None,:]
        elif color.ndim==2: # grayscale image
            self.color = np.repeat(color[:,:,None],repeats=3,axis=2).copy().astype('uint8')
        elif color.ndim==3: #rgb image
            self.color = color.copy().astype('uint8')
        else:
            print color.shape
            raise Exception("color datatype not understood")

class FontColor(object):

    def __init__(self, col_file):
        with open(col_file,'r') as f:
            self.colorsRGB = cp.load(f)
        self.ncol = self.colorsRGB.shape[0]

        # convert color-means from RGB to LAB for better nearest neighbour
        # computations:
        self.colorsLAB = np.r_[self.colorsRGB[:,0:3], self.colorsRGB[:,6:9]].astype('uint8')
        self.colorsLAB = np.squeeze(cv.cvtColor(self.colorsLAB[None,:,:],cv.cv.CV_RGB2Lab))


    def sample_normal(self, col_mean, col_std):
        """
        sample from a normal distribution centered around COL_MEAN 
        with standard deviation = COL_STD.
        """
        col_sample = col_mean + col_std * np.random.randn()
        return np.clip(col_sample, 0, 255).astype('uint8')

    def sample_from_data(self, bg_mat):
        """
        bg_mat : this is a nxmx3 RGB image.
        
        returns a tuple : (RGB_foreground, RGB_background)
        each of these is a 3-vector.
        """
        bg_orig = bg_mat.copy()
        bg_mat = cv.cvtColor(bg_mat, cv.cv.CV_RGB2Lab)
        bg_mat = np.reshape(bg_mat, (np.prod(bg_mat.shape[:2]),3))
        bg_mean = np.mean(bg_mat,axis=0)

        norms = np.linalg.norm(self.colorsLAB-bg_mean[None,:], axis=1)
        # choose a random color amongst the top 3 closest matches:
        #nn = np.random.choice(np.argsort(norms)[:3]) 
        nn = np.argmin(norms)

        ## nearest neighbour color:
        data_col = self.colorsRGB[np.mod(nn,self.ncol),:]

        col1 = self.sample_normal(data_col[:3],data_col[3:6])
        col2 = self.sample_normal(data_col[6:9],data_col[9:12])

        if nn < self.ncol:
            return (col2, col1)
        else:
            # need to swap to make the second color close to the input backgroun color
            return (col1, col2)

    def mean_color(self, arr):
        col = cv.cvtColor(arr, cv.cv.CV_RGB2HSV)
        col = np.reshape(col, (np.prod(col.shape[:2]),3))
        col = np.mean(col,axis=0).astype('uint8')
        return np.squeeze(cv.cvtColor(col[None,None,:],cv.cv.CV_HSV2RGB))

    def invert(self, rgb):
        rgb = 127 + rgb
        return rgb

    def complement(self, rgb_color):
        """
        return a color which is complementary to the RGB_COLOR.
        """
        col_hsv = np.squeeze(cv.cvtColor(rgb_color[None,None,:], cv.cv.CV_RGB2HSV))
        col_hsv[0] = col_hsv[0] + 128 #uint8 mods to 255
        col_comp = np.squeeze(cv.cvtColor(col_hsv[None,None,:],cv.cv.CV_HSV2RGB))
        return col_comp

    def triangle_color(self, col1, col2):
        """
        Returns a color which is "opposite" to both col1 and col2.
        """
        col1, col2 = np.array(col1), np.array(col2)
        col1 = np.squeeze(cv.cvtColor(col1[None,None,:], cv.cv.CV_RGB2HSV))
        col2 = np.squeeze(cv.cvtColor(col2[None,None,:], cv.cv.CV_RGB2HSV))
        h1, h2 = col1[0], col2[0]
        if h2 < h1 : h1,h2 = h2,h1 #swap
        dh = h2-h1
        if dh < 127: dh = 255-dh
        col1[0] = h1 + dh/2
        return np.squeeze(cv.cvtColor(col1[None,None,:],cv.cv.CV_HSV2RGB))

    def change_value(self, col_rgb, v_std=50):
        col = np.squeeze(cv.cvtColor(col_rgb[None,None,:], cv.cv.CV_RGB2HSV))
        x = col[2]
        vs = np.linspace(0,1)
        ps = np.abs(vs - x/255.0)
        ps /= np.sum(ps)
        v_rand = np.clip(np.random.choice(vs,p=ps) + 0.1*np.random.randn(),0,1)
        col[2] = 255*v_rand
        return np.squeeze(cv.cvtColor(col[None,None,:],cv.cv.CV_HSV2RGB))


class Colorize(object):

    def __init__(self, model_dir='data'):#, im_path):
        # # get a list of background-images:
        # imlist = [osp.join(im_path,f) for f in os.listdir(im_path)]
        # self.bg_list = [p for p in imlist if osp.isfile(p)]

        self.font_color = FontColor(col_file=osp.join(model_dir,'models/colors_new.cp'))

        # probabilities of different text-effects:
        self.p_bevel = 0.05 # add bevel effect to text
        self.p_outline = 0.05 # just keep the outline of the text
        self.p_drop_shadow = 0.15
        self.p_border = 0.15
        self.p_displacement = 0.30 # add background-based bump-mapping
        self.p_texture = 0.0 # use an image for coloring text


    def drop_shadow(self, alpha, theta, shift, size, op=0.80):
        """
        alpha : alpha layer whose shadow need to be cast
        theta : [0,2pi] -- the shadow direction
        shift : shift in pixels of the shadow
        size  : size of the GaussianBlur filter
        op    : opacity of the shadow (multiplying factor)

        @return : alpha of the shadow layer
                  (it is assumed that the color is black/white)
        """
        if size%2==0:
            size -= 1
            size = max(1,size)
        shadow = cv.GaussianBlur(alpha,(size,size),0)
        [dx,dy] = shift * np.array([-np.sin(theta), np.cos(theta)])
        shadow = op*sii.shift(shadow, shift=[dx,dy],mode='constant',cval=0)
        return shadow.astype('uint8')

    def border(self, alpha, size, kernel_type='RECT'):
        """
        alpha : alpha layer of the text
        size  : size of the kernel
        kernel_type : one of [rect,ellipse,cross]

        @return : alpha layer of the border (color to be added externally).
        """
        kdict = {'RECT':cv.MORPH_RECT, 'ELLIPSE':cv.MORPH_ELLIPSE,
                 'CROSS':cv.MORPH_CROSS}
        kernel = cv.getStructuringElement(kdict[kernel_type],(size,size))
        border = cv.dilate(alpha,kernel,iterations=1) # - alpha
        return border

    def blend(self,cf,cb,mode='normal'):
        return cf

    def merge_two(self,fore,back,blend_type=None):
        """
        merge two FOREground and BACKground layers.
        ref: https://en.wikipedia.org/wiki/Alpha_compositing
        ref: Chapter 7 (pg. 440 and pg. 444):
             http://partners.adobe.com/public/developer/en/pdf/PDFReference.pdf
        """
        a_f = fore.alpha/255.0
        a_b = back.alpha/255.0
        c_f = fore.color
        c_b = back.color

        a_r = a_f + a_b - a_f*a_b
        if blend_type != None:
            c_blend = self.blend(c_f, c_b, blend_type)
            c_r = (   ((1-a_f)*a_b)[:,:,None] * c_b
                    + ((1-a_b)*a_f)[:,:,None] * c_f
                    + (a_f*a_b)[:,:,None] * c_blend   )
        else:
            c_r = (   ((1-a_f)*a_b)[:,:,None] * c_b
                    + a_f[:,:,None]*c_f    )

        return Layer((255*a_r).astype('uint8'), c_r.astype('uint8'))

    def merge_down(self, layers, blends=None):
        """
        layers  : [l1,l2,...ln] : a list of LAYER objects.
                 l1 is on the top, ln is the bottom-most layer.
        blend   : the type of blend to use. Should be n-1.
                 use None for plain alpha blending.
        Note    : (1) it assumes that all the layers are of the SAME SIZE.
        @return : a single LAYER type object representing the merged-down image
        """
        nlayers = len(layers)
        if nlayers > 1:
            [n,m] = layers[0].alpha.shape[:2]
            out_layer = layers[-1]
            for i in range(-2,-nlayers-1,-1):
                blend=None
                if blends is not None:
                    blend = blends[i+1]
                    out_layer = self.merge_two(fore=layers[i], back=out_layer,blend_type=blend)
            return out_layer
        else:
            return layers[0]

    def resize_im(self, im, osize):
        return np.array(Image.fromarray(im).resize(osize[::-1], Image.BICUBIC))
        
    def occlude(self):
        """
        somehow add occlusion to text.
        """
        pass

    def color_border(self, col_text, col_bg):
        """
        Decide on a color for the border:
            - could be the same as text-color but lower/higher 'VALUE' component.
            - could be the same as bg-color but lower/higher 'VALUE'.
            - could be 'mid-way' color b/w text & bg colors.
        """
        choice = np.random.choice(3)

        col_text = cv.cvtColor(col_text, cv.cv.CV_RGB2HSV)
        col_text = np.reshape(col_text, (np.prod(col_text.shape[:2]),3))
        col_text = np.mean(col_text,axis=0).astype('uint8')

        vs = np.linspace(0,1)
        def get_sample(x):
            ps = np.abs(vs - x/255.0)
            ps /= np.sum(ps)
            v_rand = np.clip(np.random.choice(vs,p=ps) + 0.1*np.random.randn(),0,1)
            return 255*v_rand

        # first choose a color, then inc/dec its VALUE:
        if choice==0:
            # increase/decrease saturation:
            col_text[0] = get_sample(col_text[0]) # saturation
            col_text = np.squeeze(cv.cvtColor(col_text[None,None,:],cv.cv.CV_HSV2RGB))
        elif choice==1:
            # get the complementary color to text:
            col_text = np.squeeze(cv.cvtColor(col_text[None,None,:],cv.cv.CV_HSV2RGB))
            col_text = self.font_color.complement(col_text)
        else:
            # choose a mid-way color:
            col_bg = cv.cvtColor(col_bg, cv.cv.CV_RGB2HSV)
            col_bg = np.reshape(col_bg, (np.prod(col_bg.shape[:2]),3))
            col_bg = np.mean(col_bg,axis=0).astype('uint8')
            col_bg = np.squeeze(cv.cvtColor(col_bg[None,None,:],cv.cv.CV_HSV2RGB))
            col_text = np.squeeze(cv.cvtColor(col_text[None,None,:],cv.cv.CV_HSV2RGB))
            col_text = self.font_color.triangle_color(col_text,col_bg)

        # now change the VALUE channel:        
        col_text = np.squeeze(cv.cvtColor(col_text[None,None,:],cv.cv.CV_RGB2HSV))
        col_text[2] = get_sample(col_text[2]) # value
        return np.squeeze(cv.cvtColor(col_text[None,None,:],cv.cv.CV_HSV2RGB))

    def color_text(self, text_arr, h, bg_arr):
        """
        Decide on a color for the text:
            - could be some other random image.
            - could be a color based on the background.
                this color is sampled from a dictionary built
                from text-word images' colors. The VALUE channel
                is randomized.

            H : minimum height of a character
        """
        bg_col,fg_col,i = 0,0,0
        fg_col,bg_col = self.font_color.sample_from_data(bg_arr)
        return Layer(alpha=text_arr, color=fg_col), fg_col, bg_col


    def process(self, text_arr, bg_arr, min_h):
        """
        text_arr : one alpha mask : nxm, uint8
        bg_arr   : background image: nxmx3, uint8
        min_h    : height of the smallest character (px)

        return text_arr blit onto bg_arr.
        """
        # decide on a color for the text:
        l_text, fg_col, bg_col = self.color_text(text_arr, min_h, bg_arr)
        bg_col = np.mean(np.mean(bg_arr,axis=0),axis=0)
        l_bg = Layer(alpha=255*np.ones_like(text_arr,'uint8'),color=bg_col)

        l_text.alpha = l_text.alpha * np.clip(0.88 + 0.1*np.random.randn(), 0.72, 1.0)
        layers = [l_text]
        blends = []

        # add border:
        if np.random.rand() < self.p_border:
            if min_h <= 15 : bsz = 1
            elif 15 < min_h < 30: bsz = 3
            else: bsz = 5
            border_a = self.border(l_text.alpha, size=bsz)
            l_border = Layer(border_a, self.color_border(l_text.color,l_bg.color))
            layers.append(l_border)
            blends.append('normal')

        # add shadow:
        if np.random.rand() < self.p_drop_shadow:
            # shadow gaussian size:
            if min_h <= 15 : bsz = 1
            elif 15 < min_h < 30: bsz = 3
            else: bsz = 5

            # shadow angle:
            theta = np.pi/4 * np.random.choice([1,3,5,7]) + 0.5*np.random.randn()

            # shadow shift:
            if min_h <= 15 : shift = 2
            elif 15 < min_h < 30: shift = 7+np.random.randn()
            else: shift = 15 + 3*np.random.randn()

            # opacity:
            op = 0.50 + 0.1*np.random.randn()

            shadow = self.drop_shadow(l_text.alpha, theta, shift, 3*bsz, op)
            l_shadow = Layer(shadow, 0)
            layers.append(l_shadow)
            blends.append('normal')
        

        l_bg = Layer(alpha=255*np.ones_like(text_arr,'uint8'), color=bg_col)
        layers.append(l_bg)
        blends.append('normal')
        l_normal = self.merge_down(layers,blends)
        # now do poisson image editing:
        l_bg = Layer(alpha=255*np.ones_like(text_arr,'uint8'), color=bg_arr)
        l_out =  blit_images(l_normal.color,l_bg.color.copy())
        
        # plt.subplot(1,3,1)
        # plt.imshow(l_normal.color)
        # plt.subplot(1,3,2)
        # plt.imshow(l_bg.color)
        # plt.subplot(1,3,3)
        # plt.imshow(l_out)
        # plt.show()
        
        if l_out is None:
            # poisson recontruction produced
            # imperceptible text. In this case,
            # just do a normal blend:
            layers[-1] = l_bg
            return self.merge_down(layers,blends).color

        return l_out


    def check_perceptible(self, txt_mask, bg, txt_bg):
        """
        --- DEPRECATED; USE GRADIENT CHECKING IN POISSON-RECONSTRUCT INSTEAD ---

        checks if the text after merging with background
        is still visible.
        txt_mask (hxw) : binary image of text -- 255 where text is present
                                                   0 elsewhere
        bg (hxwx3) : original background image WITHOUT any text.
        txt_bg (hxwx3) : image with text.
        """
        bgo,txto = bg.copy(), txt_bg.copy()
        txt_mask = txt_mask.astype('bool')
        bg = cv.cvtColor(bg.copy(), cv.cv.CV_RGB2Lab)
        txt_bg = cv.cvtColor(txt_bg.copy(), cv.cv.CV_RGB2Lab)
        bg_px = bg[txt_mask,:]
        txt_px = txt_bg[txt_mask,:]
        bg_px[:,0] *= 100.0/255.0 #rescale - L channel
        txt_px[:,0] *= 100.0/255.0

        diff = np.linalg.norm(bg_px-txt_px,ord=None,axis=1)
        diff = np.percentile(diff,[10,30,50,70,90])
        print "color diff percentile :", diff
        return diff, (bgo,txto)

    def color(self, bg_arr, text_arr, hs, place_order=None, pad=20):
        """
        Return colorized text image.

        text_arr : list of (n x m) numpy text alpha mask (unit8).
        hs : list of minimum heights (scalar) of characters in each text-array. 
        text_loc : [row,column] : location of text in the canvas.
        canvas_sz : size of canvas image.
        
        return : nxmx3 rgb colorized text-image.
        """
        bg_arr = bg_arr.copy()
        if bg_arr.ndim == 2 or bg_arr.shape[2]==1: # grayscale image:
            bg_arr = np.repeat(bg_arr[:,:,None], 3, 2)

        # get the canvas size:
        canvas_sz = np.array(bg_arr.shape[:2])

        # initialize the placement order:
        if place_order is None:
            place_order = np.array(xrange(len(text_arr)))

        rendered = []
        for i in place_order[::-1]:
            # get the "location" of the text in the image:
            ## this is the minimum x and y coordinates of text:
            loc = np.where(text_arr[i])
            lx, ly = np.min(loc[0]), np.min(loc[1])
            mx, my = np.max(loc[0]), np.max(loc[1])
            l = np.array([lx,ly])
            m = np.array([mx,my])-l+1
            text_patch = text_arr[i][l[0]:l[0]+m[0],l[1]:l[1]+m[1]]

            # figure out padding:
            ext = canvas_sz - (l+m)
            num_pad = pad*np.ones(4,dtype='int32')
            num_pad[:2] = np.minimum(num_pad[:2], l)
            num_pad[2:] = np.minimum(num_pad[2:], ext)
            text_patch = np.pad(text_patch, pad_width=((num_pad[0],num_pad[2]), (num_pad[1],num_pad[3])), mode='constant')
            l -= num_pad[:2]

            w,h = text_patch.shape
            bg = bg_arr[l[0]:l[0]+w,l[1]:l[1]+h,:]

            rdr0 = self.process(text_patch, bg, hs[i])
            rendered.append(rdr0)

            bg_arr[l[0]:l[0]+w,l[1]:l[1]+h,:] = rdr0#rendered[-1]


            return bg_arr

        return bg_arr
