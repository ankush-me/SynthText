from __future__ import division

import io

# import fribidi
import numpy as np
import matplotlib.pyplot as plt 
import scipy.io as sio
import os.path as osp
import random, os
import cv2
#import cPickle as cp
import _pickle as cp
import scipy.signal as ssig
import scipy.stats as sstat
import pygame, pygame.locals
from pygame import freetype
#import Image
from PIL import Image
import math

import configuration
from common import *
import pickle
import codecs
from logger import logger, wrap, entering, exiting
import cv2 as cv
import nltk, re, pprint
from nltk import word_tokenize, sent_tokenize
from nltk.corpus.reader import *
from nltk.corpus.reader.util import *
from nltk.text import Text
from nltk.corpus.reader.chasen import *
import subprocess
import nltk
import qahirah as qah
from qahirah import     CAIRO,     Colour,     Glyph,     Vector
ft = qah.get_ft_lib()
# import fribidi as fb
# from fribidi import     FRIBIDI as FB
import harfbuzz as hb
from array import array


try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
   nltk.download('punkt')


def transform_desire(image, curveIntensity):
    '''
    will convert image to arc form.
    im1 : image.
    curveIntensity : How much curved text you desired.
    '''
    im1 = image
    ratio = 0.0001 * curveIntensity
    
    ## calculate the desired width of the image.
    height, width, channel = im1.shape
    
    x = np.linspace(0, width, width).astype(int)
    y = (ratio * ((x - (width / 2)) ** 2)).astype(int)
    ## corrosponding to an x every point will get shifted by y amount.
    
    ## time to shift.
    
    ## create canvas for new image.
    
    adder = 0
    
    if ratio >= 0:
        adder = max(y)
    else:
        adder = (-1) * min(y)
    
    retImage = (np.ones((height + adder, width, channel)) * 0).astype(np.uint8)
    
    if ratio >= 0:
        adder = 0
    
    for xs in range(width):
        ys = y[xs]
        #         print(xs,ys)
        for t in range(height):
            retImage[t + ys + adder, xs, :] = im1[t, xs, :]
    
    return retImage


def pngB_to_np(pngB):
    return np.array(Image.open(io.BytesIO(pngB)))


# In[66]:



# To get the bound of the glyph
def get_Bound_Glyph(abc, buf, hb_font, ft_face, text_size):
    # Setting Starting position of the first glyph
    glyph_pos = Vector(0, 0)
    
    # Resetting the buffer
    buf.reset()
    
    # adding string to buffer
    buf.add_str(abc)
    
    # Figuring out segmentation properties
    buf.guess_segment_properties()
    
    # Gernerating Shapes for the text in buffer using the font
    hb.shape(hb_font, buf)
    
    # Getting glyphs out of buffer (list format)
    glyphs, end_glyph_pos = buf.get_glyphs(glyph_pos)
    
    # Creating fontface
    qah_face = qah.FontFace.create_for_ft_face(ft_face)
    
    glyph_extents = (qah.Context.create_for_dummy()
                     .set_font_face(qah_face)
                     .set_font_size(text_size)
                     .glyph_extents(glyphs)
                     )
    
    # Getting the bound of the [glyphs]
    figure_bounds = math.ceil(glyph_extents.bounds)
    
    # Returning glyph and the figure bound
    return (figure_bounds, glyphs)


#### Curved Script

def to_rad(degree):
    return ((degree / 360) * 2 * (np.pi))


## Code to get Rect
def get_rect(glyphs, qah_face, text_size, angle=0):
    angle = to_rad(angle)
    ctx = qah.Context.create_for_dummy()
    ctx.set_font_face(qah_face)
    ctx.set_font_size(text_size)
    # ctx.rotate(angle)
    b = ctx.glyph_extents(glyphs)
    b.y_bearing = b.y_bearing * (-1)
    return b.bounds


def boundB( imm):
    (sx, sy) = imm.shape
    
    first = 0  ## intialize everything.
    leftx = 0
    rightx = sy
    top = 0
    bottom = sx
    
    # imm[imm > 128] = 1 # doing the binarization of image.
    # imm[imm < 128] = 0
    
    for i in range(sy):
        if np.sum(imm[:, i] != 0) > 0 and first == 0:
            leftx = i
            first = 1
            break
    
    for i in range(sy - 1, 0, -1):
        if np.sum(imm[:, i] != 0) > 0:
            rightx = i
            break
    
    fchck = 0
    for i in range(sx):
        if np.sum(imm[i, :] != 0) > 0:
            top = i
            break
    
    for i in range(sx - 1, 0, -1):
        if np.sum(imm[i, :] != 0) > 0:
            bottom = i - 1
            break
    

    
    mask = imm.swapaxes(0, 1) != 0
    xs, ys = np.where(mask)
    coords = np.c_[xs, ys].astype('float32')
    
    rect = cv2.minAreaRect(coords)
    box = np.array(cv2.boxPoints(rect))
    box = np.int0(box)
    box  = np.where(box < 0, 0, box)
    return (leftx, top, rightx - leftx + 1, bottom - top + 1 , box)

# In[639]:


# To get glyph and bounds of a glyphs
def get_Bound_Glyph_2(abc, buf, hb_font, ft_face, text_size):
    # Setting Starting position of the first glyph
    glyph_pos = Vector(0, 0)
    
    # Resetting the buffer
    buf.reset()
    
    # adding string to buffer
    buf.add_str(abc)
    
    # Figuring out segmentation properties
    buf.guess_segment_properties()
    
    # Gernerating Shapes for the text in buffer using the font
    hb.shape(hb_font, buf)
    
    # Getting glyphs out of buffer (list format)
    glyphs, end_glyph_pos = buf.get_glyphs(glyph_pos)
    
    # Creating fontface
    qah_face = qah.FontFace.create_for_ft_face(ft_face)
    
    glyph_extents = (qah.Context.create_for_dummy()
                     .set_font_face(qah_face)
                     .set_font_size(text_size)
                     .glyph_extents(glyphs)
                     )
    
    # Getting the bound of the [glyphs]
    figure_bounds = math.ceil(glyph_extents.bounds)
    
    # Returning glyph and the figure bound
    return (figure_bounds, glyphs)




@wrap(entering, exiting)
def sample_weighted(p_dict):
    ps = list(p_dict.keys())
    return p_dict[np.random.choice(ps,p=ps)]

@wrap(entering, exiting)
def move_bb(bbs, t):
    """
    Translate the bounding-boxes in by t_x,t_y.
    BB : 2x4xn
    T  : 2-long np.array
    """
    return bbs + t[:,None,None]
@wrap(entering, exiting)
def crop_safe(arr, rect, box=None, pad=0):
    """
    ARR : arr to crop
    RECT: (x,y,w,h) : area to crop to
    BBS : nx4 xywh format bounding-boxes
    PAD : percentage to pad

    Does safe cropping. Returns the cropped rectangle and
    the adjusted bounding-boxes
    """
    rect = np.array(rect)
    rect[:2] -= pad
    rect[2:] += 2*pad
    v0 = [max(0,rect[0]), max(0,rect[1])]
    v1 = [min(arr.shape[0], rect[0]+rect[2]), min(arr.shape[1], rect[1]+rect[3])]
    arr = arr[v0[0]:v1[0],v0[1]:v1[1],...]

    if len(box) > 0:
        for i in np.ndindex(box.shape[0]):
            box[i, 0] -= v0[0]
            box[i, 1] -= v0[1]
            
        return arr, box
    else:
        return arr

class BaselineState(object):
    curve = lambda this, a: lambda x: a*x*x
    differential = lambda this, a: lambda x: 2*a*x
    a = [0.50, 0.05]

    @wrap(entering, exiting)
    def get_sample(self):
        """
        Returns the functions for the curve and differential for a and b
        """
        sgn = 1.0
        if np.random.rand() < 0.5:
            sgn = -1

        a = self.a[1]*np.random.randn() + sgn*self.a[0]
        return {
            'curve': self.curve(a),
            'diff': self.differential(a),
        }

class RenderFont(object):
    """
    Outputs a rasterized font sample.
        Output is a binary mask matrix cropped closesly with the font.
        Also, outputs ground-truth bounding boxes and text string
    """

    def __init__(self, data_dir='data'):
        # distribution over the type of text:
        # whether to get a single word, paragraph or a line:
        self.p_text = {0.5 : 'WORD',
                       0.2 : 'LINE',
                       0.3 : 'PARA'}

        ## TEXT PLACEMENT PARAMETERS:
        self.f_shrink = 0.90
        self.max_shrink_trials = 5 # 0.9^5 ~= 0.6
        # the minimum number of characters that should fit in a mask
        # to define the maximum font height.
        self.min_nchar = 2
        self.min_font_h = 16 #px : 0.6*12 ~ 7px <= actual minimum height
        self.max_font_h = 120 #px
        self.p_flat = 0.10

        # curved baseline:
        self.p_curved = 1.0
        self.baselinestate = BaselineState()

        # text-source : gets english text:
        self.text_source = TextSource(min_nchar=self.min_nchar,
                                      fn=osp.join(data_dir, configuration.text_soruce),
                                    lang = configuration.lang)

        # get font-state object:
        self.font_state = FontState(data_dir)

        pygame.init()

    @wrap(entering, exiting)
    def render_multiline(self, font, text):
        """
		renders multiline TEXT on the pygame surface SURF with the
		font style FONT.
		A new line in text is denoted by \n, no other characters are
		escaped. Other forms of white-spaces should be converted to space.
		returns the updated surface, words and the character bounding boxes.
		"""
        # Adding Custom Code here by removing the Orginal Code
    
        # get the number of lines
        lines = text.split('\n')
        lengths = [len(l) for l in lines]
    
        # font parameters:
        line_spacing = font.get_sized_height() + 1
    
        # initialize the surface to proper size:
        line_bounds = font.get_rect(lines[np.argmax(lengths)])
        fsize = (round(2.0 * line_bounds.width), round(1.25 * line_spacing * len(lines)))
        # surf = pygame.Surface(fsize, pygame.locals.SRCALPHA, 32)
        space = font.get_rect('O')
        spaceWidth = space.width * 0.85  ## 0.8 has been multiplied here
        surfx, surfy = fsize
        font_path = font.path
        font_size = font.size

        # lines = di['lines']
        fsize = Vector(int(surfx + space.width), int(surfy + line_spacing))
        # line_spacing = di['line_spacing']
    
        pix = qah.ImageSurface.create(
            format=CAIRO.FORMAT_RGB24,
            dimensions=fsize
        )
    
        # Creating ft_face
        ft_face = ft.new_face(font_path)
        text_size = font_size
        # Creating Buffer
        buf = hb.Buffer.create()
        # setting char size to font face
        ft_face.set_char_size(size=text_size, resolution=qah.base_dpi)
        # ft_face.underline = font.underline
        hb_font = hb.Font.ft_create(ft_face)
        qah_face = qah.FontFace.create_for_ft_face(ft_face)
    
        ctx = qah.Context.create(pix)
        # ctx.set_source_colour(Colour.grey(0))
        # ctx.paint()
        ctx.set_source_colour(Colour.grey(1))
        ctx.set_font_face(qah_face)
        ctx.set_font_size(text_size)
    
        # Start Replacing Code from here
    
        ## for the purpose of shifting in horizontal direction.
        shiftAdditional = 0
        # By What factor shifting should be done
        shiftFactor = 1.1
    
        factor = 0
        y = 0
        bb = []
    
        ## The recent addtion code. feb 19,2019
        ## Project make faster.
        ## No need for the word wise bounding box. So, now they can be eliminated.
    
        '''
			The procedure is as follow.
			- Since we do not require individual character bounding box,
			we will discard that functionality.
			- We will place a word and create fake bounding box for it.
			- right from the top left.
			- width of the bounding box will be simply ((width of word)/total bounding box)
		'''
    
        ## Ends here feb 19,2019
    
        for l in lines:  # picking up a line.
        
            l = l.split(" ")
            l = " ".join(l)
            x = spaceWidth * 0.7  # carriage-return
            y += (line_spacing * 0.8)  # line-feed
        
            words = l.split()
        
            for w in words:
                st_bound, glyph_3 = get_Bound_Glyph(w, buf, hb_font, ft_face, text_size)
                shift = shiftFactor * (st_bound.topleft)[0] + shiftAdditional

                char_in_w = len(w)
                
                bb.append(np.array([x+shift ,y + (st_bound.topleft)[1], st_bound.width, st_bound.height]))

    
            
                ## now since we have generated fake bounding box,
                ## next task will be to render glyphs on the actual surface.
            
                ## Remember context is a pen for us.
                ## Go to the point from where you want to write.
                #bb.append(np.array([x+shift,y + (st_bound.topleft)[1], st_bound.width,st_bound.height])  )
                ctx.translate(Vector(x + shift, y))
            
                # setting the color that we wish to use.
                ctx.set_source_colour(Colour.grey(1))
                # setting the font_face
                ctx.set_font_face(qah_face)
                # defining the size.
                ctx.set_font_size(text_size)
                # rendering the glyphs on the surface.
                ctx.show_glyphs(glyph_3)
            
                ## translate back to the original position.
                ctx.translate(Vector(-(x + shift), -y))
            
                ## Shift the x to new position.
                x = x + st_bound.width + shift + spaceWidth
                ## resetting the shift.


        img = pngB_to_np(pix.to_png_bytes())

        # dicc= {}
        # dicc['img'] = img[:,:,1]
        # dicc['bb'] = bb
    
        Simg = img[:, :, 1]
    
        # self.ii = self.ii+1
        # bb = dicc['bb']
        bb = np.array(bb)
        # Simg = Simg.astype(np.uint8)
        # Simg = Simg[:,:,1]
        Simg = Simg.swapaxes(0, 1)
    
        # get the words:
        words = ' '.join(text.split())

        r0 = pygame.Rect(bb[0])
        rect_union = r0.unionall(bb)
    
        surf_arr, bbs = crop_safe(Simg, rect_union, bb, pad=5)
        surf_arr = surf_arr.swapaxes(0, 1)
        # self.visualize_bb(surf_arr,bbs)

        bbs = self.bb_xywh2coords(bbs)
        return surf_arr, words, bbs
    
    @wrap(entering, exiting)
    def render_curved(self, font, word_text):
        """
               use curved baseline for rendering word
               """

        
      
       
        
        wl = len(word_text)
        isword = len(word_text.split()) == 1
    
        # do curved iff, the length of the word <= 10
        if not isword or wl > 15 or np.random.rand() > self.p_curved:
            return self.render_multiline(font, word_text)



        # create the surface:
        lspace = font.get_sized_height() + 1
        lbound = font.get_rect(word_text)
        fsize = (round(2.0 * lbound.width), round(3 * lspace))
    
        font_path = font.path
        font_size = font.size
        space = font.get_rect('O')
        spaceWidth = space.width * 0.85
        line_spacing = font.get_sized_height() + 1
        fsize = Vector(int(fsize[0] + spaceWidth), int(fsize[1] + spaceWidth))
    
        pix = qah.ImageSurface.create(
            format=CAIRO.FORMAT_RGB24,
            dimensions=fsize
    
        )
    
        # Creating ft_face
        ft_face = ft.new_face(font_path)
        text_size = font_size
        # Creating Buffer
        buf = hb.Buffer.create()
        # setting char size to font face
        ft_face.set_char_size(size=text_size, resolution=qah.base_dpi)
        hb_font = hb.Font.ft_create(ft_face)
        qah_face = qah.FontFace.create_for_ft_face(ft_face)
    
        ctx = qah.Context.create(pix)
        ctx.set_source_colour(Colour.grey(0))
        ctx.paint()
    
        word_text = word_text.strip()
        #word_text_len = len(word_text)
        ### Added on feb, 22, 2019
    
        ## single word will be there.
        shiftFactor = 1.1
        shiftAdditional = 0
        ## Please refer to multiline for explaination.
        st_bound, glyph_3 = get_Bound_Glyph(word_text, buf, hb_font, ft_face, text_size)
    
        shift = shiftFactor * (st_bound.topleft)[0] + shiftAdditional
    
        x = spaceWidth * 0.7 + shift
        y = line_spacing * 0.8
    
        st_bound, glyph_3 = get_Bound_Glyph(word_text, buf, hb_font, ft_face, text_size)
        ctx.translate(Vector(x + shift, y))
        ctx.set_source_colour(Colour.grey(1))
        ctx.set_font_face(qah_face)
        ctx.set_font_size(text_size)
        ctx.show_glyphs(glyph_3)
    
        img = pngB_to_np(pix.to_png_bytes())
    
        curveIntensity = np.random.randint(-20, 20)
    
        img = transform_desire(img, curveIntensity)
        
        #bbox gives minimum area rectangle compared to left, top , width and height variable.
        left, top, width, height ,bbox = boundB(img[:, :, 1])
        bbs =[]
        
        bbs.append(np.array([left, top, left+width , height ]))
        
        
        bb = bbs
        bb = np.array(bb)
        Simg1 = img[:, :, 0]
        Simg1 = Simg1.swapaxes(0, 1)
        
        r0 = pygame.Rect(bb[0])
        rect_union = r0.unionall(bb)
        surf_arr, bbox,  = crop_safe(Simg1, rect_union, bbox, pad=5)

        surf_arr = surf_arr.swapaxes(0, 1)

        bbox = bbox.T
        bbox = bbox[:, :, None]
        
        return surf_arr, word_text, bbox
    
    @wrap(entering, exiting)
    def get_nline_nchar(self,mask_size,font_height,font_width):
        """
        Returns the maximum number of lines and characters which can fit
        in the MASK_SIZED image.
        """
        
        H,W = mask_size
        nline = int(np.ceil(H/(2*font_height)))
        nchar = int(np.floor(W/font_width))
        return nline,nchar

    @wrap(entering, exiting)
    def place_text(self, text_arrs, back_arr, bbs):
        areas = [-np.prod(ta.shape) for ta in text_arrs]
        order = np.argsort(areas)

        locs = [None for i in range(len(text_arrs))]
        out_arr = np.zeros_like(back_arr)
        for i in order:            
            ba = np.clip(back_arr.copy().astype(np.float), 0, 255)
            ta = np.clip(text_arrs[i].copy().astype(np.float), 0, 255)
            ba[ba > 127] = 1e8
            intersect = ssig.fftconvolve(ba,ta[::-1,::-1],mode='valid')
            safemask = intersect < 1e8

            if not np.any(safemask): # no collision-free position:
                #warn("COLLISION!!!")
                return back_arr,locs[:i],bbs[:i],order[:i]

            minloc = np.transpose(np.nonzero(safemask))
            loc = minloc[np.random.choice(minloc.shape[0]),:]
            locs[i] = loc

            # update the bounding-boxes:
            bbs[i] = move_bb(bbs[i],loc[::-1])

            # blit the text onto the canvas
            w,h = text_arrs[i].shape
            out_arr[loc[0]:loc[0]+w,loc[1]:loc[1]+h] += text_arrs[i]

        return out_arr, locs, bbs, order

    @wrap(entering, exiting)
    def robust_HW(self,mask):
        m = mask.copy()
        m = (~mask).astype('float')/255
        rH = np.median(np.sum(m,axis=0))
        rW = np.median(np.sum(m,axis=1))
        return rH,rW

    @wrap(entering, exiting)
    def sample_font_height_px(self,h_min,h_max):
        if np.random.rand() < self.p_flat:
            rnd = np.random.rand()
        else:
            rnd = np.random.beta(2.0,2.0)

        h_range = h_max - h_min
        f_h = np.floor(h_min + h_range*rnd)
        return f_h

    @wrap(entering, exiting)
    def bb_xywh2coords(self,bbs):
        """
        Takes an nx4 bounding-box matrix specified in x,y,w,h
        format and outputs a 2x4xn bb-matrix, (4 vertices per bb).
        """
        n,_ = bbs.shape
        coords = np.zeros((2,4,n))
        for i in range(n):
            coords[:,:,i] = bbs[i,:2][:,None]
            coords[0,1,i] += bbs[i,2]
            coords[:,2,i] += bbs[i,2:4]
            coords[1,3,i] += bbs[i,3]
        return coords

    @wrap(entering, exiting)
    def render_sample(self,font,mask):
        """
        Places text in the "collision-free" region as indicated
        in the mask -- 255 for unsafe, 0 for safe.
        The text is rendered using FONT, the text content is TEXT.
        """
        #H,W = mask.shape
        H,W = self.robust_HW(mask)
        f_asp = self.font_state.get_aspect_ratio(font)

        # find the maximum height in pixels:
        max_font_h = min(0.9*H, (1/f_asp)*W/(self.min_nchar+1))
        max_font_h = min(max_font_h, self.max_font_h)
        if max_font_h < self.min_font_h: # not possible to place any text here
            return #None

        # let's just place one text-instance for now
        ## TODO : change this to allow multiple text instances?
        i = 0
        while i < self.max_shrink_trials and max_font_h > self.min_font_h:
            # if i > 0:
            #     print colorize(Color.BLUE, "shrinkage trial : %d"%i, True)

            # sample a random font-height:
            f_h_px = self.sample_font_height_px(self.min_font_h, max_font_h)
            #print "font-height : %.2f (min: %.2f, max: %.2f)"%(f_h_px, self.min_font_h,max_font_h)
            # convert from pixel-height to font-point-size:
            f_h = self.font_state.get_font_size(font, f_h_px)

            # update for the loop
            max_font_h = f_h_px 
            i += 1

            font.size = f_h # set the font-size

            # compute the max-number of lines/chars-per-line:
            nline,nchar = self.get_nline_nchar(mask.shape[:2],f_h,f_h*f_asp)
            #print "  > nline = %d, nchar = %d"%(nline, nchar)

            if not( nline >= 1 and nchar >= self.min_nchar):
                continue

            # sample text:
            text_type = sample_weighted(self.p_text)
            text = self.text_source.sample(nline,nchar,text_type)

           
            
            if len(text)==0 or np.any([len(line)==0 for line in text]):
                continue

            text = text.replace('\u200c', ' ')
            text = text.replace('\u201c', ' ')
            
            # ref: https://github.com/ankush-me/SynthText/issues/48
            text_size = font.get_metrics(text.replace("\n", " "))
            supported = len(list((filter(lambda t: t is None, text_size)))) == 0
            if not supported:
                continue
            
            #print colorize(Color.GREEN, text)

            # render the text:
            txt_arr,txt, box = self.render_curved(font, text)

           
            
            # make sure that the text-array is not bigger than mask array:
            if np.any(np.r_[txt_arr.shape[:2]] > np.r_[mask.shape[:2]]):
                #warn("text-array is bigger than mask")
                continue

            # position the text within the mask:
            text_mask,loc,bb, _ = self.place_text([txt_arr], mask, [box])
            if len(loc) > 0:#successful in placing the text collision-free:
                return text_mask,loc[0],bb[0],txt
                
        return #None

    @wrap(entering, exiting)
    def visualize_bb(self, text_arr, bbs):
        ta = text_arr.copy()
        for r in bbs:
            cv.rectangle(ta, (r[0],r[1]), (r[0]+r[2],r[1]+r[3]), color=128, thickness=1)
        plt.imshow(ta,cmap='gray')
        plt.show()


class FontState(object):
    """
    Defines the random state of the font rendering  
    """
    size = [50, 10]  # normal dist mean, std
    underline = 0.05
    strong = 0.5
    oblique = 0.2
    wide = 0.5
    strength = [0.05, 0.1]  # uniform dist in this interval
    underline_adjustment = [1.0, 2.0]  # normal dist mean, std
    kerning = [2, 5, 0, 20]  # beta distribution alpha, beta, offset, range (mean is a/(a+b))
    border = 0.25
    random_caps = -1 ## don't recapitalize : retain the capitalization of the lexicon
    capsmode = [str.lower, str.upper, str.capitalize]  # lower case, upper case, proper noun
    curved = 0.2
    random_kerning = 0.2
    random_kerning_amount = 0.1

    @wrap(entering, exiting)
    def __init__(self, data_dir='data'):

        char_freq_path = osp.join(data_dir, configuration.char_freq_path)
        font_model_path = osp.join(data_dir, configuration.font_px2pt)


        
        # get character-frequencies in the English language:
        with open(char_freq_path,'rb') as f:
            self.char_freq = cp.load(f)

        # get the model to convert from pixel to font pt size:
        with open(font_model_path,'rb') as f:
            self.font_model = cp.load(f, encoding='unicode_escape')
            
        # get the names of fonts to use:
        self.FONT_LIST = osp.join(data_dir, configuration.fontlist_file)
        self.fonts = [os.path.join(data_dir,'fonts',f.strip()) for f in open(self.FONT_LIST)]

    @wrap(entering, exiting)
    def get_aspect_ratio(self, font, size=None):
        """
        Returns the median aspect ratio of each character of the font.
        """
        if size is None:
            size = 12 # doesn't matter as we take the RATIO
        chars = ''.join(self.char_freq.keys())
        w = np.array(self.char_freq.values())

        # get the [height,width] of each character:
        try:
            sizes = font.get_metrics(chars,size)
            good_idx = [i for i in range(len(sizes)) if sizes[i] is not None]
            sizes,w = [sizes[i] for i in good_idx], w[good_idx]
            sizes = np.array(sizes).astype('float')[:,[3,4]]        
            r = np.abs(sizes[:,1]/sizes[:,0]) # width/height
            good = np.isfinite(r)
            r = r[good]
            w = w[good]
            w /= np.sum(w)
            r_avg = np.sum(w*r)
            return r_avg
        except:
            return 1.0

    @wrap(entering, exiting)
    def get_font_size(self, font, font_size_px):
        """
        Returns the font-size which corresponds to FONT_SIZE_PX pixels font height.
        """
        m = self.font_model[font.name]
        return m[0]*font_size_px + m[1] #linear model

    @wrap(entering, exiting)
    def sample(self):
        """
        Samples from the font state distribution
        """
        return {
            'font': self.fonts[int(np.random.randint(0, len(self.fonts)))],
            'size': self.size[1]*np.random.randn() + self.size[0],
            'underline': np.random.rand() < self.underline,
            'underline_adjustment': max(2.0, min(-2.0, self.underline_adjustment[1]*np.random.randn() + self.underline_adjustment[0])),
            'strong': np.random.rand() < self.strong,
            'oblique': np.random.rand() < self.oblique,
            'strength': (self.strength[1] - self.strength[0])*np.random.rand() + self.strength[0],
            'char_spacing': int(self.kerning[3]*(np.random.beta(self.kerning[0], self.kerning[1])) + self.kerning[2]),
            'border': np.random.rand() < self.border,
            'random_caps': np.random.rand() < self.random_caps,
            'capsmode': random.choice(self.capsmode),
            'curved': np.random.rand() < self.curved,
            'random_kerning': np.random.rand() < self.random_kerning,
            'random_kerning_amount': self.random_kerning_amount,
        }

    @wrap(entering, exiting)
    def init_font(self,fs):
        """
        Initializes a pygame font.
        FS : font-state sample
        """
        font = freetype.Font(fs['font'], size=fs['size'])
        font.underline = fs['underline']
        font.underline_adjustment = fs['underline_adjustment']
        font.strong = fs['strong']
        font.oblique = fs['oblique']
        font.strength = fs['strength']
        #font.char_spacing = fs['char_spacing']
        font.antialiased = True
        font.origin = True
        return font


class TextSource(object):
    """
    Provides text for words, paragraphs, sentences.
    """

    @wrap(entering, exiting)
    def __init__(self, min_nchar, fn, lang="ENG"):
        """
        TXT_FN : path to file containing text data.
        """
        self.min_nchar = min_nchar
        self.fdict = {'WORD':self.sample_word,
                      'LINE':self.sample_line,
                      'PARA':self.sample_para}
        self.lang = lang
        self.characters_range=configuration.range[self.lang]
        self.words=[]
        with open(fn, 'r') as f:
            self.txt = [l.strip() for l in f.readlines()]
            for l in self.txt:
                words = l.split()
                for word in words:
                    self.words.append(word)
            
        # distribution over line/words for LINE/PARA:
        self.p_line_nline = np.array([0.85, 0.10, 0.05])
        self.p_line_nword = [4,3,12]  # normal: (mu, std)
        self.p_para_nline = [1.0,1.0]#[1.7,3.0] # beta: (a, b), max_nline
        self.p_para_nword = [1.7,3.0,10] # beta: (a,b), max_nword

        # probability to center-align a paragraph:
        self.center_para = 0.5

    def is_valid_character(self, char):
        p = any([range["from"] <= char <= range["to"] for range in self.characters_range])
        return p
    
    @wrap(entering, exiting)
    def check_symb_frac(self, txt, f=0.2):  # f =0 means no special chracter
        """
        T/F return : T iff fraction of symbol/special-charcters in
                     txt is less than or equal to f (default=0.25).
        """
        if self.lang == "ENG":
            return np.sum([not ch.isalnum() for ch in txt])/(len(txt)+0.0) <= f
        else:
            chcnt = 0
            line = txt  # .decode('utf-8')
            for ch in line:
                if ch in configuration.special_char:
                    chcnt += 1
            return float(chcnt) / (len(txt) + 0.0) <= f

    def check_valid_word(self, w):
        for ch in w:
            if not (self.is_valid_character(ch) or ch in configuration.special_char):
                return False
        return True
    
    def is_good(self, txt, f=0.35):
        """
        T/F return : T iff the lines in txt (a list of txt lines)
                     are "valid".
                     A given line l is valid iff:
                         1. It is not empty.
                         2. symbol_fraction > f
                         3. Has at-least self.min_nchar characters
                         4. Not all characters are i,x,0,O,-
        """
        def is_txt(l):
            char_ex = ['i','I','o','O','0','-']
            chs = [ch in char_ex for ch in l]
            return not np.all(chs)

        x= [ (len(l)> self.min_nchar
                and self.check_valid_word(l)
                 and self.check_symb_frac(l,f)
                 and is_txt(l)) for l in txt ]
        return x

    @wrap(entering, exiting)
    def center_align(self, lines):
        """
        PADS lines with space to center align them
        lines : list of text-lines.
        """
        ls = [len(l) for l in lines]
        max_l = max(ls)
        for i in range(len(lines)):
            l = lines[i].strip()
            dl = max_l-ls[i]
            lspace = dl//2
            rspace = dl-lspace
            lines[i] = ' '*lspace+l+' '*rspace
        return lines

    @wrap(entering, exiting)
    def get_lines(self, nline, nword, nchar_max, f=0.35, niter=100):
        def h_lines(niter=100):
            lines = ['']
            iter = 0
            while not np.all(self.is_good(lines, f)) and iter < niter:
                iter += 1
                line_start = np.random.choice(len(self.txt) - nline)
                lines = [self.txt[line_start + i] for i in range(nline)]
            return lines
    
        lines = ['']
        iter = 0
        while not np.all(self.is_good(lines, f)) and iter < niter:
            iter += 1
            lines = h_lines(niter=100)
            # get words per line:
            nline = len(lines)
            for i in range(nline):
                words = lines[i].split()
                dw = len(words) - nword[i]
                if dw > 0:
                    first_word_index = random.choice(range(dw + 1))
                    lines[i] = ' '.join(words[first_word_index:first_word_index + nword[i]])
            
                while len(lines[i]) > nchar_max:  # chop-off characters from end:
                    if not np.any([ch.isspace() for ch in lines[i]]):
                        lines[i] = ''
                    else:
                        lines[i] = lines[i][:len(lines[i]) - lines[i][::-1].find(' ')].strip()
    
        if not np.all(self.is_good(lines, f)):
            return  # None
        else:
            return lines

    @wrap(entering, exiting)
    def sample(self, nline_max,nchar_max,kind='WORD'):
        return self.fdict[kind](nline_max,nchar_max)

    @wrap(entering, exiting)
    def sample_word(self,nline_max,nchar_max,niter=100):
        rand_line = self.txt[np.random.choice(len(self.txt))]
        words = rand_line.split()
        rand_word = random.choice(words)
        
        iter = 0
        while iter < niter and (not self.is_good([rand_word])[0] or len(rand_word)>nchar_max):
            rand_line = self.txt[np.random.choice(len(self.txt))]
            words = rand_line.split()
            rand_word = random.choice(words)
            iter += 1
            
        if not self.is_good([rand_word])[0] or len(rand_word)>nchar_max:
            return []
        else:
            return rand_word

    @wrap(entering, exiting)
    def sample_line(self,nline_max,nchar_max):
        nline = nline_max+1
        while nline > nline_max:
            nline = np.random.choice([1,2,3], p=self.p_line_nline)

        # get number of words:
        nword = [self.p_line_nword[2]*sstat.beta.rvs(a=self.p_line_nword[0], b=self.p_line_nword[1])
                 for _ in range(nline)]
        nword = [max(1,int(np.ceil(n))) for n in nword]

        lines = self.get_lines(nline, nword, nchar_max, f=0.35)
        
        if lines is not None:
            return '\n'.join(lines)
        else:
            return []

    @wrap(entering, exiting)
    def sample_para(self,nline_max,nchar_max):
        # get number of lines in the paragraph:
        nline = nline_max*sstat.beta.rvs(a=self.p_para_nline[0], b=self.p_para_nline[1])
        nline = max(1, int(np.ceil(nline)))

        # get number of words:
        nword = [self.p_para_nword[2]*sstat.beta.rvs(a=self.p_para_nword[0], b=self.p_para_nword[1])
                 for _ in range(nline)]
        nword = [max(1,int(np.ceil(n))) for n in nword]

        lines = self.get_lines(nline, nword, nchar_max, f=0.35)
        if lines is not None:
            # center align the paragraph-text:
            if np.random.rand() < self.center_para:
                lines = self.center_align(lines)
            return '\n'.join(lines)
        else:
            return []
