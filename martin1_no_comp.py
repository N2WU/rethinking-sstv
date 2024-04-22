import numpy as np
from PIL import Image
import sounddevice as sd
import matplotlib.pyplot as plt

def martin1(img_array,fs):
    ms_pix = 0.4576/1000
    ms_syn = 4.862/1000
    ms_pul = 0.572/1000
    t_pix = np.arange(int(ms_pix*fs))/fs
    t_syn = np.arange(int(ms_syn*fs))/fs
    t_pul = np.arange(int(ms_pul*fs))/fs
    nlines = int(len(img_array[:,0,0]))
    npix = int(len(img_array[0,:,0]))
    nsamples = len(t_syn) + 4*len(t_pul) + 3*npix*len(t_pix)
    s = np.zeros((nlines,int(nsamples)))
    for line in range(nlines):
        # sync pulse
        ns = 0
        s[line,:len(t_syn)] = np.cos(2*np.pi*1200*t_syn)
        ns += len(t_syn)
        # sync porch
        s[line,ns:ns+len(t_pul)] = np.cos(2*np.pi*1500*t_pul)
        ns += len(t_pul)
        # green scan
        img_g = img_array[:,line,1]
        green = np.interp(img_g, (0, 256), (1500, 2300))
        for scan in range(len(green)):
            s[line,ns:ns+len(t_pix)] = np.cos(2*np.pi*green[scan]*t_pix)
            ns += len(t_pix)
        # separator pulse
        s[line,ns:ns+len(t_pul)] = np.cos(2*np.pi*1500*t_pul)
        ns += len(t_pul)
        # blue scan
        img_b = img_array[:,line,2]
        blue = np.interp(img_b, (0, 256), (1500, 2300))
        for scan in range(len(blue)):
            s[line,ns:ns+len(t_pix)] = np.cos(2*np.pi*blue[scan]*t_pix)
            ns += len(t_pix)
        # separator pulse
        s[line,ns:ns+len(t_pul)] = np.cos(2*np.pi*1500*t_pul)
        ns += len(t_pul)
        # red scan
        img_r = img_array[:,line,0]
        red = np.interp(img_r, (0, 256), (1500, 2300))
        for scan in range(len(red)):
            s[line,ns:ns+len(t_pix)] = np.cos(2*np.pi*red[scan]*t_pix)
            ns += len(t_pix)
        # separator pulse
        s[line,ns:ns+len(t_pul)] = np.cos(2*np.pi*1500*t_pul)
        ns += len(t_pul)
    s = np.reshape(s,(1,-1))
    s = s.flatten()
    s = np.real(s)
    return s

def martin1_decode(r,fs):
    r_img_array = np.zeros_like(img_array) #jank
    # assume already synchronized
    ms_pix = 0.4576/1000
    ms_syn = 4.862/1000
    ms_pul = 0.572/1000
    len_pix = int(ms_pix*fs)
    len_syn = int(ms_syn*fs)
    len_pul = int(ms_pul*fs)
    nlines = int(len(r_img_array[:,0,0]))
    npix = int(len(r_img_array[0,:,0]))
    r_array = np.reshape(r,(nlines,-1))
    r_array = r_array[:,int(len_syn+len_pul):-len_pul] #cut these off outright
    # what's left is g, pul, b, pul, r
    for line in range(nlines):
        # remove sync, porch pulses
        ns = 0
        for pix in range(npix):
            g = np.real(np.fft.fft(r_array[line,ns:ns+len_pix],int(fs)))
            g_freq = np.argmax(g)
            r_img_array[line,pix,1] = np.interp(g_freq, (1500, 2300), (0, 256))
            ns += len_pix
        ns += len_pul
        for pix in range(npix):    
            b = np.real(np.fft.fft(r_array[line,ns:ns+len_pix],int(fs)))
            b_freq = np.argmax(b)
            r_img_array[line,pix,2] = np.interp(b_freq, (1500, 2300), (0, 256))
            ns += len_pix
        ns += len_pul
        for pix in range(npix):
            r = np.real(np.fft.fft(r_array[line,ns:ns+len_pix],int(fs)))
            r_freq = np.argmax(r)
            r_img_array[line,pix,0] = np.interp(r_freq, (1500, 2300), (0, 256))
            ns += len_pix

    return r_img_array

if __name__ == "__main__":
    # constants
    fs = 48e3
    sd.default.samplerate = fs
    
    # load image
    img = Image.open('martin1_crop.jpg')
    img_array = np.asarray(img)

    martin1_sig = martin1(img_array,fs)

    #sd.play(martin1_sig,fs)
    #sd.wait()

    r_img_array = martin1_decode(martin1_sig,fs)

    r_img = Image.fromarray(r_img_array)
    r_img.show()

