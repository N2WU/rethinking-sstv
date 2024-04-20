import numpy as np
from PIL import Image
import sounddevice as sd

def martin1(img_array,fs):
    ms_pix = 0.4576/1000
    ms_syn = 4.862/1000
    ms_pul = 0.572/1000
    t_pix = np.arange(int(ms_pix*fs))/fs
    t_syn = np.arange(int(ms_syn*fs))/fs
    t_pul = np.arange(int(ms_pul*fs))/fs
    s = np.asarray([])
    for line in range(len(img_array[0,:,0])):
        # sync pulse
        s = np.append(s,np.cos(2*np.pi*1200*t_syn))
        # sync porch
        s = np.append(s,np.cos(2*np.pi*1500*t_pul))
        # green scan
        img_g = img_array[:,line,1]
        green = np.interp(img_g, (img_g.min(), img_g.max()), (1500, 2300))
        for scan in range(len(green)):
            s = np.append(s,np.cos(2*np.pi*green[scan]*t_pix/1000))
        # separator pulse
        s = np.append(s,np.cos(2*np.pi*1500*t_pul))
        # blue scan
        img_b = img_array[:,line,2]
        blue = np.interp(img_b, (img_b.min(), img_b.max()), (1500, 2300))
        for scan in range(len(blue)):
            s = np.append(s,np.cos(2*np.pi*blue[scan]*t_pix))
        # separator pulse
        s = np.append(s,np.cos(2*np.pi*1500*t_pul))
        # red scan
        img_r = img_array[:,line,0]
        red = np.interp(img_r, (img_r.min(), img_r.max()), (1500, 2300))
        for scan in range(len(red)):
            s = np.append(s,np.cos(2*np.pi*red[scan]*t_pix))
        # separator pulse
        s = np.append(s,np.cos(2*np.pi*1500*t_pul))
    s = np.real(s)
    return s

if __name__ == "__main__":
    # constants
    nlines = 256
    fs = 100e3
    sd.default.samplerate = 48e3
    
    # load image
    img = Image.open('martin1_crop.jpg')
    img_array = np.asarray(img)

    martin1_sig = martin1(img_array,fs)

    sd.play(martin1_sig)

