# Copyright (C) 2024-2025 Nithin PS.
# This file is part of Pyrebel.
#
# Pyrebel is free software: you can redistribute it and/or modify it under the terms of 
# the GNU General Public License as published by the Free Software Foundation, either 
# version 3 of the License, or (at your option) any later version.
#
# Pyrebel is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; 
# without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR 
# PURPOSE. See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with Pyrebel.
# If not, see <https://www.gnu.org/licenses/>.
#

# This is a demo of live pattern recognition running on zumo-jetson robot.
#

import numpy as np
from numba import cuda
from PIL import Image
from scipy.io.wavfile import write
import math,argparse,time,os,serial,wave,sys,itertools
import sounddevice as sd
from pyrebel.preprocess import Preprocess
from pyrebel.abstract import Abstract
from pyrebel.edge import Edge
from pyrebel.learn import Learn
from pyrebel.utils import *

@cuda.jit
def bounds_at_edge(ba_dec_d,ba_size_d,ba_size_cum_d,bound_data_d,is_over_edge_d,scaled_shape,clearance):
    ci=cuda.grid(1)
    if ci<len(ba_size_d):
        for i in range(ba_size_d[ci]):
            index=bound_data_d[ba_dec_d[ba_size_cum_d[ci]+i]]
            r=int(index/scaled_shape[1])
            c=index%scaled_shape[1]
            if r<clearance or r>(scaled_shape[0]-clearance) or c<clearance or c>(scaled_shape[1]-clearance):
                cuda.atomic.add(is_over_edge_d,ci,1)

def sort_color(bound_seed,scaled_scape,mask_learn):
    learn_array=list()
    for i in range(len(bound_seed)):
        r_=int(bound_seed[i]/scaled_shape[1]/3)
        c_=int((bound_seed[i]%scaled_shape[1])/3)
        if mask_learn[r_][c_]:
            learn_array.append(True)
        else:
            learn_array.append(False)
    return learn_array

def sort_position(seed_points,shape):
    r_array=list()
    c_array=list()
    r_min=shape[0]
    r_max=0
    c_min=shape[1]
    c_max=0
    for i in range(len(seed_points)):
        r_=int(seed_points[i]/shape[1])
        c_=seed_points[i]%shape[1]
        r_array.append(r_)
        c_array.append(c_)
        if r_<r_min:
            r_min=r_
        if c_<c_min:
            c_min=c_
        if r_>r_max:
            r_max=r_
        if c_>c_max:
            c_max=c_
    r_diff=(r_max-r_min)
    c_diff=(c_max-c_min)
    if r_diff>c_diff:
        return np.argsort(r_array)
    else:
        return np.argsort(c_array)
                                                
parser=argparse.ArgumentParser()
parser.add_argument("-i","--input",help="Input file name.")
parser.add_argument("-c","--camera",help="Input from camera.")
parser.add_argument("-at","--abs_threshold",help="Threshold of abstraction.")
parser.add_argument("-et","--edge_threshold",help="Threshold of edge detection.")
parser.add_argument("-b","--block_threshold",help="Block threshold.")
parser.add_argument("-p","--paint_threshold",help="Paint threshold.")

parser.add_argument("-lhl","--low_hue_learn",help="Minimum hue value")
parser.add_argument("-hhl","--high_hue_learn",help="Maximum hue value")
parser.add_argument("-lsl","--low_saturation_learn",help="Minimum saturation")
parser.add_argument("-hsl","--high_saturation_learn",help="Maxmimum saturation")
parser.add_argument("-lvl","--low_value_learn",help="Minimum value")
parser.add_argument("-hvl","--high_value_learn",help="Maxmimum value")

parser.add_argument("-lhr","--low_hue_recognize",help="Minimum hue value")
parser.add_argument("-hhr","--high_hue_recognize",help="Maximum hue value")
parser.add_argument("-lsr","--low_saturation_recognize",help="Minimum saturation")
parser.add_argument("-hsr","--high_saturation_recognize",help="Maxmimum saturation")
parser.add_argument("-lvr","--low_value_recognize",help="Minimum value")
parser.add_argument("-hvr","--high_value_recognize",help="Maxmimum value")

parser.add_argument("-l","--learn",help="Symbol to learn.")
parser.add_argument("-r","--recognize",help="Recognize the signature.")
parser.add_argument("-u","--user",help="Name of the user.")
args=parser.parse_args()

# Green for recognizing
low_hue_green=60
high_hue_green=120
low_saturation_green=50

# Blue for learning
low_hue_blue=150
high_hue_blue=190
low_saturation_blue=80

if args.low_hue_learn:
    low_hue_learn=int(args.low_hue_learn)
else:
    low_hue_learn=low_hue_blue
    
if args.high_hue_learn:
    high_hue_learn=int(args.high_hue_learn)
else:
    high_hue_learn=high_hue_blue
    
if args.low_saturation_learn:
    low_saturation_learn=int(args.low_saturation_learn)
else:
    low_saturation_learn=low_saturation_blue
    
if args.high_saturation_learn:
    high_saturation_learn=int(args.high_saturation_learn)
else:
    high_saturation_learn=255
    
if args.low_value_learn:
    low_value_learn=int(args.low_value_learn)
else:
    low_value_learn=0
    
if args.high_value_learn:
    high_value_learn=int(args.high_value_learn)    
else:
    high_value_learn=255


if args.low_hue_recognize:
    low_hue_recognize=int(args.low_hue_recognize)
else:
    low_hue_recognize=low_hue_green
    
if args.high_hue_recognize:
    high_hue_recognize=int(args.high_hue_recognize)
else:
    high_hue_recognize=high_hue_green
    
if args.low_saturation_recognize:
    low_saturation_recognize=int(args.low_saturation_recognize)
else:
    low_saturation_recognize=low_saturation_green
    
if args.high_saturation_recognize:
    high_saturation_recognize=int(args.high_saturation_recognize)
else:
    high_saturation_recognize=255
    
if args.low_value_recognize:
    low_value_recognize=int(args.low_value_recognize)
else:
    low_value_recognize=0
    
if args.high_value_recognize:
    high_value_recognize=int(args.high_value_recognize)
else:
    high_value_recognize=255
    
if args.abs_threshold:
    abs_threshold=int(args.abs_threshold)
else:
    abs_threshold=5

 
from jetson_utils import videoSource, videoOutput, Log
from jetson_utils import cudaAllocMapped,cudaConvertColor
from jetson_utils import cudaToNumpy,cudaDeviceSynchronize,cudaFromNumpy
def convert_color(img,output_format):
    converted_img=cudaAllocMapped(width=img.width,height=img.height,
            format=output_format)
    cudaConvertColor(img,converted_img)
    return converted_img

input_capture = videoSource("csi://0", options={'width':480,'height':360,'framerate':30,'flipMethod':'rotate-360'})
#output = videoOutput("", argv=sys.argv)
input_capture.Capture()


flip=True

if not os.path.exists('index.txt'):
    fp=open('index.txt','x')
    fp.close()
with open("index.txt","r") as fpr:
    try:
        content=str(fpr.read())
        if content=='':
            index=0
        else:
            index=int(content)
    except EOFError:
        index=0
print()
print("low_hue_learn=",low_hue_learn)
print("high_hue_learn=",high_hue_learn)
print("low_saturation_learn=",low_saturation_learn)
print("high_saturation_learn=",high_saturation_learn)
print("low_value_learn=",low_value_learn)
print("high_value_learn=",high_value_learn)
print()
print("low_hue_recognize=",low_hue_recognize)
print("high_hue_recognize=",high_hue_recognize)
print("low_saturation_recognize=",low_saturation_recognize)
print("high_saturation_recognize=",high_saturation_recognize)
print("low_value_recognize=",low_value_recognize)
print("high_value_recognize=",high_value_recognize)
print()

ser=serial.Serial('/dev/ttyACM0',115200,timeout=1,write_timeout=2)
time.sleep(2)
os.system("sudo -u "+args.user+" espeak-ng hello")

while 1:
    init_time=time.time()    
    if args.camera:
        # capture the next image
        img_array_rgb = input_capture.Capture()
        if img_array_rgb is None: # timeout
            print("No camera capture!")
            continue  
        img_gray=convert_color(img_array_rgb,'gray8')
        cudaDeviceSynchronize()
        img_array=cudaToNumpy(img_gray)
        img_rgb=cudaToNumpy(img_array_rgb)
        cudaDeviceSynchronize()
        Image.fromarray(img_rgb).save("camera.png")
        cudaDeviceSynchronize()
        img_array=img_array.reshape(1,img_array.shape[0],img_array.shape[1])[0].astype('int32')
    elif args.input:
        img_array=np.array(Image.open(args.input).convert('L'))
        img_array_rgb=np.array(Image.open(args.input).convert('RGB'))
    else:
        print("No input file.")
    block_img_h=np.full(img_array.shape,200,dtype=np.int32)
    block_hsv=np.array(Image.fromarray(img_rgb).convert('HSV'))
    img_sample=Image.new('RGB',(1,1),color='blue')
    img_sample_hsv=np.array(img_sample.convert('HSV'))
    #print(img_sample_hsv)
    blob_size=1000
    learn=False
    recognize=False
    
    lower_mask=block_hsv[:,:,0]>=low_hue_recognize
    upper_mask=block_hsv[:,:,0]<=high_hue_recognize
    low_saturation_mask=block_hsv[:,:,1]>=low_saturation_recognize
    high_saturation_mask=block_hsv[:,:,1]<=high_saturation_recognize
    low_value_mask=block_hsv[:,:,2]>=low_value_recognize
    high_value_mask=block_hsv[:,:,2]<=high_value_recognize
    mask_recognize=upper_mask*lower_mask*low_saturation_mask*high_saturation_mask*low_value_mask*high_value_mask
    if mask_recognize.sum()>blob_size:
        recognize=True
    else:
        recognize=False
    
    lower_mask=block_hsv[:,:,0]>=low_hue_learn
    upper_mask=block_hsv[:,:,0]<=high_hue_learn
    low_saturation_mask=block_hsv[:,:,1]>=low_saturation_learn
    high_saturation_mask=block_hsv[:,:,1]<=high_saturation_learn
    low_value_mask=block_hsv[:,:,2]>=low_value_learn
    high_value_mask=block_hsv[:,:,2]<=high_value_learn
    mask_learn=upper_mask*lower_mask*low_saturation_mask*high_saturation_mask*low_value_mask*high_value_mask
    if mask_learn.sum()>blob_size:
        learn=True
    else:
        learn=False
    
    block_img_masked=block_img_h*np.logical_or(mask_recognize,mask_learn)
    Image.fromarray(block_img_masked).convert('RGB').save("hsv.png")
    img_array=np.array(Image.fromarray(block_img_masked).convert('L'))

    if learn and recognize:
        try:
            ser.write("blinkledwhite_0\n".encode())
            time.sleep(0.5)
            ser.write("blinkledgreen_1\n".encode())
            time.sleep(0.5)
            ser.write("blinkledblue_1\n".encode())
            time.sleep(0.5)
        except serial.SerialTimeoutException:
            print("serial exception.")
            ser.flushInput()
            ser.flushOutput()
            pass
        print("\nFound pattern to Learn and recognize.")
        #time.sleep(2)
        #continue
    elif learn:
        try:
            ser.write("blinkledwhite_0\n".encode())
            time.sleep(0.5)
            ser.write("blinkledgreen_0\n".encode())
            time.sleep(0.5)
            ser.write("blinkledblue_1\n".encode())
            time.sleep(0.5)
        except serial.SerialTimeoutException:
            print("serial exception.")
            ser.flushInput()
            ser.flushOutput()
            pass
        print("\nFound pattern to learn. ")
        mask=mask_learn
    elif recognize:
        try:
            ser.write("blinkledwhite_0\n".encode())
            time.sleep(0.5)
            ser.write("blinkledgreen_1\n".encode())
            time.sleep(0.5)
            ser.write("blinkledblue_0\n".encode())
            time.sleep(0.5)
        except serial.SerialTimeoutException:
            print("serial exception.")
            ser.flushInput()
            ser.flushOutput()
            pass
        print("\nFound pattern to recognize.")
        mask=mask_recognize
    else:
        if flip:
            print("Looking for patterns. ",end='\r')
            flip=False
        else:
            print("Looking for patterns..",end='\r')
            flip=True
        try:
            ser.write("blinkledgreen_0\n".encode())
            time.sleep(0.5)
            ser.write("blinkledblue_0\n".encode())
            time.sleep(0.5)
            ser.write("blinkledwhite_1\n".encode())
            time.sleep(0.5)
        except serial.SerialTimeoutException:
            print("serial exception.")
            ser.flushInput()
            ser.flushOutput()
            pass
        time.sleep(1)
        continue
    
    start_time=time.time()
    prt=time.time()
    # Initialize the preprocessing class.
    pre=Preprocess(img_array)
    # Set the minimum and maximum size of boundaries of blobs in the image. Defaults to a minimum of 64.
    pre.set_bound_size(500)    
    # Perform the preprocessing to get 1D array containing boundaries of blobs in the image.
    pre.preprocess_image()
    print("preprocessed frame in",time.time()-prt)
    # Get the 1D array.
    bound_data=pre.get_bound_data()
    bound_data_d=cuda.to_device(bound_data)
    # Initialize the abstract boundary.
    init_bound_abstract=pre.get_init_abstract()
    # Get 1D array containing size of boundaries of blobs in the array.
    bound_size=pre.get_bound_size()
    if len(bound_size)<3:
        continue
    scaled_shape=pre.get_image_scaled().shape
    scaled_shape_d=cuda.to_device(scaled_shape)
    bound_seed=pre.get_bound_seed()
    bound_seed_d=cuda.to_device(bound_seed)

    print("len(bound_data)=",len(bound_data))
    print("n_blobs=",len(bound_size))
    
    # Select largest blob
    #blob_index=np.argsort(bound_size[2:])[-1]+2
    #print("blob_index=",blob_index)
    
       
    # Initialize the abstraction class
    abs=Abstract(bound_data,len(bound_size),init_bound_abstract,scaled_shape,True)
    
    n_layers=50
    # Initialize learning class
    l=Learn(n_layers,len(bound_size),3)
    i=3
    blob_over_edge=0
    blob_bounds_count=10
    print("len(know_base)=",len(l.get_know_base()))
    fst=time.time()
    while 1:
        # Do one layer of abstraction
        is_finished_abs=abs.do_abstract_one(abs_threshold)
        ba_sign=abs.get_sign()
        ba_size=abs.get_abstract_size()
        
        if i==blob_bounds_count:
            ba=abs.get_abstract()
            ba_size_d=cuda.to_device(ba_size)
            ba_size_cum_=np.cumsum(ba_size)
            ba_size_cum=np.delete(np.insert(ba_size_cum_,0,0),-1)
            ba_size_cum_d=cuda.to_device(ba_size_cum)
            ba_dec=decrement_by_one_cuda(ba)
            ba_dec_d=cuda.to_device(ba_dec)
            #draw_pixels_from_indices_cuda(ba_points_dec_d,bound_data_d,200,out_image_d)
            #out_image_h=out_image_d.copy_to_host()
            #Image.fromarray(out_image_h).convert('RGB').save("blob.png")
            is_over_edge=np.zeros(len(ba_size),dtype=np.int32)
            is_over_edge_d=cuda.to_device(is_over_edge)
            clearance=5
            bounds_at_edge[len(ba_size),1](ba_dec_d,ba_size_d,ba_size_cum_d,bound_data_d,is_over_edge_d,scaled_shape_d,clearance)
            cuda.synchronize()
            is_over_edge_h=is_over_edge_d.copy_to_host()
            #print(is_over_edge_h)
            blob_over_edge=np.count_nonzero(is_over_edge_h[2:])
            if blob_over_edge>0:
                break
            bound_size_i_d=cuda.to_device(bound_size)
            increment_by_one[len(bound_size),1](bound_size_i_d)
            cuda.synchronize()
            bound_size_i=bound_size_i_d.copy_to_host()
            bound_size_i_cum_=np.cumsum(bound_size_i)
            bound_size_i_cum=np.delete(np.insert(bound_size_i_cum_,0,0),-1)
            bound_size_i_cum_d=cuda.to_device(bound_size_i_cum)
    
            out_image=np.zeros(scaled_shape,dtype=np.int32)
            out_image_d=cuda.to_device(out_image)
    

            # Find blobs not inside another blob
            is_inside=np.zeros(len(bound_size),dtype=np.int32)            
            is_inside_d=cuda.to_device(is_inside)
            threadsperblock=(16,16)
            blockspergrid_x=math.ceil(len(bound_size)/threadsperblock[0])
            blockspergrid_y=math.ceil(len(bound_size)/threadsperblock[1])
            blockspergrid=(blockspergrid_x,blockspergrid_y)
            is_blob_inside[blockspergrid,threadsperblock](bound_size_i_d,bound_size_i_cum_d,bound_data_d,bound_seed_d,scaled_shape_d,is_inside_d)
            cuda.synchronize()
            is_inside_h=is_inside_d.copy_to_host()
            #print(is_inside_h)

            # Sort blobs by position
            sorted_indices=sort_position(bound_seed,scaled_shape)

    
            # Sort green and blue blobs
            sorted_color=sort_color(bound_seed,scaled_shape,mask_learn) # True is blue (learn) and False is green (recognize)
            #print(sorted_color)
            #blob_indices=[i for i, elem in enumerate(is_inside_h) if elem==0] # indices of outer blobs.
            #blob_indices=[i for i in sorted_indices if is_inside_h[i]==0 and i!=0 and i!=1] # indices of outer blobs in order of position.
            blue_indices=[i for i in sorted_indices if is_inside_h[i]==1 and sorted_color[i]==True and i!=0 and i!=1] # indices of blue outer blobs in order of position
            green_indices=[i for i in sorted_indices if is_inside_h[i]==1 and sorted_color[i]==False and i!=0 and i!=1] # indices of green outer blobs in order of position
            print("blue_indices=",blue_indices)
            print("green_indices=",green_indices)

            # draw blue blobs as dark grey color
            for blob in blue_indices:
                blob_index_data=bound_data[bound_size_i_cum[blob]:bound_size_i_cum[blob]+bound_size_i[blob]]
                blob_index_data_d=cuda.to_device(blob_index_data)            
                draw_pixels_cuda(blob_index_data_d,100,out_image_d)
            # draw green blobs as light grey color
            for blob in green_indices:
                blob_index_data=bound_data[bound_size_i_cum[blob]:bound_size_i_cum[blob]+bound_size_i[blob]]
                blob_index_data_d=cuda.to_device(blob_index_data)            
                draw_pixels_cuda(blob_index_data_d,250,out_image_d)
            
            out_image_h=out_image_d.copy_to_host()            
            # Save the outer blob to disk.
            Image.fromarray(out_image_h).convert('RGB').save("blob.png")
 
        # Find signatures for the layer    
        is_finished=l.find_signatures2(ba_sign,ba_size)    
        if is_finished or is_finished_abs:
            print("layers=",i-1)
            l.init_signatures()
            break
        i+=1
    print("found signatures in",time.time()-fst)
    
    if blob_over_edge:
        print("blob over edge. continuing..")
        os.system("sudo -u "+args.user+" espeak-ng \"blob over edge\"")
        continue    
    top_n=1
    if learn and recognize:
        print("Learning and recognizing..")
        print("1. recognizing..")
        rt=time.time()
        recognized=l.recognize_sym(green_indices,top_n,"image")
        print("symbols found=",recognized)
        
        if len(recognized[1][0])>0:
            sign_name_recognized=recognized[1][0][0][0]
            print("Do you want to learn "+sign_name_recognized)
            os.system("sudo -u "+args.user+" espeak-ng \"Do you want to learn \"")

            if os.path.exists(sign_name_recognized+".wav"):
                try:
                    os.system("sudo -u "+args.user+" aplay "+sign_name_recognized+".wav")
                except Exception as e:
                    pass
            else:
                os.system("sudo -u "+args.user+" espeak-ng "+sign_name_recognized)
        else:
            os.system("sudo -u "+args.user+" espeak-ng \"could not recognize.\"")
            continue
        #os.system("sudo -u "+args.user+" espeak-ng \"Looks like\"")
        print("recognize time=",time.time()-rt)
        
        #os.system("sudo -u "+args.user+" arecord -d 5 "+sign_name)
        
        freq = 44100
        duration=5
        # Start recorder with the given values 
        # of duration and sample frequency
        recording = sd.rec(int(duration * freq), 
                           samplerate=freq, channels=1,device=11)
        sd.wait()
        """
        wav_obj=wave.open(sign_name,mode='rb')
        n_frames=wav_obj.getnframes()
        frame_data=wav_obj.readframes(n_frames)
        recording=np.frombuffer(frame_data,dtype=np.uint8)
        """
        print(np.max(recording))
        if np.max(recording)<0.1:
            print("Cant hear anything.")
            os.system("sudo -u "+args.user+" espeak-ng \"Cant hear anything\"")
            continue
        
        print("Okay.")
        os.system("sudo -u "+args.user+" espeak-ng Okay")
        print("2. learning..")
        lt=time.time()
        learn_out=l.learn_sym(blue_indices,sign_name_recognized,"image")
        sign_len=len(learn_out)
   
        print("learning",sign_name_recognized,learn_out,"len(signs)=",sign_len)
        print("learn time=",time.time()-lt)
        l.write_know_base() 

    elif recognize:
        print("recognizing..")
        rt=time.time()        
        channel="all"        
        found_image=False
        found_audio=False
        wt_image=0
        wt_audio=0
        if channel=="image":
            recognized=l.recognize_sym(green_indices,top_n,"image")         
            print("channel=image","symbols found=",recognized)
            sym=list(itertools.chain(*recognized[0]))
            sym+=list(itertools.chain(*recognized[1]))
            sym.sort(key=lambda x: x[1],reverse=True)
            if len(sym)>0:
                found_image=True
                wt_image=sym[0][1]
            if len(recognized[1][0])>0:
                sign_name_recognized=recognized[1][0][0][0]
                os.system("sudo -u "+args.user+" espeak-ng \"Looks like  \""+sign_name_recognized)
            elif len(recognized[0][0])>0:
                for r in recognized[0]:
                    sign_name_recognized=r[0][0] 
                    os.system("sudo -u "+args.user+" espeak-ng \"Looks like  \""+sign_name_recognized)
            else:
                os.system("sudo -u "+args.user+" espeak-ng \"could not recognize.\"")
                continue
        elif channel=="audio":
            recognized_audio=l.recognize_sym(green_indices,top_n,"audio")
            print("channel=audio","symbols found=",recognized_audio)
            sym=list(itertools.chain(*recognized_audio[0]))
            sym+=list(itertools.chain(*recognized_audio[1]))
            sym.sort(key=lambda x: x[1],reverse=True)

            if len(sym)>0:
                found_audio=True
                wt_audio=sym[0][1]
            if len(recognized_audio[1][0])>0:
                sign_name_recognized_audio=recognized_audio[1][0][0][0]
                os.system("sudo -u "+args.user+" espeak-ng \"Looks like  \"")
                if os.path.exists(sign_name_recognized_audio):
                    try:
                        os.system("sudo -u "+args.user+" aplay "+sign_name_recognized_audio)
                    except Exception as e:
                        pass

            elif len(recognized_audio[0][0])>0:
                os.system("sudo -u "+args.user+" espeak-ng \"Looks like  \"")
                for r in recognized_audio[0]:
                    sign_name_recognized_audio=r[0][0]                    
                    if os.path.exists(sign_name_recognized_audio):
                        try:
                            os.system("sudo -u "+args.user+" aplay "+sign_name_recognized_audio)
                        except Exception as e:
                            pass
            else:
                os.system("sudo -u "+args.user+" espeak-ng \"could not recognize.\"")
                continue
        elif channel=="all":
            recognized=l.recognize_sym(green_indices,top_n,"image")            
            print("channel=image","symbols found=",recognized)
            sym=list(itertools.chain(*recognized[0]))
            sym+=list(itertools.chain(*recognized[1]))
            sym.sort(key=lambda x: x[1],reverse=True)
            if len(sym)>0:
                found_image=True
                wt_image=sym[0][1]
            
            recognized_audio=l.recognize_sym(green_indices,top_n,"audio")
            print("channel=audio","symbols found=",recognized_audio)
            sym=list(itertools.chain(*recognized_audio[0]))
            sym+=list(itertools.chain(*recognized_audio[1]))
            sym.sort(key=lambda x: x[1],reverse=True)

            if len(sym)>0:
                found_audio=True
                wt_audio=sym[0][1]
            if found_image and found_audio:
                if wt_image>wt_audio:
                    if len(recognized[1][0])>0:
                        sign_name_recognized=recognized[1][0][0][0]
                        os.system("sudo -u "+args.user+" espeak-ng \"Looks like  \""+sign_name_recognized)
                    elif len(recognized[0][0])>0:
                        for r in recognized[0]:
                            sign_name_recognized=r[0][0] 
                            os.system("sudo -u "+args.user+" espeak-ng \"Looks like  \""+sign_name_recognized)
                else:
                    if len(recognized_audio[1][0])>0:
                        sign_name_recognized_audio=recognized_audio[1][0][0][0]
                        os.system("sudo -u "+args.user+" espeak-ng \"Looks like  \"")
                        if os.path.exists(sign_name_recognized_audio):
                            try:
                                os.system("sudo -u "+args.user+" aplay "+sign_name_recognized_audio)
                            except Exception as e:
                                pass

                    elif len(recognized_audio[0][0])>0:
                        os.system("sudo -u "+args.user+" espeak-ng \"Looks like  \"")
                        for r in recognized_audio[0]:
                            sign_name_recognized_audio=r[0][0]                    
                            if os.path.exists(sign_name_recognized_audio):
                                try:
                                    os.system("sudo -u "+args.user+" aplay "+sign_name_recognized_audio)
                                except Exception as e:
                                    pass
            elif found_image:
                if len(recognized[1][0])>0:
                    sign_name_recognized=recognized[1][0][0][0]
                    os.system("sudo -u "+args.user+" espeak-ng \"Looks like  \""+sign_name_recognized)
                elif len(recognized[0][0])>0:
                    for r in recognized[0]:
                        sign_name_recognized=r[0][0] 
                        os.system("sudo -u "+args.user+" espeak-ng \"Looks like  \""+sign_name_recognized)
            elif found_audio:
                if len(recognized_audio[1][0])>0:
                    sign_name_recognized_audio=recognized_audio[1][0][0][0]
                    os.system("sudo -u "+args.user+" espeak-ng \"Looks like  \"")
                    if os.path.exists(sign_name_recognized_audio):
                        try:
                            os.system("sudo -u "+args.user+" aplay "+sign_name_recognized_audio)
                        except Exception as e:
                            pass

                elif len(recognized_audio[0][0])>0:
                    os.system("sudo -u "+args.user+" espeak-ng \"Looks like  \"")
                    for r in recognized_audio[0]:
                        sign_name_recognized_audio=r[0][0]                    
                        if os.path.exists(sign_name_recognized_audio):
                            try:
                                os.system("sudo -u "+args.user+" aplay "+sign_name_recognized_audio)
                            except Exception as e:
                                pass
            else:
                os.system("sudo -u "+args.user+" espeak-ng \"could not recognize.\"")
                continue

            
        #os.system("sudo -u "+args.user+" espeak-ng \"Looks like\"")
        print("recognize time=",time.time()-rt)
        
        if found_image and wt_image>wt_audio:
            if len(recognized[0][0])>0:
                for sign_name in recognized[0]:
                    sign_name_recognized=sign_name[0][0] 
                    if sign_name_recognized=="fw":
                        print("moving forward..")
                        ser.write("moveforward_100".encode())
                        time.sleep(0.5)
                        ser.write("blinkled_1".encode())
                    elif sign_name_recognized=="bw":
                        print("moving backward..")
                        ser.write("movebackward_100".encode())
                    elif sign_name_recognized=="sc":
                        print("turning right..")
                        ser.write("spincw_100".encode())
                    elif sign_name_recognized=="sa":
                        print("turning left..")
                        ser.write("spinacw_100".encode())
                    time.sleep(0.5)
        #time.sleep(3)
        
    elif learn:
        print("learning..")
        lt=time.time()
        sign_name=str(index)+".wav"
        print("Please tell me the name of the pattern? You have 5 seconds.")
        os.system("sudo -u "+args.user+" espeak-ng \"Please tell me the name of the pattern? You have 5 seconds.\"")
        #os.system("sudo -u "+args.user+" arecord -d 5 "+sign_name)
        
        freq = 44100
        duration=5
        # Start recorder with the given values 
        # of duration and sample frequency
        recording = sd.rec(int(duration * freq), 
                           samplerate=freq, channels=1,device=11)
        sd.wait()
        """
        wav_obj=wave.open(sign_name,mode='rb')
        n_frames=wav_obj.getnframes()
        frame_data=wav_obj.readframes(n_frames)
        recording=np.frombuffer(frame_data,dtype=np.uint8)
        """
        print(np.max(recording))
        if np.max(recording)<0.1:
            print("Cant hear anything.")
            os.system("sudo -u "+args.user+" espeak-ng \"Cant hear anything\"")
            continue
        
        norm=0.9/np.max(recording)
        recording*=norm
        write(sign_name,freq,recording)

        print("Okay.")
        os.system("sudo -u "+args.user+" espeak-ng Okay")
        learn_out=l.learn_sym(blue_indices,sign_name,"audio")
        sign_len=len(learn_out)
   
        if len(learn_out)>0:
            index+=1
            file=open('index.txt','w')
            file.write(str(index))
            file.close()
        print("learning",sign_name,learn_out,"len(signs)=",sign_len)
        print("learn time=",time.time()-lt)
        l.write_know_base() 
    
    print("Finished in",time.time()-start_time,"seconds at",float(1/(time.time()-start_time)),"fps.")
    print()
    
    
