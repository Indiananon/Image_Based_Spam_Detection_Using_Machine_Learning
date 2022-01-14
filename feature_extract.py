from PIL import Image
import os
import cv2
import numpy
import math
from pywt import dwt2
import glob
import glob
import sys
from skimage.io import imread
from skimage import color
from scipy.stats import mode
import numpy as np


download_dir = "training.csv"

csv = open(download_dir, "a") 

#columnTitleRow = "avg red value, avg green value, avg blue value, entropy of red, entropy of green, entropy of blue, average entropy, width, height, aspect ratio, image area, file size, compression ratio, bit depth, totalPixels, hue, saturation, value, hue median, saturation median, value median, output\n"
#csv.write(columnTitleRow)
out=0
sum_bpp=0.0
sum_comp=0.0
sum_totalpixels=0.0
for filename in glob.glob(os.path.join('C:\Users\ANONYMOUS\Desktop\minor_project\spam_fa\*.jpg')):
	#csv.write(filename)
	#csv.write(",")
	with Image.open(filename) as image: 
		width, height = image.size 
	#print "width",width
	#print "height",height
	aspect=float(width)/float(height)
	#print "aspect ratio",aspect
	area=width*height
	#print "image area",area
	size=os.stat(filename).st_size
	#print "file size",size
	comp=float(size)/area
	#print "compression",comp
	modeToBpp = {'1':1, 'L':8, 'P':8, 'RGB':24, 'RGBA':32, 'CMYK':32, 'YCbCr':24, 'I':32, 'F':32}
	bpp = modeToBpp[image.mode]
	#print "bit depth",bpp
	myimg = cv2.imread(filename)
	avg_color_per_row = numpy.average(myimg, axis=0)
	avg_color = numpy.average(avg_color_per_row, axis=0)
	#print "avg rgb values",(avg_color) 
	#avg_color
	#for i in range(0,3):
	csv.write(str(avg_color[0]))
	csv.write(",")
	csv.write(str(avg_color[1]))
	csv.write(",")
	csv.write(str(avg_color[2]))
	csv.write(",")
	
	try:
		img = color.rgb2hsv(imread(filename))

		hue = np.mean(img[:,:,0])
#		print hue
		sat = np.mean(img[:,:,1])
#		print sat
		val = np.mean(img[:,:,2])
#		print val
		hue_med = np.median(img[:,:,0])
#		print hue_med
		sat_med = np.median(img[:,:,1])
#		print sat_med
		val_med = np.median(img[:,:,2])
#		print val_med
	except:
		print 'error',filename



#entropy of red ,green and blue
	im = Image.open(filename)
	rgbHistogram = im.histogram()
	sum1=0.0
#	print 'Entropy for Red, Green, Blue:'
	for rgb in range(3):
		totalPixels = sum(rgbHistogram[rgb * 256 : (rgb + 1) * 256])
		ent = 0.0
		for col in range(rgb * 256, (rgb + 1) * 256):
			freq = float(rgbHistogram[col]) / totalPixels
			if freq > 0:
				ent = ent + freq * math.log(freq, 2)
		ent = -ent
		sum1 = sum1 + ent
#		print ent
		entr=str(ent) 
		csv.write(entr)
		csv.write(",")
#	print "total pixels",totalPixels
	entropy_avg=sum1 / 3.0
#	print entropy_avg

	data=str(entropy_avg) + "," +str(width) + "," + str(height) + "," + str(aspect) + "," + str(area) + "," + str(size) + "," + str(comp) + "," + str(bpp) +  "," + str(totalPixels) + "," + str(hue) + "," + str(sat) + "," + str(val) + "," + str(hue_med) + "," + str(sat_med) + "," + str(val_med) + "," + str(out) + "\n"
	csv.write(data)



