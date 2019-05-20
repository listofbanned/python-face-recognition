#imports
import time
import cv2
import os
import argparse

# data
name = "dataset/" + input("Put a name: ")
os.makedirs(name)

# getting the url from terminal
ap = argparse.ArgumentParser()
ap.add_argument("-u", "--url", required = True)
args = vars(ap.parse_args())

# I use 5 images captured to start the training
examples_amount = 180

# and then
time_start = time.time()
cap = cv2.VideoCapture(args["url"])

images_captured = 0
print ("Capture from url...\n")

while True:
    ret,frame = cap.read()
    cv2.imwrite(name + "/%d.jpg" % (images_captured + 1), frame)
    images_captured = images_captured + 1

    if (images_captured > examples_amount):
        time_end = time.time()
        cap.release()
        print ("Done")
        break

print("Starting training")

# training
import training