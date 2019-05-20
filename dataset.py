# imports
import cv2
import os

# data
name = "dataset/" + input("Put a name: ")
os.makedirs(name)
cascade = "haarcascade/haarcascade_frontalface_default.xml"
detector = cv2.CascadeClassifier(cascade)

# I use 5 images captured to start the training
examples_amount = 5

print("Please wait...")
capture = cv2.VideoCapture(0)

images_captured = 0

# dataset
while True:
	ret, frame = capture.read()
	cv2.startWindowThread()
	orig = frame.copy()
	frame = cv2.resize(frame, (0, 0), fx = 0.5, fy = 0.5)

	rects = detector.detectMultiScale(
		cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), scaleFactor = 1.1,
		minNeighbors = 5, minSize = (30, 30))

	for (x, y, w, h) in rects:
		cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
		cv2.putText(frame, "Don't let the rect disappear", (x - 25, y - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

	p = os.path.sep.join([name, "{}.png".format(str(images_captured).zfill(5))])
	cv2.imwrite(p, orig)
	images_captured +=  1

	cv2.imshow("frame", frame)
	cv2.waitKey(1)

	if images_captured > examples_amount:
		break

print("Starting training")

capture.release()
cv2.destroyAllWindows()

# training
import training