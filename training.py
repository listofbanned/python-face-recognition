# imports
import cv2
import os
import face_recognition
import pickle
from imutils import paths

def Training():
	# folder with images
	name = "dataset/"

	# training the dataset
	print("Processing...")
	images = list(paths.list_images("dataset"))

	known_encodings = []
	names = []

	# func
	for (i, path) in enumerate(images):
		print("Processing image {}/{}".format(i + 1, len(images)))
		name = path.split(os.path.sep)[-2]

		image = cv2.imread(path)
		rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

		boxes = face_recognition.face_locations(rgb)
		encodings = face_recognition.face_encodings(rgb, boxes)

		for encoding in encodings:
			known_encodings.append(encoding)
			names.append(name)

	data = {"encodings": known_encodings, "names": names}
	writer = open("encoding/dataset_encoding.pickle", "wb")
	writer.write(pickle.dumps(data))
	writer.close()

	print("Finish")
