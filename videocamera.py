# imports
import cv2
import face_recognition
import pickle

face_cascade = cv2.CascadeClassifier('haarcascade/haarcascade_frontalface_default.xml')
data = pickle.loads(open("dataset_encoding.pickle", "rb").read())
names = []

class VideoCamera(object):
	def __init__(self):
		self.video = cv2.VideoCapture(0)

	def __del__(self):
		self.video.release()

	def get_frame(self):
		success, image = self.video.read()

		faces = face_cascade.detectMultiScale(image, 1.3, 5)
		rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		rgb = cv2.resize(image, (0, 0), fx = 0.25, fy = 0.25)
		r = image.shape[1] / float(rgb.shape[1])

		boxes = face_recognition.face_locations(rgb)
		encodings = face_recognition.face_encodings(rgb, boxes)

		for encoding in encodings:
			matches = face_recognition.compare_faces(data["encodings"], encoding)
			name = "unknown"

			if True in matches:
				matchesid = [i for (i, b) in enumerate(matches) if b]
				counts = {}

				for i in matchesid:
					name = data["names"][i]
					counts[name] = counts.get(name, 0) + 1

				name = max(counts, key = counts.get)

			names.append(name)

		for ((top, right, bottom, left), name) in zip(boxes, names):
			top = int(top * r)
			right = int(right * r)
			bottom = int(bottom * r)
			left = int(left * r)

			cv2.rectangle(image, (left, top), (right, bottom), (0, 0, 255), 2)
			y = top - 15 if top - 15 > 15 else top + 15
			cv2.putText(image, name, (left, y), cv2.FONT_HERSHEY_DUPLEX, 0.75, (0, 0, 255), 2)

		ret, jpeg = cv2.imencode('.jpg', image)
		return jpeg.tobytes()
