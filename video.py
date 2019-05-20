# imports
import cv2
import face_recognition
import pickle

# load data
data = pickle.loads(open("encoding/dataset_encoding.pickle", "rb").read())
names = []

# set video capture
capture = cv2.VideoCapture(0)
capture.set(3, 640) # width
capture.set(4, 480) # height

# function
while True:
	ret, frame = capture.read()
	rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	rgb = cv2.resize(frame, (0, 0), fx = 0.25, fy = 0.25)
	r = frame.shape[1] / float(rgb.shape[1])

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

		cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
		y = top - 15 if top - 15 > 15 else top + 15
		cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_DUPLEX, 0.75, (0, 0, 255), 2)

	cv2.imshow("camera", frame)
	if cv2.waitKey(1) & 0xFF == ord("q"): break

capture.release()
cv2.destroyAllWindows()