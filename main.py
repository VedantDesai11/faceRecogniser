import numpy as np
import cv2
import face_recognition
import PIL
from PIL import Image
import keyboard

# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(0)

# initialize variables
known_face_encodings = []
known_face_names = []
faces = 0
while True:

	# Grab a single frame of video
	ret, frame = video_capture.read()

	# Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
	rgb_frame = frame[:, :, ::-1]

	k = cv2.waitKey(30)

	# press 'ESC' to quit
	if k == 27:
		break

	# press 'Spacebar' to capture
	if k == ord('c'):
		f = face_recognition.face_encodings(rgb_frame)
		known_face_encodings.append(f[0])
		faces += 1
		known_face_names.append(str(faces))

	cv2.imshow('video', frame)

video_capture.release()
cv2.destroyAllWindows()

# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(0)

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

while (True):

	# Grab a single frame of video
	ret, frame = video_capture.read()

	# Resize frame of video to 1/4 size for faster face recognition processing
	small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

	# Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
	rgb_small_frame = small_frame[:, :, ::-1]

	if process_this_frame:
		# Find all the faces and face encodings in the current frame of video

		face_locations = face_recognition.face_locations(rgb_small_frame)
		face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

		face_names = []

		for face_encoding in face_encodings:
			# See if the face is a match for the known face(s)
			matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
			name = "Unknown"

			# # If a match was found in known_face_encodings, just use the first one.
			# if True in matches:
			#     first_match_index = matches.index(True)
			#     name = known_face_names[first_match_index]

			# Or instead, use the known face with the smallest distance to the new face
			face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
			best_match_index = np.argmin(face_distances)

			if matches[best_match_index]:
				name = known_face_names[best_match_index]

			face_names.append(name)

	process_this_frame = not process_this_frame

	for (top, right, bottom, left) in face_locations:
		top *= 4
		right *= 4
		bottom *= 4
		left *= 4

		# Draw a box around the face
		cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

		# Draw a label with a name below the face
		cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
		font = cv2.FONT_HERSHEY_DUPLEX
		cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

	# Display the resulting image
	cv2.imshow('Video', frame)

	# press 'ESC' to quit
	k = cv2.waitKey(30) & 0xff
	if k == 27:
		break

video_capture.release()
cv2.destroyAllWindows()
