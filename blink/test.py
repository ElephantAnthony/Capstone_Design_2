import cv2, dlib
import numpy as np
from imutils import face_utils
from keras.models import load_model
import math
import time

IMG_SIZE = (34, 26)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('../shape_predictor_68_face_landmarks (1).dat')

model = load_model('models/2023_10_16_40_59.h5')
model.summary()

def crop_eye(img, eye_points):
    x1, y1 = np.amin(eye_points, axis=0)
    x2, y2 = np.amax(eye_points, axis=0)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

    w = (x2 - x1) * 1.2
    h = w * IMG_SIZE[1] / IMG_SIZE[0]

    margin_x, margin_y = w / 2, h / 2

    min_x, min_y = int(cx - margin_x), int(cy - margin_y)
    max_x, max_y = int(cx + margin_x), int(cy + margin_y)

    eye_rect = np.rint([min_x, min_y, max_x, max_y]).astype(int)

    eye_img = gray[eye_rect[1]:eye_rect[3], eye_rect[0]:eye_rect[2]]

    return eye_img, eye_rect


# main
cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture('../blink_mp4.mp4')

pre = time.time()
while cap.isOpened():
    ret, img_ori = cap.read()

    if not ret:
        break

    img_ori = cv2.resize(img_ori, dsize=(0, 0), fx=0.5, fy=0.5)
    gray = cv2.cvtColor(img_ori, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)

    if not faces :
        cv2.putText(img_ori, "face not detected!", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2) # 얼굴이 감지되지 않을 때
        if time.time() - pre >= 2.5:
            cv2.putText(img_ori, "face not detected! wake up!", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2) # 2.5초 동안 감지 되지 않을 때

    for face in faces:
        shapes = predictor(gray, face)
        shapes = face_utils.shape_to_np(shapes)

        eye_img_l, eye_rect_l = crop_eye(gray, eye_points=shapes[36:42])
        eye_img_r, eye_rect_r = crop_eye(gray, eye_points=shapes[42:48])

        eye_img_l = cv2.resize(eye_img_l, dsize=IMG_SIZE)
        eye_img_r = cv2.resize(eye_img_r, dsize=IMG_SIZE)
        eye_img_r = cv2.flip(eye_img_r, flipCode=1)

        eye_input_l = eye_img_l.copy().reshape((1, IMG_SIZE[1], IMG_SIZE[0], 1)).astype(np.float32) / 255.
        eye_input_r = eye_img_r.copy().reshape((1, IMG_SIZE[1], IMG_SIZE[0], 1)).astype(np.float32) / 255.

        pred_l = model.predict(eye_input_l)
        pred_r = model.predict(eye_input_r)

        # visualize
        state_l = 'O %.1f' if pred_l > 0.1 else '- %.1f'
        state_r = 'O %.1f' if pred_r > 0.1 else '- %.1f'

        state_l = state_l % pred_l
        state_r = state_r % pred_r

        # if state_l <= '0':
        #     cv2.rectangle(img_ori, pt1=tuple(eye_rect_l[0:2]), pt2=tuple(eye_rect_l[2:4]), color=(0, 0, 255), thickness=2)
        #     cv2.putText(img_ori, state_l, tuple(eye_rect_l[0:2]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        # else:
        #     cv2.rectangle(img_ori, pt1=tuple(eye_rect_l[0:2]), pt2=tuple(eye_rect_l[2:4]), color=(255, 255, 255),
        #                   thickness=2)
        #     cv2.putText(img_ori, state_l, tuple(eye_rect_l[0:2]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        #
        # if state_r <= '0':
        #     cv2.rectangle(img_ori, pt1=tuple(eye_rect_r[0:2]), pt2=tuple(eye_rect_r[2:4]), color=(0, 0, 255), thickness=2)
        #     cv2.putText(img_ori, state_r, tuple(eye_rect_r[0:2]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        # else:
        #     cv2.rectangle(img_ori, pt1=tuple(eye_rect_r[0:2]), pt2=tuple(eye_rect_r[2:4]), color=(255, 255, 255),
        #                   thickness=2)
        #     cv2.putText(img_ori, state_r, tuple(eye_rect_r[0:2]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        if state_l <= '0' and state_r <= '0':
            cv2.putText(img_ori, "eyes non detected!", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        else :
            cv2.putText(img_ori, "eyes detected!", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

        pre = time.time()

    cv2.imshow('result', img_ori)

    if cv2.waitKey(1) == ord('q'):
        break