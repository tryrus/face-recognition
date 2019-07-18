import dlib
from PyQt4 import QtCore
import numpy as np
import time


import align.detect_face
import cv2
from mtcnn_net import *


# 检测出人脸并返回人脸的四个坐标,传递给face_recognition

class Face_detector(QtCore.QThread):
    def __init__(self, textBrowser):
        super(Face_detector, self).__init__()
        #self.face_detector = dlib.get_frontal_face_detector()
        #self.ldmark_detector = dlib.shape_predictor('./dlib_model/shape_predictor_68_face_landmarks.dat')
        self.face_info = {}
        self.textBrowser = textBrowser
        self.detecting = True  # flag of if detect face
        self.ldmarking = False  # flag of if detect landmark
        self.total = 0
        self.pnet,self.rnet,self.onet = mtcnn_net()
        self.minsize = 20  # minimum size of face
        self.threshold = [0.6, 0.7, 0.7]  # three steps's threshold
        self.factor = 0.709  # scale factor

    def face_detector(self, img):
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # 左上横坐标，左上纵坐标，宽度，高度
        bounding_boxes, points = align.detect_face.detect_face(img_gray, self.minsize, self.pnet, self.rnet, self.onet,
                                                          self.threshold, self.factor)
        nrof_faces = bounding_boxes.shape[0]  # 人脸数目
        #print('找到人脸数目为：{}'.format(nrof_faces))

        # crop_faces = []
        if nrof_faces != 0:
            # crop_faces = []
            for face_position in bounding_boxes:
                face_position = face_position.astype(int)
                # print(face_position[0:4])
                # 在图上画出人脸框
                cv2.rectangle(img, (face_position[0], face_position[1]), (face_position[2], face_position[3]),
                               (0, 255, 0), 2)
                # crop = img[face_position[1]:face_position[3],
                #        face_position[0]:face_position[2], ]
                #
                # crop = cv2.resize(crop, (160, 160), interpolation=cv2.INTER_CUBIC)
                # crop_faces.append(crop)

                # 在图上画出特征点
            for num_feature in range(nrof_faces):
                point_size = 1
                point_color = (255, 0, 0)  # BGR
                thickness = 8  # 可以为 0 、4、8

                # 要画的点的坐标
                points_list = points[:, num_feature]
                for num_point in range(int(len(points_list) / 2)):
                    cv2.circle(img, (points_list[num_point], points_list[num_point + 5]), point_size, point_color,
                                thickness)

        return bounding_boxes, img


    def detect_face(self, img):
        if self.detecting:
            self.face_info = {}

            # det_start_time = time.time()
            # dets = self.face_detector(img, 0)

            if img is not None:
                dets,img= self.face_detector(img)
                # print('Detection took %s seconds.' % (time.time() - det_start_time))

                #print('Number of face detected: {}'.format(len(dets)))
                if len(dets) > 0:
                    nrof_faces = dets.shape[0]  # 人脸数目
                    self.textBrowser.append('Number of face detected: {}'.format(nrof_faces))
                num = 0
                for face_position in dets:
                    face_position = face_position.astype(int)

                    # ldmark detection
                    landmarks = []
                    if self.ldmarking:
                        shape = self.ldmark_detector(img, d)
                        landmarks = [(shape.part(i).x, shape.part(i).y) for i in range(68)]

                    crop_face = np.copy(img[max(0, face_position[1]):(face_position[3]),
                                        max(0,face_position[0]):(face_position[2]),])
                    crop_face = cv2.resize(crop_face, (160, 160), interpolation=cv2.INTER_CUBIC)
                    crop_face = facenet.prewhiten(crop_face)
                    self.total += 1

                    self.face_info[num] = ([face_position[0], face_position[1], face_position[2], face_position[3]],
                                                  landmarks[18:], crop_face)    # 0:18 are face counture
                    num +=1

                # emit signal when detection finished
            self.emit(QtCore.SIGNAL('det(PyQt_PyObject)'), [self.face_info, img])



    def startstopdet(self, checkbox):
        if checkbox.isChecked():
            self.detecting = True
        else:
            self.detecting = False

