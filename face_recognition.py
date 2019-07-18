from PyQt4 import QtCore
import glob
import pandas as pd
import numpy as np
from mtcnn_net import *

class Face_recognizer(QtCore.QThread):
    def __init__(self, textBrowser):
        super(Face_recognizer, self).__init__()
        self.recognizing = True
        self.textBrowser = textBrowser
        self.threshold = 0
        self.label = ['Stranger']
        self.db_path = './db'
        #self.db = []
        self.db = None
        # load db
        self.load_db()
        self.model = '20170512-110547/20170512-110547.pb'

        #self.predictor = dlib.shape_predictor('data/data_dlib/shape_predictor_68_face_landmarks.dat')

    def load_db(self):
        if not os.path.exists(self.db_path):
            print('Database path is not existed!')
        folders = sorted(glob.glob(os.path.join(self.db_path, '*')))
        for name in folders:
            print('loading {}:'.format(name))
            self.label.append(os.path.basename(name))

        # 处理存放所有人脸特征的 csv
        path_features_known_csv = "data/features_all.csv"
        csv_rd = pd.read_csv(path_features_known_csv, header=None)

        # 用来存放所有录入人脸特征的数组
        # the array to save the features of faces in the database
        features_known_arr = []

        # 读取已知人脸数据
        # print known faces
        for i in range(csv_rd.shape[0]):
            features_someone_arr = []
            for j in range(0, len(csv_rd.ix[i, :])):
                features_someone_arr.append(csv_rd.ix[i, :][j])
            features_known_arr.append(features_someone_arr)
        print("Faces in Database：", len(features_known_arr))

        if self.db is None:
            self.db = features_known_arr.copy()
        else:
            self.db = np.vstack((self.db, features_known_arr.copy()))

        print(self.label)


    def face_recognition(self, face_info):
        if self.recognizing:

            cord = []
            # for k, face in face_info[0].items():
            #     face_norm = face[2].astype(float)
            #     face_norm = cv2.resize(face_norm, (128, 128))
            #     img.append(face_norm)
            #     cord.append(face[0][0:2])
            face_info = face_info[0]

            for k, face in face_info.items():
                face = face_info[k][2]
                cord.append(face)

            if len(cord) != 0:
                features_all = []
                # call deep learning for classfication

                for i in range(len(cord)):
                    feature = compute_feature(self.model, np.expand_dims(cord[i], axis=0))
                    features_all.append(feature)


                # search from db find the closest
                dist=[]
                for m in range(len(features_all)):
                    d = np.sqrt(np.sum(np.square(np.subtract(self.db,features_all[m])),axis=1))
                    dist.append(d)

                # print('dist = {}'.format(dist))
                pred = np.argmin(dist, 1)
                dist = np.min(dist, 1)
                # print(dist)
                pred = [0 if dist[i] < self.threshold/100.0 else pred[i]+1 for i in range(len(dist)) ]

                # search from db find the closest
                # dist = sklearn.metrics.pairwise.cosine_similarity(features_all, self.db)
                # print('dist = {}'.format(dist))
                # pred = np.argmax(dist, 1)
                # dist = np.max(dist, 1)
                # pred = [0 if dist[i] < self.threshold / 100.0 else pred[i] + 1 for i in range(len(dist))]


                # writ on GUI
                # s=QtCore.QString
                msg = str("Face Recognition Pred: <span style='color:red'>{}</span>".format(' '.join([self.label[x] for x in pred])))
                self.textBrowser.append(msg)
                # emit signal when detection finished
                self.emit(QtCore.SIGNAL('face_id(PyQt_PyObject)'), [pred, cord])

    def set_threshold(self, th):
        self.threshold = th
        self.textBrowser.append('Threshold is changed to: {}'.format(self.threshold))

    def startstopfacerecognize(self, checkbox):
        if checkbox.isChecked():
            self.recognizing = True
        else:
            self.recognizing = False



