#conding:utf-8
import tensorflow as tf
import align.detect_face
import cv2


minsize = 20  # minimum size of face
threshold = [0.6, 0.7, 0.7]  # three steps's threshold
factor = 0.709  # scale factor
gpu_memory_fraction = 0.2

print('Creating networks and loading parameters')

with tf.Graph().as_default():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
    with sess.as_default():
        pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)

video_path = r'I:\capture_test/wasteland_cut.avi'
vc = cv2.VideoCapture(video_path)
# 待会要写的字体 font to write later
font = cv2.FONT_HERSHEY_COMPLEX

while vc.isOpened():
    flag, img = vc.read()
    kk = cv2.waitKey(1)

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if kk == ord('q'):
        break
    else:
        bounding_boxes, points = align.detect_face.detect_face(img_gray, minsize, pnet, rnet, onet, threshold, factor)

        nrof_faces = bounding_boxes.shape[0]  # 人脸数目
        #print('找到人脸数目为：{}'.format(nrof_faces))
        if nrof_faces != 0:
            # 在图上画出人脸框
            for face_position in bounding_boxes:
                face_position = face_position.astype(int)
                #print(face_position[0:4])

                cv2.rectangle(img, (face_position[0], face_position[1]), (face_position[2], face_position[3]), (0, 255, 0), 3)
            # 在图上画出特征点
            for num_feature in range(nrof_faces):
                point_size = 1
                point_color = (255, 0, 0)  # BGR
                thickness = 8  # 可以为 0 、4、8

                # 要画的点的坐标
                points_list = points[:,num_feature]

                for num_point in range(int(len(points_list)/2)):
                    cv2.circle(img, (points_list[num_point],points_list[num_point+5]), point_size, point_color, thickness)


    cv2.putText(img, "Press 'q': Quit", (20, 450), font, 0.8, (84, 255, 159), 1, cv2.LINE_AA)
    cv2.putText(img, "Face Recognition", (20, 40), font, 1, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(img, "Faces: " + str(nrof_faces), (20, 100), font, 1, (0, 0, 255), 1, cv2.LINE_AA)

    # 窗口显示 show with opencv
    cv2.imshow("camera", img)

# 释放摄像头 release camera
vc.release()

# 删除建立的窗口 delete all the windows
cv2.destroyAllWindows()
