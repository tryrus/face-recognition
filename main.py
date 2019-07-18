from PyQt4 import QtGui
import sys
from my_gui import *
from capture import *
from face_detector import *
from face_recognition import *
from functools import partial

# qt dark theme of the GUI
import qdarkstyle


def main():
    app = QtGui.QApplication(['Face_Demo'])
    app.setStyleSheet(qdarkstyle.load_stylesheet(pyside=False))

    # Create Gui Form
    form = MyGUi()

    # Create video capture thread and run
    video_path = r'I:\capture_test/wasteland_cut.avi'
    capture = Capture(video_path)
    capture.start()
    # connect GUI widgets
    form.pushButton.clicked.connect(capture.quitCapture)
    form.pushButton_2.clicked.connect(capture.startCapture)
    form.pushButton_3.clicked.connect(capture.endCapture)

    # 检测人脸并画出位置
    # Create face detector thread and run
    face_detector = Face_detector(form.textBrowser)
    # 从capture中接收'getFrame(PyQt_PyObject)'，然后送入'face_detector.detect_face'
    face_detector.connect(capture, QtCore.SIGNAL('getFrame(PyQt_PyObject)'), face_detector.detect_face)

    # Connect GUI widgets
    enable_slot_det = partial(face_detector.startstopdet, form.checkBox_2)
    form.checkBox_2.stateChanged.connect(lambda x: enable_slot_det())
    form.connect(face_detector, QtCore.SIGNAL('det(PyQt_PyObject)'), form.drawFace)


    # 识别人脸
    # Create deep net for face recognition
    form.dial.setValue(65)
    face_network = Face_recognizer(form.textBrowser)
    face_network.connect(face_detector, QtCore.SIGNAL('det(PyQt_PyObject)'), face_network.face_recognition)

    # Connect GUI Widgets
    form.dial.valueChanged.connect(face_network.set_threshold)


    form.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()