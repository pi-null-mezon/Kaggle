CONFIG -= qt

TEMPLATE = app
CONFIG += console c++11
CONFIG -= app_bundle

SOURCES += main.cpp

DESTDIR += $${PWD}/../bin

INCLUDEPATH += $${PWD}/../../../../Shared/dlibimgaugment \
               $${PWD}/../../../../Shared/opencvimgaugment \
               $${PWD}/../../../../Shared/dlibopencvconverter

include($${PWD}/../../../../Shared/dlib.pri)
include($${PWD}/../../../../Shared/opencv.pri)

include({PWD}/../../../../../../OpenIRT/Sources/Basic/imagerecognizer.pri)
include({PWD}/../../../../../../OpenIRT/Sources/Kaggle/Furniture/furniturerecognizer.pri)
