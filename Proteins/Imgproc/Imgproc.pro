CONFIG -= qt

CONFIG += c++11 console
CONFIG -= app_bundle

SOURCES += main.cpp

INCLUDEPATH += $${PWD}/../../Shared/dlibimgaugment \
               $${PWD}/../../Shared/dlibopencvconverter \
               $${PWD}/../../Shared/opencvimgaugment

include($${PWD}/../../Shared/dlib.pri)
include($${PWD}/../../Shared/opencv.pri)
