QT -= gui

TEMPLATE = app
CONFIG += console c++11
CONFIG -= app_bundle

SOURCES += main.cpp

INCLUDEPATH += $${PWD}/../../Shared/dlibimgaugment \
               $${PWD}/../../Shared/opencvimgaugment \
               $${PWD}/../../Shared/dlibopencvconverter

include($${PWD}/../../Shared/dlib.pri)
include($${PWD}/../../Shared/opencv.pri)

HEADERS += \
    customnetwork.h
