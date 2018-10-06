TEMPLATE = app
CONFIG += console c++11
CONFIG -= app_bundle
CONFIG -= qt

SOURCES += \
        main.cpp

include($${PWD}/../../Shared/opencv.pri)
include($${PWD}/../../Shared/dlib.pri)

INCLUDEPATH += $${PWD}/../../Shared/dlibopencvconverter \
               $${PWD}/../../Shared/opencvimgaugment

# Here we need to point where network defined, directory shoud contain file named customnetwork.h
INCLUDEPATH += $${PWD}/../../Whales/Dlib/Learner
