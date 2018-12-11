TEMPLATE = app
CONFIG += console c++11
CONFIG -= app_bundle

SOURCES += \
        main.cpp

include($${PWD}/../../../Shared/opencv.pri)
include($${PWD}/../../../Shared/dlib.pri)

INCLUDEPATH += $${PWD}/../../../Shared/dlibopencvconverter \
               $${PWD}/../../../Shared/opencvimgaugment \
               $${PWD}/../../../Shared/opencvimgalign \
               $${PWD}/../../../Shared/opencvmorph

SOURCES += $${PWD}/../../../Shared/opencvimgalign/opencvimgalign.cpp

# Here we need to point where network defined, directory shoud contain file named customnetwork.h
INCLUDEPATH += $${PWD}/../Learner
