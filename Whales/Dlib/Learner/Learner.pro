TEMPLATE = app
CONFIG += console c++11
CONFIG -= app_bundle
CONFIG -= qt

SOURCES += main.cpp

include($${PWD}/../../../Shared/dlib.pri)
include($${PWD}/../../../Shared/opencv.pri)

HEADERS += \
    $${PWD}/../../../Shared/dlibimgaugment/dlibimgaugment.h \
    $${PWD}/../../../Shared/opencvimgaugment/opencvimgaugment.h \
    $${PWD}/../../../Shared/dlibopencvconverter/dlibopencvconverter.h \
    customnetwork.h

INCLUDEPATH += $${PWD}/../../../Shared/dlibimgaugment \
               $${PWD}/../../../Shared/opencvimgaugment \
               $${PWD}/../../../Shared/dlibopencvconverter
