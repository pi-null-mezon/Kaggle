TEMPLATE = app
CONFIG += console c++11
CONFIG -= app_bundle

SOURCES += \
        main.cpp

include($${PWD}/../../Shared/opencv.pri)

HEADERS += $${PWD}/../../Shared/opencvimgaugment/opencvimgaugment.h

INCLUDEPATH += $${PWD}/../../Shared/opencvimgaugment
