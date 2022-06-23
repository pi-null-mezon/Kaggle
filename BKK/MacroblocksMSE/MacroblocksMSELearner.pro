TEMPLATE = app
CONFIG += console c++11
CONFIG -= app_bundle

SOURCES += main.cpp

include($${PWD}/../../Shared/dlib.pri)
include($${PWD}/../../Shared/opencv.pri)

HEADERS += customnetwork.h

INCLUDEPATH += $${PWD}/../../Shared/dlibimgaugment \
               $${PWD}/../../Shared/opencvimgaugment \
               $${PWD}/../../Shared/dlibopencvconverter

unix {
   target.path = /usr/local/bin
   INSTALLS += target
}
