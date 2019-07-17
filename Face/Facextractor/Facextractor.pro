TEMPLATE = app
CONFIG += console c++11
CONFIG -= app_bundle

SOURCES += main.cpp \
           facetracker.cpp

HEADERS += facetracker.h

include($${PWD}/../../Shared/opencv.pri)
include($${PWD}/../../Shared/dlib.pri)

unix: {
   target.path = /usr/local/bin
   INSTALLS += target
}
