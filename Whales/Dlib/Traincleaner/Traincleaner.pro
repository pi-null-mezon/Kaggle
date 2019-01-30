QT -= gui

TEMPLATE = app
CONFIG += console c++11
CONFIG -= app_bundle

SOURCES += \
        main.cpp

HEADERS += \
    $${PWD}/../../../Shared/dlibimgaugment/dlibimgaugment.h \
    $${PWD}/../../../Shared/opencvimgaugment/opencvimgaugment.h \
    $${PWD}/../../../Shared/dlibopencvconverter/dlibopencvconverter.h

INCLUDEPATH += $${PWD}/../../../Shared/dlibimgaugment \
               $${PWD}/../../../Shared/opencvimgaugment \
               $${PWD}/../../../Shared/dlibopencvconverter

include($${PWD}/../../../Shared/dlib.pri)
include($${PWD}/../../../Shared/opencv.pri)

include($${PWD}/../../../../OpenIRT/Sources/Basic/imagerecognizer.pri)
include($${PWD}/../../../../OpenIRT/Sources/Kaggle/Whales/dlibwhalesrecognizer.pri)

unix {
   target.path = /usr/local/bin
   INSTALLS += target
}
