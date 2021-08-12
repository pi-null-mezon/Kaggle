QT += core
QT -= gui

CONFIG += c++11

TARGET = tsne_mapper
CONFIG += console
CONFIG -= app_bundle

TEMPLATE = app

SOURCES += main.cpp

include($${PWD}/../../Shared/opencv.pri)
include($${PWD}/../../Shared/dlib.pri)

INCLUDEPATH += $${PWD}/../../Shared/opencvimgaugment \
               $${PWD}/../../Shared/dlibopencvconverter \
               $${PWD}/../../Shared/dlibimgaugment

# Specify where your network defined (customnetwork.h)
INCLUDEPATH += $${PWD}/../../Dialyzer/Dlib/Metric/Trainer
