#include <QCoreApplication>
#include <QThread>
#include <QMutex>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonArray>
#include <QJsonValue>

#include <QDebug>
#include <QDir>
#include <QMap>
#include <QMutexLocker>

#include "qimagedownloader.h"

bool checkAvailableThread(int _maxthreads, QMutex *_qmutex, int *_threadcounter)
{
    QMutexLocker ml(_qmutex);
    if((*_threadcounter) <= _maxthreads)
        return true;
    return false;
}

int main(int argc, char *argv[])
{
    QCoreApplication app(argc,argv);
    //--------------------------------------------
    QString inputfilename, outputdirname;
    int maxthreads = 4*QThread::idealThreadCount();
    while((--argc > 0) && ((*++argv)[0] == '-')) {
        switch(*++argv[0]) {
            case 'i':
                inputfilename = ++argv[0];
                break;

            case 'o':
                outputdirname = ++argv[0];
                break;

            case 't':
                maxthreads = QString(++argv[0]).toInt();
                break;
        }
    }
    if(inputfilename.isEmpty()) {
        qInfo() << "Empty input filename! Abort...";
        return 1;
    }
    if(outputdirname.isEmpty()) {
        qInfo() << "Empty output directory name! Abort...";
        return 2;
    }
    if(maxthreads < 1) {
        qInfo() << "Unsupported number of threads!";
        return 3;
    }
    QDir qdir;
    qdir.mkpath(outputdirname);
    qdir.cd(outputdirname);
    //--------------------------------------------
    QFile qfile(inputfilename);
    if(qfile.open(QFile::ReadOnly) == false) {
        qInfo() << "Can not open input file! Abort...";
        return 3;
    }
    //--------------------------------------------
    QMutex qmutex;
    int    threadcounter = 0;
    //--------------------------------------------
    QJsonObject qjsonobj = QJsonDocument::fromJson(qfile.readAll()).object();

    QStringList keys = qjsonobj.keys();
    if(keys.size() == 2) { // this is the case of training and validation sets

        QJsonArray annotations = qjsonobj.value("annotations").toArray();
        QMap<int,int> imgid2labelidmap; // mapping of image_id to label
        for(int j = 0; j < annotations.size(); ++j)
            imgid2labelidmap[annotations.at(j).toObject().value("image_id").toInt()] = annotations.at(j).toObject().value("label_id").toInt();

        QJsonArray images = qjsonobj.value("images").toArray();
        for(int j = 0; j < images.size(); ++j) {
            QJsonArray instances = images.at(j).toObject().value("url").toArray();
            QString _urlstr, _labelstr;
            qInfo() << "------------------------";
            for(int k = 0; k < instances.size(); ++k) {
                _urlstr = qMove(instances.at(k).toString());
                _labelstr = qMove(QString::number(imgid2labelidmap[images.at(j).toObject().value("image_id").toInt()]));
                qInfo() << j << "." << k << ")" << _urlstr << "; label_id: " << _labelstr;
                while(checkAvailableThread(maxthreads,&qmutex,&threadcounter) == false) {
                    QCoreApplication::processEvents();
                }
                downloadimage(_urlstr,qdir.absolutePath().append("/%1").arg(_labelstr),QString(),25000,&qmutex,&threadcounter);
            }
        }
    } else if(keys.size() == 1) { // this is the case of test set

        QJsonArray images = qjsonobj.value("images").toArray();
        for(int j = 0; j < images.size(); ++j) {
            int _imageid = images.at(j).toObject().value("image_id").toInt();
            QJsonArray instances = images.at(j).toObject().value("url").toArray();
            QString _urlstr;
            qInfo() << "------------------------";
            for(int k = 0; k < instances.size(); ++k) {
                _urlstr = qMove(instances.at(k).toString());
                qInfo() << j << "." << k << ")" << _urlstr;
                while(checkAvailableThread(maxthreads,&qmutex,&threadcounter) == false) {
                    QCoreApplication::processEvents();
                }
                downloadimage(_urlstr,qdir.absolutePath(),QString::number(_imageid),25000,&qmutex,&threadcounter);
            }
        }
    }

    return 0;
}

