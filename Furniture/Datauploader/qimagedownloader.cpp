#include "qimagedownloader.h"

#include <QNetworkAccessManager>
#include <QNetworkReply>
#include <QNetworkRequest>

#include <QEventLoop>
#include <QDebug>
#include <QUuid>
#include <QDir>
#include <QImage>
#include <QTimer>

QImageDownloader::QImageDownloader(const QString &_url, const QString &_targetdirname, const QString &_imageid, int _waitms, QMutex *_qmutex, int *_threadcounter, QObject *parent) :
    QThread(parent),
    url(_url),
    targetdirname(_targetdirname),
    imageid(_imageid),
    waitms(_waitms),
    qmutex(_qmutex),
    threadcounter(_threadcounter)
{
    qmutex->lock();
    (*threadcounter)++;
    qmutex->unlock();
}

QImageDownloader::~QImageDownloader()
{
    qmutex->lock();
    (*threadcounter)--;
    qmutex->unlock();
}

void QImageDownloader::run()
{
    QNetworkRequest _request;
    _request.setUrl(QUrl(url));
    QNetworkAccessManager _manager;

    QEventLoop _el;
    QNetworkReply *_reply = _manager.get(_request);
    connect(_reply,SIGNAL(finished()),&_el,SLOT(quit()));
    QTimer::singleShot(waitms,&_el,SLOT(quit())); // watch dog
    _el.exec();
    //qInfo() << "Error:  " << _reply->errorString();
    if(_reply->isFinished()) {
        QImage qimg = qMove(QImage::fromData(_reply->readAll()));
        if(!qimg.isNull()) {
            QDir _dir;
            _dir.mkpath(targetdirname);
            _dir.cd(targetdirname);
            if(imageid.isEmpty()) {
                qimg.save(_dir.absolutePath().append("/%1.jpg").arg(QUuid::createUuid().toString()));
            } else {
                qimg.save(_dir.absolutePath().append("/%1.jpg").arg(imageid));
            }
        }
    } else {
        qInfo() << "\n!!!Watch dog!!!\n";
    }
    _reply->deleteLater();
}

void downloadimage(const QString &_url, const QString &_targetdirname, const QString &_imageid, int _waitms, QMutex *_mutex, int *_threadcounter)
{
    QImageDownloader *_thread = new QImageDownloader(_url,_targetdirname,_imageid,_waitms,_mutex,_threadcounter);
    QObject::connect(_thread,SIGNAL(finished()),_thread,SLOT(deleteLater()));
    _thread->start();
}
