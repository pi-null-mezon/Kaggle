#ifndef QIMAGEDOWNLOADER_H
#define QIMAGEDOWNLOADER_H

#include <QThread>
#include <QMutex>

class QImageDownloader : public QThread
{
public:
    QImageDownloader(const QString &_url, const QString &_targetdirname, const QString &_imageid, int _waitms, QMutex *_qmutex, int *_threadcounter, QObject *parent=0);
    ~QImageDownloader();

protected:
    void run();

private:
    QString url;
    QString targetdirname;
    QString imageid;
    int     waitms;

    QMutex  *qmutex;
    int     *threadcounter;
};

void downloadimage(const QString &_url, const QString &_targetdirname, const QString &_imageid, int _waintms, QMutex *_mutex, int *_threadcounter);

#endif // QIMAGEDOWNLOADER_H
