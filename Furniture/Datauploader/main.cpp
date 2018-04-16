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

int main(int argc, char *argv[])
{
    //--------------------------------------------
    QString inputfilename, outputdirname;
    while((--argc > 0) && ((*++argv)[0] == '-')) {
        switch(*++argv[0]) {
            case 'i':
                inputfilename = ++argv[0];
                break;

            case 'o':
                outputdirname = ++argv[0];
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
    QDir qdir;
    qdir.mkpath(outputdirname);
    qdir.cd(outputdirname);
    //--------------------------------------------
    QFile qfile(inputfilename);
    if(qfile.open(QFile::ReadOnly) == false) {
        qInfo() << "Can not open input file! Abort...";
        return 3;
    }
    QJsonObject qjsonobj = QJsonDocument::fromJson(qfile.readAll()).object();

    QStringList keys = qjsonobj.keys();
    if(keys.size() == 2) { // this case of training and validation sets

        QJsonArray annotations = qjsonobj.value("annotations").toArray();
        QMap<int,int> imgid2labelidmap; // mapping of image_id to label
        for(int j = 0; j < annotations.size(); ++j)
            imgid2labelidmap[annotations.at(j).toObject().value("image_id").toInt()] = annotations.at(j).toObject().value("label_id").toInt();

        QJsonArray images = qjsonobj.value("images").toArray();
        QString _urlstr;
        for(int j = 0; j < images.size(); ++j) {
            _urlstr = qMove(images.at(j).toObject().value("url").toString());
            qInfo() << j << ")" << _urlstr << "; label_id: " << imgid2labelidmap[images.at(j).toObject().value("image_id").toInt()];
        }

    }

    return 0;
}
