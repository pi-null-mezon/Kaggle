#include <QFile>
#include <QDir>
#include <QStringList>
#include <QDebug>

int main(int argc, char *argv[])
{
    if(argc != 4) {
        qWarning() << "This app should be called with three arguments:"
                      "./app filename dirwithimages outputdir"
                      "For the instance: ./app train.csv traindir outputdir";
    }

    const QString _filename(argv[1]);
    const QString _inputdirname(argv[2]);
    const QString _outputdirname(argv[3]);

    QFile _file(_filename);
    if(_file.open(QFile::ReadOnly) == false) {
        return 1;
    } else {
        _file.readLine(); // Drop first line with header
    }

    QDir _outputdir;
    qInfo() << "Start file coping:";
    while(!_file.atEnd()) {
        QString _strline = _file.readLine();
        QString _targetdirname = QString(_outputdirname).append(QString("/%1").arg(_strline.section(',',1).simplified()));
        _outputdir.mkpath(_targetdirname);
        QString _filename = _strline.section(',',0,0);
        if(QFile::copy(QString(_inputdirname).append(QString("/%1").arg(_filename)),
                       _targetdirname.append(QString("/%1").arg(_filename)))) {
            qInfo() << _filename << " has been copied to the target location";
        }
    }
    qInfo() << "All tasks have been accomplished";
    return 0;
}
