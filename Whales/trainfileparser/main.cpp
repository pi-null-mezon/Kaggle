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
        qWarning() << "Can not open file with labels mapping! Abort...";
        return 1;
    } else {
        _file.readLine(); // Drop first line with header
    }


    qInfo() << "Input file parsing...";
    QMap<QString,QString> _labelsmap; // label to filename
    while(!_file.atEnd()) {
        QString _strline = _file.readLine();
        _labelsmap.insertMulti(_strline.section(',',1).simplified(),_strline.section(',',0,0));
    }

    QDir _outputdir(_outputdirname);
    QDir _inputdir(_inputdirname);
    QStringList _labels = _labelsmap.uniqueKeys();
    for(int i = 0; i < _labels.size(); ++i) {
        int _count = _labelsmap.count(_labels.at(i));
        if(_count > 3) { // control what classes (in terms of instances per class) should be enrolled
            qInfo("  label %s - %d instances", _labels.at(i).toUtf8().constData(), _count);
            _outputdir.mkdir(_labels.at(i));
            QStringList _files = _labelsmap.values(_labels.at(i));
            for(int j = 0; j < _files.size(); ++j) {
                QFile::copy(_inputdir.absoluteFilePath(_files.at(j)),
                            _outputdir.absolutePath().append("/%1/%2").arg(_labels.at(i),_files.at(j)));
            }
        }
    }
    qInfo() << "File copying has been accomplished";
    return 0;
}
