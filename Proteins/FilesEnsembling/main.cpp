#include <QDir>
#include <QFile>
#include <QDebug>

#include <iostream>

int main(int argc, char *argv[])
{
    if(argc != 3) {
        qInfo("This application should be called with two arguments:\n"
              "./FilesEnsembling path_to_dir_with_submissions output_filename.csv");
        return 1;
    }

    QDir dir(argv[1]);
    if(dir.exists() == false) {
        qInfo("Input directory does not exists!");
        return 2;
    }

    QStringList filesfilters;
    filesfilters << "*.csv";
    QStringList fileslist = dir.entryList(QStringList() << "*.csv",QDir::Files | QDir::NoDotAndDotDot);

    QMap<QString,QMap<int,int>> labelsmap;
    QVector<QString> guidsvector;
    guidsvector.reserve(15000);
    bool _ok;
    int _lbl;
    for(int i = 0; i < fileslist.size(); ++i) {
        QFile file(dir.absoluteFilePath(fileslist.at(i)));
        file.open(QIODevice::ReadOnly);
        QString line = file.readLine(); // skip header line
        QString guid;
        QStringList labelslist;
        while(!file.atEnd()) {
            line = file.readLine().simplified();
            guid = line.section(',',0,0);

            if(i == 0)
                guidsvector.push_back(guid);

            labelslist = line.section(',',1).split(' ');                      
            if(labelsmap.contains(guid)) {
                QMap<int,int> &_lblmap = labelsmap[guid];
                for(int j = 0; j < labelslist.size(); ++j) {
                    _lbl = labelslist.at(j).toInt(&_ok);
                    if(_ok) {
                        if(_lblmap.contains(_lbl)) {
                            _lblmap[_lbl] += 1;
                        } else {
                            _lblmap.insert(_lbl,1);
                        }
                    }
                }
            } else {
                QMap<int,int> _lblmap;
                for(int j = 0; j < labelslist.size(); ++j) {
                    _lbl = labelslist.at(j).toInt(&_ok);
                    if(_ok)
                        _lblmap.insert(_lbl,1);
                }
                labelsmap.insert(guid,qMove(_lblmap));
            }
        }
    }

    QFile _outputfile(argv[2]);
    _outputfile.open(QIODevice::WriteOnly);
    QTextStream _ots(&_outputfile);
    _ots << "Id,Predicted\n";
    QMap<int,int> _lblmap;
    for(int i = 0; i < guidsvector.size(); ++i) {
        _ots << guidsvector.at(i) << ',';
        _lblmap = labelsmap.value(guidsvector.at(i));
        QList<int> _labelslist = _lblmap.keys();
        bool first = true;
        for(int j = 0; j < _labelslist.size(); ++j) {
            if(_lblmap.value(_labelslist.at(j)) > 2) { // 4 - LB 0.485
                if(!first) {
                    _ots << ' ';
                }
                _ots << _labelslist.at(j);
                first = false;
            }
        }
        _ots << '\n';
    }
    return 0;
}
