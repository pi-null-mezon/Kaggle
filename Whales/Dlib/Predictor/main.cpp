#include <QDir>
#include <QTextStream>
#include <QStringList>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include "dlibwhalesrecognizer.h"

#include "dlibimgaugment.h"
#include "opencvimgaugment.h"
#include "dlibopencvconverter.h"

const cv::String _options = "{help h       |      | this help}"
                            "{dstthresh t  | 0.57 | distance threshold}"
                            "{inputdir i   |      | directory name where images are stored}"
                            "{model m      |      | filename/filenames of the recognitiotn model/models (use ; to separate names)}"
                            "{labels   l   |      | filename of the labels to recognize}"
                            "{outputfile o |      | output filename with the results}";

int main(int argc, char *argv[])
{
    cv::CommandLineParser _cmdparser(argc,argv,_options);
    if(_cmdparser.has("help")) {
        _cmdparser.printMessage();
        return 0;
    }
    // Conditions check
    if(_cmdparser.has("inputdir") == false) {
        qWarning("You have not provide input directory name! Abort...");
        return 1;
    }
    QString _inputdirname = _cmdparser.get<cv::String>("inputdir").c_str();

    QDir _qdir(_inputdirname);
    if(_qdir.exists() == false) {
        qWarning("Directory with the name %s is not existed! Abort...", _inputdirname.toUtf8().constData());
        return 2;
    }

    if(_cmdparser.has("model") == false) {
        qWarning("You have not provide recognition model file name! Abort...");
        return 3;
    } else {
        QStringList models_list = QString(_cmdparser.get<cv::String>("model").c_str()).split(';');
        for(int i = 0; i < models_list.size(); ++i) {
            if(QFileInfo(models_list.at(i)).exists() == false) {
                qWarning("Model '%s' does not exist! Abort...",models_list.at(i).toUtf8().constData());
                return 4;
            }
        }
    }
    QString _recmodelfilename = _cmdparser.get<cv::String>("model").c_str();

    if(_cmdparser.has("labels") == false) {
        qWarning("You have not provide labels file name! Abort...");
        return 5;
    }
    QString _labelsfilename = _cmdparser.get<cv::String>("labels").c_str();

    if(_cmdparser.has("outputfile") == false) {
        qWarning("You have not provide output file name! Abort...");
        return 6;
    }
    QString _outputfilename = _cmdparser.get<cv::String>("outputfile").c_str();

    QFile _qfile(_outputfilename);
    if(_qfile.open(QFile::WriteOnly) == false) {
        qWarning("Can not open %s! Abort...", _outputfilename.toUtf8().constData());
        return 7;
    }
    QTextStream _ts(&_qfile);

    auto _ptr = cv::oirt::createDlibWhalesRecognizer(_recmodelfilename.toUtf8().constData());
    _ptr->ImageRecognizer::load(_labelsfilename.toUtf8().constData());
    if(_ptr->empty()) {
        qWarning("No labels has been loaded from %s! Abort...", _labelsfilename.toUtf8().constData());
        return 8;
    }

    // Now we can predict labels
    QStringList _filters;
    _filters << "*.jpg" << "*.jpeg" << "*.png" << "*.bmp";
    QStringList _fileslist = _qdir.entryList(_filters, QDir::Files | QDir::NoDotAndDotDot, QDir::Name);

    _ts << "Image,Id"; // header
    const double _dstthresh = _cmdparser.get<double>("dstthresh");
    for(int i = 0; i < _fileslist.size(); ++i) {       
        _ts << "\n" << _fileslist.at(i) << ',';
        cv::Mat _cvmat = cv::imread(_qdir.absoluteFilePath(_fileslist.at(i)).toUtf8().constData());
        auto _vpredictions = _ptr->recognize(_cvmat,true);
        if(_vpredictions.size() >= 5) {
            if(_vpredictions[0].second < _dstthresh) {
                for(size_t j = 0; j < 5; j++)
                   _ts << ' ' << _ptr->getLabelInfo(_vpredictions[j].first).c_str();
            } else {
                _ts << " new_whale";
                int _pos = 0;
                for(size_t j = 0; j < _vpredictions.size(); j++) {
                    if(QString(_ptr->getLabelInfo(_vpredictions[j].first).c_str()).compare("new_whale") != 0) {
                        _ts << ' ' << _ptr->getLabelInfo(_vpredictions[j].first).c_str();
                        _pos++;
                        if(_pos == 4)
                            break;
                    }
                }
            }
        }
        qInfo("%d) %s enrolled",i,_fileslist.at(i).toUtf8().constData());
    }
    qInfo("All tasks have been accomplished successfully");
    _ts.flush();

    return 0;
}
