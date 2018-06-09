#include <QDir>
#include <QTextStream>
#include <QStringList>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include "dlibwhalesrecognizer.h"

const cv::String _options = "{help h       | | this help}"
                            "{inputdir i   | | directory name where images are stored}"
                            "{recmodel r   | | filename of the recognitiotn model}"
                            "{labels   l   | | filename of the labels to recognize}"
                            "{outputfile o | | output filename with the results}";

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
        return 1;
    }

    if(_cmdparser.has("recmodel") == false) {
        qWarning("You have not provide recognition model file name! Abort...");
        return 1;
    }
    QString _recmodelfilename = _cmdparser.get<cv::String>("recmodel").c_str();

    if(_cmdparser.has("labels") == false) {
        qWarning("You have not provide labels file name! Abort...");
        return 1;
    }
    QString _labelsfilename = _cmdparser.get<cv::String>("labels").c_str();

    if(_cmdparser.has("outputfile") == false) {
        qWarning("You have not provide output file name! Abort...");
        return 1;
    }
    QString _outputfilename = _cmdparser.get<cv::String>("outputfile").c_str();

    QFile _qfile(_outputfilename);
    if(_qfile.open(QFile::WriteOnly) == false) {
        qWarning("Can not open %s! Abort...", _outputfilename.toUtf8().constData());
        return 1;
    }
    QTextStream _ts(&_qfile);

    auto _ptr = cv::imgrec::createDlibWhalesRecognizer(_recmodelfilename.toUtf8().constData());
    _ptr->ImageRecognizer::load(_labelsfilename.toUtf8().constData());
    if(_ptr->empty()) {
        qWarning("No labels has been loaded from %s! Abort...", _labelsfilename.toUtf8().constData());
        return 1;
    }

    // Now we can predict labels
    QStringList _filters;
    _filters << "*.jpg" << "*.jpeg" << "*.png" << "*.bmp";
    QStringList _fileslist = _qdir.entryList(_filters, QDir::Files | QDir::NoDotAndDotDot, QDir::Name);

    _ts << "Image,Id"; // header
    for(int i = 0; i < _fileslist.size(); ++i) {
        qInfo("%d) %s enrolled",i,_fileslist.at(i).toUtf8().constData());
        _ts << "\n" << _fileslist.at(i) << ',';
        cv::Mat _cvmat = cv::imread(_qdir.absoluteFilePath(_fileslist.at(i)).toUtf8().constData());
        auto _vpredictions = _ptr->recognize(_cvmat,true);
        if(_vpredictions.size() > 4) {
            for(size_t j = 0; j < 5; j++) {
                if(_vpredictions[j].second < 0.50) {
                    _ts << ' ' << _ptr->getLabelInfo(_vpredictions[j].first).c_str();
                } else {
                    _ts << " new_whale";
                }
            }
        }

    }
    qInfo("All tasks have been accomplished successfully");
    _ts.flush();

    return 0;
}
