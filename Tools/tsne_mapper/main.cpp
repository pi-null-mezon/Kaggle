#include <QStringList>
#include <QTextStream>
#include <QFile>
#include <QDir>

#include <opencv2/opencv.hpp>
#include <dlib/dnn.h>

#include "dlibopencvconverter.h"
#include "opencvimgaugment.h"
#include "dlibimgaugment.h"

#include "customnetwork.h"

using namespace std;

const cv::String keys = "{inputdir i  |   | - name of the directory with images}"
                        "{icols       | 0 | - target number of input image columns}"
                        "{irows       | 0 | - target number of the input image rows}"
                        "{model m     |   | - filename of the network model's weights}"
                        "{outputdir o |   | - output directory name (where tsne_lbls.txt and tsne_dscr.csv files will be saved)}"
                        "{help h      |   | - show help}";

int main(int argc, char **argv)
{
#ifdef Q_OS_WIN
    setlocale(LC_CTYPE, "rus");
#endif
    cv::CommandLineParser cmdparser(argc, argv, keys);
    cmdparser.about("This app prepares images descriptions for t-sne plot");
    if(cmdparser.has("help")) {
        cmdparser.printMessage();
        return 0;
    }
    if(!cmdparser.has("inputdir")) {
        cout << "Empty input directory name! Abort..." << endl;
        return 1;
    }
    if(!cmdparser.has("outputdir")) {
        cout << "Empty output directory name! Abort..." << endl;
        return 2;
    }
    if(!cmdparser.has("model")) {
        cout << "Empty model filename! Abort..." << endl;
        return 3;
    }
    const int _tcols = cmdparser.get<int>("icols"),
            _trows = cmdparser.get<int>("irows");
    if((_tcols == 0) && (_trows == 0)) {
        cout << "Zero icols or irows are not allowed!" << endl;
        return 4;
    }

    anet_type net;
    try {
        dlib::deserialize(cmdparser.get<string>("model")) >> net;
    } catch(std::exception& e) {
        cout << e.what() << endl;
        return 5;
    }

    QString _outdirname = cmdparser.get<string>("outputdir").c_str();
    QFile _flbls(QString("%1/tsne_lbls.txt").arg(_outdirname));
    QFile _fdscr(QString("%1/tsne_dscr.csv").arg(_outdirname));
    if((_flbls.open(QFile::WriteOnly) == false) ||
            (_fdscr.open(QFile::WriteOnly) == false)) {
        cout << "Can not create output files! Abort..." << endl;
        return 6;
    }
    QTextStream _lblststream(&_flbls);
    QTextStream _dscrstream(&_fdscr);

    QDir _indir(cmdparser.get<string>("inputdir").c_str());
    if(!_indir.exists()) {
        cout << "Imput directory does not exist! Abort..." << endl;
        return 7;
    }

    QStringList filefilters;
    filefilters << "*.jpg" << "*.jpeg" << "*.png" << "*.bmp";
    QStringList _subdirnames = _indir.entryList(QDir::NoDotAndDotDot | QDir::Dirs);
    for(int i = 0; i < _subdirnames.size(); ++i) {
        QString _subdirname = _subdirnames.at(i);
        qInfo("%d - %s", i, _subdirname.toUtf8().constData());
        QDir _subdir(_indir.absolutePath().append("/").append(_subdirnames.at(i)));
        QStringList _filenames = _subdir.entryList(QDir::NoDotAndDotDot | QDir::Files);
        for(int j = 0; j < _filenames.size(); ++j) {
            qInfo("   %d.%d - %s", i, j, _filenames.at(j).toUtf8().constData());

            bool _isloaded = false;
            dlib::matrix<dlib::rgb_pixel> _dlibimg = load_rgb_image_with_fixed_size(_subdir.absoluteFilePath(_filenames.at(j)).toStdString(),_tcols,_trows,true,&_isloaded);
            if(_isloaded == false) {
                qInfo("Can not read image! Instance will be skipped...");
            } else {
                dlib::matrix<float,0,1> _dlibdscr = net(_dlibimg);
                cv::Mat _cvdscr = dlibmatrix2cvmat<float>(_dlibdscr);
                float *_p = _cvdscr.ptr<float>(0);
                for(size_t k = 0; k < _cvdscr.total(); ++k) {
                    if(k != (_cvdscr.total() - 1))
                        _dscrstream << _p[k] << ", ";
                    else
                        _dscrstream << _p[k] << "\n";
                }
                _lblststream << _subdir.absoluteFilePath(_filenames.at(j)) << "\n";
                _flbls.flush();
                _dscrstream.flush();
            }
        }
    }
    return 0;
}
