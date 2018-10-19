#include <QDir>
#include <QTextStream>
#include <QStringList>

#include "dlibimgaugment.h"

#include "customnetwork.h"

cv::Mat __loadImage(const std::string &_filenameprefix,int _tcols, int _trows, bool _crop, bool _center, bool _normalize, bool *_isloadded)
{
    cv::Mat _channelsmat[4];
    std::string _postfix[4] = {"_green.png", "_blue.png", "_red.png", "_yellow.png"};
    for(uint8_t i = 0; i < 4; ++i) {
        _channelsmat[i] = loadIFgraymatWsize(_filenameprefix+_postfix[i],_tcols,_trows,_crop,_center,_normalize,_isloadded);
    }
    cv::Mat _outmat;
    cv::merge(_channelsmat,4,_outmat);
    return _outmat;
}

int main(int argc, char *argv[])
{
    QString _indirname, _outfilename, _modeldirname;
    while((--argc > 0) && ((*++argv)[0] == '-'))
        switch(*++argv[0]) {
            case 'h':
                qInfo(" -h      - help");
                qInfo(" -i[str] - input directory name");
                qInfo(" -o[str] - output file name");
                qInfo(" -m[str] - directory with models");
            break;

            case 'i':
                _indirname = ++argv[0];
            break;

            case 'o':
                _outfilename = ++argv[0];
            break;

            case 'm':
                _modeldirname = ++argv[0];
            break;
        }

    qInfo("Input dir name: %s", _indirname.toUtf8().constData());
    qInfo("Output file name: %s", _outfilename.toUtf8().constData());
    if(_indirname.isEmpty()) {
        qInfo("Empty input dir name! Abort...");
        return 1;
    }
    if(_outfilename.isEmpty()) {
        qInfo("Empty output file name! Abort...");
        return 2;
    }
    if(_modeldirname.isEmpty()) {
        qInfo("Empty model directory name! Abort...");
        return 3;
    }
    QDir _modeldir(_modeldirname);
    if(_modeldir.exists() == false) {
        qInfo("Models directory (%s) does not exist! Abort...",_modeldirname.toUtf8().constData());
        return 4;
    }
    QStringList _modelfilesfilter;
    _modelfilesfilter << "*.dat";
    QStringList _modelfiles = _modeldir.entryList(_modelfilesfilter,QDir::Files|QDir::NoDotAndDotDot);
    if(_modelfiles.size() != 28) {
        qInfo("Insufficient number of models %d. Abort...", _modelfiles.size());
        return 5;
    }
    QDir indir(_indirname);
    if(!indir.exists()) {
        qInfo("Input directory does not exist! Abort...");
        return 6;
    }
    QFile _outfile(_outfilename);
    if(_outfile.open(QFile::WriteOnly) == false) {
        qInfo("Could not open output file for write! Abort...");
        return 7;
    }
    QTextStream _ots(&_outfile);
    _ots << "Id,Predicted\n";

    std::vector<anet_type> _vnets;
    _vnets.resize(static_cast<size_t>(_modelfiles.size()));
    for(int i = 0; i < _modelfiles.size(); ++i) {
        unsigned int _classid = _modelfiles.at(i).section("_(MFs",0,0).section("proteins_class_",1).toUInt();
        try {
            dlib::deserialize(_modeldir.absoluteFilePath(_modelfiles.at(i)).toStdString()) >> _vnets[_classid];
        } catch(std::exception& e) {
            cout << e.what() << endl;
        }
    }

    QStringList _filesfilter;
    _filesfilter << "*.png" << "*.tiff";
    QStringList _fileslist = indir.entryList(_filesfilter, QDir::Files | QDir::NoDotAndDotDot);
    qInfo("%d files has been found. Wait untill processing will be accomplished...", _fileslist.size());
    std::string _filenameprefix;
    for(int i = 0; i < _fileslist.size(); ++i) {
        if(_fileslist.at(i).contains("_green")) {
            bool _is_file_loaded = false;
            _filenameprefix = indir.absoluteFilePath(_fileslist.at(i)).section("_green",0,0).toStdString();           
            std::array<dlib::matrix<float>,4> _img = cvmatF2arrayofFdlibmatrix<4>(__loadImage(_filenameprefix,
                                                                                              IMG_SIZE,
                                                                                              IMG_SIZE,
                                                                                              false,
                                                                                              true,
                                                                                              false,
                                                                                              &_is_file_loaded));
            assert(_is_file_loaded);
            _ots << _fileslist.at(i).section("_green",0,0) << ',';

            bool _first_prediction = true;
            for(size_t n = 0; n < _vnets.size(); ++n) {
                unsigned long _predictedlbl = _vnets[n](_img);
                if(_predictedlbl == 1) {
                    if(_first_prediction) {
                        _first_prediction = false;
                        _ots << n;
                    } else {
                        _ots << ' ' << n;
                    }
                }
            }
            _ots << '\n';
        }
    }
    qInfo("All predictions ready");
    return 0;
}
