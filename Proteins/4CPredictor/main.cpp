#include <QDir>
#include <QTextStream>

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

std::vector<std::array<dlib::matrix<float>,4>> getAffineSet(const cv::Mat &_mat);

std::vector<std::array<dlib::matrix<float>,4>> getCrops(const cv::Mat &_mat, cv::RNG &_rng, unsigned int _crops=13);

int main(int argc, char *argv[])
{
    QString _indirname, _outfilename, _modelfilename;
    double _thresh = 0.5;
    while((--argc > 0) && ((*++argv)[0] == '-'))
        switch(*++argv[0]) {
            case 'h':
                qInfo(" -h       - help");
                qInfo(" -i[str]  - input directory name");
                qInfo(" -o[str]  - output file name");
                qInfo(" -m[str]  - model file name");
                qInfo(" -t[real] - recognition threshold (default: %f)", _thresh);
            break;

            case 'i':
                _indirname = ++argv[0];
            break;

            case 'o':
                _outfilename = ++argv[0];
            break;

            case 'm':
                _modelfilename = ++argv[0];
            break;

            case 't':
                _thresh = QString(++argv[0]).toDouble();
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
    if(_modelfilename.isEmpty()) {
        qInfo("Empty model file name! Abort...");
        return 3;
    }
    QFileInfo _modelfileinfo(_modelfilename);
    if(_modelfileinfo.exists() == false) {
        qInfo("Model file (%s) does not exist! Abort...",_modelfileinfo.absoluteFilePath().toUtf8().constData());
        return 4;
    }
    QDir indir(_indirname);
    if(!indir.exists()) {
        qInfo("Input directory does not exist! Abort...");
        return 5;
    }
    QFile _outfile(_outfilename);
    if(_outfile.open(QFile::WriteOnly) == false) {
        qInfo("Could not open output file for write! Abort...");
        return 6;
    }
    QTextStream _ots(&_outfile);
    _ots << "Id,Predicted\n";

    anet_type _net;
    try {
        dlib::deserialize(_modelfilename.toStdString()) >> _net;
    } catch(std::exception& e) {
        cout << e.what() << endl;
    }
    //std::cout << _net << std::endl;
    cv::RNG _cvrng(313);

    QStringList _filesfilter;
    _filesfilter << "*.png" << "*.tiff";
    QStringList _fileslist = indir.entryList(_filesfilter, QDir::Files | QDir::NoDotAndDotDot);
    qInfo("%d files has been found. Wait untill processing will be accomplished...", _fileslist.size());
    std::string _filenameprefix;
    for(int i = 0; i < _fileslist.size(); ++i) {
        if(_fileslist.at(i).contains("_green")) {
            bool _is_file_loaded = false;
            _filenameprefix = indir.absoluteFilePath(_fileslist.at(i)).section("_green",0,0).toStdString();
            /*cv::Mat _mat = __loadImage(_filenameprefix,
                                       IMG_SIZE,
                                       IMG_SIZE,
                                       false,
                                       true,
                                       true,
                                       &_is_file_loaded);*/
            cv::Mat _mat = __loadImage(_filenameprefix,
                                                   512,
                                                   512,
                                                   false,
                                                   true,
                                                   true,
                                                   &_is_file_loaded);
            assert(_is_file_loaded);

            std::vector<std::map<std::string,dlib::loss_multimulticlass_log_::classifier_output>> _vpmap = _net(getCrops(_mat,_cvrng)/*getAffineSet(_mat)*/);
            std::vector<double> _vprob(_net.loss_details().number_of_classifiers(),0);
            for(size_t _n = 0; _n < _vprob.size(); ++_n) {
                std::string _classname = std::to_string(_n);
                for(size_t _k = 0; _k < _vpmap.size(); ++_k) {
                    _vprob[_n] += _vpmap[_k].at(_classname).probability_of_class(0);
                }
                _vprob[_n] /= _vpmap.size();
            }

            _ots << _fileslist.at(i).section("_green",0,0) << ',';
            bool _first_prediction = true, _presented;
            for(size_t j = 0; j < _vprob.size(); ++j) {
                _presented = false;
                std::string _classname = std::to_string(j);
                if(j == 0) {
                    if(_vprob[j] > 0.40)
                        _presented = true;
                } else if(j == 27 || j == 15 || j == 10 || j == 9 || j == 8 || j == 20 || j == 17 || j == 24 || j == 26 || j == 16 || j == 13) {
                    if(_vprob[j] > 0.18)
                        _presented = true;
                } else {
                    if(_vprob[j] > 0.25)
                        _presented = true;
                }
                if(_presented) {
                    if(_first_prediction) {
                        _first_prediction = false;
                        _ots << _classname.c_str();
                    } else {
                        _ots << ' ' << _classname.c_str();
                    }
                }
            }
            _ots << '\n';
        }
    }
    qInfo("All predictions ready");
    return 0;
}

std::vector<std::array<dlib::matrix<float>,4>> getAffineSet(const cv::Mat &_mat)
{
    std::vector<std::array<dlib::matrix<float>,4>> _vout;
    double _anglestep = 45.0;
    unsigned int _steps = static_cast<unsigned int>(360.0/_anglestep);
    _vout.reserve(2*_steps);

    cv::Mat _tmpmat;
    for(unsigned int i = 0; i < _steps; ++i) {
        if(i >= 0) {
            cv::Mat _rm = cv::getRotationMatrix2D(cv::Point2f(_mat.cols/2.0f,_mat.rows/2.0f),_anglestep*i,1);
            cv::warpAffine(_mat,_tmpmat,_rm,cv::Size(_mat.cols,_mat.rows),cv::INTER_CUBIC,cv::BORDER_REFLECT101);
        } else {
            _tmpmat = _mat;
        }
        _vout.push_back(cvmatF2arrayofFdlibmatrix<4>(_tmpmat));
    }

    cv::Mat _flippedmat;
    cv::flip(_mat,_flippedmat,0);
    for(unsigned int i = 0; i < _steps; ++i) {
        if(i >= 0) {
            cv::Mat _rm = cv::getRotationMatrix2D(cv::Point2f(_mat.cols/2.0f,_mat.rows/2.0f),_anglestep*i,1);
            cv::warpAffine(_flippedmat,_tmpmat,_rm,cv::Size(_mat.cols,_mat.rows),cv::INTER_CUBIC,cv::BORDER_REFLECT101);
        } else {
            _tmpmat = _flippedmat;
        }
        _vout.push_back(cvmatF2arrayofFdlibmatrix<4>(_tmpmat));
    }

    /*cv::flip(_mat,_flippedmat,1);
    for(unsigned int i = 0; i < _steps; ++i) {
        if(i >= 0) {
            cv::Mat _rm = cv::getRotationMatrix2D(cv::Point2f(_mat.cols/2.0f,_mat.rows/2.0f),_anglestep*i,1);
            cv::warpAffine(_flippedmat,_tmpmat,_rm,cv::Size(_mat.cols,_mat.rows),cv::INTER_CUBIC,cv::BORDER_REFLECT101);
        } else {
            _tmpmat = _flippedmat;
        }
        _vout.push_back(cvmatF2arrayofFdlibmatrix<4>(_tmpmat));
    }

    cv::flip(_mat,_flippedmat,-1);
    for(unsigned int i = 0; i < _steps; ++i) {
        if(i >= 0) {
            cv::Mat _rm = cv::getRotationMatrix2D(cv::Point2f(_mat.cols/2.0f,_mat.rows/2.0f),_anglestep*i,1);
            cv::warpAffine(_flippedmat,_tmpmat,_rm,cv::Size(_mat.cols,_mat.rows),cv::INTER_CUBIC,cv::BORDER_REFLECT101);
        } else {
            _tmpmat = _flippedmat;
        }
        _vout.push_back(cvmatF2arrayofFdlibmatrix<4>(_tmpmat));
    }*/

    return _vout;
}


std::vector<std::array<dlib::matrix<float>,4>> getCrops(const cv::Mat &_mat, cv::RNG &_rng, unsigned int _crops)
{
    std::vector<std::array<dlib::matrix<float>,4>> _vout;
    _vout.reserve(4*_crops);
    for(unsigned int i = 0; i < _crops; ++i) {
        cv::Mat _tmpmat = cropimage(_mat,cv::Size(400,400),&_rng);
        //cv::resize(_tmpmat,_tmpmat,cv::Size(IMG_SIZE,IMG_SIZE),0,0,cv::INTER_AREA);
        _vout.push_back(cvmatF2arrayofFdlibmatrix<4>(_tmpmat));
    }
    /*cv::Mat _flippedmat;
    cv::flip(_mat,_flippedmat,0);
    for(unsigned int i = 0; i < _crops; ++i) {
        cv::Mat _tmpmat = cropimage(_flippedmat,cv::Size(400,400),&_rng);
        //cv::resize(_tmpmat,_tmpmat,cv::Size(IMG_SIZE,IMG_SIZE),0,0,cv::INTER_AREA);
        _vout.push_back(cvmatF2arrayofFdlibmatrix<4>(_tmpmat));
    }
    cv::flip(_mat,_flippedmat,1);
    for(unsigned int i = 0; i < _crops; ++i) {
        cv::Mat _tmpmat = cropimage(_flippedmat,cv::Size(400,400),&_rng);
        //cv::resize(_tmpmat,_tmpmat,cv::Size(IMG_SIZE,IMG_SIZE),0,0,cv::INTER_AREA);
        _vout.push_back(cvmatF2arrayofFdlibmatrix<4>(_tmpmat));
    }
    cv::flip(_mat,_flippedmat,-1);
    for(unsigned int i = 0; i < _crops; ++i) {
        cv::Mat _tmpmat = cropimage(_flippedmat,cv::Size(400,400),&_rng);
        //cv::resize(_tmpmat,_tmpmat,cv::Size(IMG_SIZE,IMG_SIZE),0,0,cv::INTER_AREA);
        _vout.push_back(cvmatF2arrayofFdlibmatrix<4>(_tmpmat));
    }*/
    return  _vout;
}
