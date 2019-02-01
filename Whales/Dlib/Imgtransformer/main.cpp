#include <QStringList>
#include <QTextStream>
#include <QFile>
#include <QDir>

#include <iostream>

#include "dlibopencvconverter.h"
#include "opencvimgaugment.h"
#include "opencvimgalign.h"
#include "opencvmorph.h"

#include "customnetwork.h"

#include <opencv2/highgui.hpp>

using namespace std;

const cv::String keys =  "{indirname  i|       | input directory with files to be processed}"
                         "{outdirname o|       | output dir where processed files should be stored}"
                         "{ofilename   |       | output filename, if specified all min area roatated rects will be saved inside this file with coordiantes in original image}"
                         "{model m     |       | filename of the model weights}"
                         "{arows       |  10   | target number of horizontal steps}"
                         "{acols       |  10   | target number of vertical steps}"
                         "{orows       |  192  | target number of rows}"
                         "{ocols       |  512  | target number of cols}"
                         "{athresh     | 0.10  | attention thresh, only regions with higher attention will be preserved}"
                         "{visualize v | false | should be processing steps showed or not}"
                         "{help h      |       | this help}";

/**
 * @brief compute auto (or self) attention map for particular input image and particualr cnn
 * @param _inmat - input image
 * @param _net   - initialized cnn with metric loss function
 * @param _netinputsize - cnn input layer size
 * @param _mapsize - attention map detalization (steps to make for both image dimensions)
 * @return attention map with size equal to input image size and CV_32FC1 type
 */
cv::Mat autoAttentionMap(const cv::Mat &_inmat, dlib::anet_type &_net, const cv::Size &_netinputsize, const cv::Size &_mapsize);

cv::Mat prepareImgForNet(const cv::Mat &_inmat, int _tcols, int _trows, bool _crop, bool _center, bool _normalize);

cv::Mat cropHighAttentionRegion(const cv::Mat &_inmat, dlib::anet_type &_net, const cv::Size &_netinputsize, const cv::Size &_attmapsize, float _attentionthresh, cv::RotatedRect *_returnrect=nullptr);

int main(int argc, char ** argv) try
{
    cv::CommandLineParser _cmd(argc,argv,keys);
    if(_cmd.has("help") || (argc == 1)) {
        _cmd.printMessage();
        return 0;
    }
    if(!_cmd.has("indirname")) {
        cout << "You have not provide input directory for analysis. Abort..." << endl;
        return 1;
    }
    if(!_cmd.has("outdirname")) {
        cout << "You have not provide output directory for enrolled images. Abort..." << endl;
        return 2;
    }
    if(!_cmd.has("model")) {
        cout << "You have not provide model for analysis. Abort..." << endl;
        return 3;
    }

    dlib::anet_type net;
    dlib::deserialize(_cmd.get<string>("model")) >> net;
    cv::Size netinputsize(IMG_WIDTH,IMG_HEIGHT);
    cv::Size attmapsize(_cmd.get<int>("acols"),_cmd.get<int>("arows"));
    const float athresh = _cmd.get<float>("athresh");

    QDir indir(_cmd.get<string>("indirname").c_str()), outdir(_cmd.get<string>("outdirname").c_str());
    if(!outdir.exists())
        outdir.mkpath(outdir.absolutePath());

    QFile _ofile;
    QTextStream _ots;
    if(_cmd.has("ofilename")) {
        _ofile.setFileName(_cmd.get<string>("ofilename").c_str());
        if(_ofile.open(QIODevice::WriteOnly)) {
            _ots.setDevice(&_ofile);
            _ots << "filename,P0_X,P0_Y,P1_X,P1_Y,P2_X,P2_Y,P3_X,P3_Y";
            _ots.setRealNumberPrecision(1);
            _ots.setRealNumberNotation(QTextStream::FixedNotation);
        }
    }

    QStringList filefilters;
    filefilters << "*.jpg" << "*.jpeg" << "*.png";


    cv::Mat _tmpmat, _transformedmat;
    bool _visualizationOn = _cmd.get<bool>("visualize");
    const cv::Size _targetsize(_cmd.get<int>("ocols"),_cmd.get<int>("orows"));
    cv::RotatedRect _rrect;
    cv::Point2f _vertices[4];
    // Let's enroll files in the root of input directory
    QStringList fileslist = indir.entryList(filefilters,QDir::Files | QDir::NoDotAndDotDot);
    for(int i = 0; i < fileslist.size(); ++i) {
        string _filename = indir.absoluteFilePath(fileslist.at(i)).toUtf8().constData();
        cout << _filename;
        _tmpmat = cv::imread(_filename,CV_LOAD_IMAGE_UNCHANGED);      
        if(!_tmpmat.empty()) {                     
            _transformedmat = cropHighAttentionRegion(_tmpmat,net,netinputsize,attmapsize,athresh,&_rrect);
            if(_ofile.isOpen()) {
                _rrect.points(_vertices);
                _ots << '\n' << fileslist.at(i);
                for(int i = 0; i < 4; ++i)
                    _ots << ',' << _vertices[i].x << ',' << _vertices[i].y;
                _ots.flush();
            }
            if(_visualizationOn) {
                cv::imshow("Transformed image", _transformedmat);
                cv::imshow("Original image", _tmpmat);
                cv::waitKey(10);
            }
            if(_transformedmat.total() != _targetsize.area()) {
                if(_transformedmat.total() > _targetsize.area())
                    cv::resize(_transformedmat,_transformedmat,_targetsize,0,0,CV_INTER_AREA);
                else
                    cv::resize(_transformedmat,_transformedmat,_targetsize,0,0,CV_INTER_CUBIC);
            }
            cv::imwrite(outdir.absolutePath().append("/%1").arg(fileslist.at(i)).toStdString(),_transformedmat);
            cout << " - enrolled" << endl;
        } else {
            cout << " - can not be loaded! Abort..." << endl;
            return 6;
        }
    }
    // Now let's enroll files in subdirs
    QStringList listofsubdirs = indir.entryList(QDir::Dirs | QDir::NoDotAndDotDot);
    for(int i = 0; i < listofsubdirs.size(); ++i) {
        cout << listofsubdirs.at(i).toStdString() << endl;
        QDir _subdir(indir.absolutePath().append("/%1").arg(listofsubdirs.at(i)));
        outdir.mkdir(listofsubdirs.at(i));
        QStringList _listoffiles = _subdir.entryList(filefilters,QDir::Files | QDir::NoDotAndDotDot);
        for(int j = 0; j < _listoffiles.size(); ++j) {
            string _filename = _subdir.absoluteFilePath(_listoffiles.at(j)).toStdString();
            cout << "  " << _filename;
            _tmpmat = cv::imread(_filename,CV_LOAD_IMAGE_UNCHANGED);
            if(!_tmpmat.empty()) {
                _transformedmat = cropHighAttentionRegion(_tmpmat,net,netinputsize,attmapsize,athresh,&_rrect);
                if(_ofile.isOpen()) {
                    _rrect.points(_vertices);
                    _ots << '\n' << listofsubdirs.at(i) << "/" << _listoffiles.at(j);
                    for(int i = 0; i < 4; ++i)
                        _ots << ',' << _vertices[i].x << ',' << _vertices[i].y;
                    _ots.flush();
                }
                if(_visualizationOn) {
                    cv::imshow("Transformed image", _transformedmat);
                    cv::imshow("Original image", _tmpmat);
                    cv::waitKey(1);
                }
                if(_transformedmat.total() != _targetsize.area()) {
                    if(_transformedmat.total() > _targetsize.area())
                        cv::resize(_transformedmat,_transformedmat,_targetsize,0,0,CV_INTER_AREA);
                    else
                        cv::resize(_transformedmat,_transformedmat,_targetsize,0,0,CV_INTER_CUBIC);
                }
                cv::imwrite(outdir.absolutePath().append("/%1/%2").arg(listofsubdirs.at(i),_listoffiles.at(j)).toStdString(),_transformedmat);
                cout << " - enrolled" << endl;
            } else {
                cout << " - can not be loaded! Abort..." << endl;
                return 6;
            }

        }
    }


    cout << "All files has been enrolled successfully" << endl;
    return 0;
}
catch(std::exception& e)
{
    cout << e.what() << endl;
}

cv::Mat prepareImgForNet(const cv::Mat &_inmat, int _tcols, int _trows, bool _crop, bool _center, bool _normalize)
{
    cv::Mat _grayscalemat;
    switch(_inmat.channels()) {
        case 4:
            cv::cvtColor(_inmat,_grayscalemat,CV_BGRA2GRAY);
            break;
        case 3:
            cv::cvtColor(_inmat,_grayscalemat,CV_BGR2GRAY);
            break;
        default:
            _grayscalemat = _inmat.clone();
            break;
    }

    if(_crop) {
        _grayscalemat = cropFromCenterAndResize(_grayscalemat,cv::Size(_tcols,_trows));
    } else if((_grayscalemat.cols != _tcols) || (_grayscalemat.rows != _trows)) {
        int resizetype = CV_INTER_AREA;
        if(_grayscalemat.cols*_grayscalemat.rows < _tcols*_trows)
            resizetype = CV_INTER_CUBIC;
        cv::resize(_grayscalemat,_grayscalemat,cv::Size(_tcols,_trows),0,0,resizetype);
    }

    _grayscalemat.convertTo(_grayscalemat,CV_32F);
    if(_center) {
        if(_normalize) {
            cv::Mat _vchannelmean, _vchannelstdev;
            cv::meanStdDev(_grayscalemat,_vchannelmean,_vchannelstdev);
            _grayscalemat = (_grayscalemat - _vchannelmean.at<const double>(0)) / (3.0*_vchannelstdev.at<const double>(0));
        } else {
            cv::Scalar _mean = cv::mean(_grayscalemat);
            _grayscalemat = (_grayscalemat - _mean[0]) / 256;
        }
    }
    return _grayscalemat;
}

cv::Mat autoAttentionMap(const cv::Mat &_inmat, dlib::anet_type &_net, const cv::Size &_netinputsize, const cv::Size &_mapsize)
{
    // Ok, here we need to load image with network's input size and proper preprocessing
    cv::Mat _tmpmat = prepareImgForNet(_inmat,_netinputsize.width,_netinputsize.height,false,true,true);

    // Let's make matrices for predictions
    std::vector<dlib::matrix<float>> _dlibmatrices;
    _dlibmatrices.reserve(_mapsize.height*_mapsize.width + 1);
    _dlibmatrices.push_back(cvmat2dlibmatrix<float>(_tmpmat)); // reference image
    float _xstep = 1.0f / (_mapsize.width-1), _ystep = 1.0f / (_mapsize.height-1);
    float _xsize = 1.25f * _xstep;
    float _ysize = 1.25f * _ystep; // as cutoutRec() count in relative coordiantes
    for(unsigned int i = 0; i < _mapsize.height; ++i) {
        for(unsigned int j = 0; j < _mapsize.width; ++j) {
            _dlibmatrices.push_back(cvmat2dlibmatrix<float>(cutoutRect(_tmpmat,j*_xstep,i*_ystep,_xsize,_ysize)));
        }
    }
    // Let's calculate descriptions
    std::vector<dlib::matrix<float,0,1>> embedded = _net(_dlibmatrices);

    // Let's prepare container for the attention map
    cv::Mat _resultmat(_mapsize.width,_mapsize.height,CV_32FC1);
    float *_dataptr = _resultmat.ptr<float>(0);
    for(unsigned long i = 0; i < _resultmat.total(); ++i)
        _dataptr[i] = dlib::length(embedded[0] - embedded[1+i]);

    /*double _min, _max;
    cv::minMaxIdx(_resultmat,&_min,&_max);
    cout << "Min distance: " << _min << endl;
    cout << "Max distance: " << _max << endl;
    */

    cv::resize(_resultmat,_resultmat,cv::Size(_inmat.cols,_inmat.rows),0,0,cv::INTER_CUBIC);
    return _resultmat;
}


cv::Mat cropHighAttentionRegion(const cv::Mat &_inmat, dlib::anet_type &_net, const cv::Size &_netinputsize, const cv::Size &_attmapsize, float _attentionthresh, cv::RotatedRect *_returnrect)
{
    cv::Mat _attentionmap, _transformedmat;
    // Let's calculate metric-loss-CNN attention map
    _attentionmap = autoAttentionMap(_inmat,_net,_netinputsize,_attmapsize);
    // Let's align image by attention map PCA directions
    _transformedmat = alignPCAWResize(_attentionmap,_inmat,cv::Size(0,0),_attentionthresh,CV_INTER_AREA,cv::BORDER_REPLICATE);
    // Let's find minimum area boundirg rect for binaryzed attention
    cv::threshold(_attentionmap,_attentionmap,_attentionthresh,1,CV_THRESH_BINARY);
    _attentionmap.convertTo(_attentionmap,CV_8U,255,0);
    openAreaAndBorders(_attentionmap,_attentionmap,8,_attentionmap.total()/20,false);
    //cv::imshow("Attention map", _attentionmap);
    std::vector<std::vector<cv::Point>> cnt;
    cv::findContours(_attentionmap,cnt,CV_RETR_EXTERNAL,CV_CHAIN_APPROX_NONE);
    if(cnt.size() > 0) {
        cv::RotatedRect _rrect;
        _rrect = cv::minAreaRect(cnt[0]);
        if(_returnrect)
            *_returnrect = _rrect;
        cv::Point2f _v[4], _w, _h;
        _rrect.points(_v);
        _w = _v[0] - _v[1];
        _h = _v[1] - _v[2];
        float _rrectwidth = std::sqrt(_w.x*_w.x + _w.y*_w.y);
        float _rrectheight = std::sqrt(_h.x*_h.x + _h.y*_h.y);
        if(_rrectwidth < _rrectheight)
           std::swap(_rrectheight,_rrectwidth);
        const float _multiplyer = 1.3; // how much region that will be cropped should be enlarged
        _rrectheight *= _multiplyer;
        _rrectwidth *= _multiplyer;
        cv::Rect2f _rect(cv::Point2f((_inmat.cols-_rrectwidth)/2.0f,(_inmat.rows-_rrectheight)/2.0f),cv::Size2f(_rrectwidth,_rrectheight));
        _rect &= cv::Rect2f(0,0,_inmat.cols,_inmat.rows);
        //cv::rectangle(_transformedmat,_rect,cv::Scalar(0,127,255),2,CV_AA);
        return _transformedmat(_rect);
    }
    return _inmat;
}
