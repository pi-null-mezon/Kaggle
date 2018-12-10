#include <QStringList>
#include <QDir>

#include <iostream>

#include "dlibopencvconverter.h"
#include "opencvimgaugment.h"
#include "opencvimgalign.h"

#include "customnetwork.h"

#include <opencv2/highgui.hpp>

using namespace std;

const cv::String keys =  "{indirname  i|     | filename of the image to be processed}"
                         "{outdirname o|     | filename of the image to be processed}"
                         "{model m     |     | filename of the model weights}"
                         "{arows       |  20 | target number of horizontal steps}"
                         "{acols       |  20 | target number of vertical steps}"
                         "{help h      |     | this help}";

/**
 * @brief compute auto (or self) attention map for particular input image and particualr cnn
 * @param _inmat - input image
 * @param _net   - initialized cnn
 * @param _netinputsize - cnn input layer size
 * @param _mapsize - attention map detalization (steps to make for both image dimensions)
 * @return attention map with size equal to input image size and CV_32FC1 type
 */
cv::Mat autoAttentionMap(const cv::Mat &_inmat, dlib::anet_type &_net, const cv::Size &_netinputsize, const cv::Size &_mapsize);

cv::Mat prepareImgForNet(const cv::Mat &_inmat, int _tcols, int _trows, bool _crop, bool _center, bool _normalize);

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

    QDir indir(_cmd.get<string>("indirname").c_str()), outdir(_cmd.get<string>("outdirname").c_str());
    if(!outdir.exists())
        outdir.mkpath(outdir.absolutePath());

    QStringList filefilters;
    filefilters << "*.jpg" << "*.jpeg" << "*.png";

    QStringList fileslist = indir.entryList(filefilters,QDir::Files | QDir::NoDotAndDotDot);
    cv::Mat _tmpmat, _attentionmap, _transformedmat;
    for(int i = 0; i < fileslist.size(); ++i) {
        string _filename = indir.absoluteFilePath(fileslist.at(i)).toUtf8().constData();
        _tmpmat = cv::imread(_filename,CV_LOAD_IMAGE_UNCHANGED);
        if(!_tmpmat.empty()) {
            cout << _filename << endl;
            _attentionmap = autoAttentionMap(_tmpmat,net,netinputsize,attmapsize);
            _transformedmat = alignPCAWResize(_attentionmap,_tmpmat,cv::Size(0,0),0.2,CV_INTER_AREA,cv::BORDER_CONSTANT);
            cv::imshow("Original image", _tmpmat);
            cv::imshow("Attention map", _attentionmap);
            cv::imshow("Transformed image", _transformedmat);
            cv::waitKey(0);
        } else {
            cout << _filename << " - can not be loaded! Abort..." << endl;
            return 6;
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
    float _xstep = 1.0f / _mapsize.width, _ystep = 1.0f / _mapsize.height;
    for(unsigned int i = 0; i < _mapsize.height; ++i) {
        for(unsigned int j = 0; j < _mapsize.width; ++j) {
            _dlibmatrices.push_back(cvmat2dlibmatrix<float>(cutoutRect(_tmpmat,j*_xstep,i*_ystep,0.1f,0.1f)));
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
    cv::normalize(_resultmat,_resultmat,1,0,cv::NORM_MINMAX);
    //cv::threshold(_resultmat,_resultmat,0.35,1,cv::THRESH_BINARY);
    return _resultmat;
}
