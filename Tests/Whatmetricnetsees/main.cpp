#include <iostream>

#include "dlibopencvconverter.h"
#include "opencvimgaugment.h"

#include "customnetwork.h"

#include <opencv2/highgui.hpp>

using namespace std;

const cv::String keys =  "{image i    |    | filename of the image to be processed}"
                         "{reference  |    | filename of the reference image, if not provided the image itself will be used}"
                         "{model m    |    | filename of the model weights}"
                         "{irows      |    | input network rows}"
                         "{icols      |    | input network cols}"
                         "{rows r     | 20 | target number of horizontal steps}"
                         "{cols c     | 20 | target number of vertical steps}"
                         "{help h     |    | this help}";

int main(int argc, char ** argv) try
{
    cv::CommandLineParser _cmd(argc,argv,keys);
    if(_cmd.has("help")) {
        _cmd.printMessage();
        return 0;
    }
    if(!_cmd.has("image")) {
        cout << "You have not provide input image for analysis. Abort..." << endl;
        return 1;
    }
    if(!_cmd.has("model")) {
        cout << "You have not provide model for analysis. Abort..." << endl;
        return 2;
    }
    if(!_cmd.has("irows")) {
        cout << "You have not provide model input rows size. Abort..." << endl;
        return 3;
    }
    if(!_cmd.has("icols")) {
        cout << "You have not provide model input cols size. Abort..." << endl;
        return 4;
    }

    cv::Size _imgsize(_cmd.get<int>("icols"),_cmd.get<int>("irows"));
    bool _loaded = false;    
    // Ok, here we need to load image with network's input size
    cv::Mat _cvmat = loadIbgrmatWsize(_cmd.get<string>("image"),_imgsize.width,_imgsize.height,true,&_loaded);
    if(_loaded == false) {
        cout << "Can not load image. Please, check that file exists. Abort..." << endl;
        return 3;
    }
    // Ok, here we need to load reference image
    cv::Mat _cvrefmat = _cvmat;
    if(_cmd.has("reference")) {
        _cvrefmat = loadIbgrmatWsize(_cmd.get<string>("reference"),_imgsize.width,_imgsize.height,true,&_loaded);
        if(_loaded == false) {
            cout << "Can not load reference image. Please, check that file exists. Abort..." << endl;
            return 4;
        }
    }

    dlib::anet_type net;
    dlib::deserialize(_cmd.get<string>("model")) >> net;

    // Let's make matrices for predictions
    std::vector<dlib::matrix<dlib::rgb_pixel>> _dlibmatrices;
    _dlibmatrices.reserve(_cmd.get<unsigned int>("rows")*_cmd.get<unsigned int>("cols") + 1);
    float _xstep = 1.0f / _cmd.get<unsigned int>("cols"),
          _ystep = 1.0f / _cmd.get<unsigned int>("rows");
    cout << "Prepare images..." << endl;
    _dlibmatrices.push_back(cvmat2dlibmatrix<dlib::rgb_pixel>(_cvrefmat)); // reference image
    for(unsigned int i = 0; i < _cmd.get<unsigned int>("rows"); ++i) {
        for(unsigned int j = 0; j < _cmd.get<unsigned int>("cols"); ++j) {
            _dlibmatrices.push_back(cvmat2dlibmatrix<dlib::rgb_pixel>(cutoutRect(_cvmat,j*_xstep,i*_ystep,0.1f,0.1f)));
        }
    }
    cout << "Inference..." << endl;
    std::vector<dlib::matrix<float,0,1>> embedded = net(_dlibmatrices);
    _dlibmatrices.clear();
    _dlibmatrices.shrink_to_fit();

    cv::Mat _resultmat(_cmd.get<int>("rows"),_cmd.get<int>("cols"),CV_32FC1);
    float *_dataptr = _resultmat.ptr<float>(0);    
    for(unsigned long i = 0; i < _resultmat.total(); ++i) {
        _dataptr[i] = dlib::length(embedded[0] - embedded[1+i]);
    }
    double _min, _max;
    cv::minMaxIdx(_resultmat,&_min,&_max);
    cout << "Min distance: " << _min << endl;
    cout << "Max distance: " << _max << endl;
    embedded.clear();
    embedded.shrink_to_fit();
    cv::resize(_resultmat,_resultmat,_imgsize,0,0,cv::INTER_CUBIC);
    cv::normalize(_resultmat,_resultmat,1,0,cv::NORM_MINMAX);
    unsigned char *_blendptr = _cvmat.ptr<unsigned char>(0);
    float *_attentionptr = _resultmat.ptr<float>(0);
    for(unsigned long i = 0; i < _cvmat.total(); ++i) {
        _blendptr[3*i+2] = static_cast<unsigned char>(_attentionptr[i]*_blendptr[3*i+2]);
        _blendptr[3*i+1] = static_cast<unsigned char>(_attentionptr[i]*_blendptr[3*i+1]);
        _blendptr[3*i] = static_cast<unsigned char>(_attentionptr[i]*_blendptr[3*i]);
    }
    cv::imshow("Image",_cvmat);
    cv::imshow("Attention map",_resultmat);
    cout << "Press any key to exit..." << endl;
    cv::waitKey(0);
    return 0;
}
catch(std::exception& e)
{
    cout << e.what() << endl;
}
