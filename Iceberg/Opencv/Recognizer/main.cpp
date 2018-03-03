// The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
/*
    This program was used to train the resnet34_1000_imagenet_classifier.dnn
    network used by the dnn_imagenet_ex.cpp example program.  

    You should be familiar with dlib's DNN module before reading this example
    program.  So read dnn_introduction_ex.cpp and dnn_introduction2_ex.cpp first.  
*/
#include <iostream>

#include <QDir>

#include <dlib/image_io.h>

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

#include "opencvimgaugment.h"

using namespace std;

string cutFileExtension(string _str) {
    return _str.substr(0,_str.rfind('.',_str.size()));
}

const cv::String options =  "{ help h       |   | show help                         }"
                            "{ prototxt p   |   | path to prototxt                  }"
                            "{ model m      |   | path to model                     }"
                            "{ dir i        |   | input directory with test images  }"
                            "{ output o     |   | output filename                   }";

int main(int argc, char** argv)
{
    cv::CommandLineParser cmdparser(argc,argv,options);
    cmdparser.about("To run this program you need a copy Kaggle Iceberg Test dataset and Learned classifier");
    if(cmdparser.has("help")) {
        cmdparser.printMessage();
        return 0;
    }
    if(cmdparser.has("dir") == false) {
        cerr << "You should provide input directory name!";
        return -1;
    }
    if(cmdparser.has("output") == false) {
        cerr << "You should provide output filename!";
        return -1;
    }

    cv::String modelTxt = cmdparser.get<cv::String>("prototxt");
    cv::String modelBin = cmdparser.get<cv::String>("model");

    cv::dnn::Net net;
    cout << "\nSCANNING CLASSIFIER...\n" << std::endl;
    try {
        //! [Read and initialize network]
        net = cv::dnn::readNetFromCaffe(modelTxt, modelBin);
        //! [Read and initialize network]
    }
    catch (cv::Exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        //! [Check that network was read successfully]
        if (net.empty()) {
            std::cerr << "Can't load network by using the following files: " << std::endl;
            std::cerr << "prototxt:   " << modelTxt << std::endl;
            std::cerr << "caffemodel: " << modelBin << std::endl;
            return -1;
        }
        //! [Check that network was read successfully]
    }
    cout << "Success, classifiers has been found" << endl;

    // Here I have used dlib's file system facilities to preserve filenames ordering (it was simplest way to save compatibility with Dlib's recognition results)
    cout << "\nSCANNING TEST DATASET...\n" << endl;
    auto _dir = dlib::directory(cmdparser.get<string>("dir"));
    auto _vfiles = _dir.get_files();
    cout << "Success (" << _vfiles.size() << " has been found). Start processing..." << endl;
    std::vector<float> vp(_vfiles.size(),0.0);
    std::vector<float> vpship(_vfiles.size(),0.0);

    cout << "Task progress (* - 5 %): ";
    cout.flush();
    size_t _progressstep = _vfiles.size()/20;

    cv::RNG _rng;
    for(size_t j = 0; j < _vfiles.size(); ++j) {

        cv::Mat imgmat = cv::imread(_vfiles[j].full_name(),CV_LOAD_IMAGE_UNCHANGED);
        if(imgmat.channels() == 4) {
            cv::cvtColor(imgmat,imgmat,CV_BGRA2BGR);
        }

        size_t _numcrops = 11;
        std::vector<cv::Mat> _vcrops;
        getImageFSCrops(imgmat, _numcrops, _vcrops, _rng,cv::Size(70,70));

        for(size_t i = 0; i < _vcrops.size(); ++i) {
            // Note that scale factor and mean substraction should be adjusted manually in the following string
            cv::Mat inputblob = cv::dnn::blobFromImage(_vcrops[i],0.2f,cv::Size(65,65),cv::mean(imgmat));
            net.setInput(inputblob,"data");
            cv::Mat prob = net.forward("softmax");
            vp[j] += prob.at<float>(0);
            vpship[j] += prob.at<float>(1);
        }
        vp[j] /= _vcrops.size();
        vpship[j] /= _vcrops.size();

        if((j % _progressstep) == 0)
            cout << '*'; cout.flush();
    }

    std::ofstream ofs;
    ofs.open(cmdparser.get<string>("output"));
    ofs << "id,is_iceberg" << std::endl;
    //ofs << std::fixed; // fixed notation of the floating point numbers
    cout << "Input processing has been accomplished successfully. Wait untill output will be saved to disk..." << endl;
    QDir _qdir(argv[2]);
    _qdir.mkdir("Iceberg");
    _qdir.mkdir("Ship");
    for (size_t j = 0; j < _vfiles.size(); ++j) {
        cout << "file#: " << j << "\tname: " << _vfiles[j].name() << std::endl;
        ofs << cutFileExtension(_vfiles[j].name()) << ',' << vp[j]  << std::endl;
    }
}



