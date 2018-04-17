// The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
/*
    This program was used to train the resnet34_1000_imagenet_classifier.dnn
    network used by the dnn_imagenet_ex.cpp example program.  

    You should be familiar with dlib's DNN module before reading this example
    program.  So read dnn_introduction_ex.cpp and dnn_introduction2_ex.cpp first.  
*/

#include <QDir>

#include <dlib/dnn.h>
#include <iostream>
#include <dlib/data_io.h>
#include <dlib/dir_nav.h>
#include <iterator>
#include <thread>

#include "dlibimgaugment.h"

using namespace std;
using namespace dlib;

#define FILTERS 16

using anet_type = loss_multiclass_log<fc<2,avg_pool_everything<dropout<
                            max_pool<2,2,2,2,relu<dropout<con<8*FILTERS,3,3,1,1,
                            max_pool<2,2,2,2,relu<dropout<con<4*FILTERS,3,3,1,1,
                            max_pool<3,3,2,2,relu<con<FILTERS,5,5,2,2,
                            input_rgb_image_sized<75>
                            >>>>>>>>>>>>>>>;

// ---------------------------------------------------------------------------------------

string cutFileExtension(string _str) {
    return _str.substr(0,_str.rfind('.',_str.size()));
}

// ----------------------------------------------------------------------------------------

int main(int argc, char** argv) try
{
    if (argc != 4)
    {
        cout << "To run this program you need a copy Kaggle Iceberg Test dataset and Learned classifier" << endl;
        cout << endl;
        cout << "With those things, you call this program like this: " << endl;
        cout << "./dnn_classifier_run /path/to/classifiers_ensemble /path/to/Testdir /path/to/outputtable.csv" << endl;
        return 1;
    }

    cout << "\nSCANNING TEST DATASET...\n" << endl;
    auto _dir = directory(argv[2]);
    auto _vfiles = _dir.get_files();
    cout << "Success (" << _vfiles.size() << " has been found). Start processing..." << endl;

    cout << "\nSCANNING CLASSIFIER...\n" << std::endl;
    auto _ensdir = directory(argv[1]);
    auto _ensvfiles = _ensdir.get_files();
    cout << "Total " << _ensvfiles.size() << " classifiers has been found, please wait until deserialization will be performed..." << endl;
    std::vector<float> vp(_vfiles.size(),0.0);
    std::vector<float> vpship(_vfiles.size(),0.0);
    unsigned short _ensemble_size = 0;
    for(size_t i = 0; i < _ensvfiles.size(); ++i) {
        anet_type net;
        deserialize(_ensvfiles[i].full_name()) >> net;
        // Now test the network on the validation dataset.  First, make a testing
        // network with softmax as the final layer. snet object will make getting
        // the predictions easy as it directly outputs the probability of each class
        // as its final output. Kaggle rules for the ICeberg competition demands
        // output as iceberg class probability for the each image. So we need SOFTMAX
        cout << "Net#: " << i <<" has been successfully loaded, wait untill all input images will be processed (* - 5 %)" << std::endl;
        _ensemble_size++;
        softmax<anet_type::subnet_type> snet;
        snet.subnet() = net.subnet();
        dlib::rand rnd(time(0)+i);
        cout << "task progress: "; cout.flush();
        size_t _progressstep = _vfiles.size()/20;
        for(size_t j = 0; j < _vfiles.size(); ++j) {
            dlib::array<matrix<rgb_pixel>> images;
            matrix<rgb_pixel> img;
            load_image(img, _vfiles[j].full_name());
            // Grab 7 random crops from the image.  We will run all of them through the
            // network and average the results.
            const size_t num_crops = 13;
            randomly_crop_image(img, images, rnd, num_crops);
            matrix<float,1,2> p1 = sum_rows(mat(snet(images.begin(), images.end())))/num_crops;
            randomly_jitter_image(img,images,num_crops,num_crops);
            matrix<float,1,2> p2 = sum_rows(mat(snet(images.begin(), images.end())))/num_crops;
            // p(i) == the probability the image contains object of class i.
            matrix<float,1,2> p = (p1+p2)/2.0f;
            vp[j] += p(0);
            vpship[j] += p(1);
            if( (j % _progressstep) == 0)
                cout << '*'; cout.flush();
        }
        cout << endl;
    }

    std::ofstream ofs;
    ofs.open(argv[3]);
    ofs << "id,is_iceberg" << std::endl;
    //ofs << std::fixed; // fixed notation of the floating point numbers
    cout << "Input processing has been accomplished successfully. Wait untill output will be saved to disk..." << endl;
    QDir _qdir(argv[2]);
    _qdir.mkdir("Iceberg");
    _qdir.mkdir("Ship");
    for (size_t j = 0; j < _vfiles.size(); ++j) {
        cout << "file#: " << j << "\tname: " << _vfiles[j].name() << std::endl;
        if(_ensemble_size > 1) {
            ofs << cutFileExtension(_vfiles[j].name()) << ',' << (vp[j]/_ensemble_size) << std::endl;
            /*if(vp[j]/_ensemble_size > 0.9) {
                QFile::copy(_vfiles[j].full_name().c_str(),_qdir.absolutePath().append("/Iceberg/").append(_vfiles[j].name().c_str()));
            } else if(vp[j]/_ensemble_size < 0.1) {
                QFile::copy(_vfiles[j].full_name().c_str(),_qdir.absolutePath().append("/Ship/").append(_vfiles[j].name().c_str()));
            }*/
        } else {
            ofs << cutFileExtension(_vfiles[j].name()) << ',' << vp[j]  << std::endl;
        }
    }
}
catch(std::exception& e)
{
    cout << e.what() << endl;
}

