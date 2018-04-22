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

#define CLASSES 128
#define IMGSIZE 313

// ----------------------------------------------------------------------------------------
template <template <int,template<typename>class,int,typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual = add_prev1<block<N,BN,1,tag1<SUBNET>>>;

template <template <int,template<typename>class,int,typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual_down = add_prev2<avg_pool<2,2,2,2,skip1<tag2<block<N,BN,2,tag1<SUBNET>>>>>>;

template <int N, template <typename> class BN, int stride, typename SUBNET>
using block  = BN<con<N,3,3,1,1,relu<BN<con<N,3,3,stride,stride,SUBNET>>>>>;


template <int N, typename SUBNET> using res       = relu<residual<block,N,bn_con,SUBNET>>;
template <int N, typename SUBNET> using ares      = relu<residual<block,N,affine,SUBNET>>;
template <int N, typename SUBNET> using res_down  = relu<residual_down<block,N,bn_con,SUBNET>>;
template <int N, typename SUBNET> using ares_down = relu<residual_down<block,N,affine,SUBNET>>;

// ----------------------------------------------------------------------------------------

template <typename SUBNET> using level1 = res<256,res<256,res_down<256,SUBNET>>>;
template <typename SUBNET> using level2 = res<128,res<128,res_down<128,SUBNET>>>;
template <typename SUBNET> using level3 = res<64,res<64,res<64,res_down<64,SUBNET>>>>;
template <typename SUBNET> using level4 = res<32,res<32,res_down<32,SUBNET>>>;

template <typename SUBNET> using alevel1 = ares<256,ares<256,ares_down<256,SUBNET>>>;
template <typename SUBNET> using alevel2 = ares<128,ares<128,ares_down<128,SUBNET>>>;
template <typename SUBNET> using alevel3 = ares<64,ares<64,ares<64,ares_down<64,SUBNET>>>>;
template <typename SUBNET> using alevel4 = ares<32,ares<32,ares_down<32,SUBNET>>>;

// training network type
using net_type = loss_multiclass_log<fc<CLASSES,avg_pool_everything<
                            level1<
                            level2<
                            level3<
                            level4<
                            relu<bn_con<con<16,5,5,2,2,
                            input_rgb_image_sized<IMGSIZE>
                            >>>>>>>>>>;

// testing network type (replaced batch normalization with fixed affine transforms)
using anet_type = loss_multiclass_log<fc<CLASSES,avg_pool_everything<
                            alevel1<
                            alevel2<
                            alevel3<
                            alevel4<
                            relu<affine<con<16,5,5,2,2,
                            input_rgb_image_sized<IMGSIZE>
                            >>>>>>>>>>;

// ----------------------------------------------------------------------------------------

std::vector<string> get_labels_names(const std::string& images_folder)
{
    std::vector<string> _labels;
    auto subdirs = directory(images_folder).get_dirs();
    std::sort(subdirs.begin(), subdirs.end());
    cout << "Labels names:" << endl;
    _labels.resize(subdirs.size());
    for(size_t i = 0; i < subdirs.size(); ++i) {
        _labels[i] = subdirs[i].name();
        cout << "label " << i << " - class name " << _labels[i] << endl;
    }
    return _labels;
}

int main(int argc, char** argv) try
{
    if (argc != 5) {
        cout << "How to run:" << endl;
        cout << "./app /path/to/classifier /path/to/traindir /path/to/testdir /path/to/output.csv" << endl;
        return 1;
    }

    cout << "\nSCANNING CLASSIFIER...\n" << std::endl;      
    anet_type net;
    deserialize(argv[1]) >> net;
    cout << "Network has been successfully loaded" << std::endl;

    cout << "SCANNING LABELS MAPPING...\n" << endl;
    auto lblname = get_labels_names(argv[2]);

    cout << "\nSCANNING TEST DATASET...\n" << endl;
    auto _dir = directory(argv[3]);
    auto _vfiles = _dir.get_files();
    cout << "Success (" << _vfiles.size() << " has been found)" << endl;

    cout << "\nGENERATING OUTPUT TABLE...\n" << std::endl;
    std::ofstream ofs;
    ofs.open(argv[4]);
    ofs << "id,predicted" << std::endl;

    matrix<rgb_pixel> img;
    bool _isloaded;
    string _label;
    string _filename;
    dlib::rand _rnd(time(0));   

    for(size_t j = 1; j <= 12800; ++j) { // cause test data contains 12800 entities
        _filename = std::to_string(j);
        cout << j << ") filename " << _filename;
        img = std::move(load_rgb_image_with_fixed_size(std::string(argv[3]) + std::string("/") + _filename + std::string(".jpg"),IMGSIZE,IMGSIZE,&_isloaded));
        if(_isloaded) {
            _label = lblname[net(img)]; // shift is caused by different indexing
            cout << "; label predicted " << _label;
        } else {
            _label = std::to_string(_rnd.get_integer_in_range(1,CLASSES));
            cout << "; label guessed " << _label;
        }
        cout << endl;
        ofs << _filename << ',' << _label << std::endl;
    }
    cout << endl;
}
catch(std::exception& e)
{
    cout << e.what() << endl;
}

