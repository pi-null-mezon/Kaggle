#include <iostream>
#include <string>

#include <dlib/dnn.h>
#include <dlib/misc_api.h>

#include <opencv2/core.hpp>

#include "dlibimgaugment.h"
#include "opencvimgaugment.h"

#include "customnetwork.h"

using namespace dlib;
using namespace std;

std::vector<std::vector<string>> load_objects_list (const string& dir)
{
    std::vector<std::vector<string>> objects;
    for (auto subdir : directory(dir).get_dirs())
    {
        std::vector<string> imgs;
        for (auto img : subdir.get_files())
            imgs.push_back(img);

        if (imgs.size() != 0)
            objects.push_back(imgs);
    }
    return objects;
}

void load_mini_batch (
    const size_t num_whales,
    const size_t samples_per_id,
    dlib::rand& rnd,
    cv::RNG & cvrng,
    const std::vector<std::vector<string>>& objs,
    std::vector<matrix<float>>& images,
    std::vector<unsigned long>& labels,
    bool _doaugmentation
)
{
    images.clear();
    labels.clear();
    DLIB_CASSERT(num_whales <= objs.size(), "The dataset doesn't have that many whales in it.");

    string obj;
    cv::Mat _tmpmat;
    bool _isloaded;
    std::vector<bool> already_selected(objs.size(), false);
    for (size_t i = 0; i < num_whales; ++i) {

        size_t id = rnd.get_random_32bit_number() % objs.size();
        while(already_selected[id]) {
            id = rnd.get_random_32bit_number() % objs.size();
        }
        already_selected[id] = true;

        for (size_t j = 0; j < samples_per_id; ++j) {

            if(objs[id].size() == samples_per_id) {
                obj = objs[id][j];
            } else {
                obj = objs[id][rnd.get_random_32bit_number()%objs[id].size()];
            }

            if(_doaugmentation) {
                _tmpmat = loadIFgraymatWsize(obj,IMG_WIDTH,IMG_HEIGHT,false,true,true,&_isloaded);
                assert(_isloaded);

                if(rnd.get_random_float() > 0.2f)
                    _tmpmat = distortimage(_tmpmat,cvrng,0.05,cv::INTER_CUBIC,cv::BORDER_REFLECT101);
                if(rnd.get_random_float() > 0.1f)
                    _tmpmat = jitterimage(_tmpmat,cvrng,cv::Size(0,0),0.05,0.05,11,cv::BORDER_REFLECT101);

                if(rnd.get_random_float() > 0.2f)
                    _tmpmat = cutoutRect(_tmpmat,rnd.get_random_float(),0,0.2f,0.4f,rnd.get_random_float()*180.0f);
                if(rnd.get_random_float() > 0.2f)
                    _tmpmat = cutoutRect(_tmpmat,rnd.get_random_float(),1,0.2f,0.4f,rnd.get_random_float()*180.0f);
                if(rnd.get_random_float() > 0.2f)
                    _tmpmat = cutoutRect(_tmpmat,0,rnd.get_random_float(),0.2f,0.4f,rnd.get_random_float()*180.0f);
                if(rnd.get_random_float() > 0.2f)
                    _tmpmat = cutoutRect(_tmpmat,1,rnd.get_random_float(),0.2f,0.4f,rnd.get_random_float()*180.0f);

                if(rnd.get_random_float() > 0.2f)
                    _tmpmat = cutoutRect(_tmpmat,rnd.get_random_float(),rnd.get_random_float(),0.1f,0.3f,rnd.get_random_float()*180.0f);

                images.push_back(cvmat2dlibmatrix<float>(_tmpmat));
            } else {
                images.push_back(load_grayscale_image_with_fixed_size(obj,IMG_WIDTH,IMG_HEIGHT,false,true,true,&_isloaded));
                assert(_isloaded);
            }

            labels.push_back(id);
        }
    }    
}

const cv::String options = "{traindir t  |      | path to directory with training data}"
                           "{validdir v  |      | path to directory with validation data}"
                           "{outputdir o |      | path to directory with output data}"
                           "{minlrthresh | 1E-4 | path to directory with output data}"
                           "{sessionguid |      | session guid}";

int main(int argc, char** argv)
{
    cv::CommandLineParser cmdparser(argc,argv,options);
    cmdparser.about("This app was designed to train identification cnn of humpback whale flukes");
    if(argc == 1) {
        cmdparser.printMessage();
        return 0;
    }
    if(!cmdparser.has("traindir")) {
        cout << "No training directory provided. Abort..." << std::endl;
        return 1;
    }
    if(!cmdparser.has("validdir")) {
        cout << "No validation directory provided. Abort..." << std::endl;
        return 2;
    }
    if(!cmdparser.has("outputdir")) {
        cout << "No output directory provided. Abort..." << std::endl;
        return 3;
    }
    string sessionguid = std::to_string(time(nullptr));
    if(cmdparser.has("sessionguid")) {
        sessionguid = cmdparser.get<string>("sessionguid");
    }

    auto trainobjs = load_objects_list(cmdparser.get<string>("traindir"));
    cout << "trainobjs.size(): "<< trainobjs.size() << endl;
    std::vector<matrix<float>> images;
    std::vector<unsigned long> labels;

    auto validobjs = load_objects_list(cmdparser.get<string>("validdir"));
    cout << "validobjs.size(): "<< validobjs.size() << endl;
    std::vector<matrix<float>> vimages;
    std::vector<unsigned long> vlabels;

    net_type net;
    dnn_trainer<net_type> trainer(net, sgd(0.0005f, 0.9f));
    trainer.set_learning_rate(0.1);
    trainer.be_verbose();
    trainer.set_synchronization_file(cmdparser.get<string>("outputdir") + string("/trainer_") + sessionguid + string("_sync") , std::chrono::minutes(10));
    trainer.set_iterations_without_progress_threshold(5000);
    const float _lratio = static_cast<float>(validobjs.size())/(validobjs.size() + trainobjs.size());
    cout << "Validation classes / Total classes: " << _lratio << endl;
    if(_lratio > 0.05) {
        trainer.set_test_iterations_without_progress_threshold(800);
    } else {
        cout << "Small validation set >> learning rate will be controlled only by training loss progress" << endl;
    }

    dlib::pipe<std::vector<matrix<float>>> qimages(4);
    dlib::pipe<std::vector<unsigned long>> qlabels(4);
    auto data_loader = [&qimages, &qlabels, &trainobjs](time_t seed)  {

        dlib::rand rnd(time(nullptr)+seed);
        cv::RNG cvrng(static_cast<uint64_t>(time(nullptr) + seed));

        std::vector<matrix<float>> images;
        std::vector<unsigned long> labels;

        while(qimages.is_enabled()) {
            try {
                load_mini_batch(90, 2, rnd, cvrng, trainobjs, images, labels,true);
                qimages.enqueue(images);
                qlabels.enqueue(labels);
            }
            catch(std::exception& e) {
                cout << "EXCEPTION IN LOADING DATA" << endl;
                cout << e.what() << endl;
            }
        }
    };
    std::thread data_loader1([data_loader](){ data_loader(1); });
    std::thread data_loader2([data_loader](){ data_loader(2); });
    std::thread data_loader3([data_loader](){ data_loader(3); });

    // Same for the test
    dlib::pipe<std::vector<matrix<float>>> testqimages(1);
    dlib::pipe<std::vector<unsigned long>> testqlabels(1);
    auto testdata_loader = [&testqimages, &testqlabels, &validobjs](time_t seed) {

        dlib::rand rnd(time(nullptr)+seed);
        cv::RNG cvrng(static_cast<uint64_t>(time(nullptr) + seed));

        std::vector<matrix<float>> images;
        std::vector<unsigned long> labels;

        while(testqimages.is_enabled()) {
            try  {
                load_mini_batch(90, 2, rnd, cvrng, validobjs, images, labels, true);
                testqimages.enqueue(images);
                testqlabels.enqueue(labels);
            }
            catch(std::exception& e)
            {
                cout << "EXCEPTION IN LOADING DATA" << endl;
                cout << e.what() << endl;
            }
        }
    };
    std::thread testdata_loader1([testdata_loader](){ testdata_loader(1); });

    size_t _step = 0;
    while(trainer.get_learning_rate() >= cmdparser.get<double>("minlrthresh"))  {
        _step++;
        images.clear();
        labels.clear();
        qimages.dequeue(images);
        qlabels.dequeue(labels);
        trainer.train_one_step(images, labels);
        if((_step % 10) == 0) {
            vimages.clear();
            vlabels.clear();
            testqimages.dequeue(vimages);
            testqlabels.dequeue(vlabels);
            trainer.test_one_step(vimages,vlabels);
        }
    }

    // stop all the data loading threads and wait for them to terminate.
    qimages.disable();
    qlabels.disable();
    data_loader1.join();
    data_loader2.join();
    data_loader3.join();

    testqimages.disable();
    testqlabels.disable();
    testdata_loader1.join();

    cout << "Training has been accomplished" << endl;

    // Wait for training threads to stop
    trainer.get_net();
    net.clean();    


    // Now, let's check how well it performs on the validation data
    anet_type anet = net;

    dlib::rand rnd(0);
    cv::RNG cvrng(0);
    const int testsnum = 50;
    float _valaccuracy = 1;
    for(int n = 0; n < testsnum; ++n) {
        load_mini_batch(90, 2, rnd, cvrng, validobjs, vimages, vlabels, true);
        std::vector<matrix<float,0,1>> embedded = anet(vimages);

        // Now, check if the embedding puts images with the same labels near each other and
        // images with different labels far apart.
        int num_right = 0;
        int num_wrong = 0;
        const float _distancethresh = anet.loss_details().get_distance_threshold();
        for (size_t i = 0; i < embedded.size(); ++i) {
            for (size_t j = i+1; j < embedded.size(); ++j)  {
                if (vlabels[i] == vlabels[j])  {
                    // The loss_metric layer will cause images with the same label to be less
                    // than net.loss_details().get_distance_threshold() distance from each
                    // other.  So we can use that distance value as our testing threshold.
                    if (length(embedded[i] - embedded[j]) < _distancethresh)
                        ++num_right;
                    else
                        ++num_wrong;
                } else {
                    if (length(embedded[i]-embedded[j]) >= _distancethresh)
                        ++num_right;
                    else
                        ++num_wrong;
                }
            }
        }
        const float _acc = static_cast<float>(num_right) / (num_right + num_wrong);
        cout << "Test iteration # " << n << endl;
        cout << "accuracy:  " << _acc << endl;
        cout << "num_right: "<< num_right << endl;
        cout << "num_wrong: "<< num_wrong << endl;
        if(_acc < _valaccuracy)
        	_valaccuracy = _acc;
    }

    string _outputfilename = string("whales_") + sessionguid + string("_VA") + std::to_string(_valaccuracy)  + string(".dat"); 
    cout << "Wait untill weights will be serialized to " << _outputfilename << endl;
    serialize(cmdparser.get<string>("outputdir") + string("/") + _outputfilename) << net;
    cout << "Done" << endl;
    return 0;
}


