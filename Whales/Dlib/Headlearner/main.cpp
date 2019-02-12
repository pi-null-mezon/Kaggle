#include <iostream>
#include <string>

#include <dlib/dnn.h>
#include <dlib/misc_api.h>

#include <opencv2/core.hpp>

#include "dlibimgaugment.h"
#include "opencvimgaugment.h"

#include "customnetwork.h"

#include "dlibwhalesrecognizer.h"

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
    cv::Ptr<cv::oirt::CNNImageRecognizer> descriptor,
    const size_t num_whales,
    const size_t samples_per_id,
    dlib::rand& rnd,
    cv::RNG & cvrng,
    const std::vector<std::vector<string>>& objs,
    std::vector<matrix<float>>& images,
    std::vector<unsigned long>& labels,
    bool _doaugmentation,
    const size_t min_samples=1 // if dir contains number of samples less than min_samples then dir will not be used
)
{
    images.clear();
    labels.clear();
    DLIB_CASSERT(num_whales <= objs.size(), "The dataset doesn't have that many whales in it.");

    string obj;
    cv::Mat _tmpmat;
    std::vector<bool> already_selected(2*objs.size(), false); // as we can effectivelly enlarge training set by horizontal flip operation
    for (size_t i = 0; i < num_whales; ++i) {

        size_t id = rnd.get_random_32bit_number() % (2*objs.size());
        while(already_selected[id] || (objs[id % objs.size()].size() < min_samples)) {
            id = rnd.get_random_32bit_number() % (2*objs.size());
        }
        already_selected[id] = true;

        for (size_t j = 0; j < samples_per_id; ++j) {

            if(objs[id % objs.size()].size() == samples_per_id) {
                obj = objs[id % objs.size()][j];
            } else {
                obj = objs[id % objs.size()][rnd.get_random_32bit_number() % objs[id % objs.size()].size()];
            }            

            if(_doaugmentation) {
                _tmpmat = cv::imread(obj,CV_LOAD_IMAGE_GRAYSCALE);
                if(id >= objs.size())
                    cv::flip(_tmpmat,_tmpmat,1);

                if(rnd.get_random_float() > 0.1f)
                    _tmpmat = jitterimage(_tmpmat,cvrng,cv::Size(0,0),0.14,0.06,14,cv::BORDER_REFLECT101,true);
                if(rnd.get_random_float() > 0.1f)
                    _tmpmat = distortimage(_tmpmat,cvrng,0.09,cv::INTER_CUBIC,cv::BORDER_REFLECT101);

                if(rnd.get_random_float() > 0.1f)
                    _tmpmat = cutoutRect(_tmpmat,rnd.get_random_float(),rnd.get_random_float(),0.1f,0.3f,rnd.get_random_float()*180.0f);
                if(rnd.get_random_float() > 0.1f)
                    _tmpmat = cutoutRect(_tmpmat,rnd.get_random_float(),rnd.get_random_float(),0.1f,0.3f,rnd.get_random_float()*180.0f);

                if(rnd.get_random_float() > 0.1f)
                    _tmpmat = cutoutRect(_tmpmat,rnd.get_random_float(),0,0.2f,0.4f,rnd.get_random_float()*180.0f);
                if(rnd.get_random_float() > 0.1f)
                    _tmpmat = cutoutRect(_tmpmat,rnd.get_random_float(),1,0.2f,0.4f,rnd.get_random_float()*180.0f);
                if(rnd.get_random_float() > 0.1f)
                    _tmpmat = cutoutRect(_tmpmat,0,rnd.get_random_float(),0.2f,0.4f,rnd.get_random_float()*180.0f);
                if(rnd.get_random_float() > 0.1f)
                    _tmpmat = cutoutRect(_tmpmat,1,rnd.get_random_float(),0.2f,0.4f,rnd.get_random_float()*180.0f);                

                if(rnd.get_random_float() > 0.1f)
                    _tmpmat = cutoutRect(_tmpmat,rnd.get_random_float(),rnd.get_random_float(),0.1f,0.3f,rnd.get_random_float()*180.0f);
                if(rnd.get_random_float() > 0.1f)
                    _tmpmat = cutoutRect(_tmpmat,rnd.get_random_float(),rnd.get_random_float(),0.1f,0.3f,rnd.get_random_float()*180.0f);

                if(rnd.get_random_float() > 0.3f)
                    cv::blur(_tmpmat,_tmpmat,cv::Size(3,3));            

                if(rnd.get_random_float() > 0.1f)
                    _tmpmat *= 0.75f + 0.5f*rnd.get_random_float();

                if(rnd.get_random_float() > 0.1f)
                    _tmpmat = addNoise(_tmpmat,cvrng,10*rnd.get_random_float(),25*rnd.get_random_float());

                images.push_back(cvmat2dlibmatrix<float>(descriptor->getImageDescription(_tmpmat)));
            } else {
                _tmpmat = cv::imread(obj,CV_LOAD_IMAGE_GRAYSCALE);
                if(id >= objs.size())
                    cv::flip(_tmpmat,_tmpmat,1);
                images.push_back(cvmat2dlibmatrix<float>(descriptor->getImageDescription(_tmpmat)));
            }

            labels.push_back(id);
        }
    }    
}

const cv::String options = "{traindir  t  |      | path to directory with training data}"
                           "{validdir  v  |      | path to directory with validation data}"
                           "{outputdir o  |      | path to directory with output data}"
                           "{model      m |      | path to descriptor model/models (separate models names by ;)}"
                           "{minlrthresh  | 1E-4 | path to directory with output data}"
                           "{sessionguid  |      | session guid}"
                           "{learningrate |      | initial learning rate}"
                           "{tiwp         | 5000 | train iterations without progress}"
                           "{viwp         | 1000 | validation iterations without progress}";


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
    if(!cmdparser.has("outputdir")) {
        cout << "No output directory provided. Abort..." << std::endl;
        return 2;
    }
    if(!cmdparser.has("model")) {
        cout << "No description models provided. Abort..." << std::endl;
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

    std::vector<std::vector<string>> validobjs;
    if(cmdparser.has("validdir"))
        validobjs = load_objects_list(cmdparser.get<string>("validdir"));
    cout << "validobjs.size(): "<< validobjs.size() << endl;
    std::vector<matrix<float>> vimages;
    std::vector<unsigned long> vlabels;

    //set_dnn_prefer_smallest_algorithms();

    net_type net;
    dnn_trainer<net_type> trainer(net, sgd());
    trainer.set_learning_rate(0.1);
    trainer.be_verbose();
    trainer.set_synchronization_file(cmdparser.get<string>("outputdir") + string("/trainer_") + sessionguid + string("_sync") , std::chrono::minutes(10));
    if(cmdparser.has("learningrate"))
        trainer.set_learning_rate(cmdparser.get<double>("learningrate"));
    trainer.set_iterations_without_progress_threshold(cmdparser.get<int>("tiwp"));
    if(validobjs.size() > 0)
        trainer.set_test_iterations_without_progress_threshold(cmdparser.get<int>("viwp"));

	set_all_bn_running_stats_window_sizes(net, 512);


    dlib::pipe<std::vector<matrix<float>>> qimages(4);
    dlib::pipe<std::vector<unsigned long>> qlabels(4);
    auto data_loader = [&qimages, &qlabels, &trainobjs, &cmdparser](time_t seed)  {

        auto descriptor = cv::oirt::createDlibWhalesRecognizer(cmdparser.get<cv::String>("model"));

        dlib::rand rnd(time(nullptr)+seed);
        cv::RNG cvrng(static_cast<uint64_t>(time(nullptr) + seed));

        std::vector<matrix<float>> images;
        std::vector<unsigned long> labels;

        while(qimages.is_enabled()) {
            try {
                if(rnd.get_random_float() > 0.01)
                    load_mini_batch(descriptor, 96, 2, rnd, cvrng, trainobjs, images, labels, true,2);
                else
                    load_mini_batch(descriptor, 64, 3, rnd, cvrng, trainobjs, images, labels, true,2);
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
    auto testdata_loader = [&testqimages, &testqlabels, &validobjs, &cmdparser](time_t seed) {

        auto descriptor = cv::oirt::createDlibWhalesRecognizer(cmdparser.get<cv::String>("model"));

        dlib::rand rnd(time(nullptr)+seed);
        cv::RNG cvrng(static_cast<uint64_t>(time(nullptr) + seed));

        std::vector<matrix<float>> images;
        std::vector<unsigned long> labels;

        while(testqimages.is_enabled()) {
            try {
                load_mini_batch(descriptor, 96, 2, rnd, cvrng, validobjs, images, labels, false,2);
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
    if(validobjs.size() == 0) {
        testqimages.disable();
        testqlabels.disable();
        testdata_loader1.join();
    }

    size_t _step = 0;
    while(trainer.get_learning_rate() >= cmdparser.get<double>("minlrthresh"))  {
        _step++;
        images.clear();
        labels.clear();
        qimages.dequeue(images);
        qlabels.dequeue(labels);
        trainer.train_one_step(images, labels);
        if((validobjs.size() > 0) && ((_step % 10) == 0)) {
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

    if(validobjs.size() > 0) {
        testqimages.disable();
        testqlabels.disable();
        testdata_loader1.join();
    }

    cout << "Training has been accomplished" << endl;

    // Wait for training threads to stop
    trainer.get_net();
    net.clean();    

    if(validobjs.size() > 0) {
        // Now, let's check how well it performs on the validation data
        auto descriptor = cv::oirt::createDlibWhalesRecognizer(cmdparser.get<cv::String>("model"));
        anet_type anet = net;

        dlib::rand rnd(0);
        cv::RNG cvrng(0);

        int testsnum = 1;
        if(validobjs.size() > 80)
            testsnum = 80;
        float _valMinF1 = 1.0f;
        for(int n = 0; n < testsnum; ++n) {
            load_mini_batch(descriptor, 90, 2, rnd, cvrng, validobjs, vimages, vlabels, false);
            std::vector<matrix<float,0,1>> embedded = anet(vimages);

            // Now, check if the embedding puts images with the same labels near each other and
            // images with different labels far apart.
            int true_positive = 0, false_positive = 0, true_negative = 0, false_negative = 0;
            const float _distancethresh = anet.loss_details().get_distance_threshold();
            for (size_t i = 0; i < embedded.size(); ++i) {
                for (size_t j = i+1; j < embedded.size(); ++j)  {
                    if (vlabels[i] == vlabels[j])  {
                        // The loss_metric layer will cause images with the same label to be less
                        // than net.loss_details().get_distance_threshold() distance from each
                        // other.  So we can use that distance value as our testing threshold.
                        if (length(embedded[i] - embedded[j]) < _distancethresh) {
                            ++true_positive;
                        } else {
                            ++false_negative;
                        }
                    } else {
                        if (length(embedded[i]-embedded[j]) >= _distancethresh) {
                            ++true_negative;
                        } else {
                            ++false_positive;
                        }
                    }
                }
            }
            const float _precision = static_cast<float>(true_positive) / (true_positive + false_positive);
            const float _recall = static_cast<float>(true_positive) / (true_positive + false_negative);
            const float _F1 = 2.0f/(1.0f/_precision + 1.0f/_recall);
            cout << "Test iteration # " << n << endl;
            cout << "-----------------------" << endl;
            cout << "true_positive: "<< true_positive << endl;
            cout << "true_negative: "<< true_negative << endl;
            cout << "false_positive: "<< false_positive << endl;
            cout << "false_negative: "<< false_negative << endl;
            cout << "F1 score: " << _F1 << endl << endl;
            if(_F1 < _valMinF1)
                _valMinF1 = _F1;
        }

        string _outputfilename = string("whales_") + sessionguid + string("_VF") + std::to_string(_valMinF1) + string(".dat");
        cout << "Wait untill weights will be serialized to " << _outputfilename << endl;
        serialize(cmdparser.get<string>("outputdir") + string("/") + _outputfilename) << net;
        cout << "Done" << endl;
    } else {
        string _outputfilename = string("whales_") /*+ sessionguid + string("_mbiter_") +  std::to_string(trainer.get_train_one_step_calls())*/ + string(".dat");
        cout << "Wait untill weights will be serialized to " << _outputfilename << endl;
        serialize(cmdparser.get<string>("outputdir") + string("/") + _outputfilename) << net;
        cout << "Done" << endl;
    }
    return 0;
}


