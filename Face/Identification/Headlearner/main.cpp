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


dlib::matrix<dlib::rgb_pixel> makeaugmentation(cv::Mat &_tmpmat, dlib::rand& rnd, cv::RNG & cvrng)
{
    if(rnd.get_random_float() > 0.5f)
        cv::flip(_tmpmat,_tmpmat,1);

    if(rnd.get_random_float() > 0.1f)
        _tmpmat = jitterimage(_tmpmat,cvrng,cv::Size(0,0),0.03,0.03,3,cv::BORDER_REFLECT101);
    if(rnd.get_random_float() > 0.1f)
        _tmpmat = distortimage(_tmpmat,cvrng,0.03,cv::INTER_CUBIC,cv::BORDER_WRAP);

    if(rnd.get_random_float() > 0.1f)
        _tmpmat = cutoutRect(_tmpmat,rnd.get_random_float(),rnd.get_random_float(),0.3f,0.3f,rnd.get_random_float()*180.0f);
    if(rnd.get_random_float() > 0.1f)
        _tmpmat = cutoutRect(_tmpmat,rnd.get_random_float(),rnd.get_random_float(),0.3f,0.3f,rnd.get_random_float()*180.0f);

    if(rnd.get_random_float() > 0.5f)
        _tmpmat = cutoutRect(_tmpmat,rnd.get_random_float(),0,0.4f,0.4f,rnd.get_random_float()*180.0f);
    if(rnd.get_random_float() > 0.5f)
        _tmpmat = cutoutRect(_tmpmat,rnd.get_random_float(),1,0.4f,0.4f,rnd.get_random_float()*180.0f);
    if(rnd.get_random_float() > 0.5f)
        _tmpmat = cutoutRect(_tmpmat,0,rnd.get_random_float(),0.4f,0.4f,rnd.get_random_float()*180.0f);
    if(rnd.get_random_float() > 0.5f)
        _tmpmat = cutoutRect(_tmpmat,1,rnd.get_random_float(),0.4f,0.4f,rnd.get_random_float()*180.0f);

    if(rnd.get_random_float() > 0.5f)
        cv::blur(_tmpmat,_tmpmat,cv::Size(3,3));

    if(rnd.get_random_float() > 0.1f)
        _tmpmat *= 0.6 + 0.8*rnd.get_random_double();

    if(rnd.get_random_float() > 0.1f)
        _tmpmat = addNoise(_tmpmat,cvrng,0,11);

    if(rnd.get_random_float() > 0.5f) {
        cv::cvtColor(_tmpmat,_tmpmat,cv::COLOR_BGR2GRAY);
        cv::Mat _chmat[] = {_tmpmat, _tmpmat, _tmpmat};
        cv::merge(_chmat,3,_tmpmat);
    }

    /*std::vector<unsigned char> _bytes;
    std::vector<int> compression_params;
    compression_params.push_back(cv::IMWRITE_JPEG_QUALITY);
    compression_params.push_back(rnd.get_integer_in_range(50,100));
    cv::imencode("*.jpg",_tmpmat,_bytes,compression_params);
    _tmpmat = cv::imdecode(_bytes,cv::IMREAD_UNCHANGED);*/

    dlib::matrix<dlib::rgb_pixel> _dlibmatrix = cvmat2dlibmatrix<dlib::rgb_pixel>(_tmpmat);
    dlib::disturb_colors(_dlibmatrix,rnd);
    return _dlibmatrix;
}


dlib::matrix<float> make_description(const dlib::matrix<dlib::rgb_pixel> &firstimage,
                                       const dlib::matrix<dlib::rgb_pixel> &secondimage,
                                       std::vector<dlib::dscrnet_type> &inets)
{
    dlib::matrix<float,2*3*128,1> dscr;
    for(size_t i = 0; i < inets.size(); ++i) {
        dlib::matrix<float,0,1> _firstdscr = inets[i](firstimage);
        std::memcpy(dscr.begin() + 2*i*128,_firstdscr.begin(),dlib::num_rows(_firstdscr)*sizeof(float));
        dlib::matrix<float,0,1> _seconddscr = inets[i](secondimage);
        std::memcpy(dscr.begin() + (2*i+1)*128,_seconddscr.begin(),dlib::num_rows(_seconddscr)*sizeof(float));
   }
   return dscr;
}

void load_mini_batch (
    const size_t num_persons,
    const size_t pairs_per_id,
    std::vector<dlib::dscrnet_type> &inets,
    dlib::rand& rnd,
    cv::RNG & cvrng,
    const std::vector<std::vector<string>>& objs,
    std::vector<matrix<float>>& dscrs,
    std::vector<unsigned long>& labels,
    bool _doaugmentation
)
{
    dscrs.clear();
    labels.clear();
    DLIB_CASSERT(num_persons <= objs.size(), "The dataset doesn't have that many persons in it.")

    string firstobj, secondobj;
    cv::Mat _firsttmpmat,_secondtmpmat;
    bool _isloaded;
    for (size_t i = 0; i < num_persons; ++i) {

        size_t id = rnd.get_random_32bit_number() % objs.size();
        // Person's pairs
        for(size_t j = 0; j < pairs_per_id; ++j) {

            firstobj = objs[id][rnd.get_random_32bit_number() % objs[id].size()];
            _firsttmpmat = loadIbgrmatWsize(firstobj,IMG_WIDTH,IMG_HEIGHT,false,&_isloaded);
            assert(_isloaded);
            secondobj = objs[id][rnd.get_random_32bit_number() % objs[id].size()];
            _secondtmpmat = loadIbgrmatWsize(secondobj,IMG_WIDTH,IMG_HEIGHT,false,&_isloaded);
            assert(_isloaded);           
            if(_doaugmentation) {
                dscrs.push_back(make_description(makeaugmentation(_firsttmpmat,rnd,cvrng),
                                                 makeaugmentation(_secondtmpmat,rnd,cvrng),
                                                 inets));
            } else {
                dscrs.push_back(make_description(cvmat2dlibmatrix<dlib::rgb_pixel>(_firsttmpmat),
                                                 cvmat2dlibmatrix<dlib::rgb_pixel>(_secondtmpmat),
                                                 inets));
            }
            labels.push_back(1);
        }
        // Pairs with imposter
        for(size_t j = 0; j < pairs_per_id; ++j) {

            firstobj = objs[id][rnd.get_random_32bit_number() % objs[id].size()];
            _firsttmpmat = loadIbgrmatWsize(firstobj,IMG_WIDTH,IMG_HEIGHT,false,&_isloaded);
            assert(_isloaded);
            size_t imposterid = rnd.get_random_32bit_number() % objs.size();
            while(imposterid == id)
                imposterid = rnd.get_random_32bit_number() % objs.size();
            secondobj = objs[imposterid][rnd.get_random_32bit_number() % objs[imposterid].size()];
            _secondtmpmat = loadIbgrmatWsize(secondobj,IMG_WIDTH,IMG_HEIGHT,false,&_isloaded);
            assert(_isloaded);
            if(_doaugmentation) {
                dscrs.push_back(make_description(makeaugmentation(_firsttmpmat,rnd,cvrng),
                                                 makeaugmentation(_secondtmpmat,rnd,cvrng),
                                                 inets));
            } else {
                dscrs.push_back(make_description(cvmat2dlibmatrix<dlib::rgb_pixel>(_firsttmpmat),
                                                 cvmat2dlibmatrix<dlib::rgb_pixel>(_secondtmpmat),
                                                 inets));
            }
            labels.push_back(0);
        }
    }    
}

const cv::String options = "{traindir  t  |       | path to directory with training data}"
                           "{validdir  v  |       | path to directory with validation data}"
                           "{outputdir o  |       | path to directory with output data}"
                           "{classes   c  |  32   | number of unique persons in minibatch}"
                           "{samples   s  |  2    | number of pairs of negative+positive pairs per class in minibatch}"
                           "{trainaugm    | true  | augmentation for train data}"
                           "{validaugm    | false | augmentation for validation data}"
                           "{minlrthresh  | 1E-5  | path to directory with output data}"
                           "{sessionguid  |       | session guid}"
                           "{learningrate |       | initial learning rate}"
                           "{tiwp         | 10000 | train iterations without progress}"
                           "{viwp         | 256   | validation iterations without progress}"
                           "{bnwsize      | 512   | batch normalization window size}"
                           "{dscrpath     |   .   | path where descriptors are stored}"
                           "{delayms      | 0     | delay of visualization}";


int main(int argc, char** argv)
{
    cv::CommandLineParser cmdparser(argc,argv,options);
    cmdparser.about("This app was designed to train face identification head net with log loss");
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
    if(cmdparser.get<int>("classes") < 2) {
        cout << "Insufficient number of classes in minibatch selected. Abort..." << std::endl;
        return 3;
    }
    if(cmdparser.get<int>("samples") < 1) {
        cout << "Insufficient number of samples per class in minibatch selected. Abort..." << std::endl;
        return 4;
    }
    string sessionguid = "default";
    if(cmdparser.has("sessionguid")) {
        sessionguid = cmdparser.get<string>("sessionguid");
    }

    cout << "Reading training data, please wait..." << endl;
    auto trainobjs = load_objects_list(cmdparser.get<string>("traindir"));
    cout << "trainobjs.size(): "<< trainobjs.size() << endl;

    std::vector<std::vector<string>> validobjs;
    if(cmdparser.has("validdir")) {
        validobjs = load_objects_list(cmdparser.get<string>("validdir"));
    }
    cout << "validobjs.size(): "<< validobjs.size() << endl;

    const size_t classes = static_cast<size_t>(cmdparser.get<int>("classes"));
    const size_t samples = static_cast<size_t>(cmdparser.get<int>("samples"));
    cout << "Number of classes per minibatch: " << classes << endl;
    cout << "Number of samples per class in minibatch: " << samples << endl;
    const bool trainaugm  = cmdparser.get<bool>("trainaugm");
    const bool validaugm = cmdparser.get<bool>("validaugm");
    cout << "Train data augmentation: " << trainaugm << endl;
    cout << "Validation data augmentation: " << validaugm << endl; 

    set_dnn_prefer_smallest_algorithms();

    net_type net;
    dnn_trainer<net_type> trainer(net, sgd(0.0001f,0.9f));
    trainer.set_learning_rate(0.1);
    trainer.be_verbose();
    trainer.set_synchronization_file(cmdparser.get<string>("outputdir") + string("/trainer_") + sessionguid + string("_sync") , std::chrono::minutes(10));
    if(cmdparser.has("learningrate"))
        trainer.set_learning_rate(cmdparser.get<double>("learningrate"));
    trainer.set_iterations_without_progress_threshold(static_cast<unsigned long>(cmdparser.get<int>("tiwp")));
    if(validobjs.size() > 0)
        trainer.set_test_iterations_without_progress_threshold(static_cast<unsigned long>(cmdparser.get<int>("viwp")));

    set_all_bn_running_stats_window_sizes(net, static_cast<unsigned long>(cmdparser.get<int>("bnwsize")));

    dlib::pipe<std::vector<matrix<float>>> train_dscr_pipe(6);
    dlib::pipe<std::vector<unsigned long>> train_lbl_pipe(6);
    auto data_loader = [&train_dscr_pipe, & train_lbl_pipe, &trainobjs, classes, samples, trainaugm, &cmdparser](time_t seed)  {

        dlib::rand rnd(time(nullptr)+seed);
        cv::RNG cvrng(static_cast<uint64_t>(time(nullptr) + seed));

        std::vector<dlib::dscrnet_type> inets(3);
        cout << "Reading face descriptor models, please wait..." << endl;
        for(size_t i = 0; i < inets.size(); ++i) {
            try {
                dlib::deserialize(cmdparser.get<std::string>("dscrpath") + "/iface_net_" + std::to_string(i) + ".dat") >> inets[i];
            } catch (const std::exception& e) {
                cout << e.what() << endl;
            }
        }

        std::vector<matrix<float>> dscrs;
        std::vector<unsigned long> labels;

        while(train_dscr_pipe.is_enabled()) {
            try {               
                load_mini_batch(classes, samples, inets, rnd, cvrng, trainobjs, dscrs, labels, trainaugm);
                train_dscr_pipe.enqueue(dscrs);
                train_lbl_pipe.enqueue(labels);
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
    std::thread data_loader4([data_loader](){ data_loader(4); });
    std::thread data_loader5([data_loader](){ data_loader(5); });

    // Same for the test
    dlib::pipe<std::vector<matrix<float>>> valid_dscr_pipe(2);
    dlib::pipe<std::vector<unsigned long>> valid_lbl_pipe(2);
    auto testdata_loader = [&valid_dscr_pipe, &valid_lbl_pipe, &validobjs, classes, samples, validaugm, &cmdparser](time_t seed) {

        dlib::rand rnd(time(nullptr)+seed);
        cv::RNG cvrng(static_cast<uint64_t>(time(nullptr) + seed));

        std::vector<dlib::dscrnet_type> inets(3);
        cout << "Reading face descriptor models, please wait..." << endl;
        for(size_t i = 0; i < inets.size(); ++i) {
            try {
                dlib::deserialize(cmdparser.get<std::string>("dscrpath") + "/iface_net_" + std::to_string(i) + ".dat") >> inets[i];
            } catch (const std::exception& e) {
                cout << e.what() << endl;
            }
        }

        std::vector<matrix<float>> dscrs;
        std::vector<unsigned long> labels;

        while(valid_dscr_pipe.is_enabled()) {
            try {
                load_mini_batch(classes, samples, inets, rnd, cvrng, validobjs, dscrs, labels, validaugm);
                valid_dscr_pipe.enqueue(dscrs);
                valid_lbl_pipe.enqueue(labels);
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
        valid_dscr_pipe.disable();
        valid_lbl_pipe.disable();
        testdata_loader1.join();
    }

    std::vector<dlib::matrix<float>> descriptions;
    std::vector<unsigned long> labels;
    size_t _step = 0;
    while(trainer.get_learning_rate() >= cmdparser.get<double>("minlrthresh"))  {
        _step++;
        descriptions.clear();
        labels.clear();
        train_dscr_pipe.dequeue(descriptions);
        train_lbl_pipe.dequeue(labels);
        trainer.train_one_step(descriptions,labels);
        if((validobjs.size() > 0) && ((_step % 10) == 0)) {
            descriptions.clear();
            labels.clear();
            valid_dscr_pipe.dequeue(descriptions);
            valid_lbl_pipe.dequeue(labels);
            trainer.test_one_step(descriptions,labels);
            cout << "step " << _step << " train loss :" << trainer.get_average_loss()
                 << ", validation loss :" << trainer.get_average_test_loss() << endl;
        }
    }

    // stop all the data loading threads and wait for them to terminate.
    train_dscr_pipe.disable();
    train_lbl_pipe.disable();
    data_loader1.join();
    data_loader2.join();
    data_loader3.join();
    data_loader4.join();
    data_loader5.join();

    if(validobjs.size() > 0) {
        valid_dscr_pipe.disable();
        valid_lbl_pipe.disable();
        testdata_loader1.join();
    }

    cout << "Training has been accomplished" << endl;

    // Wait for training threads to stop
    trainer.get_net();
    net.clean();    

    if(validobjs.size() > 0) {
        // Now, let's check how well it performs on the validation data
        anet_type anet = net;

        dlib::rand rnd(1);
        cv::RNG cvrng(1);

        std::vector<dlib::dscrnet_type> inets(3);
        cout << "Reading face descriptor models, please wait..." << endl;
        for(size_t i = 0; i < inets.size(); ++i) {
            try {
                dlib::deserialize(cmdparser.get<std::string>("dscrpath") + "/iface_net_" + std::to_string(i) + ".dat") >> inets[i];
            } catch (const std::exception& e) {
                cout << e.what() << endl;
            }
        }

        int testsnum = 50;
        float _valMinF1 = 1.0f;
        for(int n = 0; n < testsnum; ++n) {
            load_mini_batch(110, 10, inets, rnd, cvrng, validobjs, descriptions, labels, false);
            std::vector<unsigned long> predictions = anet(descriptions);

            // Now, check if the predictions for person pairs have label 1 and pair with imposter have label 0
            int true_positive = 0, false_positive = 0, true_negative = 0, false_negative = 0;
            for (size_t i = 0; i < predictions.size(); ++i) {
                switch(labels[i]) {
                    case 0:
                        switch(predictions[i]) {
                            case 0:
                                true_negative++;
                                break;
                            case 1:
                                false_positive++;
                                break;
                        }
                        break;
                    case 1:
                        switch(predictions[i]) {
                            case 0:
                                false_negative++;
                                break;
                            case 1:
                                true_positive++;
                                break;
                        }
                        break;
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

        string _outputfilename = string("net_") + sessionguid + string("_MVF") + std::to_string(_valMinF1) + string(".dat");
        cout << "Wait untill weights will be serialized to " << _outputfilename << endl;
        serialize(cmdparser.get<string>("outputdir") + string("/") + _outputfilename) << net;
        cout << "Done" << endl;
    } else {
        string _outputfilename = string("net_") + sessionguid /*+ string("_mbiter_") +  std::to_string(trainer.get_train_one_step_calls())*/ + string(".dat");
        cout << "Wait untill weights will be serialized to " << _outputfilename << endl;
        serialize(cmdparser.get<string>("outputdir") + string("/") + _outputfilename) << net;
        cout << "Done" << endl;
    }
    return 0;
}


