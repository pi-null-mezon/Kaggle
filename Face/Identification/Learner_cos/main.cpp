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

    /*if(rnd.get_random_float() > 0.1f)
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
        cv::blur(_tmpmat,_tmpmat,cv::Size(3,3));*/

    if(rnd.get_random_float() > 0.5f)
        _tmpmat *= 0.8 + 0.4*rnd.get_random_double();

    if(rnd.get_random_float() > 0.5f)
        _tmpmat = addNoise(_tmpmat,cvrng,0,1 + (int)(10*rnd.get_random_float()));

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

void load_mini_batch (
    const size_t num_persons,
    const size_t samples_per_id,
    dlib::rand& rnd,
    cv::RNG & cvrng,
    const std::vector<std::vector<string>>& objs,
    std::vector<matrix<rgb_pixel>>& images,
    std::vector<unsigned long>& labels,
    bool _doaugmentation,
    const size_t min_samples=1 // if dir contains number of samples less than min_samples then dir will not be used
)
{
    images.clear();
    labels.clear();
    DLIB_CASSERT(num_persons <= objs.size(), "The dataset doesn't have that many persons in it.")

    string obj;
    cv::Mat _tmpmat;
    bool _isloaded;
    std::vector<bool> already_selected(objs.size(), false); // as we can effectivelly enlarge training set by horizontal flip operation
    for (size_t i = 0; i < num_persons; ++i) {

        size_t id = rnd.get_random_32bit_number() % objs.size();
        while(already_selected[id] || (objs[id % objs.size()].size() < min_samples))
            id = rnd.get_random_32bit_number() % objs.size();
        already_selected[id] = true;

        for(size_t j = 0; j < samples_per_id; ++j) {

            if(objs[id % objs.size()].size() == samples_per_id)
                obj = objs[id % objs.size()][j];
            else
                obj = objs[id % objs.size()][rnd.get_random_32bit_number() % objs[id % objs.size()].size()];

            _tmpmat = loadIbgrmatWsize(obj,IMG_WIDTH,IMG_HEIGHT,false,&_isloaded);
            assert(_isloaded);
            if(_doaugmentation) {
                images.push_back(makeaugmentation(_tmpmat,rnd,cvrng));
            } else {
                images.push_back(cvmat2dlibmatrix<dlib::rgb_pixel>(_tmpmat));
            }
            labels.push_back(id);
        }
    }    
}

const cv::String options = "{traindir  t  |       | path to directory with training data}"
                           "{validdir  v  |       | path to directory with validation data}"
                           "{outputdir o  |       | path to directory with output data}"
                           "{classes   c  |  220  | number of unique persons in minibatch}"
                           "{samples   s  |  5    | number of samples per class in minibatch}"
                           "{trainaugm    | true  | augmentation for train data}"
                           "{validaugm    | false | augmentation for validation data}"
                           "{model     m  |       | path to a model (to make hard mining from training set before training)}"
                           "{minlrthresh  | 1E-5  | path to directory with output data}"
                           "{sessionguid  |       | session guid}"
                           "{learningrate |       | initial learning rate}"
                           "{tiwp         | 10000 | train iterations without progress}"
                           "{viwp         | 1000  | validation iterations without progress}"
                           "{bnwsize      | 1000  | batch normalization window size}"
                           "{delayms      | 0     | delay of visualization}";


int main(int argc, char** argv)
{
    cv::CommandLineParser cmdparser(argc,argv,options);
    cmdparser.about("This app was designed to train face identification net with cosine metric loss");
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
    std::vector<matrix<rgb_pixel>> images;
    std::vector<unsigned long> labels;

    // If user have provided model, we should make hard mining for this model
    std::vector<std::vector<string>> hardtrainobjs;
    std::vector<bool>                alreadyselected;
    if(cmdparser.has("model")) {
        hardtrainobjs.reserve(trainobjs.size());
        alreadyselected = std::vector<bool>(trainobjs.size(),false);
        anet_type _anet;
        try {
            dlib::deserialize(cmdparser.get<string>("model")) >> _anet;
        }
        catch(std::exception &_e) {
            cout << _e.what() << endl;
            return 4;
        }
        cout << endl << "Loss function distance thresh: " << _anet.loss_details().get_distance_threshold()
             << endl << "Loss function distance margin: " << _anet.loss_details().get_margin() << endl << endl;
        std::vector<matrix<float,0,1>> _valldescriptions;
        std::vector<std::string> _vfilename;
        std::vector<size_t> _valllabels;
        size_t _totalimages = 0;
        for(size_t i = 0; i < trainobjs.size(); ++i)
            _totalimages += trainobjs[i].size();
        _valldescriptions.reserve(_totalimages);
        _vfilename.reserve(_totalimages);
        _valllabels.reserve(_totalimages);
        bool _isloaded = false;
        cout << "Please wait while descriptions will be computed" << endl;
        for(size_t i = 0; i < trainobjs.size(); ++i) {
            cout << "label " << i << " (" << trainobjs[i].size() << " images)";
            std::vector<dlib::matrix<rgb_pixel>> _vdlibimages;
            _vdlibimages.reserve(trainobjs[i].size());
            for(size_t j = 0; j < trainobjs[i].size(); ++j) {
                _vfilename.push_back(trainobjs[i][j]);
                _vdlibimages.push_back(cvmat2dlibmatrix<dlib::rgb_pixel>(loadIbgrmatWsize(trainobjs[i][j],IMG_WIDTH,IMG_HEIGHT,false,&_isloaded)));
                assert(_isloaded);
            }
            std::vector<matrix<float,0,1>> _vdscrmatrices = _anet(_vdlibimages);
            for(size_t j = 0; j < _vdscrmatrices.size(); ++j) {
                _valldescriptions.push_back(std::move(_vdscrmatrices[j]));
                _valllabels.push_back(i);
            }
            cout << " - descriptions collected" << endl;
        }
        cout << "Please wait while hard mining will be performed" << endl;
        const float _distancethresh = _anet.loss_details().get_distance_threshold();
        size_t tp = 0, fp = 0, tn = 0, fn = 0;
        int _delayms = cmdparser.get<int>("delayms");
        for(size_t i = 0; i < _valldescriptions.size(); ++i) {
            for(size_t j = i+1; j < _valldescriptions.size(); ++j) {
                if(_valllabels[i] == _valllabels[j]) {
                    if(cosinedistance(_valldescriptions[i],_valldescriptions[j]) < _distancethresh) {
                        tp++;
                    } else {
                        std::cout << " Reference " << _valllabels[i]
                                  << " vs Test " << _valllabels[j]
                                  << " - dst: " << cosinedistance(_valldescriptions[i],_valldescriptions[j]) << std::endl;
                        cv::imshow("Ref.",cv::imread(_vfilename[i],cv::IMREAD_UNCHANGED));
                        cv::imshow("Test",cv::imread(_vfilename[j],cv::IMREAD_UNCHANGED));
                        cv::waitKey(_delayms);
                        if(alreadyselected[_valllabels[i]] == false) {
                            hardtrainobjs.push_back(trainobjs[_valllabels[i]]);
                            alreadyselected[_valllabels[i]] = true;
                        }
                        if(alreadyselected[_valllabels[j]] == false) {
                            hardtrainobjs.push_back(trainobjs[_valllabels[j]]);
                            alreadyselected[_valllabels[j]] = true;
                        }
                        fn++;
                    }
                } else {
                    if(cosinedistance(_valldescriptions[i],_valldescriptions[j]) >= _distancethresh) {
                        tn++;
                    } else {
                        std::cout << " Reference " << _valllabels[i]
                                  << " vs Test " << _valllabels[j]
                                  << " - dst: " << cosinedistance(_valldescriptions[i],_valldescriptions[j]) << std::endl;
                        cv::imshow("Ref.",cv::imread(_vfilename[i],cv::IMREAD_UNCHANGED));
                        cv::imshow("Test",cv::imread(_vfilename[j],cv::IMREAD_UNCHANGED));
                        cv::waitKey(_delayms);
                        if(alreadyselected[_valllabels[i]] == false) {
                            hardtrainobjs.push_back(trainobjs[_valllabels[i]]);
                            alreadyselected[_valllabels[i]] = true;
                        }
                        if(alreadyselected[_valllabels[j]] == false) {
                            hardtrainobjs.push_back(trainobjs[_valllabels[j]]);
                            alreadyselected[_valllabels[j]] = true;
                        }
                        fp++;
                    }
                }
            }
        }
        cout << "Fasle positives found: " << fp << endl;
        cout << "True  positives found: " << tp << endl;
        cout << "False negatives found: " << fn << endl;
        cout << "True  negatives found: " << tn << endl;
        trainobjs = std::move(hardtrainobjs);
        cout << "hard trainobjs.size(): "<< trainobjs.size() << endl;
    }
    // end of hard mining

    std::vector<std::vector<string>> validobjs;
    if(cmdparser.has("validdir")) {
        validobjs = load_objects_list(cmdparser.get<string>("validdir"));
    }
    cout << "validobjs.size(): "<< validobjs.size() << endl;
    std::vector<matrix<rgb_pixel>> vimages;
    std::vector<unsigned long> vlabels;

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
    dnn_trainer<net_type> trainer(net, sgd(0.0001f,0.9f)/*,{0,1}*/);
    trainer.set_learning_rate(0.1);
    trainer.be_verbose();
    trainer.set_synchronization_file(cmdparser.get<string>("outputdir") + string("/trainer_") + sessionguid + string("_sync") , std::chrono::minutes(10));
    if(cmdparser.has("learningrate"))
        trainer.set_learning_rate(cmdparser.get<double>("learningrate"));
    trainer.set_iterations_without_progress_threshold(static_cast<unsigned long>(cmdparser.get<int>("tiwp")));
    if(validobjs.size() > 0)
        trainer.set_test_iterations_without_progress_threshold(static_cast<unsigned long>(cmdparser.get<int>("viwp")));

    set_all_bn_running_stats_window_sizes(net, static_cast<unsigned long>(cmdparser.get<int>("bnwsize")));

    cout << endl << "Loss function distance thresh: " << net.loss_details().get_distance_threshold()
         << endl << "Loss function distance margin: " << net.loss_details().get_margin() << endl << endl;

    dlib::pipe<std::vector<matrix<rgb_pixel>>> qimages(6);
    dlib::pipe<std::vector<unsigned long>> qlabels(6);
    auto data_loader = [&qimages, &qlabels, &trainobjs, classes, samples, trainaugm](time_t seed)  {

        dlib::rand rnd(time(nullptr)+seed);
        cv::RNG cvrng(static_cast<uint64_t>(time(nullptr) + seed));

        std::vector<matrix<rgb_pixel>> images;
        std::vector<unsigned long> labels;

        while(qimages.is_enabled()) {
            try {               
                load_mini_batch(classes, samples, rnd, cvrng, trainobjs, images, labels, trainaugm);
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
    std::thread data_loader4([data_loader](){ data_loader(4); });
    std::thread data_loader5([data_loader](){ data_loader(5); });

    // Same for the test
    dlib::pipe<std::vector<matrix<rgb_pixel>>> testqimages(2);
    dlib::pipe<std::vector<unsigned long>> testqlabels(2);
    auto testdata_loader = [&testqimages, &testqlabels, &validobjs, classes, samples, validaugm](time_t seed) {

        dlib::rand rnd(time(nullptr)+seed);
        cv::RNG cvrng(static_cast<uint64_t>(time(nullptr) + seed));

        std::vector<matrix<rgb_pixel>> images;
        std::vector<unsigned long> labels;

        while(testqimages.is_enabled()) {
            try {
                load_mini_batch(classes, samples, rnd, cvrng, validobjs, images, labels, validaugm);
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

        /*if(trainer.get_train_one_step_calls() % 4 == 0)
            cout << trainer.get_train_one_step_calls() << ") train loss: " << trainer.get_average_loss()
                 << " test loss: " << trainer.get_average_test_loss() << std::endl;*/
    }

    // stop all the data loading threads and wait for them to terminate.
    qimages.disable();
    qlabels.disable();
    data_loader1.join();
    data_loader2.join();
    data_loader3.join();
    data_loader4.join();
    data_loader5.join();

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
        cout << endl << "Validation started, please wait...." << endl;
        // Now, let's check how well it performs on the validation data
        anet_type anet = net;

        dlib::rand rnd(0);
        cv::RNG cvrng(0);
        int testsnum = 5;
        float _valMinF1 = 0.0f;
        for(int n = 0; n < testsnum; ++n) {
            load_mini_batch(110, 10, rnd, cvrng, validobjs, vimages, vlabels, false);
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
                        if ( cosinedistance(embedded[i],embedded[j]) < _distancethresh) {
                            ++true_positive;
                        } else {
                            ++false_negative;
                        }
                    } else {
                        if ( cosinedistance(embedded[i],embedded[j]) >= _distancethresh) {
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
            _valMinF1 += _F1;
        }
        cout << "Average F1 score: " << _valMinF1 / testsnum << endl << endl;

        cout << "Final test step (on entire validation set)" << endl
             << "-----------------------" << endl
             << "Preparing templates, please wait..." << endl;
        std::vector<matrix<float,0,1>> embedded;
        vlabels.clear();
        for(size_t i = 0; i < validobjs.size(); ++i) {
            //cout << "  label: " << i << endl;
            vimages.clear();
            bool _isloaded = false;
            for(size_t j = 0; j < validobjs[i].size(); ++j) {
                cv::Mat _tmpmat = loadIbgrmatWsize(validobjs[i][j],IMG_WIDTH,IMG_HEIGHT,false,&_isloaded);
                assert(_isloaded);
                vimages.push_back(cvmat2dlibmatrix<dlib::rgb_pixel>(_tmpmat));
                vlabels.push_back(i);
            }
            std::vector<matrix<float,0,1>> _tmpembedded = anet(vimages);
            embedded.insert(embedded.end(),_tmpembedded.begin(),_tmpembedded.end());
        }

        cout << "Matching templates, please wait..."  << endl;
        // Now, check if the embedding puts images with the same labels near each other and
        // images with different labels far apart.
        unsigned long true_positive = 0, false_positive = 0, true_negative = 0, false_negative = 0;
        const float _distancethresh = anet.loss_details().get_distance_threshold();
        for (size_t i = 0; i < embedded.size(); ++i) {
            for (size_t j = i+1; j < embedded.size(); ++j)  {
                if (vlabels[i] == vlabels[j])  {
                    // The loss_metric layer will cause images with the same label to be less
                    // than net.loss_details().get_distance_threshold() distance from each
                    // other.  So we can use that distance value as our testing threshold.
                    if (cosinedistance(embedded[i],embedded[j]) < _distancethresh)
                        ++true_positive;
                    else
                        ++false_negative;
                } else {
                    if (cosinedistance(embedded[i],embedded[j]) >= _distancethresh)
                        ++true_negative;
                    else
                        ++false_positive;
                }
            }
        }
        const float _precision = static_cast<float>(true_positive) / (true_positive + false_positive);
        const float _recall = static_cast<float>(true_positive) / (true_positive + false_negative);
        _valMinF1 = 2.0f/(1.0f/_precision + 1.0f/_recall);
        cout << "Final test on entire validation set" << endl;
        cout << "-----------------------" << endl;
        cout << "true_positive: "<< true_positive << endl;
        cout << "true_negative: "<< true_negative << endl;
        cout << "false_positive: "<< false_positive << endl;
        cout << "false_negative: "<< false_negative << endl;
        cout << "F1 score: " << _valMinF1 << endl << endl;

        string _outputfilename = string("net_cos_") + sessionguid + string("_MVF") + std::to_string(_valMinF1) + string(".dat");
        cout << "Wait untill weights will be serialized to " << _outputfilename << endl;
        serialize(cmdparser.get<string>("outputdir") + string("/") + _outputfilename) << net;
        cout << "Done" << endl;
    } else {
        string _outputfilename = string("net_cos_") + sessionguid /*+ string("_mbiter_") +  std::to_string(trainer.get_train_one_step_calls())*/ + string(".dat");
        cout << "Wait untill weights will be serialized to " << _outputfilename << endl;
        serialize(cmdparser.get<string>("outputdir") + string("/") + _outputfilename) << net;
        cout << "Done" << endl;
    }
    return 0;
}


