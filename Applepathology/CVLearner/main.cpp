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

std::vector<std::vector<string>> load_classes_list (const string& dir)
{
    std::vector<std::vector<string>> objects;
    size_t _label = 0;
    for(auto subdir : directory(dir).get_dirs()) {
        std::vector<string> imgs;
        for(auto img : subdir.get_files())
            imgs.push_back(img);
        if(imgs.size() != 0) {
            std::cout << "Label " << _label << " - '" << subdir.name() << "'" << std::endl;
            objects.push_back(imgs);
            _label++;
        }
    }
    return objects;
}

std::vector<std::vector<std::vector<string>>> split_into_folds(const std::vector<std::vector<string>> &_objs, unsigned int _folds, dlib::rand& _rnd)
{
    std::vector<std::vector<std::vector<string>>> _output(_folds);
    for(size_t i = 0; i < _output.size(); ++i)
        _output[i] = std::vector<std::vector<string>>(_objs.size());

    for(size_t i = 0; i < _objs.size(); ++i)
        for(size_t j = 0; j < _objs[i].size(); ++j)
            _output[static_cast<size_t>(_rnd.get_integer(_folds))][i].push_back(_objs[i][j]);

    return _output;
}

std::vector<std::vector<string>> merge_except(const std::vector<std::vector<std::vector<string>>> &_objs, size_t _index)
{
    std::vector<std::vector<string>> _mergedobjs(_objs[0].size());
    for(size_t i = 0; i < _objs.size(); ++i)
        if(i != _index)
            for(size_t j = 0; j < _mergedobjs.size(); ++j)
                _mergedobjs[j].insert(_mergedobjs[j].end(),_objs[i][j].begin(),_objs[i][j].end());

    return _mergedobjs;
}


void load_mini_batch (
    const size_t num_classes,
    const size_t samples_per_class,
    dlib::rand& rnd,
    cv::RNG & cvrng,
    const std::vector<std::vector<string>>& objs,
    std::vector<matrix<dlib::rgb_pixel>>& images,
    std::vector<unsigned long>& labels,
    bool _doaugmentation
)
{
    images.clear();
    labels.clear();
    DLIB_CASSERT(num_classes <= objs.size(), "The dataset doesn't have that many classes!");

    string obj;
    cv::Mat _tmpmat;
    bool _isloaded;

    std::vector<bool> already_selected(objs.size(), false);

    for(size_t i = 0; i < num_classes; ++i) {

        size_t id = rnd.get_random_32bit_number() % objs.size();
        while(already_selected[id])
            id = rnd.get_random_32bit_number() % objs.size();
        already_selected[id] = true;

        for(size_t j = 0; j < samples_per_class; ++j) {

            if(objs[id].size() == samples_per_class)
                obj = objs[id][j];
            else
                obj = objs[id][rnd.get_random_32bit_number() % objs[id].size()];                        

            if(_doaugmentation) {
                _tmpmat = loadIbgrmatWsize(obj,IMG_WIDTH,IMG_HEIGHT,false,&_isloaded);
                if(!_isloaded)
                    std::cout << "Can not load: " << obj << std::endl;
                assert(_isloaded);

                if(rnd.get_random_float() > 0.5f)
                    cv::rotate(_tmpmat,_tmpmat,1);
                if(rnd.get_random_float() > 0.5f)
                    cv::rotate(_tmpmat,_tmpmat,0);
                if(rnd.get_random_float() > 0.5f)
                    cv::flip(_tmpmat,_tmpmat,1);
                if(rnd.get_random_float() > 0.5f)
                    cv::flip(_tmpmat,_tmpmat,0);

                if(rnd.get_random_float() > 0.1f)
                    _tmpmat = jitterimage(_tmpmat,cvrng,cv::Size(0,0),0.15,0.15,45,cv::BORDER_REFLECT,cv::Scalar(0),false);
                if(rnd.get_random_float() > 0.1f)
                    _tmpmat = distortimage(_tmpmat,cvrng,0.2,cv::INTER_CUBIC,cv::BORDER_WRAP,cv::Scalar(0));

                if(rnd.get_random_float() > 0.1f)
                    _tmpmat = cutoutRect(_tmpmat,rnd.get_random_float(),rnd.get_random_float(),0.5f,0.5f,rnd.get_random_float()*180.0f);
                /*if(rnd.get_random_float() > 0.1f)
                    _tmpmat = cutoutRect(_tmpmat,rnd.get_random_float(),rnd.get_random_float(),0.1f,0.3f,rnd.get_random_float()*180.0f);*/

                if(rnd.get_random_float() > 0.1f)
                    _tmpmat = cutoutRect(_tmpmat,rnd.get_random_float(),0,0.3f,0.3f,rnd.get_random_float()*180.0f);
                if(rnd.get_random_float() > 0.1f)
                    _tmpmat = cutoutRect(_tmpmat,rnd.get_random_float(),1,0.3f,0.3f,rnd.get_random_float()*180.0f);
                if(rnd.get_random_float() > 0.1f)
                    _tmpmat = cutoutRect(_tmpmat,0,rnd.get_random_float(),0.3f,0.3f,rnd.get_random_float()*180.0f);
                if(rnd.get_random_float() > 0.1f)
                    _tmpmat = cutoutRect(_tmpmat,1,rnd.get_random_float(),0.3f,0.3f,rnd.get_random_float()*180.0f);

                if(rnd.get_random_float() > 0.5f)
                    cv::blur(_tmpmat,_tmpmat,cv::Size(5,5));

                if(rnd.get_random_float() > 0.1f)
                    _tmpmat *= (0.6 + 0.8*rnd.get_random_double());

                if(rnd.get_random_float() > 0.1f)
                    _tmpmat = addNoise(_tmpmat,cvrng,0,15);

                /*if(rnd.get_random_float() > 0.5f) {
                    cv::cvtColor(_tmpmat,_tmpmat,cv::COLOR_BGR2GRAY);
                    cv::Mat _chmat[] = {_tmpmat, _tmpmat, _tmpmat};
                    cv::merge(_chmat,3,_tmpmat);
                }*/

                /*std::vector<unsigned char> _bytes;
                std::vector<int> compression_params;
                compression_params.push_back(cv::IMWRITE_JPEG_QUALITY);
                compression_params.push_back(static_cast<int>(rnd.get_integer_in_range(55,100)));
                cv::imencode("*.jpg",_tmpmat,_bytes,compression_params);
                _tmpmat = cv::imdecode(_bytes,cv::IMREAD_UNCHANGED);*/

                dlib::matrix<dlib::rgb_pixel> _dlibtmpimg = cvmat2dlibmatrix<dlib::rgb_pixel>(_tmpmat);
                dlib::disturb_colors(_dlibtmpimg,rnd);
                //cv::imshow(string("Augmented ") + to_string(std::hash<std::thread::id>{}(std::this_thread::get_id())),_tmpmat);
                //cv::waitKey(100);
                images.push_back(_dlibtmpimg);
            } else {
                _tmpmat = loadIbgrmatWsize(obj,IMG_WIDTH,IMG_HEIGHT,false,&_isloaded);
                if(!_isloaded)
                    std::cout << "Can not load: " << obj << std::endl;
                assert(_isloaded);

                //cv::imshow(string("Ordinary ") + to_string(std::hash<std::thread::id>{}(std::this_thread::get_id())),_tmpmat);
                //cv::waitKey(100);
                images.push_back(cvmat2dlibmatrix<dlib::rgb_pixel>(_tmpmat));
            }

            labels.push_back(id);
        }
    }
}

float test_accuracy_on_set(const std::vector<std::vector<string>> &_testobjs, dlib::net_type &_net, bool _beverbose,
                           const size_t _classes,
                           const size_t _samples=10,
                           const size_t _iterations=10,
                           const size_t _seed=1)
{
    anet_type   anet = _net;
    dlib::rand  rnd(static_cast<long>(_seed));
    cv::RNG     cvrng(_seed);
    std::vector<float> vacc(_iterations,0.0f);
    std::vector<uint> vright(_iterations,0);
    std::vector<uint> vwrong(_iterations,0);
    std::vector<matrix<dlib::rgb_pixel>> images;
    std::vector<unsigned long> labels;
    for(size_t i = 0; i < vacc.size(); ++i) {
        load_mini_batch(_classes, _samples, rnd, cvrng, _testobjs, images, labels, false);
        std::vector<unsigned long> predictedlabels = anet(images);
        for(size_t j = 0; j < images.size(); ++j) {
            if(predictedlabels[j] == labels[j])
                (vright[i])++;
            else
                (vwrong[i])++;
        }
        vacc[i] = static_cast<float>(vright[i]) / (vright[i] + vwrong[i]);
        if(_beverbose)
            cout << "iteration #" << i << " - accuracy: " << vacc[i] << endl;
    }
    float acc = 0.0f;
    for(size_t i = 0; i < vacc.size(); ++i)
        acc += vacc[i];

    if(vacc.size() > 0)
        return acc / vacc.size();

    return 0.0f;
}

const cv::String options = "{traindir  t  |       | path to directory with training data}"
                           "{cvfolds      |   5   | folds to use for cross validation training}"
                           "{testdir      |       | path to directory with test data}"
                           "{outputdir o  |       | path to directory with output data}"
                           "{minlrthresh  | 1E-5  | path to directory with output data}"
                           "{sessionguid  |       | session guid}"
                           "{learningrate |       | initial learning rate}"
                           "{classes c    | 4     | classes per minibatch}"
                           "{samples s    | 16    | samples per class in minibatch}"
                           "{bnwsize      | 100   | will be passed in set_all_bn_running_stats_window_sizes before net training}"
                           "{tiwp         | 5000  | train iterations without progress}"
                           "{viwp         | 250   | validation iterations without progress}"
                           "{trainaugm    | true  | apply data augmentation at train data}"
                           "{validaugm    | false | apply data augmentation at validation data}"
                           "{seed         |   1   | seed number for random generators (if equal to 0 than current time will be used)}"
                           "{psalgo       | true  | set prefer smallest algorithms}";

int main(int argc, char** argv)
{
    setlocale(LC_CTYPE,"Rus");
    cv::CommandLineParser cmdparser(argc,argv,options);
    cmdparser.about("This app was designed to train dlib's format neural network with cross validation training scheme");
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
    string sessionguid = std::to_string(0);
    if(cmdparser.has("sessionguid")) {
        sessionguid = cmdparser.get<string>("sessionguid");
    }
    time_t seed = time(nullptr);
    if(cmdparser.get<unsigned int>("seed") != 0)
        seed = static_cast<time_t>(cmdparser.get<unsigned int>("seed"));

    cout << "Trainig session guid: " << sessionguid << endl;
    cout << "Trainig session seed: " << seed << endl;
    cout << "-------------" << endl;

    auto trainobjs = load_classes_list(cmdparser.get<string>("traindir"));
    cout << "trainobjs.size(): "<< trainobjs.size() << endl;
    for(size_t i = 0; i < trainobjs.size(); ++i)
        cout << "  label " << i << " - unique samples - " << trainobjs[i].size() << endl;
    dlib::rand _foldsplitrnd(seed);
    auto allobjsfolds = split_into_folds(trainobjs,cmdparser.get<unsigned int>("cvfolds"),_foldsplitrnd);

    size_t classes_per_minibatch = static_cast<size_t>(cmdparser.get<int>("classes"));
    cout << "Classes per minibatch will be used:" << classes_per_minibatch << endl;
    size_t samples_per_class = static_cast<size_t>(cmdparser.get<int>("samples"));
    cout << "Samples per class in minibatch will be used: " << samples_per_class << endl;

    if(cmdparser.get<bool>("psalgo"))
        set_dnn_prefer_smallest_algorithms(); // larger minibatches will be available
    else
        set_dnn_prefer_fastest_algorithms();

    const bool trainaugmentation = cmdparser.get<bool>("trainaugm");
    const bool validaugmentation = cmdparser.get<bool>("validaugm");   

    for(size_t _fold = 0; _fold < allobjsfolds.size(); ++_fold) {
        cout << endl << "Split # " << _fold << endl;

        trainobjs = merge_except(allobjsfolds,_fold);
        cout << "trainobjs.size(): " << trainobjs.size() << endl;
        for(size_t i = 0; i < trainobjs.size(); ++i)
            cout << "  label " << i << " - unique samples - " << trainobjs[i].size() << endl;
        std::vector<std::vector<string>> validobjs = allobjsfolds[_fold];
        cout << "validobjs.size(): " << validobjs.size() << endl;
        for(size_t i = 0; i < validobjs.size(); ++i)
            cout << "  label " << i << " - unique samples - " << validobjs[i].size() << endl;

        net_type net;
        set_all_bn_running_stats_window_sizes(net, cmdparser.get<unsigned>("bnwsize"));
        //cout << net << endl;

        dnn_trainer<net_type> trainer(net,sgd(0.0001f,0.9f));
        trainer.set_learning_rate(0.1);
        trainer.be_verbose();
        trainer.set_synchronization_file(cmdparser.get<string>("outputdir") + string("/trainer_") + sessionguid + std::string("_split_") + std::to_string(_fold) + string("_sync") , std::chrono::minutes(10));
        if(cmdparser.has("learningrate"))
            trainer.set_learning_rate(cmdparser.get<double>("learningrate"));
        if(validobjs.size() > 0)
            trainer.set_test_iterations_without_progress_threshold(static_cast<size_t>(cmdparser.get<int>("viwp")));
        else
            trainer.set_iterations_without_progress_threshold(static_cast<size_t>(cmdparser.get<int>("tiwp")));

        dlib::pipe<std::vector<matrix<dlib::rgb_pixel>>> qimages(5);
        dlib::pipe<std::vector<unsigned long>> qlabels(5);
        auto data_loader = [classes_per_minibatch, samples_per_class, trainaugmentation, &qimages, &qlabels, &trainobjs, &seed](time_t _seed)  {

            dlib::rand rnd(seed + _seed);
            cv::RNG cvrng(static_cast<uint64_t>(seed + _seed));

            std::vector<matrix<dlib::rgb_pixel>> images;
            std::vector<unsigned long> labels;

            while(qimages.is_enabled()) {
                try {
                    load_mini_batch(classes_per_minibatch, samples_per_class, rnd, cvrng, trainobjs, images, labels, trainaugmentation);
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

        // Same for the test
        dlib::pipe<std::vector<matrix<dlib::rgb_pixel>>> testqimages(1);
        dlib::pipe<std::vector<unsigned long>> testqlabels(1);
        auto testdata_loader = [classes_per_minibatch, samples_per_class, validaugmentation, &testqimages, &testqlabels, &validobjs, &seed](time_t _seed) {
            dlib::rand rnd(seed + _seed);
            cv::RNG cvrng(static_cast<uint64_t>(seed + _seed));

            std::vector<matrix<dlib::rgb_pixel>> images;
            std::vector<unsigned long> labels;

            while(testqimages.is_enabled()) {
                try {
                    load_mini_batch(classes_per_minibatch, samples_per_class, rnd, cvrng, validobjs, images, labels, validaugmentation);
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

        std::vector<matrix<dlib::rgb_pixel>> images, vimages;
        std::vector<unsigned long> labels, vlabels;
        cout << "-------------" << endl;
        cout << "Wait while training will be accomplished:" << endl;
        while(trainer.get_learning_rate() >= cmdparser.get<double>("minlrthresh"))  {
            images.clear();
            labels.clear();
            qimages.dequeue(images);
            qlabels.dequeue(labels);
            trainer.train_one_step(images, labels);
            if((validobjs.size() > 0) && ((trainer.get_train_one_step_calls() % 10) == 0)) {
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
        data_loader4.join();

        if(validobjs.size() > 0) {
            testqimages.disable();
            testqlabels.disable();
            testdata_loader1.join();
        }

        cout << "Training has been accomplished" << endl;

        // Wait for training threads to stop
        trainer.get_net();
        net.clean();

        float acc = -1.0f;
        if(validobjs.size() > 0) {
            cout << "Accuracy evaluation on validation set:" << endl;
            acc = test_accuracy_on_set(validobjs,net,true,static_cast<size_t>(classes_per_minibatch));
            cout << "Average validation accuracy: " << acc << endl;
        }
        std::vector<std::vector<string>> testobjs;
        if(cmdparser.has("testdir")) {
            testobjs = load_classes_list(cmdparser.get<string>("testdir"));
            cout << "testdir.size(): "<< testobjs.size() << endl;
        }
        if(testobjs.size() > 0) {
            cout << "Accuracy evaluation on test set:" << endl;
            acc = test_accuracy_on_set(testobjs,net,true,static_cast<size_t>(classes_per_minibatch));
            cout << "Average test accuracy: " << acc << endl;
        }

        string _outputfilename = string("net_") + sessionguid + std::string("_split_") + std::to_string(_fold) + string(".dat");
        if((validobjs.size() > 0) || (testobjs.size() > 0))
            _outputfilename = string("net_") + sessionguid + std::string("_split_") + std::to_string(_fold) + string("_acc_") + to_string(acc) + string(".dat");
        cout << "Wait untill weights will be serialized to " << _outputfilename << endl;
        serialize(cmdparser.get<string>("outputdir") + string("/") + _outputfilename) << net;
    }
    cout << "Done" << endl;
    return 0;
}
