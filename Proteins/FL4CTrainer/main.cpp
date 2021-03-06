#include <QDir>
#include <QFile>
#include <QFileInfo>

#include <map>
#include <string>
#include <thread>
#include <vector>

#include "dlibimgaugment.h"
#include "opencvimgaugment.h"
#include "dlibopencvconverter.h"

#include "customnetwork.h"

using namespace std;
using namespace dlib;

const cv::String keys =
   "{help h           |        | app help}"
   "{classes          |   28   | number of classes (each class has two possible outcomes 'y', 'n')}"
   "{minibatchsize    |   64   | minibatch size}"
   "{traindir t       |        | training directory location}"
   "{outputdir o      |        | output directory location}"
   "{validportion v   |  0.2   | output directory location}"
   "{number n         |   1    | number of classifiers to be trained}"
   "{swptrain         | 10000  | determines after how many steps without progress (training loss) decay should be applied to learning rate}"
   "{swpvalid         |  500   | determines after how many steps without progress (test loss) decay should be applied to learning rate}"
   "{magic            | 31072  | seed value for the random number generator that controls data separation}"
   "{minlrthresh      | 1.0e-3 | minimum learning rate, determines when training should be stopped}";

std::map<std::string,std::vector<std::string>> fillLabelsMap(unsigned int _classes);

void  loadData(unsigned int _classes, const QString &_trainfilename, const QString &_traindirname,
              dlib::rand &_rnd, float _validationportion,
              std::vector<std::pair<std::string,std::map<std::string,std::string>>> &_vtrainingset,
              std::vector<std::pair<std::string,std::map<std::string,std::string>>> &_vvalidationset);

template<typename R, typename T>
R computeMacroF1Score(const std::vector<T> &_truepos, const std::vector<T> &_falsepos, const std::vector<T> &_falseneg, R _epsilon = 0.0001);

cv::Mat __loadImage(const std::string &_filenameprefix,int _tcols, int _trows, bool _crop, bool _center, bool _normalize, bool *_isloadded=nullptr);

int main(int argc, char** argv) try
{
    cv::CommandLineParser cmdparser(argc, argv, keys);
    cmdparser.about("This app has been developed for competition https://www.kaggle.com/c/human-protein-atlas-image-classification/data");
    if(cmdparser.has("help")) {
        cmdparser.printMessage();
        return 0;
    }
    if(!cmdparser.has("traindir")) {
        qInfo("You have not provide path to training directory! Abort...");
        return 1;
    }
    if(!cmdparser.has("outputdir")) {
        qInfo("You have not provide path to output directory! Abort...");
        return 2;
    }
    if(cmdparser.get<int>("number") <= 0) {
        qInfo("Number of classifiers should be greater than zero! Abort...");
        return 3;
    }
    QDir traindir(cmdparser.get<string>("traindir").c_str());
    if(!traindir.exists()) {
        qInfo("Training directory does not exist! Abort...");
        return 4;
    }
    QFileInfo trainfi(traindir.absoluteFilePath("train.csv"));
    if(!trainfi.exists()) {
        qInfo("Training dir does not contain train.csv! Abort...");
        return 5;
    }
    traindir.setPath(traindir.absolutePath().append("/train"));
    if(!traindir.exists()) {
        qInfo("Training dir does not contain /train subdir! Abort...");
        return 6;
    }
    QStringList _filesnames = traindir.entryList(QDir::Files | QDir::NoDotAndDotDot);
    if(_filesnames.size() == 0) {
        qInfo("No files has been found in /train subdir! Abort...");
        return 7;
    }
    // Let's fill our labels map
    auto labelsmap = fillLabelsMap(cmdparser.get<unsigned int>("classes"));

    const unsigned int minibatchsize = cmdparser.get<unsigned int>("minibatchsize");
    // Ok, seems we have check everithing, now we can parse files
    std::vector<std::pair<std::string,std::map<std::string,std::string>>> _trainingset;
    std::vector<std::pair<std::string,std::map<std::string,std::string>>> _validationset;
    dlib::rand _rnd(cmdparser.get<int>("magic"));
    loadData(cmdparser.get<unsigned int>("classes"),
             trainfi.absoluteFilePath(),traindir.absolutePath(),
             _rnd,cmdparser.get<float>("validportion"),
             _trainingset,_validationset);

    qInfo("Training set size: %u", static_cast<unsigned int>(_trainingset.size()));
    qInfo("Validation set size: %u", static_cast<unsigned int>(_validationset.size()));

    if(_trainingset.size() == 0 || _validationset.size() == 0) {
        qInfo("Insufficient data for training or validation. Abort...");
        return 8;
    }

    for(int n = 0; n < cmdparser.get<int>("number"); ++n) {
        net_type net(labelsmap);
        net.loss_details().set_gamma(1.5f);
        net.subnet().layer_details().set_num_outputs(static_cast<long>(net.loss_details().number_of_labels()));
        std::cout << net << std::endl;

        dnn_trainer<net_type> trainer(net,sgd());
        trainer.set_learning_rate(0.1);
        trainer.be_verbose();
        trainer.set_synchronization_file(cmdparser.get<std::string>("outputdir") + std::string("/trainer_sync_") + std::to_string(n), std::chrono::minutes(10));
        trainer.set_iterations_without_progress_threshold(cmdparser.get<unsigned int>("swptrain"));
        trainer.set_test_iterations_without_progress_threshold(cmdparser.get<unsigned int>("swpvalid"));
        // If training set very large then
        set_all_bn_running_stats_window_sizes(net, minibatchsize*4);

        // Load training data
        dlib::pipe<std::pair<std::map<std::string,std::string>,std::array<dlib::matrix<float>,4>>> trainpipe(minibatchsize*2);
        auto traindata_load = [&trainpipe,&_trainingset](time_t seed)
        {
            dlib::rand rnd(time(nullptr)+seed);
            cv::RNG    cvrng(static_cast<unsigned long long>(time(nullptr)+seed));
            std::pair<std::map<std::string,std::string>,std::array<dlib::matrix<float>,4>> _sample;
            size_t _pos;
            cv::Mat _tmpmat;
            bool _training_file_loaded = false;
            while(trainpipe.is_enabled())
            {
                _pos = rnd.get_random_32bit_number() % _trainingset.size();
                _sample.first = _trainingset[_pos].second;
                _tmpmat = __loadImage(_trainingset[_pos].first,IMG_SIZE,IMG_SIZE,false,true,false,&_training_file_loaded);                                
                assert(_training_file_loaded);
                if(rnd.get_random_float() > 0.5f)
                    cv::flip(_tmpmat,_tmpmat,0);
                if(rnd.get_random_float() > 0.5f)
                    cv::flip(_tmpmat,_tmpmat,1);
                if(rnd.get_random_float() > 0.2f)
                    _tmpmat = distortimage(_tmpmat,cvrng,0.015,cv::INTER_LANCZOS4,cv::BORDER_REFLECT_101);
                if(rnd.get_random_float() > 0.0f)
                    _tmpmat = jitterimage(_tmpmat,cvrng,cv::Size(0,0),0.015,0,180,cv::BORDER_REFLECT101);
                if(rnd.get_random_float() > 0.1f)
                    _tmpmat = cutoutRect(_tmpmat,rnd.get_random_float(),rnd.get_random_float());
                _sample.second = cvmatF2arrayofFdlibmatrix<4>(_tmpmat);
                trainpipe.enqueue(_sample);
            }
        };
        std::thread traindata_loader1([traindata_load](){ traindata_load(1); });
        std::thread traindata_loader2([traindata_load](){ traindata_load(2); });
        std::thread traindata_loader3([traindata_load](){ traindata_load(3); });
        //Load validation data
        dlib::pipe<std::pair<std::map<std::string,std::string>,std::array<dlib::matrix<float>,4>>> validpipe(minibatchsize*2);
        auto validdata_load = [&validpipe, &_validationset](time_t seed)
        {
            dlib::rand rnd(time(nullptr)+seed);
            std::pair<std::map<std::string,std::string>,std::array<dlib::matrix<float>,4>> _sample;
            cv::Mat _tmpmat;
            size_t _pos;
            bool _validation_file_loaded = false;
            while(validpipe.is_enabled())
            {
                _pos = rnd.get_random_32bit_number() % _validationset.size();
                _sample.first = _validationset[_pos].second;
                _tmpmat = __loadImage(_validationset[_pos].first,IMG_SIZE,IMG_SIZE,false,true,false,&_validation_file_loaded);
                _sample.second = cvmatF2arrayofFdlibmatrix<4>(_tmpmat);
                assert(_validation_file_loaded);
                validpipe.enqueue(_sample);
            }
        };
        std::thread validdata_loader1([validdata_load](){ validdata_load(4); });

        std::vector<std::array<dlib::matrix<float>,4>> _timages;
        std::vector<std::map<std::string,std::string>> _tlabels;
        std::vector<std::array<dlib::matrix<float>,4>> _vimages;
        std::vector<std::map<std::string,std::string>> _vlabels;

        size_t _steps = 0;
        while(trainer.get_learning_rate() >= cmdparser.get<double>("minlrthresh"))  {
            _timages.clear();
            _tlabels.clear();
            std::pair<std::map<std::string,std::string>,std::array<dlib::matrix<float>,4>> _sample;
            while(_timages.size() < minibatchsize) { // minibatch size
                trainpipe.dequeue(_sample);
                _tlabels.push_back(_sample.first);
                _timages.push_back(std::move(_sample.second));
            }
            trainer.train_one_step(_timages, _tlabels);
            _steps++;
            if((_steps % 10) == 0) {
                _vimages.clear();
                _vlabels.clear();
                while(_vimages.size() < minibatchsize) { // minibatch size
                    validpipe.dequeue(_sample);
                    _vlabels.push_back(_sample.first);
                    _vimages.push_back(std::move(_sample.second));
                }
                trainer.test_one_step(_vimages, _vlabels);
            }
        }

        trainpipe.disable();
        traindata_loader1.join();
        traindata_loader2.join();
        traindata_loader3.join();
        validpipe.disable();
        validdata_loader1.join();

        // Wait for training threads to stop
        trainer.get_net();
        net.clean();
        qInfo("Training has been accomplished");

        //Let's make testing net
        anet_type _testnet = net;

        // Now we need check score in terms of macro [F1-score](https://en.wikipedia.org/wiki/F1_score)
        qInfo("Test on validation set will be performed. Please wait...");
        std::vector<std::pair<std::string,std::map<std::string,std::string>>> _subset;
        _subset.reserve(_validationset.size());
        // Let's load portion of validation data
        for(size_t i = 0; i < _validationset.size(); ++i) {
            if(_rnd.get_random_float() < 0.5f)
                _subset.push_back(_validationset[i]);
        }

        _vimages.clear();
        _vlabels.clear();
        _vlabels.reserve(_subset.size());
        for(size_t i = 0; i < _subset.size(); ++i) {          
            _vlabels.push_back(_subset[i].second);
        }

        // We will predict by one because number of images could be big (so GPU RAM could be insufficient to handle all in one batch)
        std::vector<std::map<std::string,dlib::loss_multimulticlass_focal_::classifier_output>> _predictions;
        _predictions.reserve(_subset.size());
        for(size_t i = 0; i < _subset.size(); ++i) {
            _predictions.push_back(_testnet(cvmatF2arrayofFdlibmatrix<4>(__loadImage(_subset[i].first,IMG_SIZE,IMG_SIZE,false,true,false))));
        }

        std::vector<unsigned int> truepos(net.loss_details().number_of_classifiers(),0);
        std::vector<unsigned int> falsepos(net.loss_details().number_of_classifiers(),0);
        std::vector<unsigned int> falseneg(net.loss_details().number_of_classifiers(),0);

        for(size_t i = 0; i < _predictions.size(); ++i) {
            for(size_t j = 0; j < net.loss_details().number_of_classifiers(); ++j) {
                const std::string &_classname = std::to_string(j);
                const std::string &_predictedlabel = _predictions[i].at(_classname);
                const std::string &_truelabel = _vlabels[i].at(_classname);
                if((_truelabel.compare("y") == 0) && (_predictedlabel.compare("y") == 0)) {
                    truepos[j] += 1;
                } else if((_truelabel.compare("n") == 0) && (_predictedlabel.compare("y") == 0)) {
                    falsepos[j] += 1;
                } else if((_truelabel.compare("y") == 0) && (_predictedlabel.compare("n") == 0)) {
                    falseneg[j] += 1;
                }
            }
        }
        double _score = computeMacroF1Score<double>(truepos,falsepos,falseneg) ;
        qInfo("Macro F-score: %f", _score);
        // Save the network to disk
        serialize(cmdparser.get<std::string>("outputdir") + "/dlib_resnet_mmc_" + std::to_string(n) + "_(MFs_" + std::to_string(_score) + ").dat") << net;
        qInfo("Model has been saved on disk\n\n========\n");
    }

	return 0;
}
catch(std::exception& e)
{
    cout << e.what() << endl;
}

std::map<std::string,std::vector<std::string>> fillLabelsMap(unsigned int _classes)
{
    std::map<std::string,std::vector<std::string>> _labelsmap;
    for(unsigned int i = 0; i < _classes; ++i)
        _labelsmap[std::to_string(i)] = {"y", "n"};
    return _labelsmap;
}

void loadData(unsigned int _classes, const QString &_trainfilename, const QString &_traindirname,
              dlib::rand &_rnd, float _validationportion,
              std::vector<std::pair<std::string,std::map<std::string,std::string>>> &_vtrainingset,
              std::vector<std::pair<std::string,std::map<std::string,std::string>>> &_vvalidationset)
{
    // Let's reserve some memory
    _vtrainingset.clear();
    _vtrainingset.reserve(31072); // https://www.kaggle.com/c/human-protein-atlas-image-classification/data
    _vvalidationset.clear();
    _vvalidationset.reserve(31072);

    QFile _file(_trainfilename);
    _file.open(QFile::ReadOnly);
    _file.readLine(); // skip header data    
    while(!_file.atEnd()) {
        QString _line = _file.readLine();
        if(_line.contains(',')) {
            std::string _filename = QString("%1/%2").arg(_traindirname,_line.section(',',0,0)).toStdString();
            //qInfo("filename: %s", _filename.c_str());
            std::map<std::string,std::string> _lbls;
            for(unsigned int i = 0; i < _classes; ++i)
                _lbls[std::to_string(i)] = "n";
            QStringList _lblslist = _line.section(',',1).simplified().split(' ');
            for(int i = 0; i < _lblslist.size(); ++i) {
                const std::string &_key = _lblslist.at(i).toStdString();
                if(_lbls.count(_key) == 1)
                    _lbls[_key] = "y";
            }

            // Let's check data by eyes
            /*std::map<std::string,std::string>::const_iterator _it;
            for(_it = _lbls.begin(); _it != _lbls.end(); ++_it) {
                cout << "\tkey: " << _it->first << "; value: ";
                auto _value = _it->second;
                cout << _value << endl;
            }*/

            if(_rnd.get_random_float() > _validationportion)
                _vtrainingset.push_back(make_pair(_filename,_lbls));
            else
                _vvalidationset.push_back(make_pair(_filename,_lbls));
        }
    }
}

template<typename R, typename T>
R computeMacroF1Score(const std::vector<T> &_truepos, const std::vector<T> &_falsepos, const std::vector<T> &_falseneg, R _epsilon)
{
    R _precision = 0, _recall = 0, _p, _r;

    for(size_t i = 0; i < _truepos.size(); ++i) {
        /*qInfo("TP[%u]: %u", static_cast<unsigned int>(i), static_cast<unsigned int>(_truepos[i]));
        qInfo("FP[%u]: %u", static_cast<unsigned int>(i), static_cast<unsigned int>(_falsepos[i]));
        qInfo("FN[%u]: %u", static_cast<unsigned int>(i), static_cast<unsigned int>(_falseneg[i]));*/
        _p = static_cast<R>(_truepos[i]) / (_truepos[i] + _falsepos[i] + _epsilon);
        _r = static_cast<R>(_truepos[i]) / (_truepos[i] + _falseneg[i] + _epsilon);
        qInfo("P[%u]: %f", static_cast<unsigned int>(i), _p);
        qInfo("R[%u]: %f", static_cast<unsigned int>(i), _r);
        _precision += _p;
        _recall += _r;
    }
    _precision = _precision/_truepos.size();
    _recall = _recall/_truepos.size();
    return 2.0 / (1.0 / (_precision + _epsilon) + 1.0 / (_recall + _epsilon));
}

cv::Mat __loadImage(const std::string &_filenameprefix,int _tcols, int _trows, bool _crop, bool _center, bool _normalize, bool *_isloadded)
{
    cv::Mat _channelsmat[4];
    std::string _postfix[4] = {"_green.png", "_blue.png", "_red.png", "_yellow.png"};
    for(uint8_t i = 0; i < 4; ++i) {
        _channelsmat[i] = loadIFgraymatWsize(_filenameprefix+_postfix[i],_tcols,_trows,_crop,_center,_normalize,_isloadded);
    }
    cv::Mat _outmat;
    cv::merge(_channelsmat,4,_outmat);
    return _outmat;
}


