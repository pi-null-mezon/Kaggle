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
   "{traindir t       |        | training directory location}"
   "{outputdir o      |        | output directory location}"
   "{validportion v   |  0.2   | output directory location}"
   "{number n         |   1    | number of classifiers to be trained}"
   "{swptrain         | 5000   | determines after how many steps without progress (training loss) decay should be applied to learning rate}"
   "{swpvalid         | 1000   | determines after how many steps without progress (test loss) decay should be applied to learning rate}"
   "{minlrthresh      | 1.0e-5 | minimum learning rate, determines when training should be stopped}";

std::map<std::string,std::vector<std::string>> fillLabelsMap(unsigned int _classes);

void loadData(unsigned int _classes, const QString &_trainfilename, const QString &_traindirname, const QString &_extension,
              dlib::rand &_rnd, float _validationportion,
              std::map<std::string,std::vector<std::pair<std::string,std::map<std::string,std::string>>>> &_trainmap,
              std::map<std::string,std::vector<std::pair<std::string,std::map<std::string,std::string>>>> &_validmap);

void load_unskewed_minibatch( const std::map<std::string,std::vector<std::pair<std::string,std::map<std::string,std::string>>>> &_map,
                              dlib::rand &_rnd,
                              cv::RNG    &_cvrng,
                              std::vector<std::map<std::string,std::string>> &_vlabels,
                              std::vector<dlib::matrix<float>> &_vsamples,
                              bool _enableaugmentation)
{
    _vsamples.clear();
    _vlabels.clear();
    size_t _numofclasses = _map.size();
    _vsamples.reserve(_numofclasses);
    _vlabels.reserve(_numofclasses);
    std::string _classname;
    for(size_t i = 0; i < _numofclasses; ++i) {
        _classname = std::to_string(i);
        if(_map.at(_classname).size() > 0) {
            size_t _pos = _rnd.get_random_32bit_number() % _map.at(_classname).size();
            bool _file_loaded = false;
            if(_enableaugmentation == true) {
                //--------------------------
                cv::Mat _tmpmat = loadIgraymatWsizeCN(_map.at(_classname)[_pos].first,IMG_SIZE,IMG_SIZE,false,&_file_loaded);;
                assert(_file_loaded);
                _tmpmat = jitterimage(_tmpmat,_cvrng,cv::Size(0,0),0.02,0.1,90,cv::BORDER_REFLECT101);
                if(_rnd.get_random_float() > 0.1f)
                    _tmpmat = cutoutRect(_tmpmat,_rnd.get_random_float(),_rnd.get_random_float());
                //--------------------------
                _vsamples.push_back(cvmat2dlibmatrix<float>(_tmpmat));
            } else {
                _vsamples.push_back(load_grayscale_image_with_normalization(_map.at(_classname)[_pos].first,IMG_SIZE,IMG_SIZE,false,&_file_loaded));
                assert(_file_loaded);
            }
            _vlabels.push_back(_map.at(_classname)[_pos].second);
        }
    }
}

void showMatStat(const std::map<std::string,std::vector<std::pair<std::string,std::map<std::string,std::string>>>> &_mat);

template<typename R, typename T>
R computeMacroF1Score(const std::vector<T> &_truepos, const std::vector<T> &_falsepos, const std::vector<T> &_falseneg);

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

    // Ok, seems we have check everithing, now we can parse files
    std::map<std::string,std::vector<std::pair<std::string,std::map<std::string,std::string>>>> _trainmap;
    std::map<std::string,std::vector<std::pair<std::string,std::map<std::string,std::string>>>> _validmap;
    dlib::rand _rnd(31072); // magic number ;)
    loadData(cmdparser.get<unsigned int>("classes"),
             trainfi.absoluteFilePath(),traindir.absolutePath(),"png",
             _rnd,cmdparser.get<float>("validportion"),
             _trainmap,_validmap);

    qInfo("\n--------------\nTraining set:\n--------------\n");
    showMatStat(_trainmap);
    qInfo("\n--------------\nValidation set:\n--------------\n");
    showMatStat(_validmap);
    qInfo("\nStart training process...");

    for(int n = 0; n < cmdparser.get<int>("number"); ++n) {
        net_type net(labelsmap);
        net.subnet().layer_details().set_num_outputs(net.loss_details().number_of_labels());

        dnn_trainer<net_type> trainer(net,sgd());
        trainer.set_learning_rate(0.1);
        trainer.be_verbose();
        trainer.set_synchronization_file(cmdparser.get<std::string>("outputdir") + std::string("/trainer_sync_") + std::to_string(n), std::chrono::minutes(10));
        trainer.set_iterations_without_progress_threshold(cmdparser.get<unsigned int>("swptrain"));
        trainer.set_test_iterations_without_progress_threshold(cmdparser.get<unsigned int>("swpvalid"));
        // If training set very large then
        //set_all_bn_running_stats_window_sizes(net, 1000);

        // Load training data
        dlib::pipe<std::vector<dlib::matrix<float>>> trainpipeimg(5);
        dlib::pipe<std::vector<std::map<std::string,std::string>>> trainpipelbl(5);
        auto traindata_load = [&trainpipeimg,&trainpipelbl,&_trainmap](time_t seed)
        {
            dlib::rand rnd(time(nullptr)+seed);
            cv::RNG    cvrng(static_cast<unsigned long long>(time(nullptr)+seed));
            std::vector<dlib::matrix<float>> _vimg;
            std::vector<std::map<std::string,std::string>> _vlbl;
            while(trainpipeimg.is_enabled()) {
                load_unskewed_minibatch(_trainmap,rnd,cvrng,_vlbl,_vimg,true);
                trainpipeimg.enqueue(_vimg);
                trainpipelbl.enqueue(_vlbl);
            }
        };
        std::thread traindata_loader1([traindata_load](){ traindata_load(1); });
        std::thread traindata_loader2([traindata_load](){ traindata_load(2); });
        std::thread traindata_loader3([traindata_load](){ traindata_load(3); });
        //Load validation data
        dlib::pipe<std::vector<dlib::matrix<float>>> validpipeimg(2);
        dlib::pipe<std::vector<std::map<std::string,std::string>>> validpipelbl(2);
        auto validdata_load = [&validpipeimg,&validpipelbl,&_validmap](time_t seed)
        {
            dlib::rand rnd(time(nullptr)+seed);
            cv::RNG    cvrng(static_cast<unsigned long long>(time(nullptr)+seed));
            std::vector<dlib::matrix<float>> _vimg;
            std::vector<std::map<std::string,std::string>> _vlbl;
            while(validpipeimg.is_enabled()) {
                load_unskewed_minibatch(_validmap,rnd,cvrng,_vlbl,_vimg,false);
                validpipeimg.enqueue(_vimg);
                validpipelbl.enqueue(_vlbl);
            }
        };
        std::thread validdata_loader1([validdata_load](){ validdata_load(4); });

        std::vector<dlib::matrix<float>> _timages;
        std::vector<std::map<std::string,std::string>> _tlabels;
        std::vector<dlib::matrix<float>> _vimages;
        std::vector<std::map<std::string,std::string>> _vlabels;

        size_t _steps = 0;
        while(trainer.get_learning_rate() >= cmdparser.get<double>("minlrthresh"))  {

            trainpipeimg.dequeue(_timages);
            trainpipelbl.dequeue(_tlabels);
            trainer.train_one_step(_timages, _tlabels);
            _steps++;

            if((_steps % 4) == 0) {
                validpipeimg.dequeue(_vimages);
                validpipelbl.dequeue(_vlabels);
                trainer.test_one_step(_vimages, _vlabels);
            }
        }

        trainpipeimg.disable();
        trainpipelbl.disable();
        traindata_loader1.join();
        traindata_loader2.join();
        traindata_loader3.join();
        validpipeimg.disable();
        validpipelbl.disable();
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
        _subset.reserve(31072);
        // Let's load portion of validation data
        std::string _classname;
        for(size_t i = 0; i < _validmap.size(); ++i) {
            _classname = std::to_string(i);
            for(size_t j = 0; j < _validmap.at(_classname).size(); ++j)
                if(_rnd.get_random_float() < 0.5)
                    _subset.push_back(_validmap.at(_classname)[j]);
        }
        _vimages.clear();
        _vimages.reserve(_subset.size());
        _vlabels.clear();
        _vlabels.reserve(_subset.size());
        for(size_t i = 0; i < _subset.size(); ++i) {
            _vimages.push_back(load_grayscale_image_with_normalization(_subset[i].first,IMG_SIZE,IMG_SIZE,false));
            _vlabels.push_back(_subset[i].second);
        }

        // We will predict by one because number of images could be big (so GPU RAM could be insufficient to handle all in one batch)
        std::vector<std::map<std::string,dlib::loss_multimulticlass_log_::classifier_output>> _predictions;
        _predictions.reserve(_subset.size());
        for(size_t i = 0; i < _subset.size(); ++i) {
            _predictions.push_back(_testnet(_vimages[i]));
        }

        std::vector<unsigned int> truepos(net.loss_details().number_of_classifiers(),0);
        std::vector<unsigned int> falsepos(net.loss_details().number_of_classifiers(),0);
        std::vector<unsigned int> falseneg(net.loss_details().number_of_classifiers(),0);

        std::string _predictedlabel, _truelabel;
        for(size_t i = 0; i < _predictions.size(); ++i) {
            for(size_t j = 0; j < net.loss_details().number_of_classifiers(); ++j) {
                _classname = std::to_string(j);
                _predictedlabel = _predictions[i][_classname];
                _truelabel = _vlabels[i][_classname];
                if((_truelabel.compare("y") == 0) && (_predictedlabel.compare("y") == 0)) {
                    truepos[j] += 1;
                } else if((_truelabel.compare("n") == 0) && (_predictedlabel.compare("y") == 0)) {
                    falsepos[j] += 1;
                } else if((_truelabel.compare("y") == 0) && (_predictedlabel.compare("n") == 0)) {
                    falseneg[j] += 1;
                }
            }
        }
        float _score = computeMacroF1Score<float>(truepos,falsepos,falseneg) ;
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

void loadData(unsigned int _classes, const QString &_trainfilename, const QString &_traindirname, const QString &_extension,
              dlib::rand &_rnd, float _validationportion,
              std::map<std::string,std::vector<std::pair<std::string,std::map<std::string,std::string>>>> &_trainmap,
              std::map<std::string,std::vector<std::pair<std::string,std::map<std::string,std::string>>>> &_validmap)
{
    // Let's reserve some memory - 31072 total size of trainningset 31072 images, accordintg to https://www.kaggle.com/c/human-protein-atlas-image-classification/data
    std::string _classname;
    for(unsigned int i = 0; i < _classes; ++i) {
        _classname = std::to_string(i);
        _trainmap[_classname].reserve(12885); // I know that at most it will be 12885
        _validmap[_classname].reserve(12885);
    }

    QFile _file(_trainfilename);
    _file.open(QFile::ReadOnly);
    _file.readLine(); // skip header data    
    while(!_file.atEnd()) {
        QString _line = _file.readLine();
        if(_line.contains(',')) {
            std::string _filename = QString("%1/%2_green.%3").arg(_traindirname,_line.section(',',0,0),_extension).toStdString();
            //qInfo("filename: %s", _filename.c_str());
            std::map<std::string,std::string> _lbls;
            for(unsigned int i = 0; i < _classes; ++i) // https://www.kaggle.com/c/human-protein-atlas-image-classification/data
                _lbls[std::to_string(i)] = "n";
            QStringList _lblslist = _line.section(',',1).simplified().split(' ');
            std::string _key;
            for(int i = 0; i < _lblslist.size(); ++i) {
                _key = _lblslist.at(i).toStdString();
                if(_lbls.count(_key) == 1) {
                    _lbls[_key] = "y";
                    if(_rnd.get_random_float() > _validationportion)
                        _trainmap[_key].push_back(make_pair(_filename,_lbls));
                    else
                        _validmap[_key].push_back(make_pair(_filename,_lbls));
                }
            }
            // Let's check data by eyes
            /*std::map<std::string,std::string>::const_iterator _it;
            for(_it = _lbls.begin(); _it != _lbls.end(); ++_it) {
                cout << "\tkey: " << _it->first << "; value: " << _it->second << endl;
            }*/
        }
    }
}

template<typename R, typename T>
R computeMacroF1Score(const std::vector<T> &_truepos, const std::vector<T> &_falsepos, const std::vector<T> &_falseneg)
{
    R _precision = 0, _recall = 0;
    for(size_t i = 0; i < _truepos.size(); ++i) {
        _precision += static_cast<R>(_truepos[i]) / (_truepos[i] + _falsepos[i]);
        _recall += static_cast<R>(_truepos[i]) / (_truepos[i] + _falseneg[i]);
    }
    _precision /= _truepos.size();
    _recall /= _truepos.size();
    return 2.0 / ((1. / _precision) + (1. / _recall));
}

void showMatStat(const std::map<std::string,std::vector<std::pair<std::string,std::map<std::string,std::string>>>> &_mat)
{
    cout << "\tunique keys: " << _mat.size() << endl;
    std::map<std::string,std::vector<std::pair<std::string,std::map<std::string,std::string>>>>::const_iterator _it;
    for(_it = _mat.begin(); _it != _mat.end(); ++_it) {
        cout << "\tkey: " << _it->first << "; samples: " << _it->second.size() << endl;
    }
}


