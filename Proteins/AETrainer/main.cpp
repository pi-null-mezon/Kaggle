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
   "{minibatchsize    |   32   | size of minibatch}"
   "{traindir t       |        | training directory location}"
   "{outputdir o      |        | output directory location}"
   "{validportion v   |  0.15  | output directory location}"
   "{swptrain         | 10000  | determines after how many steps without progress (training loss) decay should be applied to learning rate}"
   "{swpvalid         | 500    | determines after how many steps without progress (test loss) decay should be applied to learning rate}"
   "{minlrthresh      | 1.0e-3 | minimum learning rate, determines when training should be stopped}";

void loadData(unsigned int _classes, const QString &_trainfilename, const QString &_traindirname,
              dlib::rand &_rnd, float _validationportion,
              std::map<std::string,std::vector<std::pair<std::string,std::map<std::string,std::string>>>> &_trainmap,
              std::map<std::string,std::vector<std::pair<std::string,std::map<std::string,std::string>>>> &_validmap);

void queryDataForClass(const unsigned int _classid, const std::map<std::string,std::vector<std::pair<std::string,std::map<std::string,std::string>>>> &_datamap, std::vector<std::string> &_vfiles, std::vector<unsigned long> &_vlabels);

cv::Mat __loadImage(const std::string &_filenameprefix,int _tcols, int _trows, bool _crop, bool _center, bool _normalize, bool *_isloadded=nullptr);

void showMapStat(const std::map<std::string,std::vector<std::pair<std::string,std::map<std::string,std::string>>>> &_mat);

template<typename R, typename T>
R computeMacroF1Score(T _truepos, T _falsepos, T _falseneg, R _epsilon=0.0001);

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
    unsigned int _minibatchsize = cmdparser.get<unsigned int>("minibatchsize");
    if(_minibatchsize == 0) {
        qInfo("Zero minibatch size! Abort...");
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

    // Ok, seems we have check everithing, now we can parse files
    std::map<std::string,std::vector<std::pair<std::string,std::map<std::string,std::string>>>> _trainmap;
    std::map<std::string,std::vector<std::pair<std::string,std::map<std::string,std::string>>>> _validmap;
    dlib::rand _rnd(31072); // magic number ;)
    loadData(cmdparser.get<unsigned int>("classes"),
             trainfi.absoluteFilePath(),traindir.absolutePath(),
             _rnd,cmdparser.get<float>("validportion"),
             _trainmap,_validmap);

    qInfo("\n--------------\nTraining set:\n--------------\n");
    showMapStat(_trainmap);
    qInfo("\n--------------\nValidation set:\n--------------\n");
    showMapStat(_validmap);
    qInfo("\n--------------\n");

    for(unsigned int n = 0; n < cmdparser.get<unsigned int>("classes"); ++n) {
        qInfo("Training for class %u has been started", n);
        std::vector<std::string>  _trainfiles;
        std::vector<unsigned long> _trainlabels;
        queryDataForClass(n,_trainmap,_trainfiles,_trainlabels);
        qInfo("Train subset size: %u", static_cast<unsigned int>(_trainfiles.size()));
        std::vector<std::string>  _validfiles;
        std::vector<unsigned long> _validlabels;
        queryDataForClass(n,_validmap,_validfiles,_validlabels);
        qInfo("Validation subset size: %u", static_cast<unsigned int>(_validfiles.size()));

        net_type net;
        dnn_trainer<net_type> trainer(net,sgd());
        trainer.set_learning_rate(0.1);
        trainer.be_verbose();
        trainer.set_synchronization_file(cmdparser.get<std::string>("outputdir") + std::string("/trainer_sync_") + std::to_string(n), std::chrono::minutes(10));
        trainer.set_iterations_without_progress_threshold(cmdparser.get<unsigned int>("swptrain"));
        trainer.set_test_iterations_without_progress_threshold(cmdparser.get<unsigned int>("swpvalid"));
        // If training set very large then
        //set_all_bn_running_stats_window_sizes(net, 1000);

        // Load training data
        dlib::pipe<std::pair<unsigned long,std::array<dlib::matrix<float>,4>>> trainpipeimg(4*_minibatchsize);
        auto traindata_load = [&trainpipeimg,&_trainfiles,&_trainlabels](time_t seed)
        {
            dlib::rand rnd(time(nullptr)+seed);
            cv::RNG    cvrng(static_cast<unsigned long long>(time(nullptr)+seed));
            cv::Mat _sample;
            size_t  _pos;
            bool _is_training_file_loaded;
            while(trainpipeimg.is_enabled()) {
                _is_training_file_loaded = false;
                _pos = rnd.get_random_32bit_number() % _trainfiles.size();
                _sample = __loadImage(_trainfiles[_pos],IMG_SIZE,IMG_SIZE,false,true,false,&_is_training_file_loaded);
                assert(_is_training_file_loaded);
                _sample = jitterimage(_sample,cvrng,cv::Size(0,0),0.02,0.1,90.0,cv::BORDER_REFLECT101);
                if(rnd.get_random_float() > 0.1f) {
                    _sample = cutoutRect(_sample,rnd.get_random_float(),rnd.get_random_float());
                }
                trainpipeimg.enqueue(std::make_pair(_trainlabels[_pos],cvmatF2arrayofFdlibmatrix<4>(_sample)));
            }
        };
        std::thread traindata_loader1([traindata_load](){ traindata_load(1); });
        std::thread traindata_loader2([traindata_load](){ traindata_load(2); });
        std::thread traindata_loader3([traindata_load](){ traindata_load(3); });
        //Load validation data
        dlib::pipe<std::pair<unsigned long, std::array<dlib::matrix<float>,4>>> validpipeimg(4*_minibatchsize);
        auto validdata_load = [&validpipeimg,&_validfiles,&_validlabels](time_t seed)
        {
            dlib::rand rnd(time(nullptr)+seed);
            cv::RNG    cvrng(static_cast<unsigned long long>(time(nullptr)+seed));
            cv::Mat _sample;
            size_t  _pos;
            bool _is_validation_file_loaded;
            while(validpipeimg.is_enabled()) {
                _is_validation_file_loaded = false;
                _pos = rnd.get_random_32bit_number() % _validfiles.size();
                _sample = __loadImage(_validfiles[_pos],IMG_SIZE,IMG_SIZE,false,true,false,&_is_validation_file_loaded);
                assert(_is_validation_file_loaded);
                validpipeimg.enqueue(std::make_pair(_validlabels[_pos],cvmatF2arrayofFdlibmatrix<4>(_sample)));
            }
        };
        std::thread validdata_loader1([validdata_load](){ validdata_load(4); });

        std::vector<std::array<dlib::matrix<float>,4>> _images;
        _images.reserve(_minibatchsize);
        std::vector<unsigned long> _labels;
        _labels.reserve(_minibatchsize);

        size_t _steps = 0;
        while(trainer.get_learning_rate() >= cmdparser.get<double>("minlrthresh"))  {           
            _images.clear();
            _labels.clear();
            std::pair<unsigned long,std::array<dlib::matrix<float>,4>> _pair;
            while(_images.size() < _minibatchsize) {
                trainpipeimg.dequeue(_pair);
                _images.push_back(std::move(_pair.second));
                _labels.push_back(_pair.first);
            }
            trainer.train_one_step(_images, _labels);
            _steps++;
            if((_steps % 10) == 0) {
                _images.clear();
                _labels.clear();
                while(_images.size() < _minibatchsize) {
                    validpipeimg.dequeue(_pair);
                    _images.push_back(std::move(_pair.second));
                    _labels.push_back(_pair.first);
                }
                trainer.test_one_step(_images, _labels);
            }
        }

        trainpipeimg.disable();
        traindata_loader1.join();
        traindata_loader2.join();
        traindata_loader3.join();
        validpipeimg.disable();
        validdata_loader1.join();

        // Wait for training threads to stop
        trainer.get_net();
        net.clean();
        qInfo("Training has been accomplished. Wait while F-score check will be performed...");

        //Let's make testing net
        anet_type _testnet = net;

        // Now we need check score in terms of macro [F1-score](https://en.wikipedia.org/wiki/F1_score)
        // We will predict by one because number of images could be big (so GPU RAM could be insufficient to handle all in one batch)
        std::vector<unsigned long> _predictions;
        _predictions.reserve(_validfiles.size());
        for(size_t i = 0; i < _validfiles.size(); ++i) {
            _predictions.push_back(_testnet(cvmatF2arrayofFdlibmatrix<4>(__loadImage(_validfiles[i],IMG_SIZE,IMG_SIZE,false,true,false))));
        }

        unsigned int truepos = 0;
        unsigned int falsepos = 0;
        unsigned int falseneg = 0;
        for(size_t i = 0; i < _predictions.size(); ++i) {            
            if((_predictions[i] == 1) && (_validlabels[i] == 1)) {
                truepos += 1;
            } else if((_predictions[i] == 1) && (_validlabels[i] == 0)) {
                falsepos += 1;
            } else if((_predictions[i] == 0) && (_validlabels[i] == 1)) {
                falseneg += 1;
            }
        }
        double _score = computeMacroF1Score<double>(truepos,falsepos,falseneg) ;
        qInfo("Macro F-score: %f", _score);
        // Save the network to disk
        serialize(cmdparser.get<std::string>("outputdir") + "/proteins_class_" + std::to_string(n) + "_(MFs_" + std::to_string(_score) + ").dat") << net;
        qInfo("Model has been saved on disk\n========================\n");
    }

	return 0;
}
catch(std::exception& e)
{
    cout << e.what() << endl;
}

void loadData(unsigned int _classes, const QString &_trainfilename, const QString &_traindirname,
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

    std::map<std::string,std::pair<size_t,size_t>> _texamples; // key -> num positive samples, num negative samples
    std::map<std::string,std::pair<size_t,size_t>> _vexamples; // key -> num positive samples, num negative samples
    for(unsigned int i = 0; i < _classes; ++i) {
        _texamples[std::to_string(i)] = std::make_pair(0,0);
        _vexamples[std::to_string(i)] = std::make_pair(0,0);
    }

    while(!_file.atEnd()) {
        QString _line = _file.readLine();
        if(_line.contains(',')) {
            std::string _filename = QString("%1/%2").arg(_traindirname,_line.section(',',0,0)).toStdString();
            //qInfo("filename: %s", _filename.c_str());
            std::map<std::string,std::string> _lbls;
            for(unsigned int i = 0; i < _classes; ++i) // https://www.kaggle.com/c/human-protein-atlas-image-classification/data
                _lbls[std::to_string(i)] = "n";

            // Let's find positive examples first
            QStringList _lblslist = _line.section(',',1).simplified().split(' ');
            for(int i = 0; i < _lblslist.size(); ++i) {
                const std::string &_key = _lblslist.at(i).toStdString();
                if(_lbls.count(_key) == 1) {
                    _lbls[_key] = "y";
                    if(_rnd.get_random_float() > _validationportion) {
                        _trainmap[_key].push_back(make_pair(_filename,_lbls));
                        _texamples[_key].first += 1;
                    } else {
                        _validmap[_key].push_back(make_pair(_filename,_lbls));
                        _vexamples[_key].first += 1;
                    }
                }
            }

            // Now we need add negative samples
            for(std::map<std::string,std::pair<size_t,size_t>>::const_iterator _it = _texamples.begin(); _it != _texamples.end(); ++_it) {
                if(_it->second.first > _it->second.second) { // positive samples more than negative
                    const std::string &_key = _it->first;
                    if(_lbls.at(_key).compare("n") == 0) {
                        if(_rnd.get_random_float() > _validationportion) {
                            _trainmap[_key].push_back(make_pair(_filename,_lbls));
                            _texamples[_key].second += 1;
                        } else {
                            _validmap[_key].push_back(make_pair(_filename,_lbls));
                            _vexamples[_key].second += 1;
                        }
                    }
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
R computeMacroF1Score(T _truepos, T _falsepos, T _falseneg, R _epsilon)
{
    R _precision = static_cast<R>(_truepos) / (_truepos + _falsepos + _epsilon);
    R _recall = static_cast<R>(_truepos) / (_truepos + _falseneg + _epsilon);
    return 2.0 / (1.0 / (_precision + _epsilon) + 1.0 / (_recall + _epsilon));
}

void showMapStat(const std::map<std::string,std::vector<std::pair<std::string,std::map<std::string,std::string>>>> &_mat)
{
    cout << "\tunique keys: " << _mat.size() << endl;
    std::map<std::string,std::vector<std::pair<std::string,std::map<std::string,std::string>>>>::const_iterator _it;
    for(_it = _mat.begin(); _it != _mat.end(); ++_it) {
        cout << "\tkey: " << _it->first << "; samples: " << _it->second.size() << endl;
    }
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

void queryDataForClass(const unsigned int _classid, const std::map<std::string,std::vector<std::pair<std::string,std::map<std::string,std::string>>>> &_datamap, std::vector<std::string> &_vfiles, std::vector<unsigned long> &_vlabels)
{
    const std::string _classname = std::to_string(_classid);
    const size_t _entries = _datamap.at(_classname).size();
    _vfiles.clear();
    _vfiles.reserve(_entries);
    _vlabels.clear();
    _vlabels.reserve(_entries);
    for(size_t i = 0; i < _entries; ++i) {
        _vfiles.push_back( _datamap.at(_classname)[i].first );
        if(_datamap.at(_classname)[i].second.at(_classname).compare("y") == 0) {
            _vlabels.push_back(1);
        } else {
            _vlabels.push_back(0);
        }
    }
}


