#include <QDir>
#include <QFile>
#include <QStringList>

#include <iostream>
#include <string>
#include <iterator>

#include <dlib/dnn.h>
#include <dlib/misc_api.h>

#include <opencv2/core.hpp>

#include "dlibimgaugment.h"
#include "opencvimgaugment.h"

#include "customnetwork.h"

using namespace dlib;
using namespace std;

struct Family {
    Family() {}

    void clear() {photosmap.clear(); relationsmap.clear();}

    static bool isvalid(const Family &_family) {
        if(_family.relationsmap.size() == 0 || _family.photosmap.size() == 0)
            return false;
        foreach(const auto &_person, _family.photosmap) {
            if(_person.second.size() == 0)
                return false;
        }
        foreach(const auto &_person, _family.relationsmap) {
            if(_person.second.size() == 0)
                return false;
        }
        return true;
    }

    static Family trimmed(const Family &_family) {
        Family _tmpfamily;
        foreach(const auto &_person,_family.relationsmap) {
            if(_family.photosmap.at(_person.first).size() > 0) {
                foreach(const auto &_kinship,_person.second) {
                    if(_family.photosmap.at(_kinship).size() > 0) {
                        _tmpfamily.relationsmap[_person.first].push_back(_kinship);
                    }
                }
            }
        }
        foreach(const auto &_person,_family.photosmap)
            if(_person.second.size() > 0)
                _tmpfamily.photosmap[_person.first] = _person.second;

        return _tmpfamily;
    }

    static std::vector<std::pair<string,string>> findnotrelated(const Family &_family) {
        std::vector<string> _vkeys(_family.relationsmap.size(),string());
        size_t i = 0;
        foreach (const auto &_person, _family.relationsmap)
            _vkeys[i++] = _person.first;
        std::vector<std::pair<string,string>> _notrelatedpairs;
        foreach (const auto &_person, _family.relationsmap) {
            const string &_key = _person.first;
            const std::vector<string> &_vkinships = _person.second;
            std::vector<bool> _vrelated(_vkeys.size(),false);
            for(size_t i = 0; i < _vkeys.size(); ++i)
                for(size_t j = 0; j < _vkinships.size(); ++j)
                    if((_vkeys[i]).compare(_vkinships[j]) == 0)
                        _vrelated[i] = true;
            for(size_t i = 0; i < _vrelated.size(); ++i)
                if((_vrelated[i] == false) && (_key.compare(_vkeys[i]) != 0))
                    _notrelatedpairs.push_back(std::make_pair(_key,_vkeys[i]));
        }
        return _notrelatedpairs;
    }

    static std::vector<string> allphotos(const Family &_family) {
        std::vector<string> vphotos;
        vphotos.reserve(_family.photosmap.size()*20); // heuristic from dataset overview
        foreach(const auto &_person, _family.photosmap)
            vphotos.insert(vphotos.end(),_person.second.begin(),_person.second.end());
        return vphotos;
    }

    std::map<string,std::vector<string>> photosmap;
    std::map<string,std::vector<string>> relationsmap;
};

std::ostream& operator<< (std::ostream& out, const Family &_family)
{
    foreach (const auto &_person, _family.relationsmap) {
        out << _person.first << "(" << _family.photosmap.at(_person.first).size() << ") kinhips: ";
        foreach (const auto &_relation, _person.second) {
            out << _relation << "(" << _family.photosmap.at(_relation).size() << ") ";
        }
        out << endl;
    }
    return out;
}

std::vector<Family> load_families(const string &_traindirname, const string &_relationsfilename)
{
    QStringList _filesfilters;
    _filesfilters << "*.jpg" << "*.png";

    QDir _qdir(_traindirname.c_str());

    QFile _qfile(_relationsfilename.c_str());
    if(_qfile.open(QIODevice::ReadOnly) == false)
        return std::vector<Family>();
    _qfile.readLine(); // drop header

    std::vector<Family> _vfamilies;

    QString _familyname;
    bool _newfamily;
    Family _family;
    while(!_qfile.atEnd()) {

        QString _line(_qfile.readLine());

        if(_line.section('/',0,0) != _familyname)
            _newfamily = true;
        else
            _newfamily = false;

        if(_newfamily) {
            _family = Family::trimmed(_family);
            if(Family::isvalid(_family)) {               
                /*std::cout << "Family: " << _familyname.toStdString() << std::endl;
                std::cout << _family << std::endl;
                std::vector<std::pair<string,string>> _vnr = Family::findnotrelated(_family);
                std::cout << "NOT RELATED PAIRS:" << std::endl;
                foreach (const auto &_pair, _vnr)
                    std::cout << _pair.first << " - " << _pair.second << std::endl;
                std::cout << std::endl;*/
                _vfamilies.push_back(std::move(_family));
            } else {
                std::cout << '\'' << _familyname.toStdString() << '\'' << " - invalid family" << endl << _family << endl;
                _family.clear();
            }            
            _familyname = _line.section('/',0,0);
        }

        string _leftname  = _line.section('/',1,1).section(',',0,0).toStdString();
        string _rightname = _line.section('/',2,2).trimmed().toStdString();
        _family.relationsmap[_leftname].push_back(_rightname);
        _family.relationsmap[_rightname].push_back(_leftname);

        if(_family.photosmap.count(_leftname) == 0) {
            QDir _qsubdir(_qdir.absolutePath().append("/%1/%2").arg(_familyname,_leftname.c_str()));
            QStringList _photosmapnames = _qsubdir.entryList(_filesfilters,QDir::Files | QDir::NoDotAndDotDot);
            for(const QString &_filename: _photosmapnames)
                _family.photosmap[_leftname].push_back(_qsubdir.absoluteFilePath(_filename).toUtf8().constData());
            if(_photosmapnames.size() == 0)
                _family.photosmap[_leftname] = std::vector<string>();
        }
        if(_family.photosmap.count(_rightname) == 0) {
            QDir _qsubdir(_qdir.absolutePath().append("/%1/%2").arg(_familyname,_rightname.c_str()));
            QStringList _photosmapnames = _qsubdir.entryList(_filesfilters,QDir::Files | QDir::NoDotAndDotDot);
            for(const QString &_filename: _photosmapnames)
                _family.photosmap[_rightname].push_back(_qsubdir.absoluteFilePath(_filename).toUtf8().constData());
            if(_photosmapnames.size() == 0)
                _family.photosmap[_rightname] = std::vector<string>();
        }
    }
    return _vfamilies;
}

std::vector<std::vector<Family>> split_into_folds(const std::vector<Family> &_objs, unsigned int _folds, dlib::rand& _rnd)
{
    std::vector<std::vector<Family>> _output(_folds);
    for(size_t i = 0; i < _objs.size(); ++i)
        _output[static_cast<size_t>(_rnd.get_integer(_folds))].push_back(_objs[i]);
    return _output;
}

std::vector<Family> merge_except(const std::vector<std::vector<Family>> &_objs, size_t _index)
{
    std::vector<Family> _mergedobjs;
    for(size_t i = 0; i < _objs.size(); ++i) {
        if(i != _index)
            for(size_t j = 0; j < _objs[i].size(); ++j)
                _mergedobjs.push_back(_objs[i][j]);
    }
    return _mergedobjs;
}

cv::Mat __loadImage(const std::string &_fileone, const std::string &_filetwo,int _tcols, int _trows, bool _crop, bool _center, bool _normalize, bool *_isloadded)
{
    cv::Mat _channels[2];
    _channels[0] = loadIFgraymatWsize(_fileone,_tcols,_trows,_crop,_center,_normalize,_isloadded);
    _channels[1] = loadIFgraymatWsize(_filetwo,_tcols,_trows,_crop,_center,_normalize,_isloadded);
    cv::Mat _outmat;
    cv::merge(_channels,2,_outmat);
    return _outmat;
}

void load_mini_batch (
    const size_t num_classes,
    const size_t num_samples,
    dlib::rand& rnd,
    cv::RNG & cvrng,
    const std::vector<Family>& objs,
    std::vector<std::array<matrix<float>,2>>& images,
    std::vector<unsigned long>& labels,
    bool _doaugmentation
)
{
    images.clear();
    labels.clear();

    cv::Mat _tmpmat;
    bool _isloaded;
    Family family_one;
    string first_filename, second_filename;
    unsigned long label;
    std::vector<bool> already_selected_family(objs.size(), false);

    size_t it1 = 0, it2;

    size_t classes_selected = 0;
    while(classes_selected < num_classes) {

        it1++;
        if(it1 == objs.size())
            cout << "it1 endless loop" << endl;

        // select random family
        size_t id = rnd.get_random_32bit_number() % objs.size();
        while(already_selected_family[id])
            id = rnd.get_random_32bit_number() % objs.size();
        already_selected_family[id] = true;

        family_one = objs[id];
        auto _it1 = family_one.relationsmap.begin();
        std::advance(_it1, rnd.get_random_32bit_number() % family_one.relationsmap.size());
        const string &person_one = _it1->first;
        /*if(family_one.photosmap.at(person_one).size() == 0)
            continue;*/
        std::vector<string> kinships_one = _it1->second;

        // select another random family
        id = rnd.get_random_32bit_number() % objs.size();
        while(already_selected_family[id])
            id = rnd.get_random_32bit_number() % objs.size();
        //already_selected_family[id] = true;
        std::vector<string> imposters = Family::allphotos(objs[id]);
        /*if(imposters.size() == 0)
            continue;*/

        size_t samples_selected = 0;
        it2 = 0;
        while(samples_selected < num_samples) {

            it2++;
            first_filename = family_one.photosmap.at(person_one)[rnd.get_random_32bit_number() % family_one.photosmap.at(person_one).size()];
            if(samples_selected % 2 == 0) {
                const string &kinship = kinships_one[rnd.get_random_32bit_number() % kinships_one.size()];
                if(it2 == objs.size())
                    cout << "it2 endless loop for " << person_one << " vs " << kinship << " in family: " << endl << family_one << endl;

                /*if(family_one.photosmap.at(kinship).size() == 0)
                    continue;               */
                second_filename = family_one.photosmap.at(kinship)[rnd.get_random_32bit_number() % family_one.photosmap.at(kinship).size()];               
                label = 1;
            } else {
                second_filename = imposters[rnd.get_random_32bit_number() % imposters.size()];
                label = 0;
            }
            if(rnd.get_random_float() > 0.5f)
                std::swap(first_filename,second_filename);

            if(_doaugmentation) {
                _tmpmat = __loadImage(first_filename,second_filename,IMG_WIDTH,IMG_HEIGHT,false,true,true,&_isloaded);
                assert(_isloaded);

                if(rnd.get_random_float() > 0.5f)
                    cv::flip(_tmpmat,_tmpmat,1);

                if(rnd.get_random_float() > 0.1f)
                    _tmpmat = jitterimage(_tmpmat,cvrng,cv::Size(0,0),0.05,0.05,5,cv::BORDER_REFLECT101,false);
                if(rnd.get_random_float() > 0.5f)
                    _tmpmat = distortimage(_tmpmat,cvrng,0.03,cv::INTER_CUBIC,cv::BORDER_REFLECT101);

                if(rnd.get_random_float() > 0.1f)
                    _tmpmat = cutoutRect(_tmpmat,rnd.get_random_float(),rnd.get_random_float(),0.2f,0.2f,rnd.get_random_float()*180.0f);

                if(rnd.get_random_float() > 0.1f)
                    _tmpmat = cutoutRect(_tmpmat,rnd.get_random_float(),0,0.2f,0.2f,rnd.get_random_float()*180.0f);
                if(rnd.get_random_float() > 0.1f)
                    _tmpmat = cutoutRect(_tmpmat,rnd.get_random_float(),1,0.2f,0.2f,rnd.get_random_float()*180.0f);
                if(rnd.get_random_float() > 0.1f)
                    _tmpmat = cutoutRect(_tmpmat,0,rnd.get_random_float(),0.2f,0.2f,rnd.get_random_float()*180.0f);
                if(rnd.get_random_float() > 0.1f)
                    _tmpmat = cutoutRect(_tmpmat,1,rnd.get_random_float(),0.2f,0.2f,rnd.get_random_float()*180.0f);

                if(rnd.get_random_float() > 0.1f)
                    _tmpmat = addNoise(_tmpmat,cvrng,0,0.05);

                if(rnd.get_random_float() > 0.5f)
                    cv::blur(_tmpmat,_tmpmat,cv::Size(3,3));               

                if(rnd.get_random_float() > 0.1f)
                    _tmpmat *= static_cast<double>((0.7f + 0.6f*rnd.get_random_float()));

                images.push_back(cvmatF2arrayofFdlibmatrix<2>(_tmpmat));
            } else {
                _tmpmat = __loadImage(first_filename,second_filename,IMG_WIDTH,IMG_HEIGHT,false,true,true,&_isloaded);
                assert(_isloaded);

                if(rnd.get_random_float() > 0.5f)
                    cv::flip(_tmpmat,_tmpmat,1);

                images.push_back(cvmatF2arrayofFdlibmatrix<2>(_tmpmat));
            }
            labels.push_back(label);
            samples_selected++;
        }
        classes_selected++;
    }
}

float test_metric_accuracy_on_set(const std::vector<Family> &_testobjs, dlib::net_type &_net, bool _beverbose,
                           const size_t _classes,
                           const size_t _samples,
                           const size_t _iterations=10,
                           const size_t _seed=1,
                           const bool _doaugmentation=true)
{
    if(_iterations == 0)
        return 0.0f;
    anet_type anet = _net;
    dlib::rand  rnd(_seed);
    cv::RNG     cvrng(_seed);
    std::vector<float> vf1(_iterations,0.0f);
    std::vector<uint> vtp(_iterations,0);
    std::vector<uint> vtn(_iterations,0);
    std::vector<uint> vfp(_iterations,0);
    std::vector<uint> vfn(_iterations,0);
    std::vector<std::array<dlib::matrix<float>,2>> images;
    std::vector<unsigned long> labels;
    for(size_t i = 0; i < _iterations; ++i) {
        load_mini_batch(_classes, _samples, rnd, cvrng, _testobjs, images, labels, _doaugmentation);
        std::vector<unsigned long> predictions = anet(images);
        for(size_t k = 0; k < images.size(); ++k) {
            if(labels[k] == 1) {
                if(predictions[k] == 1)
                    vtp[i]++;
                else
                    vfn[i]++;
            } else {
                if(predictions[k] == 1)
                    vfp[i]++;
                else
                    vtn[i]++;
            }
        }
        float _precision = static_cast<float>(vtp[i]) / (vtp[i] + vfp[i]);
        float _recall    = static_cast<float>(vtp[i]) / (vtp[i] + vfn[i]);
        vf1[i] = 2 * _precision*_recall / (_precision + _recall);
        if(_beverbose)
            cout << "iteration #" << i << (((i % 2) == 0) ? " (without kinships)" : " (with kinships)") << " - F1: " << vf1[i] << endl;
    }
    float acc = 0.0f;
    for(size_t i = 0; i < _iterations; ++i)
        acc += vf1[i];
    return acc / _iterations;
}

const cv::String options = "{traindir  t  |      | path to directory with training data}"
                           "{pairsfile p  |      | path to train_relationships.csv}"
                           "{cvfolds      |   5  | folds to use for cross validation training}"
                           "{splitseed    |   1  | seed for data folds split}"
                           "{testdir      |      | path to directory with test data}"
                           "{outputdir o  |      | path to directory with output data}"
                           "{minlrthresh  | 1E-5 | path to directory with output data}"
                           "{sessionguid  |      | session guid}"
                           "{learningrate |      | initial learning rate}"                          
                           "{classes      | 30   | classes per minibatch}"
                           "{samples      | 15   | samples per class in minibatch}"
                           "{bnwsize      | 100  | will be passed in set_all_bn_running_stats_window_sizes before net training}"
                           "{tiwp         | 5000 | train iterations without progress}"
                           "{viwp         | 1000 | validation iterations without progress}"
                           "{psalgo       | true | set prefer smallest algorithms}";

int main(int argc, char** argv)
{
    cv::CommandLineParser cmdparser(argc,argv,options);
    cmdparser.about("This app was designed to train dlib's format neural network with cross validation training scheme");
    if(argc == 1) {
        cmdparser.printMessage();
        return 0;
    }
    if(!cmdparser.has("traindir")) {
        cout << "No training directory provided! Abort..." << std::endl;
        return 1;
    }
    if(!cmdparser.has("pairsfile")) {
        cout << "No train_relationships.csv provided! Abort..." << std::endl;
        return 2;
    }
    if(!cmdparser.has("outputdir")) {
        cout << "No output directory provided! Abort..." << std::endl;
        return 3;
    }
    string sessionguid = std::to_string(0);
    if(cmdparser.has("sessionguid")) {
        sessionguid = cmdparser.get<string>("sessionguid");
    }
    cout << "Trainig session guid: " << sessionguid << endl;
    cout << "-------------" << endl;

    auto trainobjs = load_families(cmdparser.get<string>("traindir"),cmdparser.get<string>("pairsfile"));
    cout << "trainobjs.size(): "<< trainobjs.size() << endl;
    dlib::rand _foldsplitrnd(cmdparser.get<unsigned int>("splitseed"));
    auto allobjsfolds = split_into_folds(trainobjs,cmdparser.get<unsigned int>("cvfolds"),_foldsplitrnd);

    const size_t classes_per_minibatch = static_cast<unsigned long>(cmdparser.get<int>("classes"));
    cout << "Classes per minibatch will be used:" << classes_per_minibatch << endl;
    const size_t samples_per_class = static_cast<unsigned long>(cmdparser.get<int>("samples"));
    cout << "Samples per class in minibatch will be used: " << samples_per_class << endl;

    if(cmdparser.get<bool>("psalgo"))
        set_dnn_prefer_smallest_algorithms(); // larger minibatches will be available
    else
        set_dnn_prefer_fastest_algorithms();

    for(size_t _fold = 0; _fold < allobjsfolds.size(); ++_fold) {
        cout << endl << "Split # " << _fold << endl;

        trainobjs = merge_except(allobjsfolds,_fold);
        cout << "trainobjs.size(): " << trainobjs.size() << endl;
        std::vector<Family> validobjs = allobjsfolds[_fold];
        cout << "validobjs.size(): " << validobjs.size() << endl;

        net_type net;
        set_all_bn_running_stats_window_sizes(net, cmdparser.get<unsigned>("bnwsize"));

        dnn_trainer<net_type> trainer(net,sgd(0.0001f,0.9f));
        trainer.set_learning_rate(0.1);
        trainer.be_verbose();
        trainer.set_synchronization_file(cmdparser.get<string>("outputdir") + string("/trainer_") + sessionguid + std::string("_split_") + std::to_string(_fold) + string("_sync") , std::chrono::minutes(10));
        if(cmdparser.has("learningrate"))
            trainer.set_learning_rate(cmdparser.get<double>("learningrate"));
        if(validobjs.size() > 0)
            trainer.set_test_iterations_without_progress_threshold(static_cast<unsigned long>(cmdparser.get<int>("viwp")));
        else
            trainer.set_iterations_without_progress_threshold(static_cast<unsigned long>(cmdparser.get<int>("tiwp")));

        dlib::pipe<std::vector<std::array<dlib::matrix<float>,2>>> qimages(5);
        dlib::pipe<std::vector<unsigned long>> qlabels(5);
        auto data_loader = [classes_per_minibatch,samples_per_class,&qimages, &qlabels, &trainobjs](time_t seed)  {

            dlib::rand rnd(time(nullptr)+seed);
            cv::RNG cvrng(static_cast<uint64_t>(time(nullptr) + seed));

            std::vector<std::array<dlib::matrix<float>,2>> images;
            std::vector<unsigned long> labels;

            while(qimages.is_enabled()) {
                try {                    
                    load_mini_batch(classes_per_minibatch, samples_per_class, rnd, cvrng, trainobjs, images, labels, true);
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
        dlib::pipe<std::vector<std::array<dlib::matrix<float>,2>>> testqimages(1);
        dlib::pipe<std::vector<unsigned long>> testqlabels(1);
        auto testdata_loader = [classes_per_minibatch, samples_per_class,&testqimages, &testqlabels, &validobjs](time_t seed) {

            dlib::rand rnd(time(nullptr)+seed);
            cv::RNG cvrng(static_cast<uint64_t>(time(nullptr) + seed));

            std::vector<std::array<dlib::matrix<float>,2>> images;
            std::vector<unsigned long> labels;

            while(testqimages.is_enabled()) {
                try {
                    load_mini_batch(classes_per_minibatch, samples_per_class, rnd, cvrng, validobjs, images, labels, false);
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

        std::vector<std::array<dlib::matrix<float>,2>> images, vimages;
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
            acc = test_metric_accuracy_on_set(validobjs,net,true,classes_per_minibatch,2*samples_per_class,20);
            cout << "Average validation accuracy: " << acc << endl;
        }

        string _outputfilename = string("net_") + sessionguid + std::string("_split_") + std::to_string(_fold) + string(".dat");
        if(validobjs.size() > 0)
            _outputfilename = string("net_") + sessionguid + std::string("_split_") + std::to_string(_fold) + string("_acc_") + to_string(acc) + string(".dat");
        cout << "Wait untill weights will be serialized to " << _outputfilename << endl;
        serialize(cmdparser.get<string>("outputdir") + string("/") + _outputfilename) << net;
    }
    cout << "Done" << endl;
    return 0;
}
