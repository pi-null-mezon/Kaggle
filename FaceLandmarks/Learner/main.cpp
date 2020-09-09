#include <QElapsedTimer>
#include <QRegularExpression>
#include <QStringList>
#include <QFile>
#include <QDir>

#include <QFile>
#include <QJsonObject>
#include <QJsonDocument>
#include <QJsonArray>

#include "opencvimgaugment.h"
#include "dlibopencvconverter.h"

#include <opencv2/highgui.hpp>

#include "customnetwork.h"

const std::string options = "{traindir t  |       | - directory with training images}"
                            "{outputdir o |  .    | - output directory}"
                            "{mbs         |  64   | - mini batch size}"
                            "{seed        |  7    | - random number generator seed value}"
                            "{split       |  0.1  | - validation portion of data}"
                            "{learningrate|       | - learning rate}"
                            "{viwp        | 1000  | - validation iterations without progress}"
                            "{tiwp        | 10000 | - training iterations without progress}"
                            "{taug        | true  | - training time augmentation}"
                            "{minlrthresh | 1E-5  | - when learning should be stopped}"
                            "{help h      |       | - help}";

struct FaceLandmarks {
    FaceLandmarks() {}
    FaceLandmarks(FaceLandmarks &&other) {
        //qDebug("move constructor :)");
        if(this != &other) {
            filename = std::move(other.filename);
            values = std::move(other.values);
        }
    }
    std::string filename;
    std::vector<float> values;
};


void flip_labels(std::vector<float> &_lbls) {
    static unsigned char lpts[] = {1, 2, 3, 4, 5, 6, 7, 8, 18,19,20,21,22,37,38,39,40,41,42,32,33,49,50,51,61,62,68,59,60};
    static unsigned char rpts[] = {17,16,15,14,13,12,11,10,27,26,25,24,23,46,45,44,43,48,47,36,35,55,54,53,65,64,66,57,56};
    for(unsigned long i = 0; i < sizeof(lpts)/sizeof(lpts[0]); ++i) {
        std::swap(_lbls[2*(lpts[i]-1)],_lbls[2*(rpts[i]-1)]);
        std::swap(_lbls[2*(lpts[i]-1)+1],_lbls[2*(rpts[i]-1)+1]);
    }

    for(size_t i = 0; i < _lbls.size()/2; ++i) {
        _lbls[2*i] *= -1;
    }
}

void load_image(const FaceLandmarks &landmarks, matrix<rgb_pixel> &img, std::vector<float> &labels, dlib::rand &rnd, cv::RNG &cvrng, bool augment=false)
{
    bool loaded_sucessfully = false;
    cv::Mat _tmpmat = loadIbgrmatWsize(landmarks.filename,IMG_WIDTH,IMG_HEIGHT,false,&loaded_sucessfully);
    DLIB_CASSERT(loaded_sucessfully,"Can not read image!"); // TO DO understand in witch mode this macro works as it used to
    if(augment) {

        auto _tmplbls = landmarks.values;

        if(rnd.get_random_float() > 0.5f) {
            cv::flip(_tmpmat,_tmpmat,1);
            flip_labels(_tmplbls);
        }

        /*if(rnd.get_random_float() > 0.8f)
            _tmpmat = cutoutRect(_tmpmat,rnd.get_random_float(),rnd.get_random_float(),0.4f,0.4f,rnd.get_random_float()*180.0f);*/

        if(rnd.get_random_float() > 0.5f)
            cv::blur(_tmpmat,_tmpmat,cv::Size(3,3));
        if(rnd.get_random_float() > 0.1f)
            _tmpmat *= (0.8 + 0.4*rnd.get_random_double());
        if(rnd.get_random_float() > 0.1f)
            _tmpmat = addNoise(_tmpmat,cvrng,0,9);

        if(rnd.get_random_float() > 0.5f) {
            cv::cvtColor(_tmpmat,_tmpmat,cv::COLOR_BGR2GRAY);
            cv::Mat _chmat[] = {_tmpmat, _tmpmat, _tmpmat};
            cv::merge(_chmat,3,_tmpmat);
        }

        img = cvmat2dlibmatrix<dlib::rgb_pixel>(_tmpmat);
        dlib::disturb_colors(img,rnd);
        labels = _tmplbls;
    } else {
        img = cvmat2dlibmatrix<dlib::rgb_pixel>(_tmpmat);
        labels = landmarks.values;
    }
}

float mean(const std::vector<float> &_v)
{
    if(_v.size() != 0)
        return std::accumulate(_v.begin(),_v.end(),0.0f) / _v.size();
    return 0;
}

float stdev(const std::vector<float> &_v)
{
    if(_v.size() > 1) {
        const float _m = mean(_v);
        float _stdev = 0;
        for(size_t i = 0; i < _v.size(); ++i) {
            _stdev += (_v[i]-_m)*(_v[i]-_m);
        }
        return std::sqrt(_stdev/(_v.size() - 1));
    }
    return 0;
}

std::vector<float> extract_values(const QString &_filename)
{
    std::vector<float> values;
    values.reserve(136);
    cv::Mat _img = cv::imread(_filename.toStdString(),cv::IMREAD_UNCHANGED);
    if(!_img.empty()) {
        QFile jsonfile(QString("%1.json").arg(_filename.section('.',0,0)));
        if(jsonfile.open(QIODevice::ReadOnly)) {
            QJsonArray ja = QJsonDocument::fromJson(jsonfile.readAll()).object()["landmarks"].toArray();
            for(int i = 0; i < ja.size(); ++i) {
                QJsonObject _json = ja.at(i).toObject();
                values.push_back(_json["x"].toDouble()/static_cast<double>(_img.cols) - 0.5);
                values.push_back(_json["y"].toDouble()/static_cast<double>(_img.rows) - 0.5);
            }
        }
    }
    return values;
}

int main(int argc, char *argv[])
{
    cv::CommandLineParser cmdp(argc,argv,options);
    cmdp.about("Tool for face landmark detector training");
    if(cmdp.has("help") || argc == 1) {
        cmdp.printMessage();
        return 0;
    }
    if(!cmdp.has("traindir")) {
        qInfo("No training directory provided! Abort...");
        return 1;
    }
    QDir traindir(cmdp.get<std::string>("traindir").c_str());
    if(!traindir.exists()) {
        qInfo("Training directory '%s' does not exist! Abort...",traindir.absolutePath().toUtf8().constData());
        return 2;
    }

    //-------------------------------------------------------------------------------
    std::vector<FaceLandmarks> flandmarks;
    flandmarks.reserve(1E4);
    qInfo("Reading training data, please wait...");
    auto files = traindir.entryList(QStringList() << "*.jpg" << "*.jpeg" << "*.png", QDir::Files | QDir::NoDotDot);
    qInfo(" %d - total images in directory", files.size());
    for(const auto &_filename: files) {
        FaceLandmarks _facelandmarks;
        _facelandmarks.filename = traindir.absoluteFilePath(_filename).toStdString();
        _facelandmarks.values = extract_values(traindir.absoluteFilePath(_filename));
        if(_facelandmarks.values.size() > 0)
            flandmarks.push_back(std::move(_facelandmarks));
    }
    qInfo(" %lu - valid training instances found",flandmarks.size());

    const int seed = cmdp.get<int>("seed");
    const double split = cmdp.get<double>("split");
    const size_t minibatchsize = static_cast<size_t>(cmdp.get<uint>("mbs"));
    qInfo("\n minibatch size: %u", (unsigned int)minibatchsize);

    dlib::rand rnd(seed);
    std::vector<FaceLandmarks> trainingset, validationset;
    trainingset.reserve(flandmarks.size());
    validationset.reserve(flandmarks.size());
    for(size_t i = 0; i < flandmarks.size(); ++i) {
        if(rnd.get_random_double() < split)
            validationset.push_back(std::move(flandmarks[i]));
        else
            trainingset.push_back(std::move(flandmarks[i]));
    }
    flandmarks.clear();
    flandmarks.shrink_to_fit();
    qInfo(" training: %lu",trainingset.size());
    qInfo(" validation: %lu",validationset.size());

    //--------------------------------------------------------------------------------
    // DEBUGGING
    /*matrix<rgb_pixel> img;
    std::vector<float> lbls;
    cv::RNG _cvrng;
    cv::namedWindow("augmented",cv::WINDOW_NORMAL);
    cv::namedWindow("original",cv::WINDOW_NORMAL);
    for(const auto &instance : trainingset) {
        cv::Mat original = cv::imread(instance.filename);

        load_image(instance,img,lbls,rnd,_cvrng,true);
        cv::Mat augmented = dlibmatrix2cvmat(img);
        for(size_t i = 0; i < lbls.size()/2; ++i) {
            const cv::Point2f point((lbls[2*i]+0.5f)*augmented.cols,(lbls[2*i+1]+0.5f)*augmented.rows);
            //cv::circle(augmented,point,1,cv::Scalar(0,255,0),1,cv::LINE_AA);
            cv::putText(augmented,std::to_string(i+1),point-cv::Point2f(2,-1),cv::FONT_HERSHEY_SIMPLEX,0.2,cv::Scalar(0,0,255),1,cv::LINE_AA);
        }
        cv::imshow("augmented",augmented);
        cv::imshow("original",original);
        cv::waitKey(0);
    }*/

    //--------------------------------------------------------------------------------
    net_type net;
    dnn_trainer<net_type> trainer(net,sgd());
    trainer.set_learning_rate(0.01);
    trainer.be_verbose();
    trainer.set_synchronization_file(cmdp.get<string>("outputdir") + string("/trainer_sync") , std::chrono::minutes(10));
    if(cmdp.has("learningrate"))
        trainer.set_learning_rate(cmdp.get<double>("learningrate"));
    if(validationset.size() > 0)
        trainer.set_test_iterations_without_progress_threshold(static_cast<size_t>(cmdp.get<int>("viwp")));
    else
        trainer.set_iterations_without_progress_threshold(static_cast<size_t>(cmdp.get<int>("tiwp")));


    bool training_time_augmentation = cmdp.get<bool>("taug");
    qInfo(" training time augmentation: %s\n", training_time_augmentation ? "true" : "false");
    dlib::pipe<std::pair<std::vector<float>,matrix<rgb_pixel>>> trainingpipe(256);
    auto f = [&trainingpipe,&trainingset,&seed,&training_time_augmentation](time_t seed_shift) {
        dlib::rand rnd(seed+seed_shift);
        cv::RNG cvrng(seed+seed_shift);
        std::vector<float> lbls;
        matrix<rgb_pixel> img;
        while(trainingpipe.is_enabled()) {
            load_image(trainingset[rnd.get_random_32bit_number() % trainingset.size()],img,lbls,rnd,cvrng,training_time_augmentation);
            trainingpipe.enqueue(std::make_pair(lbls,img));
        }
    };
    std::thread training_data_loader1([f](){ f(1); });
    std::thread training_data_loader2([f](){ f(2); });
    std::thread training_data_loader3([f](){ f(3); });
    std::thread training_data_loader4([f](){ f(4); });

    dlib::pipe<std::pair<std::vector<float>,matrix<rgb_pixel>>> validationpipe(256);
    auto vf = [&validationpipe,&validationset,&seed](time_t seed_shift) {
        dlib::rand rnd(seed+seed_shift);
        cv::RNG cvrng(seed+seed_shift);
        std::vector<float> lbls;
        matrix<rgb_pixel> img;
        while(validationpipe.is_enabled()) {
            load_image(validationset[rnd.get_random_32bit_number() % validationset.size()],img,lbls,rnd,cvrng,false);
            validationpipe.enqueue(std::make_pair(lbls,img));
        }
    };
    std::thread validation_data_loader1([vf](){ vf(1); });
    if(validationset.size() == 0) {
        validationpipe.disable();
        validation_data_loader1.join();
    }

    std::vector<matrix<float>> trainlbls, validlbls;
    std::vector<matrix<rgb_pixel>> trainimgs, validimgs;
    std::pair<std::vector<float>,matrix<rgb_pixel>> tmp;
    while (trainer.get_learning_rate() >= cmdp.get<double>("minlrthresh") ) {

        trainlbls.clear();
        trainimgs.clear();
        while(trainlbls.size() < minibatchsize) {
            trainingpipe.dequeue(tmp);
            trainlbls.push_back(dlib::mat(tmp.first));
            trainimgs.push_back(std::move(tmp.second));
        }
        trainer.train_one_step(trainimgs,trainlbls);
        if((validationset.size() > 0) && ((trainer.get_train_one_step_calls() % 10) == 0)) {
            validlbls.clear();
            validimgs.clear();
            while(validimgs.size() < minibatchsize) {
                validationpipe.dequeue(tmp);
                validlbls.push_back(dlib::mat(tmp.first));
                validimgs.push_back(std::move(tmp.second));
            }
            trainer.test_one_step(validimgs,validlbls);

        }
        if((trainer.get_train_one_step_calls() % 100) == 0)
            qInfo(" #%llu - lr: %f,  loss: %f / %f",
                  trainer.get_train_one_step_calls(),
                  trainer.get_learning_rate(),
                  trainer.get_average_loss(),
                  trainer.get_average_test_loss());
    }

    trainingpipe.disable();
    training_data_loader1.join();
    training_data_loader2.join();
    training_data_loader3.join();
    training_data_loader4.join();
    validationpipe.disable();
    validation_data_loader1.join();
    trainer.get_net();
    net.clean();
    qInfo("Training has been accomplished");

    anet_type anet = net;
    cv::RNG cvrng;
    cv::namedWindow("prediction",cv::WINDOW_NORMAL);
    QElapsedTimer qet;
    std::vector<float> dv;
    dv.reserve(136*validationset.size());

    qInfo("MAE test on whole validation set:");
    for(const auto &instance : validationset) {
        matrix<rgb_pixel> img;
        std::vector<float> lbls;
        load_image(instance,img,lbls,rnd,cvrng,false);
        qet.start();
        matrix<float> prediction = anet(img);
        qInfo("prediction time: %f us",(qet.nsecsElapsed()/1000.0));
        for(size_t i = 0; i < lbls.size(); ++i)
            dv.push_back(std::abs(lbls[i]-prediction(i)));

        /*cv::Mat _tmpmat = dlibmatrix2cvmat(img);
        for(size_t i = 0; i < lbls.size()/2; ++i) {
            const cv::Point2f truepoint((lbls[2*i]+0.5f)*_tmpmat.cols,(lbls[2*i+1]+0.5f)*_tmpmat.rows);
            const cv::Point2f predpoint((prediction(2*i)+0.5f)*_tmpmat.cols,(prediction(2*i+1)+0.5f)*_tmpmat.rows);
            cv::circle(_tmpmat,truepoint,1,cv::Scalar(0,255,0),1,cv::LINE_AA);
            cv::circle(_tmpmat,truepoint,3,cv::Scalar(0,125,255),1,cv::LINE_AA);
        }
        cv::imshow("prediction",_tmpmat);
        cv::waitKey(0);*/
    }
    qInfo("-------------");
    const float mean_err = mean(dv), stdev_err = stdev(dv);
    qInfo("Mean abs err: %.3f ± %.3f", mean_err, 2.0f*stdev_err);

    qInfo("Serialization...");
    serialize(cmdp.get<std::string>("outputdir") +
              std::string("/FaceLandmarks_net_mae_") +
              std::to_string(mean_err) +
              std::string("_stdev_") +
              std::to_string(stdev_err) +
              std::string(".dat")) << net;
    qInfo("Done");

    return 0;
}
