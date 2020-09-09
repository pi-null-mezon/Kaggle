#include <QElapsedTimer>
#include <QRegularExpression>
#include <QStringList>
#include <QFile>
#include <QDir>

#include <QDebug>

#include "opencvimgaugment.h"
#include "dlibopencvconverter.h"

#include <opencv2/highgui.hpp>

#include "customnetwork.h"

const std::string options = "{traindir t  |       | - directory with training images}"
                            "{outputdir o |  .    | - output directory}"
                            "{mbs         |  64   | - mini batch size}"
                            "{seed        |  7    | - random number generator seed value}"
                            "{split       |  0.2  | - validation portion of data}"
                            "{learningrate|       | - learning rate}"
                            "{viwp        | 1000  | - validation iterations without progress}"
                            "{tiwp        | 10000 | - training iterations without progress}"
                            "{taug        | true  | - training time augmentation}"
                            "{minlrthresh | 1E-5  | - when learning should be stopped}"
                            "{help h      |       | - help}";

struct HeadPose {
    HeadPose() {}
    std::string filename;
    std::vector<float> angles; // two angles (relative to 90.0), first is pan, second is tilt
};


void flip_lbls_horizontally(std::vector<float> &_angles)
{
    // only tilt should be modified
    _angles[1] *= -1;
}


void load_image(const HeadPose &headpose, matrix<rgb_pixel> &img, std::vector<float> &labels, dlib::rand &rnd, cv::RNG &cvrng, bool augment=false)
{
    bool loaded_sucessfully = false;
    cv::Mat _tmpmat = loadIbgrmatWsize(headpose.filename,IMG_WIDTH,IMG_HEIGHT,false,&loaded_sucessfully);
    DLIB_CASSERT(loaded_sucessfully,"Can not read image!"); // TO DO understand in witch mode this macro works as it used to
    if(augment) {

        auto _tmplbls = headpose.angles;
        if(rnd.get_random_float() > 0.5f) {
            cv::flip(_tmpmat,_tmpmat,1);
            flip_lbls_horizontally(_tmplbls);
        }

        /*if(rnd.get_random_float() > 0.5f)
            _tmpmat = jitterimage(_tmpmat,cvrng,cv::Size(0,0),0.03,0.03,0,cv::BORDER_REFLECT,cv::Scalar(0),false);
        if(rnd.get_random_float() > 0.5f)
            _tmpmat = cutoutRect(_tmpmat,rnd.get_random_float(),rnd.get_random_float(),0.4f,0.4f,rnd.get_random_float()*180.0f);*/

        if(rnd.get_random_float() > 0.5f)
            cv::blur(_tmpmat,_tmpmat,cv::Size(3,3));
        if(rnd.get_random_float() > 0.1f)
            _tmpmat *= (0.8 + 0.4*rnd.get_random_double());
        if(rnd.get_random_float() > 0.1f)
            _tmpmat = addNoise(_tmpmat,cvrng,0,7);        

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
        labels = headpose.angles;
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

std::vector<float> extract_pitch_and_yaw(const QString &_filename)
{
    std::vector<float> headangles(2,0.f);
    if(_filename.contains('}')) {
        QStringList sections = _filename.section('}',1,1).section(".jpg",0,0).split(QRegularExpression("(\\+|-)"),QString::SkipEmptyParts);
        headangles[0] = QString("%1%2").arg(QString(_filename.section(sections.at(0),0,0).back()),sections.at(0)).toFloat() / 90.0f;
        headangles[1] = QString("%1%2").arg(QString(_filename.section(sections.at(1),0,0).back()),sections.at(1)).toFloat() / 90.0f;
    } else {
        QStringList sections = _filename.split(QRegularExpression("\\b"));
        headangles[0] = QString("%1%2").arg(sections.at(2),sections.at(3)).toFloat() / 90.0f;
        headangles[1] = QString("%1%2").arg(sections.at(4),sections.at(5)).toFloat() / 90.0f;
    }
    return headangles;
}

int main(int argc, char *argv[])
{
    cv::CommandLineParser cmdp(argc,argv,options);
    cmdp.about("Tool for head pose (pan and tilt angles) regressor training");
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
    std::vector<HeadPose> hposes;
    hposes.reserve(1E4);
    qInfo("Reading training data, please wait...");
    auto files = traindir.entryList(QStringList() << "*.jpg" << "*.jpeg" << "*.png", QDir::Files | QDir::NoDotDot);
    for(const auto &_filename: files) {
        HeadPose _headpose;
        _headpose.filename = traindir.absoluteFilePath(_filename).toStdString();
        _headpose.angles = extract_pitch_and_yaw(_filename);
        //qDebug() << _filename;
        //qDebug() << _headpose.angles;
        hposes.push_back(std::move(_headpose));
    }
    qInfo(" total: %lu - training instances has been found",hposes.size());

    const int seed = cmdp.get<int>("seed");
    const double split = cmdp.get<double>("split");
    const size_t minibatchsize = static_cast<size_t>(cmdp.get<uint>("mbs"));
    qInfo(" \nminibatch size: %u", (unsigned int)minibatchsize);

    dlib::rand rnd(seed);
    std::vector<HeadPose> trainingset, validationset;
    trainingset.reserve(hposes.size());
    validationset.reserve(hposes.size());
    for(size_t i = 0; i < hposes.size(); ++i) {
        if(rnd.get_random_double() < split)
            validationset.push_back(std::move(hposes[i]));
        else
            trainingset.push_back(std::move(hposes[i]));
    }
    hposes.clear();
    hposes.shrink_to_fit();
    qInfo(" training: %lu",trainingset.size());
    qInfo(" validation: %lu",validationset.size());

    //--------------------------------------------------------------------------------
    // DEBUGGING
    /*matrix<rgb_pixel> img;
    std::vector<float> lbls;
    cv::RNG _cvrng;
    for(const auto &instance : trainingset) {
        cv::Mat original = cv::imread(instance.filename);
        cv::putText(original,
                    QString("%1; %2").arg(QString::number(instance.angles[0]*90.0f,'f',1),
                                          QString::number(instance.angles[1]*90.0f,'f',1)).toStdString(),
                    cv::Point(5,15),cv::FONT_HERSHEY_SIMPLEX,0.5,cv::Scalar(125,255,125),1,cv::LINE_AA);

        load_image(instance,img,lbls,rnd,_cvrng,true);
        cv::Mat augmented = dlibmatrix2cvmat(img);
        cv::putText(augmented,
                    QString("%1; %2").arg(QString::number(lbls[0]*90.0f,'f',1),
                                          QString::number(lbls[1]*90.0f,'f',1)).toStdString(),
                    cv::Point(5,15),cv::FONT_HERSHEY_SIMPLEX,0.5,cv::Scalar(125,255,125),1,cv::LINE_AA);
        cv::imshow("augmented",augmented);
        cv::imshow("train",original);
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
    std::vector<float> dpan, dtilt;    
    dtilt.reserve(validationset.size()); // also known as pitch
    dpan.reserve(validationset.size());  // also known as yaw
    qInfo("MAE test on whole validation set:");
    for(const auto &instance : validationset) {
        matrix<rgb_pixel> img;
        std::vector<float> lbls;
        load_image(instance,img,lbls,rnd,cvrng,false);
        qet.start();
        matrix<float> prediction = anet(img);
        qInfo("prediction time: %f us",(qet.nsecsElapsed()/1000.0));
        dtilt.push_back(lbls[0]-prediction(0));
        dpan.push_back(lbls[1]-prediction(1));
        /*cv::Mat _tmpmat = dlibmatrix2cvmat(img);
        cv::putText(_tmpmat,
                    QString("%1; %2").arg(QString::number(lbls[0]*90.0f,'f',1),QString::number(lbls[1]*90.0f,'f',1)).toStdString(),
                    cv::Point(5,15),cv::FONT_HERSHEY_SIMPLEX,0.5,cv::Scalar(125,255,125),1,cv::LINE_AA);
        cv::putText(_tmpmat,
                    QString("%1; %2").arg(QString::number(prediction(0)*90.0f,'f',1),QString::number(prediction(1)*90.0f,'f',1)).toStdString(),
                    cv::Point(5,35),cv::FONT_HERSHEY_SIMPLEX,0.5,cv::Scalar(125,125,255),1,cv::LINE_AA);
        cv::imshow("prediction",_tmpmat);
        cv::waitKey(0);*/
    }
    qInfo("-------------");
    const float mean_err_pan = 90.0f*mean(dpan), stdev_err_pan = 90.0f*stdev(dpan);
    qInfo("Pan(Yaw): %.2f ± %.2f deg", mean_err_pan, 2.0f*stdev_err_pan);
    const float mean_err_tilt = 90.0f*mean(dtilt), stdev_err_tilt = 90.0f*stdev(dtilt);
    qInfo("Tilt(Pitch): %.2f ± %.2f deg", mean_err_tilt, 2.0f*stdev_err_tilt);

    qInfo("Serialization...");
    serialize(cmdp.get<std::string>("outputdir") +
              std::string("/headpose_net_mae_") +
              std::to_string((mean_err_pan + mean_err_tilt)/2) +
              std::string("_2stdev_") +
              std::to_string((stdev_err_pan + stdev_err_tilt)) +
              std::string(".dat")) << net;
    qInfo("Done");

    return 0;
}
