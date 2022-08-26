#include <QElapsedTimer>
#include <QJsonDocument>
#include <QJsonObject>
#include <QStringList>
#include <QFile>
#include <QDir>

#include <QDebug>

#include "opencvimgaugment.h"
#include "dlibopencvconverter.h"

#include <opencv2/highgui.hpp>

#include "customnetwork.h"

const int min_value = 1;
const int max_value = 3;

using namespace dlib;

const std::string options = "{traindir t  |       | - directory with training images}"
                            "{outputdir o |  .    | - output directory}"
                            "{mbs         |  128  | - mini batch size}"
                            "{seed        |  7    | - random number generator seed value}"
                            "{split       |  0.1  | - validation portion of data}"
                            "{learningrate|       | - learning rate}"
                            "{viwp        | 1000  | - validation iterations without progress}"
                            "{tiwp        | 10000 | - training iterations without progress}"
                            "{taug        | true  | - training time augmentation}"
                            "{minlrthresh | 1E-5  | - when learning should be stopped}"
                            "{help h      |       | - help}";


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


void load_image(const std::string &filename, matrix<rgb_pixel> &img, float &label, dlib::rand &rnd, cv::RNG &cvrng, bool augment=false)
{
    bool loaded_sucessfully = false;
    cv::Mat _tmpmat = loadIbgrmatWsize(filename,IMG_WIDTH,IMG_HEIGHT,false,&loaded_sucessfully);
    DLIB_CASSERT(loaded_sucessfully,"Can not read image!"); // TO DO understand in witch mode this macro works as it used to

    if(augment) {
        if(rnd.get_random_float() > 0.5f)
            cv::flip(_tmpmat,_tmpmat,1);
        if(rnd.get_random_float() > 0.5f)
            _tmpmat = jitterimage(_tmpmat,cvrng,cv::Size(0,0),0.05,0.05,5,cv::BORDER_REFLECT,cv::Scalar(0),false);

        _tmpmat *= (0.7 + 0.6*rnd.get_random_double());

        /*float rf = 0.85f + 0.15f*rnd.get_random_float();
        cv::resize(_tmpmat,_tmpmat,cv::Size(),rf,rf);*/
    }

    int power = rnd.get_integer_in_range(2, max_value + 1);
    if(rnd.get_random_float() < 0.5f) {
        _tmpmat = applyMotionBlur(_tmpmat,90.0f*rnd.get_random_float(),power);
        _tmpmat = applyMotionBlur(_tmpmat,90.0f*rnd.get_random_float(),power);
    } else {
        power = min_value;
    }
    /*switch(rnd.get_integer_in_range(0,4)) {
        case 0:
            cv::blur(_tmpmat,_tmpmat,cv::Size(power,power));
            break;
        case 1:
            _tmpmat = applyMotionBlur(_tmpmat,90.0f*rnd.get_random_float(),power);
            _tmpmat = applyMotionBlur(_tmpmat,90.0f*rnd.get_random_float(),power);
            break;
        case 2: {
            int size = power % 2 == 1 ? power : power + 1;
            cv::GaussianBlur(_tmpmat,_tmpmat,cv::Size(size,size),power);
        } break;
        default:
            power = min_value;
            break;
    }*/

    if(augment) {
        _tmpmat = addNoise(_tmpmat,cvrng,0,rnd.get_integer_in_range(1,8));

        if(rnd.get_random_float() > 0.5f) {
            cv::cvtColor(_tmpmat,_tmpmat,cv::COLOR_BGR2GRAY);
            cv::Mat _chmat[] = {_tmpmat, _tmpmat, _tmpmat};
            cv::merge(_chmat,3,_tmpmat);
        }

        std::vector<unsigned char> _bytes;
        std::vector<int> compression_params;
        compression_params.push_back(cv::IMWRITE_JPEG_QUALITY);
        compression_params.push_back(static_cast<int>(rnd.get_integer_in_range(50,100)));
        cv::imencode("*.jpg",_tmpmat,_bytes,compression_params);
        _tmpmat = cv::imdecode(_bytes,cv::IMREAD_UNCHANGED);

        if(_tmpmat.cols != IMG_WIDTH)
            cv::resize(_tmpmat,_tmpmat,cv::Size(IMG_WIDTH,IMG_HEIGHT));
        /*if(rnd.get_random_float() > 0.5f)
            _tmpmat = cutoutRect(_tmpmat,rnd.get_random_float(),rnd.get_random_float(),0.5f,0.5f,rnd.get_random_float()*180.0f);*/

        img = cvmat2dlibmatrix<dlib::rgb_pixel>(_tmpmat);
        dlib::disturb_colors(img,rnd);       
    } else {
        img = cvmat2dlibmatrix<dlib::rgb_pixel>(_tmpmat);
    }
    label = ((float)(power - min_value) / (max_value - min_value)) - 0.5f;
}

int main(int argc, char *argv[])
{
    cv::CommandLineParser cmdp(argc,argv,options);
    cmdp.about("Tool for head pose regressor training");
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
    std::vector<std::string> filenames;
    filenames.reserve(1E4);
    qInfo("Reading training data, please wait...");
    const auto files = traindir.entryList(QStringList() << "*.jpg" << "*.jpeg" << "*.png", QDir::Files | QDir::NoDotDot);
    for(const auto &_filename: files)
        filenames.emplace_back(traindir.absoluteFilePath(_filename).toStdString());
    qInfo(" total: %lu - training instances has been found",filenames.size());

    const int seed = cmdp.get<int>("seed");
    const double split = cmdp.get<double>("split");
    const size_t minibatchsize = static_cast<size_t>(cmdp.get<uint>("mbs"));
    qInfo(" \nminibatch size: %u", (unsigned int)minibatchsize);

    dlib::rand rnd(seed);
    std::vector<std::string> trainingset, validationset;
    trainingset.reserve(filenames.size());
    validationset.reserve(filenames.size());
    for(size_t i = 0; i < filenames.size(); ++i) {
        if(rnd.get_random_double() < split)
            validationset.push_back(std::move(filenames[i]));
        else
            trainingset.push_back(std::move(filenames[i]));
    }
    filenames.clear();
    filenames.shrink_to_fit();
    qInfo(" training: %lu",trainingset.size());
    qInfo(" validation: %lu",validationset.size());

    //--------------------------------------------------------------------------------
    // DEBUGGING
    /*matrix<rgb_pixel> img;
    float lbl;
    cv::RNG _cvrng;
    for(const auto &instance : trainingset) {
        load_image(instance,img,lbl,rnd,_cvrng,true);
        cv::Mat augmented = dlibmatrix2cvmat(img);
        cv::putText(augmented,
                    QString("%1").arg(QString::number(lbl,'f',1)).toStdString(),
                    cv::Point(5,15),cv::FONT_HERSHEY_SIMPLEX,0.5,cv::Scalar(125,255,125),1,cv::LINE_AA);
        cv::imshow("augmented",augmented);
        cv::waitKey(0);
    }*/
    //--------------------------------------------------------------------------------
    net_type net;
    dnn_trainer<net_type> trainer(net,sgd(0.0001, 0.9));
    //trainer.set_learning_rate(0.01);
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
    dlib::pipe<std::pair<float,matrix<rgb_pixel>>> trainingpipe(256);
    auto f = [&trainingpipe,&trainingset,&seed,&training_time_augmentation](time_t seed_shift) {
        dlib::rand rnd(seed+seed_shift);
        cv::RNG cvrng(seed+seed_shift);
        float lbl;
        matrix<rgb_pixel> img;
        while(trainingpipe.is_enabled()) {
            load_image(trainingset[rnd.get_random_32bit_number() % trainingset.size()],img,lbl,rnd,cvrng,training_time_augmentation);
            trainingpipe.enqueue(std::make_pair(lbl,img));
        }
    };
    std::thread training_data_loader1([f](){ f(1); });
    std::thread training_data_loader2([f](){ f(2); });
    std::thread training_data_loader3([f](){ f(3); });
    std::thread training_data_loader4([f](){ f(4); });

    dlib::pipe<std::pair<float,matrix<rgb_pixel>>> validationpipe(256);
    auto vf = [&validationpipe,&validationset,&seed](time_t seed_shift) {
        dlib::rand rnd(seed+seed_shift);
        cv::RNG cvrng(seed+seed_shift);
        float lbl;
        matrix<rgb_pixel> img;
        while(validationpipe.is_enabled()) {
            load_image(validationset[rnd.get_random_32bit_number() % validationset.size()],img,lbl,rnd,cvrng,false);
            validationpipe.enqueue(std::make_pair(lbl,img));
        }
    };
    std::thread validation_data_loader1([vf](){ vf(1); });
    if(validationset.size() == 0) {
        validationpipe.disable();
        validation_data_loader1.join();
    }

    std::vector<float> trainlbls, validlbls;
    std::vector<matrix<rgb_pixel>> trainimgs, validimgs;
    std::pair<float,matrix<rgb_pixel>> tmp;
    while (trainer.get_learning_rate() >= cmdp.get<double>("minlrthresh") ) {

        trainlbls.clear();
        trainimgs.clear();
        while(trainlbls.size() < minibatchsize) {
            trainingpipe.dequeue(tmp);
            trainlbls.push_back(tmp.first);
            trainimgs.push_back(std::move(tmp.second));
        }
        trainer.train_one_step(trainimgs,trainlbls);
        if((validationset.size() > 0) && ((trainer.get_train_one_step_calls() % 10) == 0)) {
            validlbls.clear();
            validimgs.clear();
            while(validimgs.size() < minibatchsize) {
                validationpipe.dequeue(tmp);
                validlbls.push_back(tmp.first);
                validimgs.push_back(std::move(tmp.second));
            }
            trainer.test_one_step(validimgs,validlbls);

        }
        if((trainer.get_train_one_step_calls() % 200) == 0)
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
    std::vector<float> timens, differences;
    timens.reserve(validationset.size());
    differences.reserve(validationset.size());
    qInfo("MAE test on whole validation set:");
    bool firstinference = true;
    for(const auto &instance : validationset) {
        matrix<rgb_pixel> img;
        float lbl;
        load_image(instance,img,lbl,rnd,cvrng,false);
        qet.start();
        float prediction = anet(img);
        qInfo("%.2f vs %.2f", lbl, prediction);
        if(firstinference == false)
            timens.push_back(qet.nsecsElapsed());
        else
            firstinference = false;
        differences.push_back(prediction - lbl);
    }
    qInfo("Average inference time: %f us",mean(timens)/1000.0f);
    qInfo("-------------");
    const float mean_err = (max_value - min_value)*mean(differences), stdev_err = (max_value - min_value)*stdev(differences);
    qInfo("Difference: %.1f ± %.1f", mean_err, 2*stdev_err);
    qInfo("Serialization...");
    serialize(cmdp.get<std::string>("outputdir") +
              std::string("/blur_net_mae_") +
              std::to_string(mean_err) +
              std::string("_stdev_") +
              std::to_string(stdev_err) +
              std::string(".dat")) << net;
    qInfo("Done");

    return 0;
}
