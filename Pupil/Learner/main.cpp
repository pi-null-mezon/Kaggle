#include <QElapsedTimer>
#include <QStringList>
#include <QFile>
#include <QDir>

#include "opencvimgaugment.h"
#include "dlibopencvconverter.h"

#include <opencv2/highgui.hpp>

#include "customnetwork.h"

const std::string options = "{traindir t  |       | - directory with training images and markup file}"
                            "{markupfile  | QImageMarker.csv | - name of the markup file}"
                            "{outputdir o |  .    | - output directory}"
                            "{mbs         |  16   | - mini batch size}"
                            "{seed        |  7    | - random number generator seed value}"
                            "{split       |  0.2  | - validation portion of data}"
                            "{learningrate|       | - learning rate}"
                            "{viwp        | 200   | - validation iterations without progress}"
                            "{tiwp        | 10000 | - training iterations without progress}"
                            "{taug        | true  | - training time augmentation}"
                            "{minlrthresh | 1E-5  | - when learning should be stopped}"
                            "{help h      |       | - help}";

struct Pupil {
    Pupil() {}
    std::string filename;
    std::vector<float> points;
};

void drawCircle(cv::Mat &_tmpmat, const std::vector<float> &points, const cv::Scalar &color)
{
    cv::circle(_tmpmat,cv::Point2f(_tmpmat.cols*points[0],_tmpmat.rows*points[1]),
              (_tmpmat.cols+_tmpmat.rows)*points[2]/4.0f,color,1,cv::LINE_AA);
}

void flip_lbls_horizontally(std::vector<float> &points)
{
    points[0] = 1.0f - points[0];
}

void flip_lbls_vertically(std::vector<float> &points)
{
    points[1] = 1.0f - points[1];
}

void apply_random_clip(cv::Mat &img, std::vector<float> &lbls,float minportion, float maxportion, dlib::rand &rnd)
{
    const int size = (rnd.get_random_float()*(maxportion-minportion) + minportion)*img.cols;
    const int shiftx = rnd.get_random_32bit_number() % (img.cols - size);
    const int shifty = rnd.get_random_32bit_number() % (img.rows - size);

    cv::Mat _tmpmat;
    cv::resize(img(cv::Rect(shiftx,shifty,size,size)),_tmpmat,cv::Size(img.cols,img.rows),0,0,cv::INTER_CUBIC);
    lbls[0] = (lbls[0]*img.cols - static_cast<float>(shiftx)) / size;
    lbls[1] = (lbls[1]*img.rows - static_cast<float>(shifty)) / size;
    lbls[2] = lbls[2] * img.cols / size;
    img= _tmpmat;
}

void apply_central_rotation(cv::Mat &img, std::vector<float> &lbls, float maxangle, dlib::rand &rnd)
{
    cv::Mat rm = cv::getRotationMatrix2D(cv::Point2f(img.cols*lbls[0],img.rows*lbls[1]),rnd.get_random_float()*maxangle,1);
    int border = cv::BORDER_CONSTANT;
    if(rnd.get_random_float() > 0.5f)
        border = cv::BORDER_REFLECT;
    cv::warpAffine(img,img,rm,cv::Size(img.cols,img.rows),cv::INTER_CUBIC,border,cv::mean(img));
}

void load_image(const Pupil &pupil, matrix<rgb_pixel> &img, std::vector<float> &labels, dlib::rand &rnd, cv::RNG &cvrng, bool augment=false)
{
    bool loaded_sucessfully = false;
    cv::Mat _tmpmat = loadIbgrmatWsize(pupil.filename,IMG_WIDTH,IMG_HEIGHT,false,&loaded_sucessfully);
    DLIB_CASSERT(loaded_sucessfully,"Can not read image!"); // TO DO understand in witch mode this macro works as it used to
    if(augment) {
        std::vector<float> _tmplbls = pupil.points;

        /*drawCircle(_tmpmat,pupil.points,cv::Scalar(0,255,0,100));
        cv::imshow("orig",_tmpmat);*/

        if(rnd.get_random_float() > 0.5f)
            _tmpmat = cutoutRect(_tmpmat,rnd.get_random_float(),rnd.get_random_float(),0.3f,0.3f,rnd.get_random_float()*180.0f);
        if(rnd.get_random_float() > 0.5f)
            cv::blur(_tmpmat,_tmpmat,cv::Size(3,3));
        if(rnd.get_random_float() > 0.1f)
            _tmpmat *= (0.8 + 0.4*rnd.get_random_double());
        if(rnd.get_random_float() > 0.1f)
            _tmpmat = addNoise(_tmpmat,cvrng,0,7);
        if(rnd.get_random_float() > 0.5f) {
            cv::flip(_tmpmat,_tmpmat,1);
            flip_lbls_horizontally(_tmplbls);
        }
        if(rnd.get_random_float() > 0.5f) {
            cv::flip(_tmpmat,_tmpmat,0);
            flip_lbls_vertically(_tmplbls);
        }
        if(rnd.get_random_float() > 0.5f)
            apply_central_rotation(_tmpmat,_tmplbls,30,rnd);
        if(rnd.get_random_float() > 0.5)
            apply_random_clip(_tmpmat,_tmplbls,0.75f,1.0f,rnd);

        if(rnd.get_random_float() > 0.5f) {
            cv::cvtColor(_tmpmat,_tmpmat,cv::COLOR_BGR2GRAY);
            cv::Mat _chmat[] = {_tmpmat, _tmpmat, _tmpmat};
            cv::merge(_chmat,3,_tmpmat);
        }

        /*drawCircle(_tmpmat,_tmplbls,cv::Scalar(0,0,255));
        cv::imshow("aug",_tmpmat);
        cv::waitKey(0);*/

        img = cvmat2dlibmatrix<dlib::rgb_pixel>(_tmpmat);
        dlib::disturb_colors(img,rnd);
        labels = _tmplbls;
    } else {
        img = cvmat2dlibmatrix<dlib::rgb_pixel>(_tmpmat);
        labels = pupil.points;
    }
}

int main(int argc, char *argv[])
{
    cv::CommandLineParser cmdp(argc,argv,options);
    cmdp.about("Tool for the pupil position and diameter measurements convolutional network training");
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
    const QString filename(traindir.absoluteFilePath(cmdp.get<std::string>("markupfile").c_str()));
    QFile markupfile(filename);
    if(!markupfile.exists()) {
        qInfo("Markup file '%s' not found! Abort...", filename.toUtf8().constData());
        return 3;
    }
    //-------------------------------------------------------------------------------
    std::vector<Pupil> pupils;
    pupils.reserve(4096);
    qInfo("Reading training data, please wait...");
    markupfile.open(QIODevice::ReadOnly);
    while (!markupfile.atEnd()) {
        QString line = markupfile.readLine().simplified();
        QStringList parts = line.split(',');
        if(parts.size() == 5) {
            Pupil pupil;
            pupil.filename = traindir.absoluteFilePath(parts.at(0)).toStdString();
            if(QFileInfo(pupil.filename.c_str()).exists()) {
                std::vector<float> values(3,0);
                values[0] = (parts.at(1).toFloat() + parts.at(3).toFloat()) / 2.0f;
                values[1] = (parts.at(2).toFloat() + parts.at(4).toFloat()) / 2.0f;
                values[2] = (std::abs(parts.at(3).toFloat() - parts.at(1).toFloat()) +
                             std::abs(parts.at(4).toFloat() - parts.at(2).toFloat())) / 2.0f;
                pupil.points = std::move(values);
                pupils.push_back(std::move(pupil));
            }
        } else {
            qInfo("Line '%s' - insufficient data! Abort...", line.toUtf8().constData());
            return 4;
        }
    }
    qInfo(" total: %lu - training instances has been found",pupils.size());

    const int seed = cmdp.get<int>("seed");
    const double split = cmdp.get<double>("split");
    const size_t minibatchsize = static_cast<size_t>(cmdp.get<uint>("mbs"));
    qInfo(" \nminibatch size: %u", (unsigned int)minibatchsize);

    dlib::rand rnd(seed);
    std::vector<Pupil> trainingset, validationset;
    trainingset.reserve(pupils.size());
    validationset.reserve(pupils.size());
    for(size_t i = 0; i < pupils.size(); ++i) {
        if(rnd.get_random_double() < split)
            validationset.push_back(std::move(pupils[i]));
        else
            trainingset.push_back(std::move(pupils[i]));
    }
    pupils.clear();
    pupils.shrink_to_fit();
    qInfo(" training: %lu",trainingset.size());
    qInfo(" validation: %lu\n",validationset.size());

    //--------------------------------------------------------------------------------
    // DEBUGGING
    /*matrix<rgb_pixel> img;
    std::vector<float> lbls;
    cv::RNG cvrng;
    cv::namedWindow("orig",cv::WINDOW_NORMAL);
    cv::namedWindow("aug",cv::WINDOW_NORMAL);
    for(const auto &instance : trainingset)
        load_image(instance,img,lbls,rnd,cvrng,true);*/

    //--------------------------------------------------------------------------------
    net_type net;
    dnn_trainer<net_type> trainer(net,sgd());
    trainer.set_learning_rate(0.01);
    trainer.be_verbose();
    trainer.set_synchronization_file(cmdp.get<string>("outputdir") + string("/trainer_sync") , std::chrono::minutes(2));
    if(cmdp.has("learningrate"))
        trainer.set_learning_rate(cmdp.get<double>("learningrate"));
    if(validationset.size() > 0)
        trainer.set_test_iterations_without_progress_threshold(static_cast<size_t>(cmdp.get<int>("viwp")));
    else
        trainer.set_iterations_without_progress_threshold(static_cast<size_t>(cmdp.get<int>("tiwp")));


    bool training_time_augmentation = cmdp.get<bool>("taug");
    dlib::pipe<std::pair<std::vector<float>,matrix<rgb_pixel>>> trainingpipe(256);
    auto f = [&trainingpipe,&trainingset,&seed,&training_time_augmentation](time_t seed_shift) {
        dlib::rand rnd(seed+seed_shift);
        cv::RNG cvrng(seed+seed_shift);
        Pupil pupil;
        std::vector<float> lbls;
        matrix<rgb_pixel> img;
        while(trainingpipe.is_enabled()) {
            pupil = trainingset[rnd.get_random_32bit_number() % trainingset.size()];
            load_image(pupil,img,lbls,rnd,cvrng,training_time_augmentation);
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
        Pupil pupil;
        std::vector<float> lbls;
        matrix<rgb_pixel> img;
        while(validationpipe.is_enabled()) {
            pupil = validationset[rnd.get_random_32bit_number() % validationset.size()];
            load_image(pupil,img,lbls,rnd,cvrng,false);
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
    serialize(cmdp.get<std::string>("outputdir") +
              std::string("/pupil_net_vl_") +
              std::to_string(trainer.get_average_test_loss()) +
              std::string(".dat")) << net;

    anet_type anet = net;
    cv::RNG cvrng;
    cv::namedWindow("prediction",cv::WINDOW_NORMAL);
    QElapsedTimer qet;
    for(const auto &instance : validationset) {
        matrix<rgb_pixel> img;
        std::vector<float> lbls;
        load_image(instance,img,lbls,rnd,cvrng,false);
        qet.start();
        matrix<float> prediction = anet(img);
        qInfo("prediction time: %f us",(qet.nsecsElapsed()/1000.0));
        std::vector<float> points(prediction.begin(),prediction.end());
        cv::Mat _tmpmat = dlibmatrix2cvmat(img);
        drawCircle(_tmpmat,points,cv::Scalar(255,0,0));
        cv::imshow("prediction",_tmpmat);
        cv::waitKey(0);
    }

    return 0;
}
