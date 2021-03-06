#include <iostream>
#include <dlib/data_io.h>
#include <dlib/dir_nav.h>
#include <iterator>
#include <thread>

#include <dlib/opencv.h>
#include <opencv2/opencv.hpp>

#include "dlibimgaugment.h"
#include "opencvimgaugment.h"

#include "customnetwork.h"

using namespace std;
using namespace dlib;

// Load list of all file names
std::vector<std::vector<string>> load_objects_list (const string& dir);

// This function takes the output of load_objects_list() as input and randomly
// selects images for training.  It should also be pointed out that it's really
// important that each mini-batch contain multiple images of each calss.  This
// is because the metric learning algorithm needs to consider pairs of images
// that should be close (i.e. images of the same calss) as well as pairs of
// images that should be far apart (i.e. images of different classees) during each
// training step.
void load_mini_batch (
    const size_t num_classees,     // how many different classees to include
    const size_t samples_per_id,   // how many images per calss to select.
    dlib::rand& rnd,
    cv::RNG& cvrng,
    const std::vector<std::vector<string>>& objs,
    std::vector<matrix<rgb_pixel>>& images,
    std::vector<unsigned long>& labels,
    bool _applyaugmentation
)
{
    DLIB_CASSERT(num_classees <= objs.size(), "The dataset doesn't have that many classees in it.");
    images.clear();
    images.reserve(num_classees*samples_per_id);
    labels.clear();
    labels.reserve(num_classees*samples_per_id);

    std::vector<bool> already_selected(objs.size(), false);
    for (size_t i = 0; i < num_classees; ++i)
    {
        size_t id = rnd.get_random_32bit_number()%objs.size();
        // don't pick a calss we already added to the mini-batch
        while(already_selected[id])
            id = rnd.get_random_32bit_number()%objs.size();
        already_selected[id] = true;

        for (size_t j = 0; j < samples_per_id; ++j)
        {
            string obj;
            if(objs[id].size() == samples_per_id) {
                obj = objs[id][j];
            } else {
                obj = objs[id][rnd.get_random_32bit_number()%objs[id].size()];
            }

            if(_applyaugmentation) { // You might want to do some data augmentation at this point
                cv::Mat _tmpmat = loadIbgrmatWsize(obj,400,300,true);
                if(rnd.get_random_double() > 0.5)
                    _tmpmat = jitterimage(_tmpmat,cvrng,cv::Size(0,0),0.13,0.07,13,cv::BORDER_REPLICATE);
                else
                    _tmpmat = jitterimage(_tmpmat,cvrng,cv::Size(0,0),0.13,0.07,13,cv::BORDER_REFLECT);

                if(rnd.get_random_float() > 0.5) {
                    _tmpmat = distortimage(_tmpmat,cvrng,0.2,CV_INTER_CUBIC,cv::BORDER_REPLICATE);
                } else {
                    _tmpmat = distortimage(_tmpmat,cvrng,0.2,CV_INTER_CUBIC,cv::BORDER_REFLECT);
                }
                if(rnd.get_random_float() > 0.1f) {
                    _tmpmat = cutoutRect(_tmpmat,rnd.get_random_float(),0);
                    _tmpmat = cutoutRect(_tmpmat,rnd.get_random_float(),1);
                }

                /*if(rnd.get_random_float() > 0.1f)
                    _tmpmat = cutoutRect(_tmpmat,rnd.get_random_float(),rnd.get_random_float(),0.3f);*/

                /*if(rnd.get_random_float() > 0.5f)
                    cv::blur(_tmpmat,_tmpmat,cv::Size(3,3));*/

                /*if(rnd.get_random_float() > 0.1f) {
                    cv::Scalar _mean = cv::mean(_tmpmat);
                    for(int y = 0; y < _tmpmat.rows; ++y) {
                        unsigned char *_p = _tmpmat.ptr<unsigned char>(y);
                        for(int x = 0; x < _tmpmat.cols; ++x) {
                            double _m = std::abs(y - _tmpmat.rows/2.0)/(_tmpmat.rows/2.0);
                            _m *= _m;
                            _p[3*x]   = static_cast<unsigned char>(_m*_mean[0] + (1.0 - _m)*_p[3*x]);
                            _p[3*x+1] = static_cast<unsigned char>(_m*_mean[1] + (1.0 - _m)*_p[3*x+1]);
                            _p[3*x+2] = static_cast<unsigned char>(_m*_mean[2] + (1.0 - _m)*_p[3*x+2]);
                        }
                    }
                }*/

                matrix<rgb_pixel> _dlibimgmatrix = cvmat2dlibmatrix<rgb_pixel>(_tmpmat);
                if(rnd.get_random_double() > 0.0) {
                    disturb_colors(_dlibimgmatrix, rnd);
                }
                images.push_back(std::move(_dlibimgmatrix));
            } else {
                images.push_back(load_rgb_image_with_fixed_size(obj,400,300,true));
            }
            labels.push_back(id);
        }
    }

    // All the images going into a mini-batch have to be the same size.  And really, all
    // the images in your entire training dataset should be the same size for what we are
    // doing to make the most sense.
    DLIB_CASSERT(images.size() > 0);
    for (auto&& img : images)
    {
        DLIB_CASSERT(img.nr() == images[0].nr() && img.nc() == images[0].nc(),
            "All the images in a single mini-batch must be the same size.");
    }
}

// ----------------------------------------------------------------------------------------
const cv::String keys =
       "{help h           |        | print this message}"
       "{traindirpath   t |        | training directory location}"
       "{validdirpath   v |        | validation directory location}"
       "{testdirpath      |        | accuracy test directory location}"
       "{outputdirpath  o |        | output directory location}"
       "{number n         |   1    | number of classifiers to be trained}"
       "{swptrain         | 5000   | determines after how many steps without progress (training loss) decay should be applied to learning rate}"
       "{swpvalid         | 1000   | determines after how many steps without progress (test loss) decay should be applied to learning rate}"
       "{minlrthresh      | 1.0e-5 | minimum learning rate, determines when training should be stopped}";
// -----------------------------------------------------------------------------------------

int main(int argc, char** argv) try
{
    cv::CommandLineParser cmdparser(argc, argv, keys);
    cmdparser.about("Application name dnn_classifier_learner");
    if(cmdparser.has("help")) {
        cmdparser.printMessage();
        return 0;
    }
    if(!cmdparser.has("traindirpath")) {
        cout << "You have not provide path to training directory!";
        return 1;
    }
    if(!cmdparser.has("validdirpath")) {
        cout << "You have not provide path to validation directory!";
        return 2;
    }
    if(!cmdparser.has("testdirpath")) {
        cout << "You have not provide path to test directory!";
        return 3;
    }
    if(!cmdparser.has("outputdirpath")) {
        cout << "You have not provide path to output directory!";
        return 4;
    }
    if(cmdparser.get<int>("number") <= 0) {
        cout << "Number of classifiers should be greater than 0!";
        return 5;
    }

    for(int n = 0; n < cmdparser.get<int>("number"); ++n) {
        cout << "\nNET #" << n << "\n------------------------------------------------------" << endl;

        auto objstrain = load_objects_list(cmdparser.get<std::string>("traindirpath"));
        auto objsvalid = load_objects_list(cmdparser.get<std::string>("validdirpath"));
        auto objstest =  load_objects_list(cmdparser.get<std::string>("testdirpath"));

        cout << "train objs.size(): "<< objstrain.size() << endl;
        cout << "valid objs.size(): "<< objsvalid.size() << endl;
        cout << "test  objs.size(): "<< objstest.size()  << endl;

        std::vector<matrix<rgb_pixel>> imagestrain, imagesvalid;
        std::vector<unsigned long> labelstrain, labelsvalid;

        net_type net;

        dnn_trainer<net_type> trainer(net, sgd());
        trainer.set_learning_rate(0.1);
        trainer.be_verbose();
        trainer.set_synchronization_file(cmdparser.get<std::string>("outputdirpath") + std::string("/metric_sync_") + std::to_string(n), std::chrono::minutes(2));
        trainer.set_iterations_without_progress_threshold(cmdparser.get<unsigned int>("swptrain"));
        trainer.set_test_iterations_without_progress_threshold(cmdparser.get<unsigned int>("swpvalid"));

        // TRAINING =============================
        dlib::pipe<std::vector<matrix<rgb_pixel>>> qimagestrain(4);
        dlib::pipe<std::vector<unsigned long>> qlabelstrain(4);
        auto traindata_loader = [&qimagestrain, &qlabelstrain, &objstrain](time_t seed)
        {
           dlib::rand rnd(time(nullptr)+seed);
           cv::RNG    cvrnd(static_cast<uint64_t>(time(nullptr)+seed));
           std::vector<matrix<rgb_pixel>> images;
           std::vector<unsigned long> labels;
           while(qimagestrain.is_enabled())
           {
               try
               {
                   load_mini_batch(22, 5, rnd, cvrnd, objstrain, images, labels, true);
                   qimagestrain.enqueue(images);
                   qlabelstrain.enqueue(labels);
               }
               catch(std::exception& e)
               {
                   cout << "EXCEPTION IN LOADING DATA" << endl;
                   cout << e.what() << endl;
               }
           }
        };
        // Run the traindata_loader from 5 threads.  You should set the number of threads
        // relative to the number of CPU cores you have.
        std::thread traindata_loader1([traindata_loader](){ traindata_loader(1); });
        std::thread traindata_loader2([traindata_loader](){ traindata_loader(2); });
        std::thread traindata_loader3([traindata_loader](){ traindata_loader(3); });
        // TRAINING =============================

        // VALIDATION ===========================
        dlib::pipe<std::vector<matrix<rgb_pixel>>> qimagesvalid(2);
        dlib::pipe<std::vector<unsigned long>> qlabelsvalid(2);
        auto validdata_loader = [&qimagesvalid, &qlabelsvalid, &objsvalid](time_t seed)
        {
           dlib::rand rnd(time(nullptr)+seed);
           cv::RNG    cvrnd(static_cast<uint64_t>(time(nullptr)+seed));
           std::vector<matrix<rgb_pixel>> images;
           std::vector<unsigned long> labels;
           while(qimagesvalid.is_enabled())
           {
               try
               {
                   load_mini_batch(22, 5, rnd, cvrnd, objsvalid, images, labels, false);
                   qimagesvalid.enqueue(images);
                   qlabelsvalid.enqueue(labels);
               }
               catch(std::exception& e)
               {
                   cout << "EXCEPTION IN LOADING DATA" << endl;
                   cout << e.what() << endl;
               }
           }
        };
        std::thread validdata_loader1([validdata_loader](){ validdata_loader(1); });
        // VALIDATION ===========================

        // Here we do the training.  We keep passing mini-batches to the trainer until the
        // learning rate has dropped low enough.
        size_t _step = 0;
        while(trainer.get_learning_rate() >= cmdparser.get<double>("minlrthresh"))
        {
           _step ++;
           qimagestrain.dequeue(imagestrain);
           qlabelstrain.dequeue(labelstrain);
           trainer.train_one_step(imagestrain, labelstrain);
           if((_step % 4) == 0) {
               qimagesvalid.dequeue(imagesvalid);
               qlabelsvalid.dequeue(labelsvalid);
               trainer.test_one_step(imagesvalid, labelsvalid);
           }
        }

        // Wait for training threads to stop
        trainer.get_net();       
        net.clean();
        cout << "Training has been finished" << endl;

        // stop all the data loading threads and wait for them to terminate.
        qimagestrain.disable();
        qlabelstrain.disable();
        traindata_loader1.join();
        traindata_loader2.join();
        traindata_loader3.join();
        qimagesvalid.disable();
        qlabelsvalid.disable();
        validdata_loader1.join();

        // Now, let's check how well it performs on the validation data.
        dlib::rand rnd(2308);
        cv::RNG    cvrnd(2308);
        cout << "Validation subset:" << endl;
        load_mini_batch(22, 5, rnd, cvrnd, objsvalid, imagesvalid, labelsvalid, false);
        // Let's acquire a non-batch-normalized version of the network
        anet_type testing_net = net;
        // Run all the images through the network to get their vector embeddings.
        std::vector<matrix<float,0,1>> embedded = testing_net(imagesvalid);
        // Now, check if the embedding puts images with the same labels near each other and
        // images with different labels far apart.
        int num_right = 0;
        int num_wrong = 0;
        for (size_t i = 0; i < embedded.size(); ++i)
        {
            for (size_t j = i+1; j < embedded.size(); ++j)
            {
                if (labelsvalid[i] == labelsvalid[j])
                {
                    // The loss_metric layer will cause images with the same label to be less
                    // than net.loss_details().get_distance_threshold() distance from each
                    // other.  So we can use that distance value as our testing threshold.
                    if (length(embedded[i]-embedded[j]) < testing_net.loss_details().get_distance_threshold())
                        ++num_right;
                    else
                        ++num_wrong;
                }
                else
                {
                    if (length(embedded[i]-embedded[j]) >= testing_net.loss_details().get_distance_threshold())
                        ++num_right;
                    else
                        ++num_wrong;
                }
            }
        }       
        cout << "num_right: "<< num_right << endl;
        cout << "num_wrong: "<< num_wrong << endl;
        auto _accuracy = static_cast<float>(num_right)/(num_wrong+num_right);
        cout << "accuracy: "<< _accuracy << endl;

        cout << "Test set:" << endl;
        load_mini_batch(22, 5, rnd, cvrnd, objstest, imagesvalid, labelsvalid, false);
        // Run all the images through the network to get their vector embeddings.
        embedded = testing_net(imagesvalid);
        // Now, check if the embedding puts images with the same labels near each other and
        // images with different labels far apart.
        num_right = 0;
        num_wrong = 0;
        for (size_t i = 0; i < embedded.size(); ++i)
        {
            for (size_t j = i+1; j < embedded.size(); ++j)
            {
                if (labelsvalid[i] == labelsvalid[j])
                {
                    // The loss_metric layer will cause images with the same label to be less
                    // than net.loss_details().get_distance_threshold() distance from each
                    // other.  So we can use that distance value as our testing threshold.
                    if (length(embedded[i]-embedded[j]) < testing_net.loss_details().get_distance_threshold())
                        ++num_right;
                    else
                        ++num_wrong;
                }
                else
                {
                    if (length(embedded[i]-embedded[j]) >= testing_net.loss_details().get_distance_threshold())
                        ++num_right;
                    else
                        ++num_wrong;
                }
            }
        }
        cout << "num_right: "<< num_right << endl;
        cout << "num_wrong: "<< num_wrong << endl;
        _accuracy = static_cast<float>(num_right)/(num_wrong+num_right);
        cout << "accuracy: "<< _accuracy << endl;
        // Save the network to disk
        serialize(cmdparser.get<std::string>("outputdirpath") + std::string("/net_") + std::to_string(n) + std::string("_(TA_") + std::to_string(_accuracy) + std::string(").dat")) << net;
    }
	return 0;
}
catch(std::exception& e)
{
    cout << e.what() << endl;
}

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
