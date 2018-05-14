// The contents of this file are in the public domain

#include <dlib/dnn.h>
#include <iostream>
#include <dlib/data_io.h>
#include <dlib/dir_nav.h>
#include <iterator>
#include <thread>

#include <dlib/opencv.h>
#include <opencv2/opencv.hpp>

#include "dlibimgaugment.h"

using namespace std;
using namespace dlib;

#define CLASSES 128
#define IMGSIZE 227

// ----------------------------------------------------------------------------------------
// The next page of code defines a ResNet network.  It's basically copied
// and pasted from the dnn_imagenet_ex.cpp example, except we replaced the loss
// layer with loss_metric and make the network somewhat smaller.

template <template <int,template<typename>class,int,typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual = add_prev1<block<N,BN,1,tag1<SUBNET>>>;

template <template <int,template<typename>class,int,typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual_down = add_prev2<avg_pool<2,2,2,2,skip1<tag2<block<N,BN,2,tag1<SUBNET>>>>>>;

template <int N, template <typename> class BN, int stride, typename SUBNET>
using block  = BN<con<N,3,3,1,1,relu<BN<con<N,3,3,stride,stride,SUBNET>>>>>;


template <int N, typename SUBNET> using res       = relu<residual<block,N,bn_con,SUBNET>>;
template <int N, typename SUBNET> using ares      = relu<residual<block,N,affine,SUBNET>>;
template <int N, typename SUBNET> using res_down  = relu<residual_down<block,N,bn_con,SUBNET>>;
template <int N, typename SUBNET> using ares_down = relu<residual_down<block,N,affine,SUBNET>>;

// ----------------------------------------------------------------------------------------

template <typename SUBNET> using level0 = res_down<256,SUBNET>;
template <typename SUBNET> using level1 = res<256,res<256,res_down<256,SUBNET>>>;
template <typename SUBNET> using level2 = res<128,res<128,res_down<128,SUBNET>>>;
template <typename SUBNET> using level3 = res<64,res<64,res<64,res_down<64,SUBNET>>>>;
template <typename SUBNET> using level4 = res<32,res<32,res<32,SUBNET>>>;

template <typename SUBNET> using alevel0 = ares_down<256,SUBNET>;
template <typename SUBNET> using alevel1 = ares<256,ares<256,ares_down<256,SUBNET>>>;
template <typename SUBNET> using alevel2 = ares<128,ares<128,ares_down<128,SUBNET>>>;
template <typename SUBNET> using alevel3 = ares<64,ares<64,ares<64,ares_down<64,SUBNET>>>>;
template <typename SUBNET> using alevel4 = ares<32,ares<32,ares<32,SUBNET>>>;


// training network type
using net_type = loss_metric<fc_no_bias<128,avg_pool_everything<
                            level0<
                            level1<
                            level2<
                            level3<
                            level4<
                            max_pool<3,3,2,2,relu<bn_con<con<32,7,7,2,2,
                            input_rgb_image
                            >>>>>>>>>>>>;

// testing network type (replaced batch normalization with fixed affine transforms)
using anet_type = loss_metric<fc_no_bias<128,avg_pool_everything<
                            alevel0<
                            alevel1<
                            alevel2<
                            alevel3<
                            alevel4<
                            max_pool<3,3,2,2,relu<affine<con<32,7,7,2,2,
                            input_rgb_image
                            >>>>>>>>>>>>;

// ----------------------------------------------------------------------------------------

// ----------------------------------------------------------------------------------------

// We will need to create some functions for loading data.  This program will
// expect to be given a directory structured as follows:
//    top_level_directory/
//        person1/
//            image1.jpg
//            image2.jpg
//            image3.jpg
//        person2/
//            image4.jpg
//            image5.jpg
//            image6.jpg
//        person3/
//            image7.jpg
//            image8.jpg
//            image9.jpg
//
// The specific folder and image names don't matter, nor does the number of folders or
// images.  What does matter is that there is a top level folder, which contains
// subfolders, and each subfolder contains images of a single person.

// This function spiders the top level directory and obtains a list of all the
// image files.
std::vector<std::vector<string>> load_objects_list (
    const string& dir
)
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

// This function takes the output of load_objects_list() as input and randomly
// selects images for training.  It should also be pointed out that it's really
// important that each mini-batch contain multiple images of each person.  This
// is because the metric learning algorithm needs to consider pairs of images
// that should be close (i.e. images of the same person) as well as pairs of
// images that should be far apart (i.e. images of different people) during each
// training step.
void load_mini_batch (
    const size_t num_people,     // how many different people to include
    const size_t samples_per_id, // how many images per person to select.
    dlib::rand& rnd,
    const std::vector<std::vector<string>>& objs,
    std::vector<matrix<rgb_pixel>>& images,
    std::vector<unsigned long>& labels
)
{
    images.clear();
    labels.clear();
    DLIB_CASSERT(num_people <= objs.size(), "The dataset doesn't have that many people in it.");

    std::vector<bool> already_selected(objs.size(), false);
    matrix<rgb_pixel> image;
    for (size_t i = 0; i < num_people; ++i)
    {
        size_t id = rnd.get_random_32bit_number()%objs.size();
        // don't pick a person we already added to the mini-batch
        while(already_selected[id])
            id = rnd.get_random_32bit_number()%objs.size();
        already_selected[id] = true;

        for (size_t j = 0; j < samples_per_id; ++j)
        {
            const auto& obj = objs[id][rnd.get_random_32bit_number()%objs[id].size()];
            load_image(image, obj);
            images.push_back(std::move(image));
            labels.push_back(id);
        }
    }

    // You might want to do some data augmentation at this point.  Here we do some simple
    // color augmentation.
    dlib::array<dlib::matrix<dlib::rgb_pixel>> _vcrops;
    for (auto&& crop : images)
    {
        randomly_crop_image(crop,_vcrops,rnd,1,0.75,0.95,IMGSIZE,IMGSIZE);
        crop = std::move(_vcrops[0]);
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
       "{outputdirpath  o |        | output directory location}"
       "{number n         |   1    | number of classifiers to be trained}"
       "{lossthresh       | 0.20   | testset loss threshold for network saving}"
       "{swptrain         | 5000   | determines after how many steps without progress (training loss) decay should be applied to learning rate}"
       "{swpvalid         | 1000   | determines after how many steps without progress (test loss) decay should be applied to learning rate}"
       "{minlrthresh      | 1.0e-4 | minimum learning rate, determines when trining should be stopped}";
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
        return 2;
    }
    if(!cmdparser.has("validdirpath")) {
        cout << "You have not provide path to validation directory!";
        return 2;
    }
    if(!cmdparser.has("outputdirpath")) {
        cout << "You have not provide path to output directory!";
        return 2;
    }
    if(cmdparser.get<int>("number") <= 0) {
        cout << "Number of classifiers should be greater than 0!";
        return 2;
    }

    for(int n = 0; n < cmdparser.get<int>("number"); ++n) {
        cout << "\nNET #" << n << "\n------------------------------------------------------" << endl;

        auto objstrain = load_objects_list(cmdparser.get<std::string>("traindirpath"));
        auto objsvalid = load_objects_list(cmdparser.get<std::string>("validdirpath"));

        cout << "train objs.size(): "<< objstrain.size() << endl;
        cout << "valid objs.size(): "<< objsvalid.size() << endl;

        std::vector<matrix<rgb_pixel>> imagestrain, imagesvalid;
        std::vector<unsigned long> labelstrain, labelsvalid;

        net_type net;

        dnn_trainer<net_type> trainer(net, sgd(0.0001, 0.9));
        trainer.set_learning_rate(0.1);
        trainer.be_verbose();
        trainer.set_synchronization_file(cmdparser.get<std::string>("outputdirpath") + std::string("/face_metric_sync_") + std::to_string(n), std::chrono::minutes(10));
        trainer.set_iterations_without_progress_threshold(cmdparser.get<unsigned long>("swptrain"));
        trainer.set_test_iterations_without_progress_threshold(cmdparser.get<unsigned long>("swpvalid"));

        // TRAINING =============================
        dlib::pipe<std::vector<matrix<rgb_pixel>>> qimagestrain(4);
        dlib::pipe<std::vector<unsigned long>> qlabelstrain(4);
        auto traindata_loader = [&qimagestrain, &qlabelstrain, &objstrain](time_t seed)
        {
           dlib::rand rnd(time(0)+seed);
           std::vector<matrix<rgb_pixel>> images;
           std::vector<unsigned long> labels;
           while(qimagestrain.is_enabled())
           {
               try
               {
                   load_mini_batch(20, 10, rnd, objstrain, images, labels);
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
        std::thread traindata_loader4([traindata_loader](){ traindata_loader(4); });
        std::thread traindata_loader5([traindata_loader](){ traindata_loader(5); });
        // TRAINING =============================

        // VALIDATION ===========================
        dlib::pipe<std::vector<matrix<rgb_pixel>>> qimagesvalid(1);
        dlib::pipe<std::vector<unsigned long>> qlabelsvalid(1);
        auto validdata_loader = [&qimagesvalid, &qlabelsvalid, &objsvalid](time_t seed)
        {
           dlib::rand rnd(time(0)+seed);
           std::vector<matrix<rgb_pixel>> images;
           std::vector<unsigned long> labels;
           while(qimagesvalid.is_enabled())
           {
               try
               {
                   load_mini_batch(20, 10, rnd, objsvalid, images, labels);
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
           if((_step % 15) == 0) {
               qimagesvalid.dequeue(imagesvalid);
               qlabelsvalid.dequeue(labelsvalid);
               trainer.test_one_step(imagesvalid, labelsvalid);
           }
        }

        // Wait for training threads to stop
        trainer.get_net();
        cout << "done training" << endl;

        // Save the network to disk
        net.clean();
        serialize(cmdparser.get<std::string>("outputdirpath") + std::string("/metric_network_resnet_") + std::to_string(n) + std::string(".dat")) << net;

        // stop all the data loading threads and wait for them to terminate.
        qimagestrain.disable();
        qlabelstrain.disable();
        traindata_loader1.join();
        traindata_loader2.join();
        traindata_loader3.join();
        traindata_loader4.join();
        traindata_loader5.join();
        qimagesvalid.disable();
        qlabelsvalid.disable();
        validdata_loader1.join();

        // Now, just to show an example of how you would use the network, let's check how well
        // it performs on the training data.
        dlib::rand rnd(time(0));
        load_mini_batch(20, 10, rnd, objsvalid, imagesvalid, labelsvalid);

        // Normally you would use the non-batch-normalized version of the network to do
        // testing, which is what we do here.
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

        serialize(cmdparser.get<std::string>("outputdirpath") + "/net_" + std::to_string(n) + "_(" + std::to_string(trainer.get_average_loss()) + " - " + std::to_string(trainer.get_average_test_loss()) + ").dat") << testing_net;

        // Also wait for threaded processing to stop in the trainer.

        /*cout << "Network #" << n << " (trainset loss: " << trainer.get_average_loss() << "; test loss: " << trainer.get_average_test_loss() << ")" << endl;
        if(trainer.get_average_test_loss() < cmdparser.get<double>("lossthresh")) {
            //serialize(cmdparser.get<std::string>("outputdirpath") + "/net_" + std::to_string(n) + "_(" + std::to_string(trainer.get_average_loss()) + " - " + std::to_string(trainer.get_average_test_loss()) + ").dat") << net;
            // Now test the network on the validation dataset.  First, make a testing
            // network with softmax as the final layer.  We don't have to do this if we just wanted
            // to test the "top1 accuracy" since the normal network outputs the class prediction.
            // But this snet object will make getting the top5 predictions easy as it directly
            // outputs the probability of each class as its final output.
            softmax<anet_type::subnet_type> snet; snet.subnet() = net.subnet();
            cout << "Testing network on train dataset..." << endl;
            int num_right_top1 = 0;
            int num_wrong_top1 = 0;
            dlib::rand rnd(0);
            // loop over all the imagenet validation images
            double logloss = 0.0;
            for (auto l : validationset) {
                matrix<rgb_pixel> img;
                load_image(img,l.filename);
                // Grab N random crops from the image.  We will run all of them through the
                // network and average the results.
                dlib::array<matrix<rgb_pixel>> images;
                const size_t num_crops = 3;
                randomly_crop_image(img,images,rnd,num_crops,0.85,0.99,IMGSIZE,IMGSIZE);
                matrix<float,1,CLASSES> p = sum_rows(mat(snet(images.begin(), images.end())))/num_crops;
                // p(i) is the probability that the image contains object of class i.
                // update log loss
                logloss += -std::log(p(l.numeric_label));
                // check top 1 accuracy
                if(index_of_max(p) == l.numeric_label) {
                    ++num_right_top1;
                } else {
                    ++num_wrong_top1;
                    if(cmdparser.get<int>("number") == 1) {
                        cv::Mat _imgmat = dlibmatrix2cvmat(img);
                        cv::putText(_imgmat, string("true:") + l.label, cv::Point(6,11), CV_FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(0,0,0), 1, CV_AA);
                        cv::putText(_imgmat, string("true:") + l.label, cv::Point(5,10), CV_FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(255,255,255), 1, CV_AA);
                        cv::namedWindow("Wrong", CV_WINDOW_NORMAL);
                        cv::imshow("Wrong", _imgmat);
                        cv::waitKey(1);
                        cout << "True label: " << l.numeric_label << "; "
                             << "predicted: " << index_of_max(p) << endl;
                    }
                }
            }
            logloss /= validationset.size();
            cout << "Average testset log loss:  " << logloss << endl;
            cout << "Test accuracy: " << 1.0 - (double)num_wrong_top1/(num_right_top1+num_wrong_top1) << endl;
            cout << "Wrong / Total: " << num_wrong_top1 << "/" << (num_right_top1+num_wrong_top1) << endl;
            serialize(cmdparser.get<std::string>("outputdirpath") + "/net_" + std::to_string(n) + "_(" + std::to_string(trainer.get_average_loss()) + " - " + std::to_string(trainer.get_average_test_loss()) + " - " + std::to_string(logloss) + ").dat") << net;
        } else {
            cout << "Training session is not well enough, this network will not be saved..." << endl;
        }*/
    }
}
catch(std::exception& e)
{
    cout << e.what() << endl;
}

