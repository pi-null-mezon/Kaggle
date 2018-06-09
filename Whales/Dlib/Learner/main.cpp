#include <dlib/dnn.h>
#include <dlib/misc_api.h>

#include "dlibimgaugment.h"

#include <opencv2/imgcodecs.hpp>

using namespace dlib;
using namespace std;

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

// This function spiders the top level directory and obtains a list of all the image files

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
    cv::RNG & cvrng,
    const std::vector<std::vector<string>>& objs,
    std::vector<matrix<rgb_pixel>>& images,
    std::vector<unsigned long>& labels
)
{
    images.clear();
    labels.clear();
    DLIB_CASSERT(num_people <= objs.size(), "The dataset doesn't have that many people in it.");
    cv::Mat _tmpmat;
    dlib::matrix<dlib::rgb_pixel> _tmpmatrix;
    dlib::array<matrix<dlib::rgb_pixel>> _vcrops;
    std::vector<bool> already_selected(objs.size(), false);
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
            if(rnd.get_random_double() > 0.2) {
                _tmpmat = cv::imread(obj,CV_LOAD_IMAGE_COLOR);
                if(_tmpmat.cols*_tmpmat.rows > 100000)
                    cv::resize(_tmpmat,_tmpmat,cv::Size(512,192),0,0,CV_INTER_AREA);
                else
                    cv::resize(_tmpmat,_tmpmat,cv::Size(512,192),0,0,CV_INTER_CUBIC);
                if(rnd.get_random_double() > 0.7)
                    cv::blur(_tmpmat,_tmpmat,cv::Size(5,5));
                _tmpmatrix = std::move(cvmat2dlibmatrix<dlib::rgb_pixel>(_tmpmat));
            } else {
                _tmpmatrix = std::move(load_rgb_image_with_fixed_size(obj,512,192,true));
            }
            if(rnd.get_random_double() > 0.8) {
                randomly_crop_image(_tmpmatrix,_vcrops,rnd,1,0.800,0.999,0,0,true);
                _tmpmatrix = std::move(_vcrops[0]);
            }
            images.push_back(std::move(_tmpmatrix));
            labels.push_back(id);
        }
    }

    // You might want to do some data augmentation at this point
    for (auto&& crop : images)
    {        
        disturb_colors(crop,rnd);
        if(rnd.get_random_double() > 0.1) {
            randomly_jitter_image(crop,_vcrops,rnd.get_integer(LONG_MAX),1,0,0,1.11,0.05,15.0);
            crop = std::move(_vcrops[0]);
        }
        if(rnd.get_random_double() > 0.2) {
            randomly_cutout_rect(crop,_vcrops,rnd,1,0.5,0.5,rnd.get_random_double()*90.0);
            crop = std::move(_vcrops[0]);
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

template <typename SUBNET> using level0 = res<512,res_down<512,SUBNET>>;
template <typename SUBNET> using level1 = res_down<256,SUBNET>;
template <typename SUBNET> using level2 = res_down<128,SUBNET>;
template <typename SUBNET> using level3 = res_down<64,SUBNET>;
template <typename SUBNET> using level4 = res_down<32,SUBNET>;

template <typename SUBNET> using alevel0 = ares<512,ares_down<512,SUBNET>>;
template <typename SUBNET> using alevel1 = ares_down<256,SUBNET>;
template <typename SUBNET> using alevel2 = ares_down<128,SUBNET>;
template <typename SUBNET> using alevel3 = ares_down<64,SUBNET>;
template <typename SUBNET> using alevel4 = ares_down<32,SUBNET>;


// training network type
using net_type = loss_metric<fc_no_bias<128,avg_pool_everything<
                            level0<
                            level1<
                            level2<
                            level3<
                            level4<
                            relu<bn_con<con<32,7,7,2,2,
                            input_rgb_image
                            >>>>>>>>>>>;

// testing network type (replaced batch normalization with fixed affine transforms)
using anet_type = loss_metric<fc_no_bias<128,avg_pool_everything<
                            alevel0<
                            alevel1<
                            alevel2<
                            alevel3<
                            alevel4<
                            relu<affine<con<32,7,7,2,2,
                            input_rgb_image
                            >>>>>>>>>>>;

// ----------------------------------------------------------------------------------------

int main(int argc, char** argv)
{
    if (argc != 3)
    {
        cout << "Give a folder as input.  It should contain sub-folders of images and we will " << endl;
        cout << "learn to distinguish between these sub-folders with metric learning.  " << endl;
        cout << "that comes with dlib by running this command:" << endl;
        cout << "   ./learner whalestrain whalestest" << endl;
        return 1;
    }

    auto trainobjs = load_objects_list(argv[1]);
    auto testobjs = load_objects_list(argv[2]);

    cout << "trainobjs.size(): "<< trainobjs.size() << endl;
    cout << "testobjs.size(): "<< testobjs.size() << endl;

    std::vector<matrix<rgb_pixel>> images;
    std::vector<unsigned long> labels;

    std::vector<matrix<rgb_pixel>> testimages;
    std::vector<unsigned long> testlabels;

    net_type net;

    dnn_trainer<net_type> trainer(net, sgd(0.0001, 0.9));
    trainer.set_learning_rate(0.1);
    trainer.be_verbose();
    trainer.set_synchronization_file("whales_metric_sync", std::chrono::minutes(10));

    // I've set this to something really small to make the example terminate
    // sooner.  But when you really want to train a good model you should set
    // this to something like 10000 so training doesn't terminate too early.
    trainer.set_iterations_without_progress_threshold(50000);
    trainer.set_test_iterations_without_progress_threshold(1000);

    // If you have a lot of data then it might not be reasonable to load it all
    // into RAM.  So you will need to be sure you are decompressing your images
    // and loading them fast enough to keep the GPU occupied.  I like to do this
    // using the following coding pattern: create a bunch of threads that dump
    // mini-batches into dlib::pipes.  
    dlib::pipe<std::vector<matrix<rgb_pixel>>> qimages(4);
    dlib::pipe<std::vector<unsigned long>> qlabels(4);
    auto data_loader = [&qimages, &qlabels, &trainobjs](time_t seed)
    {
        dlib::rand rnd(time(0)+seed);
        cv::RNG cvrng(time(0)+seed);
        std::vector<matrix<rgb_pixel>> images;
        std::vector<unsigned long> labels;
        while(qimages.is_enabled())
        {
            try
            {
                load_mini_batch(25, 8, rnd, cvrng, trainobjs, images, labels);
                qimages.enqueue(images);
                qlabels.enqueue(labels);
            }
            catch(std::exception& e)
            {
                cout << "EXCEPTION IN LOADING DATA" << endl;
                cout << e.what() << endl;
            }
        }
    };
    // Run the data_loader from 5 threads.  You should set the number of threads
    // relative to the number of CPU cores you have.
    std::thread data_loader1([data_loader](){ data_loader(1); });
    std::thread data_loader2([data_loader](){ data_loader(2); });
    std::thread data_loader3([data_loader](){ data_loader(3); });
    std::thread data_loader4([data_loader](){ data_loader(4); });
    std::thread data_loader5([data_loader](){ data_loader(5); });

    // Same for the test
    dlib::pipe<std::vector<matrix<rgb_pixel>>> testqimages(1);
    dlib::pipe<std::vector<unsigned long>> testqlabels(1);
    auto testdata_loader = [&testqimages, &testqlabels, &testobjs](time_t seed)
    {
        dlib::rand rnd(time(0)+seed);
        cv::RNG cvrng(time(0)+seed);
        std::vector<matrix<rgb_pixel>> images;
        std::vector<unsigned long> labels;
        while(testqimages.is_enabled())
        {
            try
            {
                load_mini_batch(25, 8, rnd, cvrng, testobjs, images, labels);
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
    // Run the data_loader from 5 threads.  You should set the number of threads
    // relative to the number of CPU cores you have.
    std::thread testdata_loader1([testdata_loader](){ testdata_loader(1); });


    // Here we do the training.  We keep passing mini-batches to the trainer until the
    // learning rate has dropped low enough.
    size_t _step = 0;
    while(trainer.get_learning_rate() >= 1e-6)
    {
        _step++;
        qimages.dequeue(images);
        qlabels.dequeue(labels);
        trainer.train_one_step(images, labels);
        if((_step % 11) == 0) {
            testqimages.dequeue(testimages);
            testqlabels.dequeue(testlabels);
            trainer.test_one_step(testimages,testlabels);
        }
    }

    // Wait for training threads to stop
    trainer.get_net();
    cout << "done training" << endl;

    // Save the network to disk
    net.clean();
    serialize("whales_metric_network_resnet.dat") << net;

    // stop all the data loading threads and wait for them to terminate.
    qimages.disable();
    qlabels.disable();
    data_loader1.join();
    data_loader2.join();
    data_loader3.join();
    data_loader4.join();
    data_loader5.join();
    testqimages.disable();
    testqlabels.disable();
    testdata_loader1.join();

    // Now, just to show an example of how you would use the network, let's check how well
    // it performs on the training data.
    dlib::rand rnd(time(0));
    cv::RNG cvrng(time(0));
    load_mini_batch(25, 8, rnd, cvrng, testobjs, images, labels);

    // Normally you would use the non-batch-normalized version of the network to do
    // testing, which is what we do here.
    anet_type testing_net = net;

    // Run all the images through the network to get their vector embeddings.
    std::vector<matrix<float,0,1>> embedded = testing_net(images);

    // Now, check if the embedding puts images with the same labels near each other and
    // images with different labels far apart.
    int num_right = 0;
    int num_wrong = 0;
    for (size_t i = 0; i < embedded.size(); ++i)
    {
        for (size_t j = i+1; j < embedded.size(); ++j)
        {
            if (labels[i] == labels[j])
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
}


