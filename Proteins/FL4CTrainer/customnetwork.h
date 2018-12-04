#ifndef CUSTOMNETWORK_H
#define CUSTOMNETWORK_H

#include <dlib/dnn.h>

#include "focalloss.h"

using namespace std;
using namespace dlib;
//-----------------------------------------------------------------------------------------
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

template <typename SUBNET> using level1 = res_down<512,SUBNET>;
template <typename SUBNET> using level2 = res<256,res<256,res_down<256,SUBNET>>>;
template <typename SUBNET> using level3 = res<128,res_down<128,SUBNET>>;
template <typename SUBNET> using level4 = res<64,res<64,res<64,SUBNET>>>;

template <typename SUBNET> using alevel1 = ares_down<512,SUBNET>;
template <typename SUBNET> using alevel2 = ares<256,ares<256,ares_down<256,SUBNET>>>;
template <typename SUBNET> using alevel3 = ares<128,ares_down<128,SUBNET>>;
template <typename SUBNET> using alevel4 = ares<64,ares<64,ares<64,SUBNET>>>;

#define IMG_SIZE 256
// training network type
using net_type = loss_multimulticlass_focal<fc<56,avg_pool_everything<
                            //level1<
                            level2<
                            level3<
                            level4<
                            avg_pool<3,3,2,2,relu<bn_con<con<32,3,3,1,1,relu<bn_con<con<32,5,5,2,2,
                            input<std::array<matrix<float>,4>>
                            >>>>>>>>>>>>>;

// testing network type (replaced batch normalization with fixed affine transforms)
using anet_type = loss_multimulticlass_focal<fc<56,avg_pool_everything<
                            //alevel1<
                            alevel2<
                            alevel3<
                            alevel4<
                            avg_pool<3,3,2,2,relu<affine<con<32,3,3,1,1,relu<affine<con<32,5,5,2,2,
                            input<std::array<matrix<float>,4>>
                            >>>>>>>>>>>>>;

#endif // CUSTOMNETWORK_H
