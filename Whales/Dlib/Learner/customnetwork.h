#ifndef CUSTOMNETWORK_H
#define CUSTOMNETWORK_H

#include <dlib/dnn.h>

#define IMG_WIDTH  512
#define IMG_HEIGHT 192

namespace dlib {

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

template <typename SUBNET> using level0 = res<512,res_down<512,SUBNET>>;
template <typename SUBNET> using level1 = res<256,res_down<256,SUBNET>>;
template <typename SUBNET> using level2 = res<128,res_down<128,SUBNET>>;
template <typename SUBNET> using level3 = res<64,res_down<64,SUBNET>>;
template <typename SUBNET> using level4 = res<32,res_down<32,SUBNET>>;

template <typename SUBNET> using alevel0 = ares<512,ares_down<512,SUBNET>>;
template <typename SUBNET> using alevel1 = ares<256,ares_down<256,SUBNET>>;
template <typename SUBNET> using alevel2 = ares<128,ares_down<128,SUBNET>>;
template <typename SUBNET> using alevel3 = ares<64,ares_down<64,SUBNET>>;
template <typename SUBNET> using alevel4 = ares<32,ares_down<32,SUBNET>>;

// training network type
using net_type = loss_metric<fc_no_bias<128,avg_pool_everything<
                            level0<
                            level1<
                            level2<
                            level3<
                            level4<
                            relu<bn_con<con<16,7,7,2,2,
                            input<matrix<float>>
                            >>>>>>>>>>>;

// testing network type (replaced batch normalization with fixed affine transforms)
using anet_type = loss_metric<fc_no_bias<128,avg_pool_everything<
                            alevel0<
                            alevel1<
                            alevel2<
                            alevel3<
                            alevel4<
                            relu<affine<con<16,7,7,2,2,
                            input<matrix<float>>
                            >>>>>>>>>>>;
}

#endif // CUSTOMNETWORK_H
