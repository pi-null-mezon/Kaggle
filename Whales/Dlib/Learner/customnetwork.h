#ifndef CUSTOMNETWORK_H
#define CUSTOMNETWORK_H

#include <dlib/dnn.h>

#define FNUM       32
#define IMG_WIDTH  500
#define IMG_HEIGHT 200

namespace dlib {
template <template <int,template<typename>class,int,typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual = add_prev1<block<N,BN,1,tag1<SUBNET>>>;

template <template <int,template<typename>class,int,typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual_down = add_prev2<max_pool<2,2,2,2,skip1<tag2<block<N,BN,2,tag1<SUBNET>>>>>>;

template <int N, template <typename> class BN, int stride, typename SUBNET>
using block  = BN<con<N,3,3,1,1,relu<BN<con<N,3,3,stride,stride,SUBNET>>>>>;

template <int N, typename SUBNET> using res       = relu<residual<block,N,bn_con,SUBNET>>;
template <int N, typename SUBNET> using ares      = relu<residual<block,N,affine,SUBNET>>;
template <int N, typename SUBNET> using res_down  = relu<residual_down<block,N,bn_con,SUBNET>>;
template <int N, typename SUBNET> using ares_down = relu<residual_down<block,N,affine,SUBNET>>;

// ----------------------------------------------------------------------------------------

template <typename SUBNET> using level0 = res<32*FNUM,res_down<32*FNUM,SUBNET>>;
template <typename SUBNET> using level1 = res_down<16*FNUM,SUBNET>;
template <typename SUBNET> using level2 = res<8*FNUM,res_down<8*FNUM,SUBNET>>;
template <typename SUBNET> using level3 = res<4*FNUM,res<4*FNUM,res_down<4*FNUM,SUBNET>>>;
template <typename SUBNET> using level4 = res<2*FNUM,res<2*FNUM,res<2*FNUM,res_down<2*FNUM,SUBNET>>>>;

template <typename SUBNET> using alevel0 = ares<32*FNUM,ares_down<32*FNUM,SUBNET>>;
template <typename SUBNET> using alevel1 = ares_down<16*FNUM,SUBNET>;
template <typename SUBNET> using alevel2 = ares<8*FNUM,ares_down<8*FNUM,SUBNET>>;
template <typename SUBNET> using alevel3 = ares<4*FNUM,ares<4*FNUM,ares_down<4*FNUM,SUBNET>>>;
template <typename SUBNET> using alevel4 = ares<2*FNUM,ares<2*FNUM,ares<2*FNUM,ares_down<2*FNUM,SUBNET>>>>;
// training network type
using net_type = loss_metric<fc_no_bias<64,avg_pool_everything<
                            level1<
                            level2<
                            level3<
                            level4<
                            max_pool<3,3,2,2,relu<bn_con<con<FNUM,7,7,2,2,
                            input<matrix<float>>
                            >>>>>>>>>>>;

// testing network type (replaced batch normalization with fixed affine transforms)
using anet_type = loss_metric<fc_no_bias<64,avg_pool_everything<
                            alevel1<
                            alevel2<
                            alevel3<
                            alevel4<
                            max_pool<3,3,2,2,relu<affine<con<FNUM,7,7,2,2,
                            input<matrix<float>>
                            >>>>>>>>>>>;
}

#endif // CUSTOMNETWORK_H
