#ifndef COSSIM_METRIC_LOSS_H
#define COSSIM_METRIC_LOSS_H

#include <dlib/dnn/loss.h>

namespace dlib {

template <
    typename EXP
    >
typename disable_if_c<std::numeric_limits<typename EXP::type>::is_integer, const typename EXP::type>::type cosinedistance (
    const matrix_exp<EXP>& m1,
    const matrix_exp<EXP>& m2)
{
    return (1.0f - dot(m1,m2)/(length(m1)*length(m2)));
}

class loss_metric_cossim_
    {
public:

    typedef unsigned long training_label_type;
    typedef matrix<float,0,1> output_label_type;

    loss_metric_cossim_() = default;

    loss_metric_cossim_(
        float margin_,
        float dist_thresh_
        ) : margin(margin_), dist_thresh(dist_thresh_)
        {
        DLIB_CASSERT(margin_ > 0);
        DLIB_CASSERT(dist_thresh_ > 0);
    }

    template <
        typename SUB_TYPE,
        typename label_iterator
        >
    void to_label (
        const tensor& input_tensor,
        const SUB_TYPE& sub,
        label_iterator iter
        ) const
    {
        const tensor& output_tensor = sub.get_output();
        DLIB_CASSERT(sub.sample_expansion_factor() == 1);
        DLIB_CASSERT(input_tensor.num_samples() != 0);
        DLIB_CASSERT(input_tensor.num_samples()%sub.sample_expansion_factor() == 0);
        DLIB_CASSERT(input_tensor.num_samples() == output_tensor.num_samples());
        DLIB_CASSERT(output_tensor.nr() == 1 &&
                         output_tensor.nc() == 1);

        const float* p = output_tensor.host();
        for (long i = 0; i < output_tensor.num_samples(); ++i)
        {
            *iter = mat(p,output_tensor.k(),1);

            ++iter;
            p += output_tensor.k();
        }
    }


    float get_margin() const { return margin; }
    float get_distance_threshold() const { return dist_thresh; }

    template <
        typename const_label_iterator,
        typename SUBNET
        >
    double compute_loss_value_and_gradient (
        const tensor& input_tensor,
        const_label_iterator truth,
            SUBNET& sub
        ) const
    {
        const tensor& output_tensor = sub.get_output();
        tensor& grad = sub.get_gradient_input();

        DLIB_CASSERT(sub.sample_expansion_factor() == 1);
        DLIB_CASSERT(input_tensor.num_samples() != 0);
        DLIB_CASSERT(input_tensor.num_samples()%sub.sample_expansion_factor() == 0);
        DLIB_CASSERT(input_tensor.num_samples() == grad.num_samples());
        DLIB_CASSERT(input_tensor.num_samples() == output_tensor.num_samples());
        DLIB_CASSERT(output_tensor.nr() == 1 &&
                         output_tensor.nc() == 1);
        DLIB_CASSERT(grad.nr() == 1 &&
                         grad.nc() == 1);



        temp.set_size(output_tensor.num_samples(), output_tensor.num_samples());
        grad_mul.copy_size(temp);

        tt::gemm(0, temp, 1, output_tensor, false, output_tensor, true);


        std::vector<double> temp_threshs;
        const float* d = temp.host();
        double loss = 0;
        double num_pos_samps = 0.0001;
        double num_neg_samps = 0.0001;
        for (long r = 0; r < temp.num_samples(); ++r)
        {
            auto xx = d[r*temp.num_samples() + r];
            const auto x_label = *(truth + r);
            for (long c = r+1; c < temp.num_samples(); ++c)
            {
                const auto y_label = *(truth + c);
                if (x_label == y_label)
                {
                    ++num_pos_samps;
                }
                else
                {
                    ++num_neg_samps;

                    // Figure out what distance threshold, when applied to the negative pairs,
                    // causes there to be an equal number of positive and negative pairs.
                    auto yy = d[c*temp.num_samples() + c];
                    auto xy = d[r*temp.num_samples() + c];
                    // compute the distance between x and y samples.
                    auto cosdst = 1.0 - xy / std::sqrt(xx * yy);
                    if (cosdst <= 0)
                        cosdst = 0;
                    temp_threshs.push_back(cosdst);
                }
            }
        }
        // The whole objective function is multiplied by this to scale the loss
        // relative to the number of things in the mini-batch.
        const double scale = 0.5/num_pos_samps;
        DLIB_CASSERT(num_pos_samps>=1, "Make sure each mini-batch contains both positive pairs and negative pairs");
        DLIB_CASSERT(num_neg_samps>=1, "Make sure each mini-batch contains both positive pairs and negative pairs");

        std::sort(temp_threshs.begin(), temp_threshs.end());
        const float neg_thresh = temp_threshs[std::min(num_pos_samps,num_neg_samps)-1];

        // loop over all the pairs of training samples and compute the loss and
        // gradients.  Note that we only use the hardest negative pairs and that in
        // particular we pick the number of negative pairs equal to the number of
        // positive pairs so everything is balanced.
        float* gm = grad_mul.host();
        for (long r = 0; r < temp.num_samples(); ++r)
        {
            gm[r*temp.num_samples() + r] = 0;
            const auto x_label = *(truth + r);
            auto xx = d[r*temp.num_samples() + r];
            for (long c = 0; c < temp.num_samples(); ++c)
            {
                if (r==c)
                    continue;
                const auto y_label = *(truth + c);
                auto yy = d[c*temp.num_samples() + c];
                auto xy = d[r*temp.num_samples() + c];

                // compute the distance between x and y samples.
                auto cosdst = 1.0f - xy / (std::sqrt(xx * yy) + 0.0001f);
                if (cosdst <= 0)
                    cosdst = 0;

                // It should be noted that the derivative of cosdist(x,y) = 1 - xy / (|x||y|) with respect
                // to the x vector is the vector (xy*x - xx*y / (xx|x||y|)).
                // If you stare at the code below long enough you will see that it's just an
                // application of this formula.

                if (x_label == y_label)
                {
                    // Things with the same label should have distances < dist_thresh between
                    // them.  If not then we experience non-zero loss.
                    if (cosdst < dist_thresh-margin)
                    {
                        gm[r*temp.num_samples() + c] = 0;
                    }
                    else
                    {
                        loss += scale*(cosdst - (dist_thresh-margin));
                        auto _tmp = xx * std::sqrt(xx * yy);
                        gm[r*temp.num_samples() + r] += scale * xy / _tmp;
                        gm[r*temp.num_samples() + c] = -scale * xx / _tmp;
                    }
                }
                else
                {
                    // Things with different labels should have distances > dist_thresh between
                    // them.  If not then we experience non-zero loss.
                    if (cosdst > dist_thresh+margin || cosdst > neg_thresh)
                    {
                        gm[r*temp.num_samples() + c] = 0;
                    }
                    else
                    {
                        loss += scale*((dist_thresh+margin) - cosdst);
                        auto _tmp = xx * std::sqrt(xx * yy);
                        // don't divide by zero (or a really small number)
                        if(std::abs(_tmp) < 0.0001f) {
                            if(_tmp >= 0)
                                _tmp = 0.0001f;
                            else
                                _tmp = -0.0001f;
                        }
                        gm[r*temp.num_samples() + r] -= scale * xy / _tmp;
                        gm[r*temp.num_samples() + c] = scale * xx / _tmp;
                    }
                }
            }
        }


        tt::gemm(0, grad, 1, grad_mul, false, output_tensor, false);

        return loss;
    }

    friend void serialize(const loss_metric_cossim_& item, std::ostream& out)
    {
        serialize("loss_metric_cossim_", out);
        serialize(item.margin, out);
        serialize(item.dist_thresh, out);
    }

    friend void deserialize(loss_metric_cossim_& item, std::istream& in)
    {
        std::string version;
        deserialize(version, in);
        if (version == "loss_metric_cossim_")
        {
            deserialize(item.margin, in);
            deserialize(item.dist_thresh, in);
        }
        else
        {
            throw serialization_error("Unexpected version found while deserializing dlib::loss_metric_cossim_.  Instead found " + version);
        }
    }

    friend std::ostream& operator<<(std::ostream& out, const loss_metric_cossim_& item )
    {
        out << "loss_metric_cossim (margin="<<item.margin<<", distance_threshold="<<item.dist_thresh<<")";
        return out;
    }

    friend void to_xml(const loss_metric_cossim_& item, std::ostream& out)
    {
        out << "<loss_metric_cossim margin='"<<item.margin<<"' distance_threshold='"<<item.dist_thresh<<"'/>";
    }

private:
    float margin = 0.15;
    float dist_thresh = 0.6;


    // These variables are only here to avoid being reallocated over and over in
    // compute_loss_value_and_gradient()
    mutable resizable_tensor temp, grad_mul;

};

template <typename SUBNET>
using loss_metric_cossim = add_loss_layer<loss_metric_cossim_, SUBNET>;

}

#endif // COSSIM_METRIC_LOSS_H
