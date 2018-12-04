#ifndef FOCALLOSS_H
#define FOCALLOSS_H

#include <dlib/dnn/loss.h>

namespace dlib {


class loss_multimulticlass_focal_
{

public:

    loss_multimulticlass_focal_ () = default;

    loss_multimulticlass_focal_ (
        const std::map<std::string,std::vector<std::string>>& labels
    )
    {
        for (auto& l : labels)
        {
            possible_labels[l.first] = std::make_shared<decltype(l.second)>(l.second);
            DLIB_CASSERT(l.second.size() >= 2, "Each classifier must have at least two possible labels.");

            for (size_t i = 0; i < l.second.size(); ++i)
            {
                label_idx_lookup[l.first][l.second[i]] = i;
                ++total_num_labels;
            }
        }
    }

    unsigned long number_of_labels() const { return total_num_labels; }

    unsigned long number_of_classifiers() const { return possible_labels.size(); }

    void set_gamma(float _gamma) {gamma = _gamma;}      

    float get_gamma() const { return gamma; }

    void set_alpha(float _alpha) {alpha = _alpha;}

    float get_alpha() const { return alpha; }

    std::map<std::string,std::vector<std::string>> get_labels (
    ) const
    {
        std::map<std::string,std::vector<std::string>> info;
        for (auto& i : possible_labels)
        {
            for (auto& label : *i.second)
                info[i.first].emplace_back(label);
        }
        return info;
    }

    class classifier_output
    {

    public:
        classifier_output() = default;

        size_t num_classes() const { return class_probs.size(); }

        double probability_of_class (
            size_t i
        ) const
        {
            DLIB_CASSERT(i < num_classes());
            return class_probs(i);
        }

        const std::string& label(
            size_t i
        ) const
        {
            DLIB_CASSERT(i < num_classes());
            return (*_labels)[i];
        }

        operator std::string(
        ) const
        {
            DLIB_CASSERT(num_classes() != 0);
            return (*_labels)[index_of_max(class_probs)];
        }

        friend std::ostream& operator<< (std::ostream& out, const classifier_output& item)
        {
            DLIB_ASSERT(item.num_classes() != 0);
            out << static_cast<std::string>(item);
            return out;
        }

    private:

        friend class loss_multimulticlass_focal_;

        template <typename EXP>
        classifier_output(
            const matrix_exp<EXP>& class_probs,
            const std::shared_ptr<std::vector<std::string>>& _labels
        ) :
            class_probs(class_probs),
            _labels(_labels)
        {
        }

        matrix<float,1,0> class_probs;
        std::shared_ptr<std::vector<std::string>> _labels;
    };

    typedef std::map<std::string,std::string> training_label_type;
    typedef std::map<std::string,classifier_output> output_label_type;

    template <
        typename SUB_TYPE,
        typename label_iterator
        >
    void to_label (
        const tensor& input_tensor,
        const SUB_TYPE& sub,
        label_iterator iter_begin
    ) const
    {
        const tensor& output_tensor = sub.get_output();
        DLIB_CASSERT(sub.sample_expansion_factor() == 1);
        DLIB_CASSERT(output_tensor.nr() == 1 &&
                     output_tensor.nc() == 1 );
        DLIB_CASSERT(input_tensor.num_samples() == output_tensor.num_samples());

        DLIB_CASSERT(number_of_labels() != 0, "You must give the loss_multimulticlass_focal_'s constructor label data before you can use it!");
        DLIB_CASSERT(output_tensor.k() == (long)number_of_labels(), "The output tensor must have " << number_of_labels() << " channels.");


        long k_offset = 0;
        for (auto& l : possible_labels)
        {
            auto iter = iter_begin;
            const std::string& classifier_name = l.first;
            const auto& labels = (*l.second);
            scratch.set_size(output_tensor.num_samples(), labels.size());
            tt::copy_tensor(false, scratch, 0, output_tensor, k_offset, labels.size());

            tt::softmax(scratch, scratch);

            for (long i = 0; i < scratch.num_samples(); ++i)
                (*iter++)[classifier_name] = classifier_output(rowm(mat(scratch),i), l.second);

            k_offset += labels.size();
        }
    }


    template <
        typename const_label_iterator,
        typename SUBNET
        >
    double compute_loss_value_and_gradient (
        const tensor& input_tensor,
        const_label_iterator truth_begin,
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
        DLIB_CASSERT(number_of_labels() != 0, "You must give the loss_multimulticlass_focal_'s constructor label data before you can use it!");
        DLIB_CASSERT(output_tensor.k() == (long)number_of_labels(), "The output tensor must have " << number_of_labels() << " channels.");

        // The loss we output is the average loss over the mini-batch.
        const double scale = 1.0/output_tensor.num_samples();
        double loss = 0;
        long k_offset = 0;
        for (auto& l : label_idx_lookup)
        {
            const std::string& classifier_name = l.first;
            const auto& int_labels = l.second;
            scratch.set_size(output_tensor.num_samples(), int_labels.size());
            tt::copy_tensor(false, scratch, 0, output_tensor, k_offset, int_labels.size());

            tt::softmax(scratch, scratch);


            auto truth = truth_begin;
            float* g = scratch.host();
            for (long i = 0; i < scratch.num_samples(); ++i)
            {
                const long y = int_labels.at(truth->at(classifier_name));
                ++truth;

                float _pt = g[i*scratch.k()+y]; // aka prob of true label
                float _multiplier = scale*alpha*std::exp(safe_log(1-_pt)*(gamma-1))*(1-_pt-gamma*_pt*safe_log(_pt));

                for (long k = 0; k < scratch.k(); ++k)
                {
                    const unsigned long idx = i*scratch.k()+k;
                    if (k == y)
                    {
                        loss += scale*-alpha*std::exp(safe_log(1-g[idx])*gamma)*safe_log(g[idx]);
                        g[idx] = scale*alpha*std::exp(safe_log(1-g[idx])*gamma)*(gamma*g[idx]*safe_log(g[idx])-1+g[idx]);
                    }
                    else
                    {
                        g[idx] = _multiplier*g[idx];
                    }
                }
            }

            tt::copy_tensor(false, grad, k_offset, scratch, 0, int_labels.size());

            k_offset += int_labels.size();
        }
        return loss;
    }


    friend void serialize(const loss_multimulticlass_focal_& item, std::ostream& out)
    {
        serialize("loss_multimulticlass_focal_", out);
        serialize(item.get_gamma(), out);
        serialize(item.get_alpha(), out);
        serialize(item.get_labels(), out);
    }

    friend void deserialize(loss_multimulticlass_focal_& item, std::istream& in)
    {
        std::string version;
        deserialize(version, in);
        if (version != "loss_multimulticlass_focal_")
            throw serialization_error("Unexpected version found while deserializing dlib::loss_multimulticlass_focal_.");

        float _gamma;
        deserialize(_gamma, in);
        float _alpha;
        deserialize(_alpha, in);
        std::map<std::string,std::vector<std::string>> info;
        deserialize(info, in);
        item = loss_multimulticlass_focal_(info);
        item.set_gamma(_gamma);
        item.set_alpha(_alpha);
    }

    friend std::ostream& operator<<(std::ostream& out, const loss_multimulticlass_focal_& item)
    {
        out << "loss_multimulticlass_focal, labels={";
        for (auto i = item.possible_labels.begin(); i != item.possible_labels.end(); )
        {
            auto& category = i->first;
            auto& labels = *(i->second);
            out << category << ":(";
            for (size_t j = 0; j < labels.size(); ++j)
            {
                out << labels[j];
                if (j+1 < labels.size())
                    out << ",";
            }

            out << ")";
            if (++i != item.possible_labels.end())
                out << ", ";
        }
        out << "}, gamma=" << item.get_gamma()
            << ", alpha=" << item.get_alpha();
        return out;
    }

    friend void to_xml(const loss_multimulticlass_focal_& item, std::ostream& out)
    {
        out << "<loss_multimulticlass_focal>\n";
        out << item;
        out << "\n</loss_multimulticlass_focal>";
    }

private:

    std::map<std::string,std::shared_ptr<std::vector<std::string>>> possible_labels;
    unsigned long total_num_labels = 0;
    float gamma = 2.0;
    float alpha = 0.25;

    // We make it true that: possible_labels[classifier][label_idx_lookup[classifier][label]] == label
    std::map<std::string, std::map<std::string,long>> label_idx_lookup;


    // Scratch doesn't logically contribute to the state of this object.  It's just
    // temporary scratch space used by this class.
    mutable resizable_tensor scratch;
};

template <typename SUBNET>
using loss_multimulticlass_focal = add_loss_layer<loss_multimulticlass_focal_, SUBNET>;

}
#endif // FOCALLOSS_H
