#include "tensor.h"
#include "optimizer.h"
#include "gpt.h"
#include "dataloader.h"
#include "function.h"

#include <cstring>
#include <cmath>
#include <cassert>
#include <algorithm>

using namespace tllm;

int main(int argc, char* argv[]) {
    if (argc <= 4) {
        std::cout << "Usage: ./train [model_path] [start_epoch] [start_iter] [ckpt_folder]" << std::endl;
        exit(0);
    }

    string model_path = argv[1];
    int start_epoch = atoi(argv[2]);
    int start_iter = atoi(argv[3]);
    string ckpt_folder = "../ckpt";
    if (argc > 4) {
        ckpt_folder = argv[4];
    }


    GPT gpt(6, 64, 4, 4096, 256, 0.2, false);
    std::cout << "GPT model " << gpt.get_num_params() / 1e6 << "M" <<std::endl;

    ParamsDict decay_params;
    ParamsDict nodecay_params;
    for (auto iter : gpt.parameters()) {
        if (iter.second.get().ndim() >= 2) {
            decay_params.insert(iter);
        }
        else {
            nodecay_params.insert(iter);
        }
    }
    std::cout << "loading paramaters done!" <<std::endl;
    AdamW adamw(decay_params, nodecay_params, 0.001, 0.9, 0.95, "cuda");
    if( start_epoch > 0 ){
        gpt.load(argv[1]);
        adamw.load(argv[1]);
    }

    gpt.cuda();

    const int n_epoch = 3;
    const int iter_per_epoch = TinyStoriesLoader("../data/tok4096/", 128, 256).get_iter_len();
    const int total_iters = n_epoch * iter_per_epoch;

    // Linear warmup then cosine decay. A constant LR makes the first steps
    // (when the gradients are largest) thrash the freshly-initialised weights.
    const float max_lr = 1e-3;
    const float min_lr = 1e-4;
    const int warmup_iters = 200;
    auto lr_at = [&](int it) -> float {
        if (it < warmup_iters) {
            return max_lr * (it + 1) / (float)warmup_iters;
        }
        float decay = (it - warmup_iters) / (float)std::max(1, total_iters - warmup_iters);
        decay = std::min(1.0f, decay);
        return min_lr + 0.5f * (1.0f + (float)std::cos(M_PI * decay)) * (max_lr - min_lr);
    };

    string ckpt_temp_path= "";
    for (int e = start_epoch; e < n_epoch; ++e) {
        TinyStoriesLoader loader("../data/tok4096/", 128, 256);
        float loss_g = 0;
        for (int i = 0; i < loader.get_iter_len(); ++i) {
            if (start_iter > 0) {
                --start_iter;
                continue;
            }
            adamw.set_lr(lr_at(e * iter_per_epoch + i));

            auto ret = loader.next();
            Tensor& data = ret.first;
            Tensor& label = ret.second;
            data.to("cuda");
            label.to("cuda");
            label.view({label.dsize()});

            Tensor loss = gpt.forward(data, label);
            loss.backward();

            if (i % 1 == 0) {
                loss.cpu();
                std::cout << "[" << e << "/" << i << "/" << loader.get_iter_len() << "] " << "loss: " << loss[0] <<std::endl;
                loss_g = loss[0];
            }

            if (i != 0 && i % 500 == 0) {
                ckpt_temp_path = ckpt_folder + "/epoch" + std::to_string(e) + "_" + std::to_string(i) + "_" + std::to_string(loss[0]) + "/";
                gpt.save(ckpt_temp_path);
                adamw.save(ckpt_temp_path);

                std::cout << "model saved" << std::endl;
            }

            adamw.step();
        }
        std::cout << "saving model" << std::endl;
        gpt.save( ckpt_folder + "/0_epoch" + std::to_string(e) + "_" + std::to_string(loss_g) + "/");
        adamw.save(ckpt_folder + "/0_epoch" + std::to_string(e) + "_" + std::to_string(loss_g) + "/");
        std::cout << "model saved" << std::endl;
    }
    return 0;
}
