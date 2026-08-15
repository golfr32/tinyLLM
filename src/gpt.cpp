#include "gpt.h"

#include <cassert>
#include <thread>
#include <cstring>
#include <queue>
#include <cmath>
#include <cfloat>
#include <algorithm>

using namespace tllm;

GPT::GPT(index_t n_layer, index_t n_embd, index_t n_head, index_t vocab_size, index_t block_size, float dropout, bool bias)
  : wpe(block_size, n_embd),
    // drop(dropout),
    ln_f(n_embd, bias),
    lm_head(n_embd, vocab_size, false),
    wte(vocab_size, n_embd, lm_head.parameters()["weight"]),
    block_size_(block_size),
    vocab_size_(vocab_size) {
    for (int i = 0; i < n_layer; ++i) {
        blocks.push_back(std::move(std::make_unique<nn::TransformerBlock>(n_embd, n_head, dropout, bias)));
    }
    init_weight();
}

Tensor GPT::forward(Tensor& idx) {
    auto shape = idx.shape();
    index_t batch = shape[0];
    index_t T = shape[1];
    assert(T < block_size_);
    idx.disable_grad();
    Tensor pos_ids = get_pos_ids(T);
    pos_ids.to(device());
    Tensor pos_emb = wpe(pos_ids);
    Tensor tok_emb = wte(idx);
    Tensor x = tok_emb + pos_emb;
    // Tensor x = drop(tok_emb);
    std::vector<Tensor> vec;
    vec.push_back(std::move(x));
    for (int i = 0; i < blocks.size(); ++i) {
        Tensor temp = (*blocks[i])(vec[vec.size() - 1]);
        vec.push_back(std::move(temp));
    }
    Tensor x_l = ln_f(vec[vec.size() - 1]);
    Tensor logits = lm_head(x_l); // (1, T, vocab_size)
    return logits;
}

struct node {
    node(float p, int i) : prob(p), id(i) {}
    float prob;
    int id;
};

bool cmp(node& p1, node& p2){
    return p1.prob > p2.prob;
}

void GPT::generate(string text, Tokenizer& tokenizer, int top_k, float temperature) {
    Tensor tokens = tokenizer.encode(text, 1, 0);
    index_t count = tokens.dsize();
    std::default_random_engine ge(time(0));
    std::uniform_real_distribution<float> d(0, 1);
    int prev = tokens[count - 1];
    while (count < block_size_) {
        tokens.to(device());
        Tensor ret = forward(tokens);
        ret.cpu();
        index_t last_index = ret.shape()[1] - 1;

        std::priority_queue<node, std::vector<node>, decltype(&cmp)> heap(cmp);
        for (index_t i = 0; i < vocab_size_; ++i) {
            heap.push({ret[{0, last_index, i}], (int)i});
            if (heap.size() > (size_t)top_k) {
                heap.pop();
            }
        }
        std::vector<node> topk;
        while (!heap.empty()) {
            topk.push_back(heap.top());
            heap.pop();
        }

        // forward() returns raw logits, so turn the top-k into an actual
        // distribution (max-shifted softmax) before sampling from it.
        float max_logit = -FLT_MAX;
        for (auto& iter : topk) {
            max_logit = std::max(max_logit, iter.prob);
        }
        float sum = 0;
        for (auto& iter : topk) {
            iter.prob = std::exp((iter.prob - max_logit) / temperature);
            sum += iter.prob;
        }
        // Replace each entry with the running CDF.
        float accum = 0;
        for (auto& iter : topk) {
            accum += iter.prob / sum;
            iter.prob = accum;
        }

        int next = topk.back().id;  // fallback: highest-probability token
        float rand = d(ge);
        for (auto& iter : topk) {
            if (rand < iter.prob) {
                next = iter.id;
                break;
            }
        }
        if (next == 1 || next == 2) break;
        tokenizer.saft_print(tokenizer.decode(prev, next));
        prev = next;
        ++count;
        Tensor new_tokens({1, count});
        tokens.cpu();
        for (int i = 0; i < count; ++i) {
            if (i == count - 1) {
                new_tokens[i] = next;
            }
            else {
                new_tokens[i] = tokens[i];
            }
        }
        tokens = new_tokens;
    }
}

Tensor GPT::forward(Tensor& idx, Tensor& targets) {
    auto shape = idx.shape();
    index_t batch = shape[0];
    index_t T = shape[1];
    assert(T < block_size_);
    idx.disable_grad();

    Tensor pos_ids = get_pos_ids(T);
    pos_ids.to(device());
    Tensor pos_emb = wpe(pos_ids);
    Tensor tok_emb = wte(idx);
    Tensor x = tok_emb + pos_emb;
    
    // Tensor x = drop(tok_emb);
    std::vector<Tensor> vec;
    vec.push_back(std::move(x));
    for (int i = 0; i < blocks.size(); ++i) {
        Tensor temp = (*blocks[i])(vec[vec.size() - 1]);
        vec.push_back(std::move(temp));
    }
    Tensor x_l = ln_f(vec[vec.size() - 1]);
    Tensor logits = lm_head(x_l); // (B, T, vocab_size)
    auto shape_l = logits.shape();
    logits.view({logits.dsize() / vocab_size_, vocab_size_}); 
    Tensor loss = F::cross_entropy(logits, targets);

    logits.view(shape_l);
    return loss;
}

ParamsDict GPT::parameters() {
    ParamsDict blocks_parm{};
    for (int i = 0; i < blocks.size(); ++i) {
        blocks_parm.insert_parmdict("block" + std::to_string(i), blocks[i]->parameters());
    }
    return {
        {"wpe", wpe.parameters()},
        {"ln_f", ln_f.parameters()},
        {"lm_head", lm_head.parameters()},
        {"blocks", blocks_parm}
    };
}

Tensor GPT::get_pos_ids(index_t T) {
    Tensor pos({T}, "cpu", false);
    for (int i = 0; i < T; ++i) {
        pos[i] = i;
    }
    return pos;
}

void random_gen(float* t, index_t dsize, float std) {
    std::thread::id tid = std::this_thread::get_id();
    std::default_random_engine e(time(0) + *(unsigned int*)&tid);
    std::normal_distribution<float> n(0, std);
    for (int i = 0; i < dsize; ++i) {
        t[i] = n(e);
    }
}

void constant_gen(float* t, index_t dsize, float val) {
    for (int i = 0; i < dsize; ++i) {
        t[i] = val;
    }
}

static bool ends_with(const string& s, const string& suffix) {
    return s.size() >= suffix.size() &&
           s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

void GPT::init_weight() {
    std::vector<std::thread> threads;
    for (auto iter : parameters()) {
        const string& name = iter.first;
        Tensor& weight = iter.second.get();

        // LayerNorm is an affine identity at init: gain 1, shift 0. Drawing it
        // from N(0, 0.02) like the other tensors scales every residual branch
        // to ~zero and the model has to claw the gain back before it can learn.
        if (name.find("ln_") != string::npos || name.find("ln_f") != string::npos) {
            threads.push_back(std::thread(constant_gen, weight.data(), weight.dsize(),
                                          ends_with(name, ".bias") ? 0.f : 1.f));
        }
        else if (ends_with(name, ".bias")) {
            threads.push_back(std::thread(constant_gen, weight.data(), weight.dsize(), 0.f));
        }
        else if (ends_with(name, "c_proj.weight")) {
            // GPT-2 residual scaling: keep the variance of the residual stream
            // constant as the n_layer branches accumulate into it.
            threads.push_back(std::thread(random_gen, weight.data(), weight.dsize(),
                                          0.02f / std::sqrt(2.f * blocks.size())));
        }
        else {
            threads.push_back(std::thread(random_gen, weight.data(), weight.dsize(), 0.02f));
        }
    }
    for (std::thread& t : threads) {
        t.join();
    }
}