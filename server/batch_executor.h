#pragma once

#include "runtime.h"

#include <algorithm>
#include <atomic>
#include <functional>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <future>
#include <memory>
#include <mutex>
#include <thread>

namespace questwend {

class BatchExecutor {
public:
    struct Lease {
        BatchExecutor & executor;
        int slot;
        bool fresh = true;
        std::atomic<bool> cancelled{false};
        void cancel() { cancelled = true; executor.ready_.notify_all(); }
        Lease(BatchExecutor & owner, int index) : executor(owner), slot(index) {}
        Lease(const Lease &) = delete;
        Lease & operator=(const Lease &) = delete;
        ~Lease() {
            std::lock_guard<std::mutex> lock(executor.mutex_);
            executor.busy_[slot] = false;
            executor.ready_.notify_all();
        }
    };

    BatchExecutor(Runtime & runtime, int slots, int prefill_chunk, int decode_rounds)
        : runtime_(runtime), busy_(slots, false), prefill_chunk_(prefill_chunk),
          decode_rounds_(std::max(1, decode_rounds)) {
        runtime_.configure_batch_slots(slots);
        worker_ = std::thread([this] { run(); });
    }
    ~BatchExecutor() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stopping_ = true;
            ready_.notify_all();
        }
        worker_.join();
    }

    std::shared_ptr<Lease> acquire_for(std::chrono::milliseconds timeout) {
        std::unique_lock<std::mutex> lock(mutex_);
        if (!ready_.wait_for(lock, timeout, [&] {
                return stopping_ || std::find(busy_.begin(), busy_.end(), false) != busy_.end();
            }) || stopping_) return nullptr;
        const int slot = (int) (std::find(busy_.begin(), busy_.end(), false) - busy_.begin());
        busy_[slot] = true;
        return std::make_shared<Lease>(*this, slot);
    }

    Runtime::BatchOutput decode(Lease & lease, const std::vector<int32_t> & tokens,
                                bool greedy = false, const std::function<bool()> & keep_running = {}) {
        if (tokens.empty()) throw std::runtime_error("batch decode: empty input");
        if (&lease.executor != this || lease.cancelled) throw std::runtime_error("batch decode: invalid or cancelled lease");
        auto job = std::make_shared<Job>(lease);
        job->input = {lease.slot, tokens, lease.fresh, greedy};
        auto completed = job->output.get_future();
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (stopping_) throw std::runtime_error("batch executor stopping");
            pending_.push_back(job);
            ready_.notify_all();
        }
        if (keep_running) {
            try {
                while (completed.wait_for(std::chrono::milliseconds(0)) != std::future_status::ready) {
                    if (!keep_running()) { lease.cancel(); break; }
                    completed.wait_for(std::chrono::milliseconds(100));
                }
            } catch (...) {
                lease.cancel();
                completed.wait();
                throw;
            }
        }
        auto output = completed.get();
        lease.fresh = false;
        return output;
    }

private:
    struct Job {
        explicit Job(Lease & owner) : lease(owner) {}
        Lease & lease;
        Runtime::BatchInput input;
        size_t offset = 0;
        std::promise<Runtime::BatchOutput> output;
    };
    Runtime & runtime_;
    std::vector<bool> busy_;
    int prefill_chunk_;
    int decode_rounds_;
    std::mutex mutex_;
    std::condition_variable ready_;
    std::deque<std::shared_ptr<Job>> pending_;
    bool stopping_ = false;
    std::thread worker_;

    void run() {
        int decode_rounds_remaining = 0;
        for (;;) {
            std::vector<std::shared_ptr<Job>> jobs;
            bool prefill = false;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                ready_.wait(lock, [&] { return stopping_ || !pending_.empty(); });
                if (stopping_ && pending_.empty()) return;
                for (auto job = pending_.begin(); job != pending_.end();) {
                    if ((*job)->lease.cancelled) {
                        (*job)->output.set_exception(std::make_exception_ptr(std::runtime_error("batch request cancelled")));
                        job = pending_.erase(job);
                    } else ++job;
                }
                if (pending_.empty()) continue;
                // Providers submit at most one job per pinned slot. Wait briefly
                // for the other ready decoders instead of launching separate graphs.
                ready_.wait_for(lock, std::chrono::milliseconds(2), [&] {
                    return stopping_ || pending_.size() == (size_t) std::count(busy_.begin(), busy_.end(), true);
                });
                const bool decoder_ready = std::any_of(pending_.begin(), pending_.end(),
                        [](const auto & job) { return job->input.tokens.size() - job->offset == 1; });
                const bool admit_prefill = !decoder_ready || decode_rounds_remaining == 0;
                for (auto job = pending_.begin(); job != pending_.end();) {
                    const bool is_prefill = (*job)->input.tokens.size() - (*job)->offset > 1;
                    if (!is_prefill || (!prefill && admit_prefill)) {
                        prefill |= is_prefill;
                        jobs.push_back(*job);
                        job = pending_.erase(job);
                    } else {
                        ++job;
                    }
                }
            }
            std::sort(jobs.begin(), jobs.end(), [](const auto & left, const auto & right) {
                return left->input.slot < right->input.slot;
            });
            std::vector<Runtime::BatchInput> inputs;
            const int decoders = (int) std::count_if(jobs.begin(), jobs.end(),
                    [](const auto & job) { return job->input.tokens.size() - job->offset == 1; });
            const int prefill_rows = std::max(1, prefill_chunk_ - decoders);
            for (const auto & job : jobs) {
                const size_t end = std::min(job->input.tokens.size(), job->offset + prefill_rows);
                inputs.push_back({job->input.slot,
                        std::vector<int32_t>(job->input.tokens.begin() + job->offset, job->input.tokens.begin() + end),
                        job->input.reset, job->input.greedy, end == job->input.tokens.size()});
            }
            try {
                auto outputs = runtime_.decode_batch(inputs);
                if (prefill) decode_rounds_remaining = decode_rounds_ - 1;
                else if (decode_rounds_remaining > 0) --decode_rounds_remaining;
                for (size_t index = 0; index < jobs.size(); ++index) {
                    auto & job = jobs[index];
                    job->offset += inputs[index].tokens.size();
                    job->input.reset = false;
                    if (job->lease.cancelled) {
                        job->output.set_exception(std::make_exception_ptr(std::runtime_error("batch request cancelled")));
                    } else if (job->offset == job->input.tokens.size()) {
                        job->output.set_value(std::move(outputs[index]));
                    } else {
                        std::lock_guard<std::mutex> lock(mutex_);
                        pending_.push_back(job);
                        ready_.notify_all();
                    }
                }
            } catch (...) {
                for (const auto & job : jobs) job->output.set_exception(std::current_exception());
            }
        }
    }
};

} // namespace questwend
