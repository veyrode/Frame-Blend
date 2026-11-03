#include "fb_media_reader.h"

#include "fb_gpu.h"

#define NOMINMAX
#include <windows.h>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/hwcontext.h>
#include <libavutil/hwcontext_cuda.h>
#include <libavutil/pixdesc.h>
}

#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <iterator>
#include <map>
#include <mutex>
#include <new>
#include <set>
#include <thread>
#include <tuple>

namespace fb::media {
namespace {

// Decoded frames are kept within a memory budget (host memory for software
// frames, device memory for NVDEC surfaces); the decoder runs ahead of the
// latest request by up to 60% of that, and frames well behind it are dropped.
constexpr double kHostBudgetBytes = 1024.0 * 1024 * 1024;
constexpr double kDeviceBudgetBytes = 384.0 * 1024 * 1024;
constexpr double kPinnedBudgetBytes = 512.0 * 1024 * 1024;  // page-locked memory is taken from the whole system
constexpr int64_t kKeepBehind = 160;     // frames kept behind the latest request
constexpr int64_t kForwardDecode = 600;  // decode forward instead of seeking for gaps up to this
// Playback asks for ascending runs of frames and earns the full read-ahead
// gradually; an isolated request (scrubbing, a render after an edit) reads only
// this far ahead, so it does not keep every core busy decoding unused frames.
constexpr int64_t kScrubAhead = 32;
constexpr int64_t kSequentialGap = 64;   // a request at most this far past the run continues it
// ... or up to one second of source: playback that cannot keep up drops frames,
// and its requests then jump hundreds of frames ahead (0.3-0.8 s at 1130 fps,
// Premiere 2023 host test 2026-10-22). Treating those jumps as scrubbing cut the
// read-ahead to kScrubAhead, so every render waited for the frames it skipped.
constexpr int64_t kReverseGap = 256;     // a run starting at most this far below the last one plays backwards
// A project opens a reader for every recording on its timelines, so readers
// decode nothing before a render asks (no warm-up at the clip's in point: with
// 32 recordings that alone filled 16 GB). They share one budget
// (frame_memory_budget): a reader that goes over it takes memory back from
// readers nothing has asked for in kReleaseIdleMs. A reader idle for
// kCloseIdleMs also closes its decoder (about 65 MB with frame threads at
// 720p) and reopens it on the next request.
constexpr uint64_t kReleaseIdleMs = 1000;
constexpr uint64_t kCloseIdleMs = 3000;

std::atomic<int64_t> g_frame_bytes{0};

void account(const HostMemory* memory, int64_t delta) {
    if (memory) {
        memory->bytes += delta;
    }
    g_frame_bytes += delta;
}

// Snaps a container rate within 100 ppm of an integer or NTSC rate to it
// (MP4 time bases can turn 720/1 into 720.046).
AVRational exact_rate(AVRational rate) {
    if (rate.num <= 0 || rate.den <= 0) {
        return rate;
    }
    const double fps = av_q2d(rate);
    const double integer = std::round(fps);
    if (integer >= 1.0 && std::fabs(fps - integer) <= fps * 1e-4) {
        return AVRational{static_cast<int>(integer), 1};
    }
    const double k = std::round(fps * 1001.0 / 1000.0);
    if (k >= 1.0 && std::fabs(fps - k * 1000.0 / 1001.0) <= fps * 1e-4) {
        return AVRational{static_cast<int>(k * 1000.0), 1001};
    }
    return rate;
}

std::string av_error(int code) {
    char text[AV_ERROR_MAX_STRING_SIZE] = {};
    av_strerror(code, text, sizeof(text));
    return text;
}

enum AVPixelFormat pick_cuda(AVCodecContext*, const enum AVPixelFormat* formats) {
    for (const enum AVPixelFormat* f = formats; *f != AV_PIX_FMT_NONE; ++f) {
        if (*f == AV_PIX_FMT_CUDA) {
            return *f;
        }
    }
    return formats[0];  // no NVDEC for this stream: software
}

// Decoded NV12 buffers of dropped frames, reused for new frames: a fresh
// 1.4 MB (720p) buffer per frame costs the system zeroing its pages first,
// which halved the decoding rate (649 frames/s against 1204 for the ffmpeg
// command line on the same 720 fps recording, 2026-10-06).
class BufferPool {
public:
    explicit BufferPool(void* pinned_context) { memory_.cuda_context = pinned_context; }

    HostBuffer take() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (free_.empty()) {
            return HostBuffer(HostAllocator<uint8_t>(&memory_));
        }
        HostBuffer buffer = std::move(free_.back());
        free_.pop_back();
        return buffer;
    }

    void give(HostBuffer&& buffer) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!buffer.empty() && free_.size() < kMaxFree) {
            free_.push_back(std::move(buffer));
        }
    }

    // Frees the buffers kept for reuse.
    void clear() {
        std::vector<HostBuffer> dropped;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            dropped.swap(free_);
        }
    }

    bool pinned() const { return memory_.cuda_context != nullptr; }
    HostMemory& memory() { return memory_; }

private:
    static constexpr size_t kMaxFree = 48;
    HostMemory memory_;  // before free_: outlives the buffers it allocated
    std::mutex mutex_;
    std::vector<HostBuffer> free_;
};

class FfmpegReader;

// Open readers, for taking memory back from idle ones (take_back_memory).
// Lock order: this mutex, then a reader's; a reader never takes it while
// holding its own.
std::mutex g_registry_mutex;
std::set<FfmpegReader*> g_registry;

void take_back_memory(FfmpegReader* self);

class FfmpegReader final : public Reader {
public:
    FfmpegReader(std::string path_utf8, void* cuda_context, void* pinned_context)
        : path_(std::move(path_utf8)), cuda_context_(cuda_context),
          pool_(std::make_shared<BufferPool>(cuda_context ? nullptr : pinned_context)) {}

    ~FfmpegReader() override {
        {
            std::lock_guard<std::mutex> registry(g_registry_mutex);
            g_registry.erase(this);
        }
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stop_ = true;
        }
        cv_.notify_all();
        if (worker_.joinable()) {
            worker_.join();
        }
    }

    bool start(std::string& error) {
        worker_ = std::thread([this] { run(); });
        {
            std::unique_lock<std::mutex> lock(mutex_);
            cv_.wait(lock, [this] { return opened_ || !open_error_.empty(); });
            error = open_error_;
            if (!opened_) {
                return false;
            }
        }
        std::lock_guard<std::mutex> registry(g_registry_mutex);
        g_registry.insert(this);
        return true;
    }

    const Info& info() const override { return info_; }

    // Whether nothing has asked this reader for a frame recently; `last`
    // orders the candidates, oldest first (0: never asked).
    bool idle(uint64_t now, uint64_t& last, int64_t& bytes) const {
        std::lock_guard<std::mutex> lock(mutex_);
        last = last_request_ms_;
        bytes = pool_->memory().bytes;
        return requests_.empty() && (last == 0 || now - last >= kReleaseIdleMs);
    }

    // Drops the cached frames (those a render still holds are freed when it
    // lets go of them), stops reading ahead and closes the decoder.
    void release() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            for (auto it = cache_.begin(); it != cache_.end();) {
                it = requested(it->first) || it->first == producing_ ? std::next(it) : cache_.erase(it);
            }
            target_ = std::min(target_, latest_);
            close_decoder_ = true;
            ++releases_;
        }
        pool_->clear();
        cv_.notify_all();
    }

    std::shared_ptr<const Frame> get(int64_t index) override {
        std::unique_lock<std::mutex> lock(mutex_);
        if (index < 0 || (info_.frames > 0 && index >= info_.frames) || failed_) {
            return nullptr;
        }
        requests_.insert(index);
        last_request_ms_ = last_activity_ms_ = GetTickCount64();
        close_decoder_ = false;
        // Renders running in parallel interleave slightly, so a run also
        // continues with requests a little behind its end.
        const bool continues = index >= run_start_ && index <= run_end_ + playback_gap_;
        if (!continues) {
            // Reversed clips play backwards: each render starts a little below
            // the previous one. Nothing is read ahead then, and the frames
            // kept are those below the latest request (see drop_stale).
            backward_ = index < run_start_ && run_start_ - index <= kReverseGap;
            run_start_ = index;
            run_end_ = index;
        }
        run_end_ = std::max(run_end_, index);
        const int64_t ahead = backward_ ? 0 : std::min(read_ahead_, kScrubAhead + 2 * (run_end_ - run_start_));
        latest_ = index;
        // A jump drops the read-ahead target of the previous position.
        target_ = continues ? std::max(target_, index + ahead) : index + ahead;
        cv_.notify_all();
        cv_.wait(lock, [&] { return cache_.count(resolve(index)) || failed_ || stop_ || (end_ >= 0 && index >= end_); });
        requests_.erase(index);
        auto it = cache_.find(resolve(index));
        return it == cache_.end() ? nullptr : it->second;
    }

    std::string stats() const override {
        std::lock_guard<std::mutex> lock(mutex_);
        return "decoded=" + std::to_string(decoded_) + " seeks=" + std::to_string(seeks_) +
               " cached=" + std::to_string(cache_.size()) + "/" + std::to_string(capacity_) +
               " ahead=" + std::to_string(next_ - latest_) + " nvdec=" + std::to_string(info_.nvdec ? 1 : 0) +
               " held_mb=" + std::to_string(pool_->memory().bytes >> 20) +
               " all_readers_mb=" + std::to_string(g_frame_bytes >> 20) + "/" + std::to_string(frame_memory_budget() >> 20) +
               " releases=" + std::to_string(releases_) + " decoder_opens=" + std::to_string(decoder_opens_) +
               (error_.empty() ? "" : " error=" + error_);
    }

private:
    // ---- worker thread ----

    void run() {
        std::string error;
        const bool ok = open_file(error);
        {
            std::lock_guard<std::mutex> lock(mutex_);
            opened_ = ok;
            open_error_ = ok ? "" : error;
        }
        cv_.notify_all();
        while (ok) {
            int64_t want = -1;
            bool requested = false;
            bool close = false;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                for (;;) {
                    drop_stale();
                    if (stop_ || !pending().empty() || reading_ahead()) {
                        break;
                    }
                    if (codec_ && (close_decoder_ || GetTickCount64() - last_activity_ms_ >= kCloseIdleMs)) {
                        close = true;
                        break;
                    }
                    if (codec_) {
                        cv_.wait_for(lock, std::chrono::milliseconds(500));
                    } else {
                        cv_.wait(lock);
                    }
                }
                if (stop_) {
                    break;
                }
                if (!close) {
                    const std::set<int64_t> missing = pending();
                    requested = !missing.empty();
                    // Read ahead from the latest request, never from stale positions.
                    want = requested ? *missing.begin() : std::max(next_, latest_);
                    producing_ = want;
                }
            }
            if (close) {
                close_codec();
                pool_->clear();
                continue;
            }
            const bool ok_frame = produce(want);
            {
                std::lock_guard<std::mutex> lock(mutex_);
                producing_ = -1;
                if (!ok_frame && requested && end_ < 0) {
                    failed_ = true;  // a requested frame cannot be decoded
                }
                if (!ok_frame && !requested) {
                    target_ = latest_;  // stop reading ahead until the next request
                }
            }
            cv_.notify_all();
        }
        {
            std::lock_guard<std::mutex> lock(mutex_);
            cache_.clear();
        }
        av_frame_free(&frame_);
        av_packet_free(&packet_);
        avcodec_free_context(&codec_);
        avformat_close_input(&format_);
        av_buffer_unref(&device_);
    }

    // Whether the worker should decode ahead of the latest request (caller
    // holds the mutex).
    bool reading_ahead() const {
        return next_ <= target_ && end_ < 0 && cache_.size() < capacity_;
    }

    std::set<int64_t> pending() const {
        std::set<int64_t> out;
        for (int64_t r : requests_) {
            const int64_t frame = resolve(r);
            if (!cache_.count(frame) && (end_ < 0 || r < end_)) {
                out.insert(frame);
            }
        }
        return out;
    }

    // Variable frame rate recordings (a capture that dropped frames or froze)
    // leave indices no frame rounds to; they show the frame before them, as a
    // player does. Returns the index whose frame `index` shows, once the
    // decoder has passed it (caller holds the mutex).
    int64_t resolve(int64_t index) const {
        auto it = holds_.upper_bound(index);
        if (it == holds_.begin()) {
            return index;
        }
        --it;
        return index < it->second.end ? it->second.frame : index;
    }

    // Whether a pending request shows frame `index` (caller holds the mutex).
    bool requested(int64_t index) const {
        for (int64_t r : requests_) {
            if (r == index || resolve(r) == index) {
                return true;
            }
        }
        return false;
    }

    bool open_file(std::string& error) {
        int r = avformat_open_input(&format_, path_.c_str(), nullptr, nullptr);
        if (r < 0) {
            error = "cannot open: " + av_error(r);
            return false;
        }
        if ((r = avformat_find_stream_info(format_, nullptr)) < 0) {
            error = "no stream info: " + av_error(r);
            return false;
        }
        const AVCodec* decoder = nullptr;
        stream_ = av_find_best_stream(format_, AVMEDIA_TYPE_VIDEO, -1, -1, &decoder, 0);
        if (stream_ < 0 || !decoder) {
            error = "no decodable video stream";
            return false;
        }
        AVStream* stream = format_->streams[stream_];
        for (unsigned i = 0; i < format_->nb_streams; ++i) {
            format_->streams[i]->discard = static_cast<int>(i) == stream_ ? AVDISCARD_DEFAULT : AVDISCARD_ALL;
        }
        time_base_ = stream->time_base;
        rate_ = exact_rate(stream->avg_frame_rate.num > 0 ? stream->avg_frame_rate : stream->r_frame_rate);
        // Gaps measured in source frames scale with the rate (see kSequentialGap).
        const int64_t one_second = static_cast<int64_t>(std::ceil(av_q2d(rate_)));
        playback_gap_ = std::max(kSequentialGap, one_second);
        forward_decode_ = std::max(kForwardDecode, 2 * one_second);
        if (rate_.num <= 0 || rate_.den <= 0) {
            error = "no frame rate";
            return false;
        }
        decoder_ = decoder;
        if (cuda_context_) {
            cuda_stream_ = fb::gpu::cuda_create_stream(cuda_context_);
            device_ = av_hwdevice_ctx_alloc(AV_HWDEVICE_TYPE_CUDA);
            if (device_ && cuda_stream_) {
                auto* hw = static_cast<AVCUDADeviceContext*>(reinterpret_cast<AVHWDeviceContext*>(device_->data)->hwctx);
                hw->cuda_ctx = static_cast<CUcontext>(cuda_context_);
                hw->stream = static_cast<CUstream>(cuda_stream_);
                if (av_hwdevice_ctx_init(device_) != 0) {
                    av_buffer_unref(&device_);
                }
            } else {
                av_buffer_unref(&device_);
            }
        }
        if (!open_codec(error)) {
            return false;
        }
        frame_ = av_frame_alloc();
        packet_ = av_packet_alloc();
        info_.width = codec_->width & ~1;
        info_.height = codec_->height & ~1;
        info_.fps = FbRational{rate_.num, rate_.den};
        info_.codec = decoder->name;
        info_.full_range = codec_->color_range == AVCOL_RANGE_JPEG || codec_->pix_fmt == AV_PIX_FMT_YUVJ420P;
        info_.bt601 = codec_->colorspace == AVCOL_SPC_BT470BG || codec_->colorspace == AVCOL_SPC_SMPTE170M;
        // Indices count time at the average rate, so a variable rate recording
        // can span more indices than it has frames; the end of the stream
        // settles the count once it is reached.
        if (format_->duration > 0) {
            info_.frames = av_rescale_q(format_->duration, AVRational{1, AV_TIME_BASE}, av_inv_q(rate_));
        }
        info_.frames = std::max<int64_t>(info_.frames, stream->nb_frames);
        // The first decoded frame defines frame 0.
        std::string first_error;
        if (!decode_one(first_error)) {
            error = "cannot decode the first frame: " + first_error;
            return false;
        }
        const double frame_bytes = 1.5 * info_.width * info_.height;
        const double own_budget = info_.nvdec ? kDeviceBudgetBytes : pool_->pinned() ? kPinnedBudgetBytes : kHostBudgetBytes;
        std::lock_guard<std::mutex> lock(mutex_);
        capacity_ = static_cast<size_t>(
            std::max(64.0, std::min(own_budget, static_cast<double>(frame_memory_budget())) / frame_bytes));
        read_ahead_ = static_cast<int64_t>(capacity_ * 6 / 10);
        last_activity_ms_ = GetTickCount64();
        return true;
    }

    // Worker thread only: the decoder, closed while the reader is idle.
    bool open_codec(std::string& error) {
        codec_ = avcodec_alloc_context3(decoder_);
        if (!codec_) {
            error = "cannot allocate the decoder";
            return false;
        }
        avcodec_parameters_to_context(codec_, format_->streams[stream_]->codecpar);
        codec_->pkt_timebase = time_base_;
        if (device_) {
            codec_->hw_device_ctx = av_buffer_ref(device_);
            codec_->get_format = pick_cuda;
        } else {
            codec_->thread_count = 0;  // all cores
            codec_->thread_type = FF_THREAD_FRAME | FF_THREAD_SLICE;
        }
        const int r = avcodec_open2(codec_, decoder_, nullptr);
        if (r < 0) {
            avcodec_free_context(&codec_);
            error = "cannot open decoder: " + av_error(r);
            return false;
        }
        ++decoder_opens_;
        return true;
    }

    void close_codec() {
        avcodec_free_context(&codec_);
        draining_ = false;
        std::lock_guard<std::mutex> lock(mutex_);
        need_seek_ = true;  // the demuxer is wherever the old decoder left it
        close_decoder_ = false;
    }

    // 1: a frame in frame_; 0: end of stream; -1: error.
    int receive(std::string& error) {
        for (;;) {
            int r = avcodec_receive_frame(codec_, frame_);
            if (r == 0) {
                return 1;
            }
            if (r == AVERROR_EOF) {
                return 0;
            }
            if (r != AVERROR(EAGAIN)) {
                error = "decode: " + av_error(r);
                return -1;
            }
            if (draining_) {
                return 0;
            }
            r = av_read_frame(format_, packet_);
            if (r < 0) {
                avcodec_send_packet(codec_, nullptr);
                draining_ = true;
                continue;
            }
            if (packet_->stream_index == stream_) {
                r = avcodec_send_packet(codec_, packet_);
                if (r < 0 && r != AVERROR(EAGAIN) && r != AVERROR_INVALIDDATA) {
                    av_packet_unref(packet_);
                    error = "send: " + av_error(r);
                    return -1;
                }
            }
            av_packet_unref(packet_);
        }
    }

    // Decodes the next frame and caches it under its own index.
    bool decode_one(std::string& error) {
        const int r = receive(error);
        if (r < 0) {
            return false;
        }
        if (r == 0) {
            std::lock_guard<std::mutex> lock(mutex_);
            end_ = next_ < 0 ? 0 : next_;
            if (info_.frames <= 0 || info_.frames > end_) {
                info_.frames = end_;
            }
            cv_.notify_all();
            return true;
        }
        const int64_t pts = frame_->best_effort_timestamp != AV_NOPTS_VALUE ? frame_->best_effort_timestamp : frame_->pts;
        if (origin_ == AV_NOPTS_VALUE) {
            origin_ = pts;
        }
        const int64_t index = av_rescale_q_rnd(pts - origin_, time_base_, av_inv_q(rate_), AV_ROUND_NEAR_INF);
        std::shared_ptr<Frame> frame(new Frame(), [pool = pool_](Frame* f) {
            pool->give(std::move(f->nv12));
            delete f;
        });
        frame->index = index;
        frame->timestamp_hns = av_rescale_q(pts - origin_, time_base_, AVRational{1, 10000000});
        if (frame_->format == AV_PIX_FMT_CUDA) {
            auto* frames_ctx = reinterpret_cast<AVHWFramesContext*>(frame_->hw_frames_ctx->data);
            if (frames_ctx->sw_format != AV_PIX_FMT_NV12) {
                error = std::string("NVDEC surface format ") + av_get_pix_fmt_name(frames_ctx->sw_format) + " is not NV12";
                av_frame_unref(frame_);
                return false;
            }
            // The surface copy is enqueued on our stream; finish it before use.
            fb::gpu::cuda_sync(cuda_context_, cuda_stream_);
            AVFrame* held = av_frame_clone(frame_);
            frame->luma = reinterpret_cast<unsigned long long>(held->data[0]);
            frame->chroma = reinterpret_cast<unsigned long long>(held->data[1]);
            frame->pitch = held->linesize[0];
            const int64_t surface_bytes = static_cast<int64_t>(held->linesize[0]) * info_.height * 3 / 2;
            account(&pool_->memory(), surface_bytes);
            frame->hold = std::shared_ptr<void>(held, [pool = pool_, surface_bytes](void* p) {
                account(&pool->memory(), -surface_bytes);
                AVFrame* f = static_cast<AVFrame*>(p);
                av_frame_free(&f);
            });
            info_.nvdec = true;
        } else if (!to_nv12(*frame_, frame->nv12 = pool_->take(), error)) {
            av_frame_unref(frame_);
            return false;
        }
        av_frame_unref(frame_);
        {
            std::lock_guard<std::mutex> lock(mutex_);
            ++decoded_;
            if (next_ > 0 && index > next_) {
                holds_[next_] = Hold{index, next_ - 1};
            }
            next_ = index + 1;
            cache_[index] = std::move(frame);
            evict(index);
        }
        cv_.notify_all();
        if (g_frame_bytes > frame_memory_budget()) {
            take_back_memory(this);
        }
        return true;
    }

    // Software frames (planar 4:2:0 or NV12) to compact NV12.
    bool to_nv12(const AVFrame& f, HostBuffer& out, std::string& error) {
        const int w = info_.width;
        const int h = info_.height;
        out.resize(static_cast<size_t>(w) * h * 3 / 2);
        for (int y = 0; y < h; ++y) {
            std::memcpy(out.data() + static_cast<size_t>(y) * w, f.data[0] + static_cast<ptrdiff_t>(y) * f.linesize[0], w);
        }
        uint8_t* uv = out.data() + static_cast<size_t>(w) * h;
        if (f.format == AV_PIX_FMT_NV12) {
            for (int y = 0; y < h / 2; ++y) {
                std::memcpy(uv + static_cast<size_t>(y) * w, f.data[1] + static_cast<ptrdiff_t>(y) * f.linesize[1], w);
            }
            return true;
        }
        if (f.format != AV_PIX_FMT_YUV420P && f.format != AV_PIX_FMT_YUVJ420P) {
            error = std::string("unsupported pixel format ") + av_get_pix_fmt_name(static_cast<AVPixelFormat>(f.format));
            return false;
        }
        for (int y = 0; y < h / 2; ++y) {
            const uint8_t* u = f.data[1] + static_cast<ptrdiff_t>(y) * f.linesize[1];
            const uint8_t* v = f.data[2] + static_cast<ptrdiff_t>(y) * f.linesize[2];
            uint8_t* row = uv + static_cast<size_t>(y) * w;
            for (int x = 0; x < w / 2; ++x) {
                row[2 * x] = u[x];
                row[2 * x + 1] = v[x];
            }
        }
        return true;
    }

    // Makes frame `index` available, seeking only when decoding forward is
    // not cheaper. Returns false on a decoding error.
    bool produce(int64_t index) {
        bool seek = false;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            index = resolve(index);
            if (cache_.count(index)) {
                return true;
            }
            seek = need_seek_ || index < next_ || index > next_ + forward_decode_;
        }
        if (!codec_) {
            std::string error;
            if (!open_codec(error)) {
                std::lock_guard<std::mutex> lock(mutex_);
                error_ = error;
                return false;
            }
        }
        // A seek lands on a keyframe at or before the target; if the decoder
        // still resumes past the frame, seek again further back.
        for (int attempt = 0; attempt < 4; ++attempt) {
            if (seek && !seek_to(index - attempt * forward_decode_)) {
                std::lock_guard<std::mutex> lock(mutex_);
                error_ = "seek to frame " + std::to_string(index) + " failed";
                return false;
            }
            const int result = decode_until(index);
            if (result >= 0) {
                return result == 1;
            }
            seek = true;
        }
        std::lock_guard<std::mutex> lock(mutex_);
        error_ = "decoder kept resuming past frame " + std::to_string(index) + " (next " + std::to_string(next_) + ")";
        return false;
    }

    bool seek_to(int64_t index) {
        index = std::max<int64_t>(0, index);
        const int64_t ts = origin_ + av_rescale_q(index, av_inv_q(rate_), time_base_);
        if (av_seek_frame(format_, stream_, ts, AVSEEK_FLAG_BACKWARD) < 0) {
            return false;
        }
        avcodec_flush_buffers(codec_);
        draining_ = false;
        std::lock_guard<std::mutex> lock(mutex_);
        ++seeks_;
        need_seek_ = false;
        next_ = -1;
        end_ = -1;
        return true;
    }

    // 1: frame cached or past the end; 0: decoding error; -1: decoder resumed past it.
    int decode_until(int64_t index) {
        for (int guard = 0; guard < 1000000; ++guard) {
            {
                std::lock_guard<std::mutex> lock(mutex_);
                if (cache_.count(resolve(index)) || (end_ >= 0 && index >= end_)) {
                    return 1;
                }
                if (next_ > index) {
                    return -1;
                }
            }
            std::string error;
            if (!decode_one(error)) {
                std::lock_guard<std::mutex> lock(mutex_);
                error_ = error;
                return 0;
            }
        }
        return 0;
    }

    // Keeps requested frames and the newest ones (caller holds the mutex).
    // Frames far behind the latest request will not be needed again during
    // playback (caller holds the mutex).
    void drop_stale() {
        auto keep = [this](int64_t index) { return requested(index) || index == producing_; };
        if (backward_) {
            for (auto it = cache_.upper_bound(latest_ + kKeepBehind); it != cache_.end();) {
                it = keep(it->first) ? std::next(it) : cache_.erase(it);
            }
        } else {
            for (auto it = cache_.begin(); it != cache_.end() && it->first < latest_ - kKeepBehind;) {
                it = keep(it->first) ? std::next(it) : cache_.erase(it);
            }
        }
    }

    void evict(int64_t newest) {
        drop_stale();
        while (cache_.size() > capacity_) {
            auto victim = cache_.end();
            for (auto it = cache_.begin(); it != cache_.end(); ++it) {
                if (!requested(it->first) && it->first != newest) {
                    victim = it;
                    break;  // smallest index first
                }
            }
            if (victim == cache_.end()) {
                return;
            }
            cache_.erase(victim);
        }
    }

    std::string path_;
    void* cuda_context_ = nullptr;
    void* cuda_stream_ = nullptr;
    std::thread worker_;
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    bool stop_ = false;
    bool opened_ = false;
    bool failed_ = false;
    std::string open_error_;
    std::string error_;
    Info info_;
    AVFormatContext* format_ = nullptr;
    AVCodecContext* codec_ = nullptr;
    AVBufferRef* device_ = nullptr;
    AVFrame* frame_ = nullptr;
    AVPacket* packet_ = nullptr;
    int stream_ = -1;
    AVRational time_base_{1, 1};
    AVRational rate_{0, 1};
    bool draining_ = false;
    int64_t origin_ = AV_NOPTS_VALUE;
    int64_t next_ = 0;    // index the decoder produces next (-1 right after a seek)
    int64_t end_ = -1;    // first index past the end, once reached
    int64_t target_ = 0;  // read ahead up to here
    int64_t latest_ = 0;  // most recent request
    int64_t producing_ = -1;  // frame the worker is currently decoding towards
    int64_t run_start_ = 0;   // current ascending run of requests
    int64_t run_end_ = -1;
    bool backward_ = false;   // requests move towards the file's start (reversed clip)
    size_t capacity_ = 112;
    int64_t read_ahead_ = 64;
    int64_t playback_gap_ = kSequentialGap;   // see kSequentialGap
    int64_t forward_decode_ = kForwardDecode;  // see kForwardDecode
    std::set<int64_t> requests_;
    std::map<int64_t, std::shared_ptr<const Frame>> cache_;
    // Indices [start, end) that show `frame`, keyed by start (see resolve).
    struct Hold {
        int64_t end;
        int64_t frame;
    };
    std::map<int64_t, Hold> holds_;
    std::shared_ptr<BufferPool> pool_;
    uint64_t decoded_ = 0;
    uint64_t seeks_ = 0;
    // Sharing the memory budget (see kReleaseIdleMs).
    const AVCodec* decoder_ = nullptr;
    bool need_seek_ = false;         // the decoder was reopened: the demuxer must seek first
    bool close_decoder_ = false;     // release(): close the decoder once idle
    uint64_t last_request_ms_ = 0;   // GetTickCount64 of the last get(); 0: never asked
    uint64_t last_activity_ms_ = 0;  // ... of the last get()
    uint64_t releases_ = 0;
    std::atomic<uint64_t> decoder_opens_{0};
};

// Over the budget: releases the idle readers holding memory, the one asked
// least recently first (never asked: first of all), until all
// readers are back under the budget. Readers in use are left alone, so a few
// clips playing together can go over it, each within its own capacity.
void take_back_memory(FfmpegReader* self) {
    std::lock_guard<std::mutex> registry(g_registry_mutex);
    std::set<FfmpegReader*> released;
    const uint64_t now = GetTickCount64();
    while (g_frame_bytes > frame_memory_budget()) {
        FfmpegReader* victim = nullptr;
        uint64_t victim_last = 0;
        int64_t victim_bytes = 0;
        for (FfmpegReader* reader : g_registry) {
            uint64_t last = 0;
            int64_t bytes = 0;
            if (reader == self || released.count(reader) || !reader->idle(now, last, bytes) || bytes <= 0) {
                continue;
            }
            if (!victim || last < victim_last || (last == victim_last && bytes > victim_bytes)) {
                victim = reader;
                victim_last = last;
                victim_bytes = bytes;
            }
        }
        if (!victim) {
            return;
        }
        victim->release();
        released.insert(victim);
    }
}

std::mutex g_readers_mutex;
std::map<std::tuple<std::wstring, void*, void*>, std::weak_ptr<Reader>> g_readers;

std::string narrow(const std::wstring& wide) {
    const int n = WideCharToMultiByte(CP_UTF8, 0, wide.c_str(), -1, nullptr, 0, nullptr, nullptr);
    std::string utf8(n > 0 ? n - 1 : 0, '\0');
    if (n > 1) {
        WideCharToMultiByte(CP_UTF8, 0, wide.c_str(), -1, utf8.data(), n, nullptr, nullptr);
    }
    return utf8;
}

}  // namespace

// Every block starts with a header saying whether it is page-locked (the
// first frames may get page-locked memory and later ones ordinary memory if
// the system runs short).
// The header also holds the block's size, for the memory accounting.
namespace {
constexpr size_t kHostHeader = 64;  // keeps the data 64-byte aligned
struct HostHeader {
    bool pinned;
    size_t bytes;
};
static_assert(sizeof(HostHeader) <= kHostHeader, "host block header");
}

int64_t frame_memory_bytes() {
    return g_frame_bytes;
}

// An eighth of the physical memory, from 512 MB to 2 GB: 2 GB on 16 GB. One
// reader alone may hold up to its own capacity (kPinnedBudgetBytes...).
int64_t frame_memory_budget() {
    static const int64_t budget = [] {
        MEMORYSTATUSEX status{};
        status.dwLength = sizeof(status);
        const int64_t total = GlobalMemoryStatusEx(&status) ? static_cast<int64_t>(status.ullTotalPhys) : 0;
        return std::clamp<int64_t>(total / 8, 512LL << 20, 2048LL << 20);
    }();
    return budget;
}

void* host_allocate(const HostMemory* memory, size_t bytes) {
    void* block = memory && memory->cuda_context ? fb::gpu::cuda_host_alloc(memory->cuda_context, bytes + kHostHeader)
                                                 : nullptr;
    const bool pinned = block != nullptr;
    if (!block) {
        block = ::operator new(bytes + kHostHeader, std::align_val_t(kHostHeader));
    }
    *static_cast<HostHeader*>(block) = HostHeader{pinned, bytes};
    account(memory, static_cast<int64_t>(bytes));
    return static_cast<uint8_t*>(block) + kHostHeader;
}

void host_free(const HostMemory* memory, void* p) {
    if (!p) {
        return;
    }
    void* block = static_cast<uint8_t*>(p) - kHostHeader;
    const HostHeader header = *static_cast<HostHeader*>(block);
    account(memory, -static_cast<int64_t>(header.bytes));
    if (header.pinned) {
        fb::gpu::cuda_host_free(memory ? memory->cuda_context : nullptr, block);
    } else {
        ::operator delete(block, std::align_val_t(kHostHeader));
    }
}

std::string video_codec(const std::wstring& path) {
    std::string error;
    if (!load_ffmpeg(error)) {
        return "";
    }
    AVFormatContext* format = nullptr;
    if (avformat_open_input(&format, narrow(path).c_str(), nullptr, nullptr) < 0) {
        return "";
    }
    std::string name;
    if (avformat_find_stream_info(format, nullptr) >= 0) {
        const int stream = av_find_best_stream(format, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
        if (stream >= 0) {
            name = avcodec_get_name(format->streams[stream]->codecpar->codec_id);
        }
    }
    avformat_close_input(&format);
    return name;
}

bool load_ffmpeg(std::string& error) {
    static std::once_flag once;
    static std::string result;
    std::call_once(once, [] {
        HMODULE self = nullptr;
        GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                           reinterpret_cast<LPCWSTR>(&load_ffmpeg), &self);
        wchar_t path[MAX_PATH] = {};
        GetModuleFileNameW(self, path, MAX_PATH);
        std::wstring dir(path);
        dir = dir.substr(0, dir.find_last_of(L'\\') + 1);
        // Dependency order; the delay-loaded imports then bind to these modules.
        for (const wchar_t* name : {L"avutil-60.dll", L"swresample-6.dll", L"avcodec-62.dll", L"avformat-62.dll"}) {
            if (!LoadLibraryExW((dir + name).c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH)) {
                result = "cannot load " + narrow(dir + name) + " (error " + std::to_string(GetLastError()) + ")";
                return;
            }
        }
    });
    error = result;
    return result.empty();
}

std::shared_ptr<Reader> Reader::open(const std::wstring& path, void* cuda_context, void* pinned_context,
                                     std::string& error) {
    if (!load_ffmpeg(error)) {
        return nullptr;
    }
    std::lock_guard<std::mutex> lock(g_readers_mutex);
    const auto key = std::make_tuple(path, cuda_context, pinned_context);
    if (auto existing = g_readers[key].lock()) {
        return existing;
    }
    auto reader = std::make_shared<FfmpegReader>(narrow(path), cuda_context, pinned_context);
    if (!reader->start(error)) {
        return nullptr;
    }
    g_readers[key] = reader;
    return reader;
}

}  // namespace fb::media
