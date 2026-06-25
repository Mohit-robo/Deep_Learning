#pragma once

#include <algorithm>
#include <deque>
#include <cmath>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudafilters.hpp>

namespace ocrpipe
{
inline constexpr float kCraftTextThreshold = 0.6f;
inline constexpr float kCraftLinkThreshold = 0.2f;
inline constexpr int kParseqSequenceLength = 26;
inline constexpr int kParseqExpectedVocab = 95;
inline constexpr const char* kDefaultCharset =
    "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~";

struct CraftPreprocessMeta
{
    float scale = 1.0f;
    int pad_x = 0;
    int pad_y = 0;
    int orig_width = 0;
    int orig_height = 0;
};

inline std::mutex& craft_preprocess_meta_mutex()
{
    static std::mutex mutex;
    return mutex;
}

inline std::deque<CraftPreprocessMeta>& craft_preprocess_meta_queue()
{
    static std::deque<CraftPreprocessMeta> queue;
    return queue;
}

inline void push_craft_preprocess_meta(const CraftPreprocessMeta& meta)
{
    std::lock_guard<std::mutex> lock(craft_preprocess_meta_mutex());
    craft_preprocess_meta_queue().push_back(meta);
}

inline bool pop_craft_preprocess_meta(CraftPreprocessMeta* meta)
{
    if (meta == nullptr) {
        return false;
    }

    std::lock_guard<std::mutex> lock(craft_preprocess_meta_mutex());
    auto& queue = craft_preprocess_meta_queue();
    if (queue.empty()) {
        return false;
    }
    *meta = queue.front();
    queue.pop_front();
    return true;
}

inline cv::Point2f safe_point(float x, float y)
{
    return cv::Point2f(x, y);
}

inline std::vector<cv::Point2f> order_points(const std::vector<cv::Point2f>& pts)
{
    std::vector<cv::Point2f> rect(4);
    if (pts.size() != 4) {
        return rect;
    }

    std::vector<cv::Point2f> in = pts;
    std::sort(in.begin(), in.end(), [](const cv::Point2f& a, const cv::Point2f& b) {
        return (a.x + a.y) < (b.x + b.y);
    });
    rect[0] = in[0];
    rect[2] = in[3];

    std::sort(in.begin(), in.end(), [](const cv::Point2f& a, const cv::Point2f& b) {
        return (a.y - a.x) < (b.y - b.x);
    });
    rect[1] = in[0];
    rect[3] = in[3];
    return rect;
}

inline float softmax_probability(const float* logits, int vocab_size, int max_index)
{
    if (logits == nullptr || vocab_size <= 0 || max_index < 0 || max_index >= vocab_size) {
        return 0.0f;
    }

    float max_logit = logits[0];
    for (int i = 1; i < vocab_size; ++i) {
        max_logit = std::max(max_logit, logits[i]);
    }

    float denom = 0.0f;
    for (int i = 0; i < vocab_size; ++i) {
        denom += std::exp(logits[i] - max_logit);
    }
    if (denom <= 0.0f) {
        return 0.0f;
    }
    return std::exp(logits[max_index] - max_logit) / denom;
}

inline std::vector<std::vector<cv::Point2f>> craft_heatmap_to_boxes(
    const float* output,
    int height,
    int width,
    int channels,
    float text_threshold = kCraftTextThreshold,
    float link_threshold = kCraftLinkThreshold)
{
    std::vector<std::vector<cv::Point2f>> boxes;
    if (output == nullptr || height <= 0 || width <= 0 || channels < 2) {
        return boxes;
    }

    cv::cuda::GpuMat d_heatmap(height, width, CV_32FC(channels), const_cast<float*>(output));
    std::vector<cv::cuda::GpuMat> planes;
    cv::cuda::split(d_heatmap, planes);

    const float low_text = 0.4f;
    cv::cuda::GpuMat d_text_mask;
    cv::cuda::GpuMat d_link_mask;
    cv::cuda::GpuMat d_combined_mask;

    cv::cuda::threshold(planes[0], d_text_mask, low_text, 1.0, cv::THRESH_BINARY);
    cv::cuda::threshold(planes[1], d_link_mask, link_threshold, 1.0, cv::THRESH_BINARY);
    d_text_mask.convertTo(d_text_mask, CV_8U);
    d_link_mask.convertTo(d_link_mask, CV_8U);
    cv::cuda::bitwise_or(d_text_mask, d_link_mask, d_combined_mask);

    cv::Mat text_mask;
    cv::Mat link_mask;
    cv::Mat combined_mask;
    d_text_mask.download(text_mask);
    d_link_mask.download(link_mask);
    d_combined_mask.download(combined_mask);

    if (cv::countNonZero(combined_mask) == 0) {
        return boxes;
    }

    cv::Mat labels, stats, centroids;
    const int n_labels = cv::connectedComponentsWithStats(combined_mask, labels, stats, centroids, 4, CV_32S);
    for (int k = 1; k < n_labels; ++k) {
        const int area = stats.at<int>(k, cv::CC_STAT_AREA);
        if (area < 4) {
            continue;
        }

        cv::Mat label_mask = labels == k;
        double max_text = 0.0;
        cv::minMaxLoc(planes[0], nullptr, &max_text, nullptr, nullptr, label_mask);
        if (max_text < text_threshold) {
            continue;
        }

        cv::Mat segmap = cv::Mat::zeros(height, width, CV_8U);
        segmap.setTo(255, label_mask);
        segmap.setTo(0, (link_mask == 1) & (text_mask == 0));

        const int x = stats.at<int>(k, cv::CC_STAT_LEFT);
        const int y = stats.at<int>(k, cv::CC_STAT_TOP);
        const int w = stats.at<int>(k, cv::CC_STAT_WIDTH);
        const int h = stats.at<int>(k, cv::CC_STAT_HEIGHT);
        if (w <= 0 || h <= 0) {
            continue;
        }

        const int niter = std::max(
            0,
            static_cast<int>(std::sqrt(static_cast<double>(area) * static_cast<double>(std::min(w, h)) /
                                       static_cast<double>(w * h)) * 2.0));

        int sx = std::max(0, x - niter);
        int ex = std::min(width, x + w + niter + 1);
        int sy = std::max(0, y - niter);
        int ey = std::min(height, y + h + niter + 1);
        if (ex <= sx || ey <= sy) {
            continue;
        }

        cv::Mat roi = segmap(cv::Rect(sx, sy, ex - sx, ey - sy)).clone();
        const cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(1 + niter, 1 + niter));
        cv::dilate(roi, roi, kernel);
        roi.copyTo(segmap(cv::Rect(sx, sy, ex - sx, ey - sy)));

        std::vector<cv::Point> nonzero;
        cv::findNonZero(segmap, nonzero);
        if (nonzero.empty()) {
            continue;
        }

        cv::RotatedRect rr = cv::minAreaRect(nonzero);
        cv::Point2f pts[4];
        rr.points(pts);
        std::vector<cv::Point2f> quad = {pts[0], pts[1], pts[2], pts[3]};

        const float box_w = std::hypot(quad[0].x - quad[1].x, quad[0].y - quad[1].y);
        const float box_h = std::hypot(quad[1].x - quad[2].x, quad[1].y - quad[2].y);
        const float box_ratio = std::max(box_w, box_h) / (std::min(box_w, box_h) + 1e-5f);
        if (std::abs(1.0f - box_ratio) <= 0.1f) {
            const auto minmax_x = std::minmax_element(nonzero.begin(), nonzero.end(), [](const cv::Point& a, const cv::Point& b) {
                return a.x < b.x;
            });
            const auto minmax_y = std::minmax_element(nonzero.begin(), nonzero.end(), [](const cv::Point& a, const cv::Point& b) {
                return a.y < b.y;
            });
            const float l = static_cast<float>(minmax_x.first->x);
            const float r = static_cast<float>(minmax_x.second->x);
            const float t = static_cast<float>(minmax_y.first->y);
            const float b = static_cast<float>(minmax_y.second->y);
            quad = {
                safe_point(l, t),
                safe_point(r, t),
                safe_point(r, b),
                safe_point(l, b),
            };
        }

        const std::vector<cv::Point2f> ordered = order_points(quad);
        boxes.push_back(ordered);
    }

    return boxes;
}

inline std::string parseq_decode_logits(
    const float* logits,
    int sequence_length,
    int vocab_size,
    const std::string& charset = kDefaultCharset,
    float* avg_confidence = nullptr)
{
    std::string result;
    if (logits == nullptr || sequence_length <= 0 || vocab_size <= 0) {
        return result;
    }

    const int used_vocab = std::min(vocab_size, static_cast<int>(charset.size()) + 3);
    float sum_conf = 0.0f;
    int steps = 0;

    for (int t = 0; t < sequence_length; ++t) {
        const float* step = logits + (static_cast<std::size_t>(t) * static_cast<std::size_t>(vocab_size));
        int max_index = 0;
        float max_logit = step[0];
        for (int i = 1; i < used_vocab; ++i) {
            if (step[i] > max_logit) {
                max_logit = step[i];
                max_index = i;
            }
        }

        const float prob = softmax_probability(step, used_vocab, max_index);
        sum_conf += prob;
        ++steps;

        if (max_index == 0) {
            break;
        }
        if (max_index == used_vocab - 2 || max_index == used_vocab - 1) {
            continue;
        }

        const int char_index = max_index - 1;
        if (char_index >= 0 && char_index < static_cast<int>(charset.size())) {
            result.push_back(charset[static_cast<std::size_t>(char_index)]);
        }
    }

    if (avg_confidence != nullptr) {
        *avg_confidence = (steps > 0) ? (sum_conf / static_cast<float>(steps)) : 0.0f;
    }
    return result;
}
}  // namespace ocrpipe
