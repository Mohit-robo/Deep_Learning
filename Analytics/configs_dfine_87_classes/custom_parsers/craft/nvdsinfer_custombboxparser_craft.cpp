#include "nvdsinfer_custom_impl.h"
#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>
#include <cmath>
#include <string>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudafilters.hpp>

extern "C" bool NvDsInferParseCustomCRAFT(
    std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
    NvDsInferNetworkInfo const &networkInfo,
    NvDsInferParseDetectionParams const &detectionParams,
    std::vector<NvDsInferParseObjectInfo> &objectList)
{
    // Find the output layer
    const NvDsInferLayerInfo *outputLayer = nullptr;
    for (const auto &layer : outputLayersInfo) {
        if (std::string(layer.layerName) == "output") {
            outputLayer = &layer;
            break;
        }
    }

    if (!outputLayer) {
        std::cerr << "ERROR: NvDsInferParseCustomCRAFT could not find 'output' layer" << std::endl;
        return false;
    }

    // outputLayer->dims should be [H, W, 2] or [2, H, W]
    // Let's assume the Triton config had [-1, -1, -1, 2] (NHWC) so dims are [H, W, 2]
    int H = 0, W = 0;
    
    if (outputLayer->inferDims.numElements == 0) return false;
    
    if (outputLayer->inferDims.numDims == 3) {
        if (outputLayer->inferDims.d[2] == 2) {
            H = outputLayer->inferDims.d[0];
            W = outputLayer->inferDims.d[1];
        } else if (outputLayer->inferDims.d[0] == 2) {
            H = outputLayer->inferDims.d[1];
            W = outputLayer->inferDims.d[2];
        }
    }

    if (H == 0 || W == 0) {
        std::cerr << "ERROR: Unexpected dims for CRAFT output" << std::endl;
        return false;
    }

    float* outputData = (float*)outputLayer->buffer;

    // CRAFT Post-processing thresholds
    float text_threshold = 0.6f;
    float link_threshold = 0.2f;
    float low_text = 0.4f;
    int lowest_allowed_area = 4;

    // We need to extract score_text and score_link
    cv::Mat textmap(H, W, CV_32FC1);
    cv::Mat linkmap(H, W, CV_32FC1);

    if (outputLayer->inferDims.d[2] == 2) {
        // HWC format
        for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) {
                textmap.at<float>(y, x) = outputData[(y * W + x) * 2 + 0];
                linkmap.at<float>(y, x) = outputData[(y * W + x) * 2 + 1];
            }
        }
    } else {
        // CHW format
        for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) {
                textmap.at<float>(y, x) = outputData[0 * H * W + y * W + x];
                linkmap.at<float>(y, x) = outputData[1 * H * W + y * W + x];
            }
        }
    }

    // Thresholding using CUDA
    cv::cuda::GpuMat d_textmap(textmap);
    cv::cuda::GpuMat d_linkmap(linkmap);
    cv::cuda::GpuMat d_text_score, d_link_score;

    cv::cuda::threshold(d_textmap, d_text_score, low_text, 1.0, cv::THRESH_BINARY);
    cv::cuda::threshold(d_linkmap, d_link_score, link_threshold, 1.0, cv::THRESH_BINARY);

    // Combine
    cv::cuda::GpuMat d_text_score_comb;
    cv::cuda::add(d_text_score, d_link_score, d_text_score_comb);
    cv::cuda::threshold(d_text_score_comb, d_text_score_comb, 1.0, 1.0, cv::THRESH_TRUNC);
    
    cv::cuda::GpuMat d_text_score_comb_8u;
    d_text_score_comb.convertTo(d_text_score_comb_8u, CV_8UC1);

    cv::Mat text_score_comb_8u;
    d_text_score_comb_8u.download(text_score_comb_8u);

    // We also need text_score and link_score on CPU for the loop below
    cv::Mat text_score, link_score;
    d_text_score.download(text_score);
    d_link_score.download(link_score);

    // Connected Components
    cv::Mat labels, stats, centroids;
    int nLabels = cv::connectedComponentsWithStats(text_score_comb_8u, labels, stats, centroids, 4);

    cv::Mat segmap = cv::Mat::zeros(H, W, CV_8UC1);

    for (int k = 1; k < nLabels; ++k) {
        int size = stats.at<int>(k, cv::CC_STAT_AREA);
        if (size < lowest_allowed_area) continue;

        // Check if max text score in this component >= text_threshold
        double maxVal = 0;
        cv::Mat mask = (labels == k);
        cv::minMaxLoc(textmap, nullptr, &maxVal, nullptr, nullptr, mask);
        if (maxVal < text_threshold) continue;

        segmap.setTo(0);
        segmap.setTo(255, mask);

        // Remove link area (link_score == 1 AND text_score == 0)
        cv::Mat link_only = (link_score == 1.0) & (text_score == 0.0);
        segmap.setTo(0, link_only);

        int x = stats.at<int>(k, cv::CC_STAT_LEFT);
        int y = stats.at<int>(k, cv::CC_STAT_TOP);
        int w = stats.at<int>(k, cv::CC_STAT_WIDTH);
        int h = stats.at<int>(k, cv::CC_STAT_HEIGHT);

        int niter = (int)(sqrt(size * std::min(w, h) / (float)(w * h)) * 2);
        int sx = std::max(0, x - niter);
        int sy = std::max(0, y - niter);
        int ex = std::min(W, x + w + niter + 1);
        int ey = std::min(H, y + h + niter + 1);

        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(1 + niter, 1 + niter));
        cv::dilate(segmap(cv::Rect(sx, sy, ex - sx, ey - sy)), segmap(cv::Rect(sx, sy, ex - sx, ey - sy)), kernel);

        // Find contours
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(segmap, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        if (contours.empty()) continue;

        // Use the largest contour
        int largest_idx = 0;
        double max_area = 0;
        for (size_t i = 0; i < contours.size(); i++) {
            double area = cv::contourArea(contours[i]);
            if (area > max_area) {
                max_area = area;
                largest_idx = i;
            }
        }

        // DeepStream requires axis-aligned rectangles for SGIE cropping
        cv::Rect bounding_rect = cv::boundingRect(contours[largest_idx]);

        // Map the coordinates back to the original network input size (networkInfo.width/height)
        // Note: The output map (H x W) is usually half the size of the network input.
        float ratio_w = (float)networkInfo.width / W;
        float ratio_h = (float)networkInfo.height / H;

        NvDsInferParseObjectInfo obj;
        obj.classId = 0; // Word class
        obj.detectionConfidence = maxVal;
        
        // Expand slightly for OCR context
        int expand_x = 2;
        int expand_y = 2;

        obj.left = std::max(0.0f, (bounding_rect.x - expand_x) * ratio_w);
        obj.top = std::max(0.0f, (bounding_rect.y - expand_y) * ratio_h);
        obj.width = std::min((float)networkInfo.width - obj.left, (bounding_rect.width + 2*expand_x) * ratio_w);
        obj.height = std::min((float)networkInfo.height - obj.top, (bounding_rect.height + 2*expand_y) * ratio_h);

        objectList.push_back(obj);
    }

    return true;
}

// DeepStream plugin symbol
CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseCustomCRAFT);
