#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "nvdsinfer_custom_impl.h"
#include "ocr_pipeline_common.hpp"

namespace
{
const NvDsInferLayerInfo* findLayer(
    const std::vector<NvDsInferLayerInfo>& outputLayersInfo,
    const char* layerName)
{
    for (const auto& layer : outputLayersInfo) {
        if (layer.layerName != nullptr && std::strcmp(layer.layerName, layerName) == 0) {
            return &layer;
        }
    }
    return nullptr;
}

std::string uppercase_ascii(std::string s)
{
    for (char& ch : s) {
        ch = static_cast<char>(std::toupper(static_cast<unsigned char>(ch)));
    }
    return s;
}
}  // namespace

extern "C" bool NvDsInferParseCustomParseq(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    float classifierThreshold,
    std::vector<NvDsInferAttribute>& attrList,
    std::string& descString)
{
    (void)networkInfo;
    attrList.clear();
    descString.clear();

    const NvDsInferLayerInfo* outputLayer = nullptr;
    if (!outputLayersInfo.empty()) {
        // Since the ONNX output layer name might be dynamic (e.g. '5851'), just use the first one
        outputLayer = &outputLayersInfo[0];
    }
    if (outputLayer == nullptr || outputLayer->buffer == nullptr) {
        return false;
    }

    const NvDsInferDims& dims = outputLayer->inferDims;
    int sequenceLength = 0;
    int vocabSize = 0;

    if (dims.numDims == 2) {
        sequenceLength = dims.d[0];
        vocabSize = dims.d[1];
    } else if (dims.numDims == 3) {
        sequenceLength = dims.d[1];
        vocabSize = dims.d[2];
    } else {
        return false;
    }

    if (sequenceLength <= 0 || vocabSize <= 0) {
        return false;
    }

    const float* tensor = static_cast<const float*>(outputLayer->buffer);
    float avgConfidence = 0.0f;
    std::string plateText = ocrpipe::parseq_decode_logits(
        tensor,
        std::min(sequenceLength, ocrpipe::kParseqSequenceLength),
        vocabSize,
        ocrpipe::kDefaultCharset,
        &avgConfidence);

    if (plateText.empty()) {
        return false;
    }

    if (classifierThreshold > 0.0f && avgConfidence < classifierThreshold) {
        return false;
    }

    descString = uppercase_ascii(plateText);
    
    NvDsInferAttribute attr;
    attr.attributeConfidence = avgConfidence;
    attr.attributeValue = 0;
    attr.attributeLabel = strdup(descString.c_str());
    attrList.push_back(attr);
    
    // fprintf(stderr, "[PARSeq] plate=\"%s\" confidence=%.3f\n",
    //     descString.c_str(), avgConfidence);
    fflush(stderr);
    return true;
}

CHECK_CUSTOM_CLASSIFIER_PARSE_FUNC_PROTOTYPE(NvDsInferParseCustomParseq);
