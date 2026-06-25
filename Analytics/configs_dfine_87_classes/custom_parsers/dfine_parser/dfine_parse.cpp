#include "nvdsinfer_custom_impl.h"
#include <cstring>
#include <iostream>
#include <vector>

// Helper function to find the class with the highest confidence
static std::pair<int, float> get_max_confidence_class(const float *logits,
                                                      int num_classes) {
  int max_class_id = -1;
  float max_confidence = 0.0f;

  // Skip class 0 (background)
  for (int i = 0; i < num_classes; ++i) {
    if (logits[i] > max_confidence) {
      max_confidence = logits[i];
      max_class_id = i;
    }
  }

  return std::make_pair(max_class_id, max_confidence);
}

extern "C" bool
NvDsInferParseDFineBbox(std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
                        NvDsInferNetworkInfo const &networkInfo,
                        NvDsInferParseDetectionParams const &detectionParams,
                        std::vector<NvDsInferObjectDetectionInfo> &objectList) {

  const NvDsInferLayerInfo *logits_layer = nullptr;
  const NvDsInferLayerInfo *boxes_layer = nullptr;
  // constexpr int NUM_CLASSES = 87; // As per the logits tensor shape

  // Identify the logits and boxes layers by name
  for (const auto &layer : outputLayersInfo) {
    if (strcmp(layer.layerName, "logits") == 0) {
      logits_layer = &layer;
    } else if (strcmp(layer.layerName, "boxes") == 0) {
      boxes_layer = &layer;
    }
  }

  if (!logits_layer || !boxes_layer) {
    std::cerr << "Error: Could not find 'logits' or 'boxes' output layer."
              << std::endl;
    return false;
  }

  if (logits_layer->inferDims.numDims != 2 ||
      boxes_layer->inferDims.numDims != 2) {
    std::cerr << "Error: Unexpected tensor dimensions." << std::endl;
    return false;
  }

  int NUM_CLASSES = logits_layer->inferDims.d[1];
  const unsigned int num_detections =
      logits_layer->inferDims.d[0]; // Should be 300
  const float *logits_data = static_cast<const float *>(logits_layer->buffer);
  const float *boxes_data = static_cast<const float *>(boxes_layer->buffer);

  for (unsigned int i = 0; i < num_detections; ++i) {
    const float *current_logits = logits_data + (i * NUM_CLASSES);

    auto const [class_id, confidence] =
        get_max_confidence_class(current_logits, NUM_CLASSES);

    if (class_id < 0) {
      continue;
    }
    // Apply a confidence threshold
    if (confidence > detectionParams.perClassPreclusterThreshold[class_id]) {
      const float *bbox = boxes_data + (i * 4);

      NvDsInferObjectDetectionInfo object;
      object.classId = class_id;
      object.detectionConfidence = confidence;

      // Assuming the box format is [x_center, y_center, width, height] and
      // normalized Convert to top-left coordinates and scale to network input
      // dimensions
      object.left = (bbox[0] - bbox[2] / 2.0f) * networkInfo.width;

      object.top = (bbox[1] - bbox[3] / 2.0f) * networkInfo.height;

      object.width = std::min(bbox[2] * networkInfo.width,
                              networkInfo.width - object.left);

      object.height = std::min(bbox[3] * networkInfo.height,
                               networkInfo.height - object.top);

      objectList.push_back(object);

      // print object
      // std::cout << "Object " << i << ": " << object.classId << " "
      //           << object.detectionConfidence << " " << object.left << " "
      //           << object.top << " " << object.width << " " << object.height
      //           << std::endl;
    }
  }

  return true;
}

// Macro to validate the function signature
CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseDFineBbox);
