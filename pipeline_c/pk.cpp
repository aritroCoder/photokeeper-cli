#include "pk.h"

#include <algorithm>

#include <opencv2/imgcodecs.hpp>

#include "pipeline.h"

static_assert(PK_N == pk::N, "pk.h feature count drifted from pipeline.h");

static int finish(const cv::Mat& bgr, double out[PK_N]) {
    if (bgr.empty()) return -1;
    pk::Features f = pk::analyze(bgr);
    std::copy(f.begin(), f.end(), out);
    return 0;
}

void pk_init(const char* models_dir) { pk::MODELS = models_dir; }

int pk_analyze_path(const char* path, double out[PK_N]) {
    try {
        return finish(pk::load(path), out);
    } catch (const cv::Exception&) {  // undecodable
        return -1;
    }
}

int pk_analyze_encoded(const unsigned char* data, size_t len, double out[PK_N]) {
    try {
        return finish(cv::imdecode(cv::Mat(1, int(len), CV_8U, const_cast<unsigned char*>(data)), cv::IMREAD_COLOR), out);
    } catch (const cv::Exception&) {
        return -1;
    }
}

int pk_keep(const double features[PK_N]) {
    pk::Features f;
    std::copy(features, features + PK_N, f.begin());
    return pk::keep(f);
}
