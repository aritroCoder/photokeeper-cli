#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>

#include <opencv2/imgcodecs.hpp>

#include "pipeline.h"

namespace pk {

static uint32_t be(const unsigned char* p, int n) {
    uint32_t v = 0;
    for (int i = 0; i < n; ++i) v = v << 8 | p[i];
    return v;
}
static uint32_t le(const unsigned char* p, int n) {
    uint32_t v = 0;
    while (n--) v = v << 8 | p[n];
    return v;
}

// Long edge read from the file header alone, no pixels decoded. 0 if unknown.
// shortcut: JPEG, PNG and TIFF only, anything else decodes at full size; add a format when a gallery shows it
static int header_long_edge(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    unsigned char h[24];
    if (!f.read((char*)h, sizeof h)) return 0;
    if (!std::memcmp(h, "\x89PNG", 4)) return std::max(be(h + 16, 4), be(h + 20, 4));
    if (!std::memcmp(h, "II*\0", 4) || !std::memcmp(h, "MM\0*", 4)) {
        auto rd = [little = h[0] == 'I'](const unsigned char* p, int n) { return little ? le(p, n) : be(p, n); };
        unsigned char e[12];
        f.seekg(rd(h + 4, 4));
        f.read((char*)e, 2);
        uint32_t w = 0, hgt = 0;
        for (uint32_t k = rd(e, 2); k-- && f.read((char*)e, 12);) {
            uint32_t tag = rd(e, 2), val = rd(e + 8, rd(e + 2, 2) == 3 ? 2 : 4);  // type 3 is SHORT, left-justified
            if (tag == 256) w = val;
            if (tag == 257) hgt = val;
        }
        return std::max(w, hgt);
    }
    if (h[0] == 0xFF && h[1] == 0xD8) {  // JPEG: walk segments to the first SOF, which holds height then width
        f.seekg(2);
        unsigned char s[9];
        while (f.read((char*)s, 4) && s[0] == 0xFF) {
            if (s[1] == 0xFF) {  // fill byte
                f.seekg(-3, std::ios::cur);
                continue;
            }
            if (s[1] >= 0xC0 && s[1] <= 0xCF && s[1] != 0xC4 && s[1] != 0xC8 && s[1] != 0xCC)
                return f.read((char*)s + 4, 5) ? std::max(be(s + 5, 2), be(s + 7, 2)) : 0;
            f.seekg(be(s + 2, 2) - 2, std::ios::cur);
        }
    }
    return 0;
}

cv::Mat load(const std::string& path) {
    int long_edge = header_long_edge(path);
    // codec-side downscale; JPEG skips the inverse DCT for dropped frequencies, so this is far cheaper than resize
    int flag = long_edge >= 8 * SIDE   ? cv::IMREAD_REDUCED_COLOR_8
               : long_edge >= 4 * SIDE ? cv::IMREAD_REDUCED_COLOR_4
               : long_edge >= 2 * SIDE ? cv::IMREAD_REDUCED_COLOR_2
                                       : cv::IMREAD_COLOR;
    return thumbnail(cv::imread(path, flag));
}

}  // namespace pk
