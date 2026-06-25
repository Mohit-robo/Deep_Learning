#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <unordered_map>

struct WordInfo {
    std::string text;
    float cx, cy, w, h;
    std::vector<float> confs;
    std::vector<std::pair<float, float>> poly;
};

struct PlateResult {
    char* text;
    float* confs;
    int num_confs;
    char* status;
};

// Levenshtein ratio for sequence matching
float sequence_matcher_ratio(const std::string& a, const std::string& b) {
    if (a.empty() || b.empty()) return 0.0f;
    int len_a = a.length();
    int len_b = b.length();
    std::vector<std::vector<int>> d(len_a + 1, std::vector<int>(len_b + 1));
    for (int i = 0; i <= len_a; ++i) d[i][0] = i;
    for (int j = 0; j <= len_b; ++j) d[0][j] = j;
    for (int i = 1; i <= len_a; ++i) {
        for (int j = 1; j <= len_b; ++j) {
            int cost = (a[i - 1] == b[j - 1]) ? 0 : 1;
            d[i][j] = std::min({ d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost });
        }
    }
    return static_cast<float>(len_a + len_b - d[len_a][len_b]) / (len_a + len_b);
}

std::string text_correction(std::string txt) {
    std::vector<std::string> state_name_list = {
        "AP", "AR", "AS", "BR", "CG", "GA", "GJ", "HR", "HP", "JH", "KA", "KL", "MP", 
        "MH", "MN", "ML", "MZ", "NL", "OD", "PB", "RJ", "SK", "TN", "TG", "TR", "UP", 
        "UK", "WB", "AN", "CH", "DN", "DD", "DL", "JK", "LA", "LD", "PY"
    };
    char number_list_replace[] = {'0','0','0','1','1','1','4','4','2','3','4','4','5','5','6','6','7','8','9','9'};
    char alpa_list_replace[]   = {'O','o','D','I','i','E','L','Z','z','H','A','u','S','s','G','b','T','B','q','P'};
    
    if (txt.length() == 10) {
        std::string state_code = txt.substr(0, 2);
        
        bool is_alpha = isalpha(state_code[0]) && isalpha(state_code[1]);
        bool found = false;
        
        if (is_alpha) {
            for (const auto& s : state_name_list) {
                if (state_code == s) { found = true; break; }
            }
        }
        
        if (!found) {
            float max_ratio = 0;
            std::string best_match = "";
            for (const auto& s : state_name_list) {
                float r = sequence_matcher_ratio(state_code, s);
                if (r > max_ratio) { max_ratio = r; best_match = s; }
            }
            if (max_ratio >= 0.5f && !best_match.empty()) {
                txt.replace(0, 2, best_match);
            }
        }
        
        for (int i = 2; i < 4; ++i) {
            if (!isdigit(txt[i])) {
                for (int j = 0; j < sizeof(alpa_list_replace); ++j) {
                    if (txt[i] == alpa_list_replace[j]) {
                        txt[i] = number_list_replace[j];
                        break;
                    }
                }
            }
        }
        
        for (int i = 4; i < 6; ++i) {
            if (!isalpha(txt[i])) {
                for (int j = 0; j < sizeof(number_list_replace); ++j) {
                    if (txt[i] == number_list_replace[j]) {
                        txt[i] = alpa_list_replace[j];
                        break;
                    }
                }
            }
        }
        
        for (int i = 6; i < 10; ++i) {
            if (!isdigit(txt[i])) {
                for (int j = 0; j < sizeof(alpa_list_replace); ++j) {
                    if (txt[i] == alpa_list_replace[j]) {
                        txt[i] = number_list_replace[j];
                        break;
                    }
                }
            }
        }
    }
    return txt;
}

std::string correct_numberplate(std::string txt) {
    std::string clean = "";
    for (char c : txt) {
        if (isalnum(c)) clean += c;
    }
    return text_correction(clean);
}

extern "C" PlateResult* process_plate(
    const char** words_text, 
    float** words_confs, 
    int* words_num_confs,
    float** words_polys,
    int num_words
) {
    std::vector<WordInfo> words;
    for (int i = 0; i < num_words; ++i) {
        WordInfo w;
        w.text = words_text[i];
        
        // Exclude invalid short alphabet-only outputs
        bool all_alpha = true;
        for (char c : w.text) { if (!isalpha(c)) { all_alpha = false; break; } }
        if (w.text.length() >= 4 && all_alpha) {
            w.text = "";
        }
        
        for (int j = 0; j < words_num_confs[i]; ++j) {
            w.confs.push_back(words_confs[i][j]);
        }
        
        float min_x = 10000, max_x = 0, min_y = 10000, max_y = 0;
        float sum_x = 0, sum_y = 0;
        for (int j = 0; j < 4; ++j) {
            float px = words_polys[i][j*2];
            float py = words_polys[i][j*2+1];
            w.poly.push_back({px, py});
            min_x = std::min(min_x, px);
            max_x = std::max(max_x, px);
            min_y = std::min(min_y, py);
            max_y = std::max(max_y, py);
            sum_x += px;
            sum_y += py;
        }
        w.cx = sum_x / 4.0f;
        w.cy = sum_y / 4.0f;
        w.w = max_x - min_x;
        w.h = max_y - min_y;
        
        if (!w.text.empty()) {
            words.push_back(w);
        }
    }

    if (words.empty()) {
        PlateResult* res = new PlateResult;
        res->text = strdup("");
        res->num_confs = 0;
        res->confs = nullptr;
        res->status = strdup("single");
        return res;
    }

    std::vector<std::vector<WordInfo>> lines;
    for (auto& w : words) {
        bool added = false;
        for (auto& line : lines) {
            float line_cy = line.back().cy;
            float line_h = line.back().h;
            if (std::abs(line_cy - w.cy) < line_h / 2.0f) {
                line.push_back(w);
                added = true;
                break;
            }
        }
        if (!added) {
            lines.push_back({w});
        }
    }

    std::sort(lines.begin(), lines.end(), [](const std::vector<WordInfo>& a, const std::vector<WordInfo>& b) {
        return a.front().cy < b.front().cy;
    });

    std::string final_string = "";
    std::vector<float> final_confs;
    
    for (auto& line : lines) {
        std::sort(line.begin(), line.end(), [](const WordInfo& a, const WordInfo& b) {
            return a.cx < b.cx;
        });
        
        for (auto& w : line) {
            final_string += w.text;
            final_confs.insert(final_confs.end(), w.confs.begin(), w.confs.end());
        }
    }

    std::string corrected_string = correct_numberplate(final_string);

    PlateResult* res = new PlateResult;
    res->text = strdup(corrected_string.c_str());
    res->num_confs = final_confs.size();
    if (res->num_confs > 0) {
        res->confs = new float[res->num_confs];
        for (int i = 0; i < res->num_confs; ++i) {
            res->confs[i] = final_confs[i];
        }
    } else {
        res->confs = nullptr;
    }
    res->status = strdup((lines.size() == 1) ? "single" : "double");
    
    return res;
}

extern "C" void free_plate_result(PlateResult* res) {
    if (res) {
        if (res->text) free(res->text);
        if (res->status) free(res->status);
        if (res->confs) delete[] res->confs;
        delete res;
    }
}
