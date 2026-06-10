#pragma once

#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

inline std::string jsonString(const std::string& s) {
    std::string out = "\"";
    for (char c : s) {
        if (c == '"') {
            out += "\\\"";
        } else if (c == '\\') {
            out += "\\\\";
        } else if (c == '\n') {
            out += "\\n";
        } else if (c == '\r') {
            out += "\\r";
        } else if (c == '\t') {
            out += "\\t";
        } else {
            out += c;
        }
    }
    return out + '"';
}

inline std::string jsonVector(const std::vector<float>& v) {
    std::ostringstream ss;
    ss << '[';
    for (size_t i = 0; i < v.size(); i++) {
        if (i) {
            ss << ',';
        }
        ss << std::fixed << std::setprecision(4) << v[i];
    }
    ss << ']';
    return ss.str();
}

inline std::vector<float> parseVectorCsv(const std::string& s) {
    std::vector<float> v;
    std::istringstream ss(s);
    std::string token;
    while (std::getline(ss, token, ',')) {
        try {
            v.push_back(std::stof(token));
        } catch (...) {
        }
    }
    return v;
}

inline std::string extractJsonString(const std::string& body, const std::string& key) {
    size_t p = body.find('"' + key + '"');
    if (p == std::string::npos) {
        return "";
    }
    p = body.find(':', p);
    if (p == std::string::npos) {
        return "";
    }
    p++;
    while (p < body.size() && (body[p] == ' ' || body[p] == '\t')) {
        p++;
    }
    if (p >= body.size() || body[p] != '"') {
        return "";
    }
    p++;

    std::string result;
    while (p < body.size()) {
        if (body[p] == '"') {
            break;
        }
        if (body[p] == '\\' && p + 1 < body.size()) {
            p++;
            switch (body[p]) {
                case '"':
                    result += '"';
                    break;
                case '\\':
                    result += '\\';
                    break;
                case 'n':
                    result += '\n';
                    break;
                case 'r':
                    result += '\r';
                    break;
                case 't':
                    result += '\t';
                    break;
                default:
                    result += body[p];
                    break;
            }
        } else {
            result += body[p];
        }
        p++;
    }
    return result;
}

inline int extractJsonInt(const std::string& body, const std::string& key, int fallback = 0) {
    size_t p = body.find('"' + key + '"');
    if (p == std::string::npos) {
        return fallback;
    }
    p = body.find(':', p);
    if (p == std::string::npos) {
        return fallback;
    }
    p++;
    while (p < body.size() && (body[p] == ' ' || body[p] == '\t')) {
        p++;
    }
    try {
        return std::stoi(body.substr(p));
    } catch (...) {
        return fallback;
    }
}

inline std::vector<float> extractJsonFloatArray(const std::string& body, const std::string& key) {
    size_t p = body.find('"' + key + '"');
    if (p == std::string::npos) {
        return {};
    }
    p = body.find('[', p);
    if (p == std::string::npos) {
        return {};
    }
    size_t e = body.find(']', p);
    if (e == std::string::npos) {
        return {};
    }
    return parseVectorCsv(body.substr(p + 1, e - p - 1));
}

inline bool parseVectorItemBody(
    const std::string& body,
    std::string& metadata,
    std::string& category,
    std::vector<float>& embedding)
{
    metadata = extractJsonString(body, "metadata");
    category = extractJsonString(body, "category");
    embedding = extractJsonFloatArray(body, "embedding");
    return !metadata.empty() && !embedding.empty();
}
