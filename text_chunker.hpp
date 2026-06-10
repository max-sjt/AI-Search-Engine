#pragma once

#include <algorithm>
#include <sstream>
#include <string>
#include <vector>

inline std::vector<std::string> chunkText(
    const std::string& text,
    int chunkWords = 250,
    int overlapWords = 30)
{
    std::istringstream ss(text);
    std::vector<std::string> words;
    std::string word;
    while (ss >> word) {
        words.push_back(word);
    }

    if (words.empty()) {
        return {};
    }
    if ((int)words.size() <= chunkWords) {
        return {text};
    }

    std::vector<std::string> chunks;
    const int step = std::max(1, chunkWords - overlapWords);
    for (int i = 0; i < (int)words.size(); i += step) {
        const int end = std::min(i + chunkWords, (int)words.size());
        std::string chunk;
        for (int j = i; j < end; j++) {
            if (j > i) {
                chunk += ' ';
            }
            chunk += words[j];
        }
        chunks.push_back(chunk);
        if (end == (int)words.size()) {
            break;
        }
    }
    return chunks;
}
