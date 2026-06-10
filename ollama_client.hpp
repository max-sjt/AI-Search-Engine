#pragma once

#ifndef _WIN32_WINNT
#define _WIN32_WINNT 0x0A00
#endif

#include "httplib.h"
#include "json_utils.hpp"

#include <string>
#include <vector>

class OllamaClient {
public:
    std::string embedModel = "nomic-embed-text";
    std::string genModel = "llama3.2:1b";

    explicit OllamaClient(const std::string& host = "127.0.0.1", int port = 11434)
        : host_(host), port_(port)
    {
    }

    bool isAvailable() const {
        httplib::Client cli(host_, port_);
        cli.set_connection_timeout(2, 0);
        auto res = cli.Get("/api/tags");
        return res && res->status == 200;
    }

    std::vector<float> embed(const std::string& text) const {
        httplib::Client cli(host_, port_);
        cli.set_connection_timeout(3, 0);
        cli.set_read_timeout(30, 0);

        const std::string body =
            "{\"model\":\"" + embedModel + "\",\"prompt\":\"" + escapeJson(text) + "\"}";
        auto res = cli.Post("/api/embeddings", body, "application/json");
        if (!res || res->status != 200) {
            return {};
        }
        return parseEmbedding(res->body);
    }

    std::string generate(const std::string& prompt) const {
        httplib::Client cli(host_, port_);
        cli.set_connection_timeout(3, 0);
        cli.set_read_timeout(180, 0);

        const std::string body =
            "{\"model\":\"" + genModel + "\","
            "\"prompt\":\"" + escapeJson(prompt) + "\","
            "\"stream\":false}";
        auto res = cli.Post("/api/generate", body, "application/json");
        if (!res || res->status != 200) {
            return "ERROR: Ollama unavailable. Run: ollama serve";
        }
        return extractJsonString(res->body, "response");
    }

private:
    std::string host_;
    int port_;

    static std::string escapeJson(const std::string& s) {
        std::string out;
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
        return out;
    }

    static std::vector<float> parseEmbedding(const std::string& body) {
        size_t p = body.find("\"embedding\"");
        if (p == std::string::npos) {
            return {};
        }
        p = body.find('[', p);
        if (p == std::string::npos) {
            return {};
        }

        size_t e = p + 1;
        int depth = 1;
        while (e < body.size() && depth > 0) {
            if (body[e] == '[') {
                depth++;
            } else if (body[e] == ']') {
                depth--;
            }
            e++;
        }
        return parseVectorCsv(body.substr(p + 1, e - p - 2));
    }
};
