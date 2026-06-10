#pragma once

#ifndef _WIN32_WINNT
#define _WIN32_WINNT 0x0A00
#endif

#include "httplib.h"
#include "json_utils.hpp"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

class AiProvider {
public:
    virtual ~AiProvider() = default;

    virtual std::string name() const = 0;
    virtual std::string embeddingModel() const = 0;
    virtual std::string generationModel() const = 0;
    virtual std::string setupHint() const = 0;
    virtual bool isAvailable() const = 0;
    virtual std::vector<float> embed(const std::string& text) const = 0;
    virtual std::string generate(const std::string& prompt) const = 0;
};

inline std::string getenvOr(const char* key, const std::string& fallback) {
    const char* value = std::getenv(key);
    return value && *value ? std::string(value) : fallback;
}

inline std::string lowerCopy(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
        return (char)std::tolower(c);
    });
    return s;
}

class OllamaProvider : public AiProvider {
public:
    OllamaProvider(
        const std::string& host = getenvOr("OLLAMA_HOST", "127.0.0.1"),
        int port = std::stoi(getenvOr("OLLAMA_PORT", "11434")))
        : host_(host),
          port_(port),
          embedModel_(getenvOr("OLLAMA_EMBED_MODEL", "nomic-embed-text")),
          genModel_(getenvOr("OLLAMA_GEN_MODEL", "llama3.2:1b"))
    {
    }

    std::string name() const override {
        return "ollama";
    }

    std::string embeddingModel() const override {
        return embedModel_;
    }

    std::string generationModel() const override {
        return genModel_;
    }

    std::string setupHint() const override {
        return "Run: ollama serve; ollama pull " + embedModel_ + "; ollama pull " + genModel_;
    }

    bool isAvailable() const override {
        httplib::Client cli(host_, port_);
        cli.set_connection_timeout(2, 0);
        auto res = cli.Get("/api/tags");
        return res && res->status == 200;
    }

    std::vector<float> embed(const std::string& text) const override {
        httplib::Client cli(host_, port_);
        cli.set_connection_timeout(3, 0);
        cli.set_read_timeout(30, 0);

        const std::string body =
            "{\"model\":\"" + embedModel_ + "\",\"prompt\":" + jsonString(text) + "}";
        auto res = cli.Post("/api/embeddings", body, "application/json");
        if (!res || res->status != 200) {
            return {};
        }
        return parseEmbedding(res->body);
    }

    std::string generate(const std::string& prompt) const override {
        httplib::Client cli(host_, port_);
        cli.set_connection_timeout(3, 0);
        cli.set_read_timeout(180, 0);

        const std::string body =
            "{\"model\":\"" + genModel_ + "\","
            "\"prompt\":" + jsonString(prompt) + ","
            "\"stream\":false}";
        auto res = cli.Post("/api/generate", body, "application/json");
        if (!res || res->status != 200) {
            return "ERROR: Ollama unavailable. " + setupHint();
        }
        return extractJsonString(res->body, "response");
    }

private:
    std::string host_;
    int port_;
    std::string embedModel_;
    std::string genModel_;

    static std::vector<float> parseEmbedding(const std::string& body) {
        return extractJsonFloatArray(body, "embedding");
    }
};

class QwenProvider : public AiProvider {
public:
    QwenProvider()
        : host_(getenvOr("QWEN_API_HOST", "dashscope.aliyuncs.com")),
          basePath_(getenvOr("QWEN_API_BASE_PATH", "/compatible-mode/v1")),
          apiKey_(getenvOr("QWEN_API_KEY", getenvOr("DASHSCOPE_API_KEY", ""))),
          embedModel_(getenvOr("QWEN_EMBED_MODEL", "text-embedding-v4")),
          genModel_(getenvOr("QWEN_GEN_MODEL", "qwen-plus"))
    {
    }

    std::string name() const override {
        return "qwen";
    }

    std::string embeddingModel() const override {
        return embedModel_;
    }

    std::string generationModel() const override {
        return genModel_;
    }

    std::string setupHint() const override {
        return "Set QWEN_API_KEY or DASHSCOPE_API_KEY. Build with CPPHTTPLIB_OPENSSL_SUPPORT for HTTPS.";
    }

    bool isAvailable() const override {
        if (apiKey_.empty()) {
            return false;
        }
#ifdef CPPHTTPLIB_OPENSSL_SUPPORT
        httplib::SSLClient cli(host_, 443);
        configure(cli);
        auto res = cli.Get((basePath_ + "/models").c_str());
        return res && res->status >= 200 && res->status < 300;
#else
        return false;
#endif
    }

    std::vector<float> embed(const std::string& text) const override {
        if (apiKey_.empty()) {
            return {};
        }
#ifdef CPPHTTPLIB_OPENSSL_SUPPORT
        httplib::SSLClient cli(host_, 443);
        configure(cli);
        const std::string body =
            "{\"model\":" + jsonString(embedModel_) + ",\"input\":" + jsonString(text) + "}";
        auto res = cli.Post((basePath_ + "/embeddings").c_str(), body, "application/json");
        if (!res || res->status < 200 || res->status >= 300) {
            return {};
        }
        return extractJsonFloatArray(res->body, "embedding");
#else
        return {};
#endif
    }

    std::string generate(const std::string& prompt) const override {
        if (apiKey_.empty()) {
            return "ERROR: Qwen API key missing. " + setupHint();
        }
#ifdef CPPHTTPLIB_OPENSSL_SUPPORT
        httplib::SSLClient cli(host_, 443);
        configure(cli);
        const std::string body =
            "{\"model\":" + jsonString(genModel_) + ","
            "\"messages\":[{\"role\":\"user\",\"content\":" + jsonString(prompt) + "}],"
            "\"stream\":false}";
        auto res = cli.Post((basePath_ + "/chat/completions").c_str(), body, "application/json");
        if (!res || res->status < 200 || res->status >= 300) {
            return "ERROR: Qwen API unavailable. " + setupHint();
        }
        return extractJsonString(res->body, "content");
#else
        return "ERROR: Qwen provider requires HTTPS support. " + setupHint();
#endif
    }

private:
    std::string host_;
    std::string basePath_;
    std::string apiKey_;
    std::string embedModel_;
    std::string genModel_;

#ifdef CPPHTTPLIB_OPENSSL_SUPPORT
    void configure(httplib::SSLClient& cli) const {
        cli.set_connection_timeout(5, 0);
        cli.set_read_timeout(180, 0);
        cli.enable_server_certificate_verification(true);
        cli.set_bearer_token_auth(apiKey_);
    }
#endif
};

inline std::unique_ptr<AiProvider> createAiProviderFromEnv() {
    const std::string provider = lowerCopy(getenvOr("AI_PROVIDER", "ollama"));
    if (provider == "qwen" || provider == "dashscope") {
        return std::make_unique<QwenProvider>();
    }
    return std::make_unique<OllamaProvider>();
}
