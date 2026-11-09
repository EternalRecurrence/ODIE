// Binary stdin/stdout bridge to the installed JBIG-KIT; no CUDA dependency.
extern "C" {
#include <jbig.h>
}
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

static void append(unsigned char* data, size_t size, void* destination) {
    auto& output = *static_cast<std::vector<unsigned char>*>(destination);
    output.insert(output.end(), data, data + size);
}

int main(int argc, char** argv) {
    try {
        if (argc != 4) throw std::runtime_error("encode|decode width height");
        const unsigned long width = std::stoul(argv[2]), height = std::stoul(argv[3]);
        if (!width || !height) throw std::runtime_error("Empty image");
        const size_t expected = ((width + 7) / 8) * height;
        std::vector<unsigned char> input((std::istreambuf_iterator<char>(std::cin)), {});
        if (std::string(argv[1]) == "encode") {
            if (input.size() != expected) throw std::runtime_error("Bitmap length mismatch");
            unsigned char* planes[] = {input.data()};
            std::vector<unsigned char> output;
            jbg_enc_state state;
            jbg_enc_init(&state, width, height, 1, planes, append, &output);
            jbg_enc_layers(&state, 0);
            jbg_enc_options(&state, 0, JBG_TPBON, 128, 8, 0);
            jbg_enc_out(&state);
            jbg_enc_free(&state);
            std::cout.write(reinterpret_cast<const char*>(output.data()), output.size());
        } else if (std::string(argv[1]) == "decode") {
            jbg_dec_state state;
            jbg_dec_init(&state);
            jbg_dec_maxsize(&state, width, height);
            size_t consumed = 0;
            int result = jbg_dec_in(&state, input.data(), input.size(), &consumed);
            bool valid = result == JBG_EOK && consumed == input.size()
                && jbg_dec_getwidth(&state) == width && jbg_dec_getheight(&state) == height
                && jbg_dec_getplanes(&state) == 1 && jbg_dec_getsize(&state) == expected;
            if (valid)
                std::cout.write(reinterpret_cast<const char*>(jbg_dec_getimage(&state, 0)), expected);
            jbg_dec_free(&state);
            if (!valid) throw std::runtime_error("Invalid or truncated JBIG mask");
        } else {
            throw std::runtime_error("Unknown operation");
        }
        if (!std::cout) throw std::runtime_error("Output failed");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
