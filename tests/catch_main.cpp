#include <catch2/catch_session.hpp>

int main(int argc, char* argv[]) {
    Catch::Session session;
    int result = session.applyCommandLine(argc, argv);
    if (result != 0) return result;
    return session.run();
}
