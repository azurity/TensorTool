#include <catch2/catch.hpp>
#include "TensorTool/seq.hpp"

using namespace TensorTool;

TEST_CASE("splice operator", "[seq][utils]")
{
    Eigen::DSizes<int, 3> data{0, 1, 2};
    SECTION("splice can delete elements.")
    {
        auto res = splice<1, 1, 0>(data, {});
        REQUIRE(res.size() == 2);
        REQUIRE(res[0] == 0);
        REQUIRE(res[1] == 2);
    }

    SECTION("splice can cut off elements.")
    {
        auto res = splice<2, 1, 0>(data, {});
        REQUIRE(res.size() == 2);
        REQUIRE(res[0] == 0);
        REQUIRE(res[1] == 1);
    }

    SECTION("splice can insert elements.")
    {
        auto res = splice<1, 0, 1>(data, {-1});
        REQUIRE(res.size() == 4);
        REQUIRE(res[0] == 0);
        REQUIRE(res[1] == -1);
        REQUIRE(res[2] == 1);
        REQUIRE(res[3] == 2);
    }

    SECTION("splice can append elements.")
    {
        auto res = splice<3, 0, 1>(data, {-1});
        REQUIRE(res.size() == 4);
        REQUIRE(res[0] == 0);
        REQUIRE(res[1] == 1);
        REQUIRE(res[2] == 2);
        REQUIRE(res[3] == -1);
    }

    SECTION("splice can replace elements.")
    {
        auto res = splice<1, 1, 1>(data, {-1});
        REQUIRE(res.size() == 3);
        REQUIRE(res[0] == 0);
        REQUIRE(res[1] == -1);
        REQUIRE(res[2] == 2);
    }
}
