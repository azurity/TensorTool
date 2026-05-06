#include <catch2/catch_test_macros.hpp>
#include "TensorTool/slice.hpp"

using namespace TensorTool;

TEST_CASE("slice operator", "[tensor][slice]")
{
    auto data = Eigen::Tensor<int, 2>(3, 4).setValues({{0, 1, 2, 3}, {4, 5, 6, 7}, {8, 9, 10, 11}});

    SECTION("slice can cut a block:: data[1:2, 2:4]")
    {
        Eigen::Tensor<int, 2> res = slice(data, Range{1, 2}, Range{2, 4});
        REQUIRE(res.rank() == 2);
        REQUIRE(res.dimension(0) == 1);
        REQUIRE(res.dimension(1) == 2);
        REQUIRE(res(0, 0) == 6);
        REQUIRE(res(0, 1) == 7);
    }

    SECTION("slice can use less-index:: data[1:2, ]")
    {
        Eigen::Tensor<int, 2> res = slice(data, Range{1, 2});
        REQUIRE(res.rank() == 2);
        REQUIRE(res.dimension(0) == 1);
        REQUIRE(res.dimension(1) == 4);
        REQUIRE(res(0, 0) == 4);
        REQUIRE(res(0, 1) == 5);
    }

    SECTION("slice can use neg-index and open-bracket:: data[-1:, ]")
    {
        Eigen::Tensor<int, 2> res = slice(data, Range{-1, std::nullopt});
        REQUIRE(res.rank() == 2);
        REQUIRE(res.dimension(0) == 1);
        REQUIRE(res.dimension(1) == 4);
        REQUIRE(res(0, 0) == 8);
        REQUIRE(res(0, 1) == 9);
    }

    SECTION("slice can use neg-index and open-bracket:: data[-1:, ]")
    {
        Eigen::Tensor<int, 2> res = slice(data, Range{-1, std::nullopt});
        REQUIRE(res.rank() == 2);
        REQUIRE(res.dimension(0) == 1);
        REQUIRE(res.dimension(1) == 4);
        REQUIRE(res(0, 0) == 8);
        REQUIRE(res(0, 1) == 9);
    }

    SECTION("slice can use reverse-index:: data[-1:-3, ]")
    {
        Eigen::Tensor<int, 2> res = slice(data, Range{-1, -3});
        REQUIRE(res.rank() == 2);
        REQUIRE(res.dimension(0) == 2);
        REQUIRE(res.dimension(1) == 4);
        REQUIRE(res(0, 0) == 8);
        REQUIRE(res(1, 0) == 4);
    }

    SECTION("slice can use more-index:: data[..., 1:2]")
    {
        Eigen::Tensor<int, 2> res = slice(data, More{}, Range{1, 2});
        REQUIRE(res.rank() == 2);
        REQUIRE(res.dimension(0) == 3);
        REQUIRE(res.dimension(1) == 1);
        REQUIRE(res(0, 0) == 1);
        REQUIRE(res(1, 0) == 5);
        REQUIRE(res(2, 0) == 9);
    }
}

TEST_CASE("slice-ref operator", "[tensor][slice]")
{
    auto data = Eigen::Tensor<int, 2>(3, 4).setValues({{0, 1, 2, 3}, {4, 5, 6, 7}, {8, 9, 10, 11}});
    SECTION("slice-ref can assign value:: data[1:2, 1:@] = *")
    {
        auto data1 = data;
        slice_ref(data1, Range{1, 2}, Range{1, 3}).setZero();
        REQUIRE(data1(1, 1) == 0);
        REQUIRE(data1(1, 2) == 0);
    }

    SECTION("slice-ref cannot assign to reverse-index:: data[-1:-3, ] = *")
    {
        auto data1 = data;
        REQUIRE_THROWS(slice_ref(data1, Range{-1, -3}) = Eigen::Tensor<int, 2>(2, 4).setConstant(100));
    }
}
