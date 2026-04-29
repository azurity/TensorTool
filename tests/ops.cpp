#include <catch2/catch.hpp>
#include "TensorTool/ops.hpp"

using namespace TensorTool;

TEST_CASE("cwise operators", "[tensor][ops]")
{
    auto data0 = Eigen::Tensor<int, 2>(2, 1).setConstant(2);
    auto data1 = Eigen::Tensor<int, 2>(1, 2).setConstant(3);
    auto data2 = Eigen::Tensor<int, 2>(3, 1).setConstant(3);

    SECTION("cwise-op can auto broadcast")
    {
        Eigen::Tensor<int, 2> res = cwise_add(data0, data1);
        REQUIRE(res.dimension(0) == 2);
        REQUIRE(res.dimension(1) == 2);
        REQUIRE(res(0, 0) == 5);
    }

    SECTION("cwise-op must has matched dims")
    {
        REQUIRE_THROWS(cwise_add(data0, data2));
    }
}

TEST_CASE("unary cwise operators", "[tensor][ops]")
{
    auto data = Eigen::Tensor<double, 1>(2).setConstant(M_PI_4);

    SECTION("sin can op tensor")
    {
        Eigen::Tensor<double, 1> res = sin(data);
        REQUIRE(res.dimension(0) == 2);
        REQUIRE(res(0) == std::sin(M_PI_4));
    }
}
