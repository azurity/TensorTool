#include <catch2/catch.hpp>
#include "TensorTool/dims.hpp"

using namespace TensorTool;

template <typename Derived, typename C = std::enable_if_t<std::is_base_of_v<Eigen::TensorBase<Derived>, Derived>>>
inline auto fast_eval(Derived &&exp)
{
    return Eigen::Tensor<typename Derived::Scalar, Derived::NumDimensions>(exp);
}

TEST_CASE("squeeze operator", "[tensor][dims][squeeze]")
{
    auto data = Eigen::Tensor<int, 3>(2, 1, 3).setZero();
    SECTION("squeeze can remove dim[size=1]")
    {
        auto res = fast_eval(squeeze<1>(data));
        REQUIRE(res.rank() == 2);
        REQUIRE(res.dimension(0) == 2);
        REQUIRE(res.dimension(1) == 3);
    }

    SECTION("squeeze cannot remove dim[size!=1]")
    {
        REQUIRE_THROWS(fast_eval(squeeze<0>(data)));
    }

    SECTION("squeeze can remove negative dim[size=1]")
    {
        auto res = fast_eval(squeeze<-2>(data));
        REQUIRE(res.rank() == 2);
        REQUIRE(res.dimension(0) == 2);
        REQUIRE(res.dimension(1) == 3);
    }
}

TEST_CASE("unsqueeze operator", "[tensor][dims][unsqueeze]")
{
    auto data = Eigen::Tensor<int, 2>(2, 3).setZero();
    SECTION("unsqueeze can expand dim front")
    {
        auto res = fast_eval(unsqueeze<0>(data));
        REQUIRE(res.rank() == 3);
        REQUIRE(res.dimension(0) == 1);
        REQUIRE(res.dimension(1) == 2);
        REQUIRE(res.dimension(2) == 3);
    }

    SECTION("unsqueeze can expand dim back")
    {
        auto res = fast_eval(unsqueeze<2>(data));
        REQUIRE(res.rank() == 3);
        REQUIRE(res.dimension(0) == 2);
        REQUIRE(res.dimension(1) == 3);
        REQUIRE(res.dimension(2) == 1);
    }

    SECTION("unsqueeze can insert dim")
    {
        auto res = fast_eval(unsqueeze<1>(data));
        REQUIRE(res.rank() == 3);
        REQUIRE(res.dimension(0) == 2);
        REQUIRE(res.dimension(1) == 1);
        REQUIRE(res.dimension(2) == 3);
    }

    SECTION("unsqueeze can expand negative dim")
    {
        auto res = fast_eval(unsqueeze<-1>(data));
        REQUIRE(res.rank() == 3);
        REQUIRE(res.dimension(0) == 2);
        REQUIRE(res.dimension(1) == 3);
        REQUIRE(res.dimension(2) == 1);
    }
}

TEST_CASE("stack operator", "[tensor][dims][stack]")
{
    auto data0 = Eigen::Tensor<int, 2>(2, 3).setZero();
    auto data1 = Eigen::Tensor<int, 2>(2, 3).setZero();
    auto data2 = Eigen::Tensor<int, 2>(2, 3).setZero();

    SECTION("stack can stack tensors")
    {
        auto res = stack<0>(data0, data1, data2);
        REQUIRE(res.rank() == 3);
        REQUIRE(res.dimension(0) == 3);
        REQUIRE(res.dimension(1) == 2);
        REQUIRE(res.dimension(2) == 3);
    }
}
