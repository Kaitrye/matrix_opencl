#include <gtest/gtest.h>
#include <cstdlib>
#include <ctime>
#include "../src/matrix.h"

TEST(MatrixMultiplicationTest, SmallMatrix) {
    Matrix A(2, 3);
    Matrix B(3, 2);

    int value = 1;
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 3; ++j)
            A(i, j) = value++;

    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 2; ++j)
            B(i, j) = value++;

    Matrix expected(2, 2);
    expected(0, 0) = 58; expected(0, 1) = 64;
    expected(1, 0) = 139; expected(1, 1) = 154;

    Matrix C = A * B;

    ASSERT_EQ(C.rows(), 2);
    ASSERT_EQ(C.cols(), 2);

    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j)
            EXPECT_EQ(C(i, j), expected(i, j));
}

TEST(MatrixMultiplicationTest, IdentityMatrix) {
    srand(static_cast<unsigned int>(time(nullptr)));

    const int N = 3;
    Matrix A(N, N);
    Matrix I(N, N);

    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            A(i, j) = rand();

    for (int i = 0; i < N; ++i)
        I(i, i) = 1;

    Matrix B = A * I;

    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            EXPECT_EQ(B(i, j), A(i, j));
}

TEST(MatrixMultiplicationTest, DimensionMismatchThrows) {
    Matrix A(2, 3);
    Matrix B(4, 2);

    EXPECT_THROW({
        A *= B;
    }, std::runtime_error);
}

TEST(MatrixMultiplicationTest, OneByOneMatrix) {
    Matrix A(1, 1);
    Matrix B(1, 1);
    A(0, 0) = 7;
    B(0, 0) = 3;

    Matrix C = A * B;

    ASSERT_EQ(C.rows(), 1);
    ASSERT_EQ(C.cols(), 1);
    EXPECT_EQ(C(0, 0), 21);
}

TEST(MatrixMultiplicationTest, ZeroMatrixResult) {
    Matrix A(2, 3);
    Matrix B(3, 4);

    Matrix C = A * B;

    for (size_t i = 0; i < C.rows(); ++i)
        for (size_t j = 0; j < C.cols(); ++j)
            EXPECT_EQ(C(i, j), 0);
}
