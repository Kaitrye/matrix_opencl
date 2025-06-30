#include <cassert>
#include <iostream>
#include <cstdlib>
#include <ctime>
#include "../matrix.h"

void test_multiplication_small() {
    Matrix A(2, 3);
    Matrix B(3, 2);

    int value = 1;
    for (int i=0; i<2; ++i) {
        for (int j=0; j<3; ++j) {
            A(i, j) = value++;
        }
    }

    for (int i=0; i<3; ++i) {
        for (int j=0; j<2; ++j) {
            B(i, j) = value++;
        }
    }

    Matrix expected(2, 2);
    expected(0, 0) = 58; expected(0, 1) = 64;
    expected(1, 0) = 139; expected(1, 1) = 154;

    Matrix C = A * B;

    assert(C.rows() == 2 && C.cols() == 2);
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j)
            assert(C(i, j) == expected(i, j));

    std::cout << "test_multiplication_small passed\n";
}

void test_multiplication_by_identity() {

    srand(static_cast<unsigned int>(time(nullptr)));

    int N = 3;
    Matrix A(N, N);
    Matrix I(N, N);

    for (int i=0; i<N; ++i) {
        for (int j=0; j<N; ++j) {
            A(i, j) = rand();
        }
    }

    for (int i = 0; i < N; ++i) {
        I(i, i) = 1;
    }

    Matrix B = A * I;

    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            assert(B(i, j) == A(i, j));

    std::cout << "test_multiplication_by_identity passed\n";
}

void test_multiply_dimension_mismatch() {
    Matrix A(2, 3);
    Matrix B(4, 2);

    bool caught = false;
    try {
        A *= B;
    } catch (const std::runtime_error& e) {
        caught = true;
    }

    assert(caught);
    std::cout << "test_multiply_dimension_mismatch passed\n";
}

void test_1x1_matrix() {
    Matrix A(1, 1);
    Matrix B(1, 1);
    A(0, 0) = 7;
    B(0, 0) = 3;

    Matrix C = A * B;
    assert(C.rows() == 1 && C.cols() == 1);
    assert(C(0, 0) == 21);

    std::cout << "test_1x1_matrix passed\n";
}

void test_zero_matrix() {
    Matrix A(2, 3);
    Matrix B(3, 4);
    Matrix C = A * B;

    for (size_t i = 0; i < C.rows(); ++i)
        for (size_t j = 0; j < C.cols(); ++j)
            assert(C(i, j) == 0);

    std::cout << "test_zero_matrix passed\n";
}

int main() {
    test_multiplication_small();
    test_multiplication_by_identity();
    test_multiply_dimension_mismatch();
    test_1x1_matrix();
    test_zero_matrix();

    std::cout << "All tests passed!\n";
    return 0;
}
