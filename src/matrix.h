#include <vector>

#ifndef MATRIX_
#define MATRIX_

class Matrix {
  std::vector<int> data_;
  int rows_;
  int cols_;

  static std::string loadKernel(const char *filename);
  void multiply_opencl(const Matrix &other);

public:
  Matrix(int rows = 1, int cols = 1);

  Matrix(const Matrix &);
  Matrix(Matrix &&) noexcept;

  Matrix &operator=(const Matrix &);
  Matrix &operator=(Matrix &&) noexcept;
  Matrix &operator*=(const Matrix &);

  size_t rows() const;
  size_t cols() const;
  int &operator()(size_t, size_t);
  int operator()(size_t, size_t) const;

  const int *data() const;

  ~Matrix() = default;
};

Matrix operator*(const Matrix &, const Matrix &);

std::ostream &operator<<(std::ostream &, const Matrix &);
std::istream &operator>>(std::istream &, Matrix &);

#endif // MATRIX_
