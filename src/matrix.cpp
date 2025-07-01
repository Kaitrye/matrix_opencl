#define CL_HPP_TARGET_OPENCL_VERSION 200
#define CL_HPP_ENABLE_EXCEPTIONS

#include "matrix.h"

#include <CL/opencl.hpp>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>

std::string Matrix::loadKernel(const char *filename) {
  std::ifstream file(filename);
  if (!file.is_open()) {
    throw std::runtime_error("Cannot open kernel file");
  }
  std::stringstream ss;
  ss << file.rdbuf();

  return ss.str();
}

Matrix::Matrix(int rows, int cols)
    : rows_(rows), cols_(cols), data_(rows * cols) {}

Matrix::Matrix(const Matrix &other)
    : rows_(other.rows_), cols_(other.cols_), data_(other.data_) {}

Matrix::Matrix(Matrix &&other) noexcept
    : rows_(other.rows_), cols_(other.cols_), data_(std::move(other.data_)) {
  other.rows_ = 0;
  other.cols_ = 0;
}

Matrix &Matrix::operator=(const Matrix &other) {
  if (this != &other) {
    rows_ = other.rows_;
    cols_ = other.cols_;
    data_ = other.data_;
  }
  return *this;
}

Matrix &Matrix::operator=(Matrix &&other) noexcept {
  if (this != &other) {
    std::swap(rows_, other.rows_);
    std::swap(cols_, other.cols_);
    std::swap(data_, other.data_);
  }
  return *this;
}

void Matrix::multiply_opencl(const Matrix &other) {
  int M = rows_, K = cols_, N = other.cols_;
  Matrix result(M, N);

  std::vector<cl::Platform> platforms;
  cl::Platform::get(&platforms);
  if (platforms.empty()) {
    throw std::runtime_error("No OpenCL platforms found");
  }
  cl::Platform platform = platforms[0];

  std::vector<cl::Device> devices;
  platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);
  if (devices.empty()) {
    throw std::runtime_error("No OpenCL devices found");
  }
  cl::Device device = devices[0];

  cl::Context context(device);
  cl::CommandQueue queue(context, device);

  std::string kernel_code = loadKernel("src/matmul.cl");
  cl::Program::Sources sources{{kernel_code.c_str(), kernel_code.length()}};
  cl::Program program(context, sources);
  program.build({device});

  cl::Kernel kernel(program, "matmul");

  cl::Buffer bufferA(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                     sizeof(int) * M * K, data_.data());
  cl::Buffer bufferB(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                     sizeof(int) * K * N, (void *)other.data());
  cl::Buffer bufferC(context, CL_MEM_WRITE_ONLY, sizeof(int) * M * N);

  kernel.setArg(0, bufferA);
  kernel.setArg(1, bufferB);
  kernel.setArg(2, bufferC);
  kernel.setArg(3, M);
  kernel.setArg(4, N);
  kernel.setArg(5, K);

  cl::NDRange global(M, N);
  queue.enqueueNDRangeKernel(kernel, cl::NullRange, global);
  queue.enqueueReadBuffer(bufferC, CL_TRUE, 0, sizeof(int) * M * N,
                          (void *)result.data());

  *this = std::move(result);
}

Matrix &Matrix::operator*=(const Matrix &other) {
  if (cols_ != other.rows_) {
    throw std::runtime_error("Matrix dimensions mismatch for multiplication");
  }

  multiply_opencl(other);
  return *this;
}

size_t Matrix::rows() const { return rows_; }
size_t Matrix::cols() const { return cols_; }

int &Matrix::operator()(size_t i, size_t j) { return data_[i * cols_ + j]; }

int Matrix::operator()(size_t i, size_t j) const {
  return data_[i * cols_ + j];
}

std::ostream &operator<<(std::ostream &out, const Matrix &M) {
  int rows = M.rows();
  int cols = M.cols();

  std::vector<int> lens(cols, 0);
  for (size_t i = 0; i < rows; ++i) {
    for (size_t j = 0; j < cols; ++j) {
      int len = std::to_string(M(i, j)).size();
      if (lens[j] < len) lens[j] = len;
    }
  }

  for (size_t i = 0; i < rows; ++i) {
    for (size_t j = 0; j < cols; ++j) {
      if (j != 0) out << " ";
      out << std::setw(lens[j]) << M(i, j);
    }
    out << std::endl;
  }

  return out;
}

std::istream &operator>>(std::istream &in, Matrix &M) {
  for (size_t i = 0; i < M.rows(); ++i) {
    for (size_t j = 0; j < M.cols(); ++j) {
      in >> M(i, j);
    }
  }

  return in;
}

Matrix operator*(const Matrix &left, const Matrix &right) {
  Matrix tmp(left);
  return tmp *= right;
}

const int *Matrix::data() const { return data_.data(); }
