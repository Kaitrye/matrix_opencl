# Умножение целочисленных матриц с OpenCL на C++

Этот проект реализует класс Matrix на C++ с перегрузкой оператора умножения * и *=, используя OpenCL для ускоренного перемножения целочисленных матриц произвольной размерности. В качестве обёртки используется официальная opencl.hpp.

---

## Структура проекта

- `matrix.h` / `matrix.cpp` — реализация класса `Matrix`:
  - хранение матрицы в `std::vector<int>`
  - доступ к элементам через `operator()` и `const int* data()`
  - перегрузка операторов `<<`, `>>`, `*`, `*=`
  - умножение через OpenCL (`multiply_opencl`)
- `matmul.cl` — OpenCL-ядро умножения матриц
- `tests/test_matrix.cpp` — автотесты через `assert`
- `CMakeLists.txt` — сборочная система CMake
- `.github/workflows/ci.yml` — CI на **GitHub Actions** с использованием **POCL**

---

## Используемые технологии

- **C++17**
- **OpenCL 2.0** (через `opencl.hpp`)
- **POCL** — Portable OpenCL (работает на CPU, используется в CI)
- **CMake** — кроссплатформенная сборка
- **GitHub Actions** — для автоматического тестирования

---

## Принцип работы

1. Входные матрицы (из std::vector<int>) копируются в OpenCL-буферы
2. Запускается ядро matmul.cl:
3. Ядро OpenCL (`matmul.cl`) принимает 3 буфера и 3 размерности:
   ```c
   __kernel void matmul(__global const int* A,
                     __global const int* B,
                     __global int* C,
                     const int M, const int N, const int K) {
    int row = get_global_id(0);
    int col = get_global_id(1);

    int sum = 0;
    for (int i = 0; i < K; ++i)
        sum += A[row * K + i] * B[i * N + col];

    C[row * N + col] = sum;
   }
   ```
4. Внутри ядра каждая ячейка `C[i][j]` вычисляется параллельно по формуле:

   ![formula](https://latex.codecogs.com/svg.image?\dpi{150}\bg_black%20{\color{white}C[i][j]=\sum_{k=0}^{K-1}A[i][k]\cdot%20B[k][j]})

5. Результат копируется обратно в объект Matrix.

## Тестирование

Тесты реализованы с использованием assert и проверяют:

- корректность умножения матриц
- умножение с нулевой и единичной матрицами
- граничные случаи (1×1, несовместимые размерности и др.)

CI использует POCL (Portable OpenCL) — позволяет запускать тесты даже на CPU без GPU.

---

## Лицензия

Проект распространяется под лицензией **MIT**. См. файл [`LICENSE`](./LICENSE).

---
