#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <string>
#include <iomanip>
#include <chrono>
#include <cstdint>
#include <cstdlib>

//  ГПСЧ 1:
//
//  Два состояния a и b обновляют друг друга на каждом шаге:
//    a = A * a + b
//    b = B * b ^ a
//  Выход — XOR старших частей обоих состояний
struct CrossState {
    uint64_t a;
    uint64_t b;
    static constexpr uint64_t A = 6364136223846793005ULL;
    static constexpr uint64_t B = 2685821657736338717ULL;

    CrossState(uint64_t seed = 123456789ULL)
        : a(seed | 1ULL), b(~seed | 1ULL) {
    }

    uint64_t next() {
        a = A * a + b;
        b = B * b ^ a;
        return a ^ (b >> 16);
    }

    int nextInt(int range) {
        return static_cast<int>((next() >> 17) % static_cast<uint64_t>(range));
    }
};


//  ГПСЧ 2: XorShift с acc
//
//  XorShift64 порождает линейную последовательность
//  Добавляем  acc, который монотонно растёт ,суммируя каждое состояние:
//    state — три XOR-сдвига
//    acc  += state
//    output = state ^ acc
//  acc нелинейно смещает выход на каждом шаге, ломая период
struct XorShiftACC {
    uint64_t state;
    uint64_t acc;

    XorShiftACC(uint64_t seed = 987654321ULL)
        : state(seed | 1ULL), acc(0ULL) {
    }

    uint64_t next() {
        state ^= state << 11;
        state ^= state >> 5;
        state ^= state << 23;
        acc += state;
        return state ^ acc;
    }

    int nextInt(int range) {
        return static_cast<int>((next() >> 17) % static_cast<uint64_t>(range));
    }
};


//  ГПСЧ 3: Квадратичный XOR
//
//  На каждом шаге возводим состояние в квадрат, XOR со сдвигом и нечётной константой C:
//    state = state * state ^ (state >> 17) ^ C
//  Сдвиг + XOR перемешивают биты
struct QuadXOR {
    uint64_t state;
    static constexpr uint64_t C = 0xddddddddbbbbbbbbULL;

    QuadXOR(uint64_t seed = 2024777ULL) : state(seed | 1ULL) {}

    uint64_t next() {
        state = state * state ^ (state >> 17) ^ C;
        return state;
    }

    int nextInt(int range) {
        return static_cast<int>((next() >> 17) % static_cast<uint64_t>(range));
    }
};


//  Статистические функции
//  среднее
double calcMean(const std::vector<int>& v) {
    double s = 0;
    for (int x : v) s += x;
    return s / static_cast<double>(v.size());
}
//  стандартное отклонение
double calcStdDev(const std::vector<int>& v, double mean) {
    double s = 0;
    for (int x : v) s += (x - mean) * (x - mean);
    return std::sqrt(s / static_cast<double>(v.size()));
}
//  коэфицент вариации 
double calcCV(double sd, double m) {
    return m > 1e-9 ? sd / m * 100.0 : 0.0;
}

// Аппроксимация p-value хи-квадрат
double chiPValue(double chi2, double df) {
    if (df <= 0 || chi2 < 0) return 0.0;
    double x = std::pow(chi2 / df, 1.0 / 3.0);
    double mu = 1.0 - 2.0 / (9.0 * df);
    double sigma = std::sqrt(2.0 / (9.0 * df));
    if (sigma < 1e-15) return 0.0;
    return 0.5 * std::erfc((x - mu) / (sigma * std::sqrt(2.0)));
}

//  Критерий хи-квадрат: равномерность распределения
struct ChiResult {
    double chi2;
    int    df;
    double pValue;
    bool   uniform; // H0 не отвергается при p > 0.05
};
//  хи-квадрат
ChiResult chiSquareTest(const std::vector<int>& data, int range, int bins = 20) {
    int n = static_cast<int>(data.size());
    std::vector<int> obs(bins, 0);
    for (int x : data) {
        int b = static_cast<int>(static_cast<long long>(x) * bins / range);
        if (b >= bins) b = bins - 1;
        obs[b]++;
    }
    double expected = static_cast<double>(n) / bins;
    double chi2 = 0;
    for (int c : obs) { double d = c - expected; chi2 += d * d / expected; }
    double pv = chiPValue(chi2, bins - 1);
    return { chi2, bins - 1, pv, pv > 0.05 };
}

//  NIST SP 800-22 тесты
struct TestResult {
    std::string name;
    double      statistic;
    double      pValue;
    bool        passed; // уровень значимости NIST: a = 0.01
};

// 16 младших бит каждого числа -> битовая последовательность
std::vector<int> toBits(const std::vector<int>& data) {
    std::vector<int> bits;
    bits.reserve(data.size() * 16);
    for (int x : data)
        for (int b = 15; b >= 0; b--)
            bits.push_back((x >> b) & 1);
    return bits;
}

// Тест 1. Frequency / Monobit (NIST SP 800-22)
// Проверяет: число нулей == числу единиц во всей последовательности
TestResult frequencyTest(const std::vector<int>& bits) {
    int n = static_cast<int>(bits.size());
    double S = 0;
    for (int b : bits) S += b ? 1.0 : -1.0;
    double Sobs = std::abs(S) / std::sqrt(static_cast<double>(n));
    double pv = std::erfc(Sobs / std::sqrt(2.0));
    return { "Frequency (Monobit)", Sobs, pv, pv >= 0.01 };
}

// Тест 2. Block Frequency (NIST SP 800-22)
// Доля единиц в каждом блоке длины M должна быть == 1/2
TestResult blockFrequencyTest(const std::vector<int>& bits, int M = 128) {
    int n = static_cast<int>(bits.size());
    int N = n / M;
    if (N == 0) return { "Block Frequency", 0, 0, false };
    double chi2 = 0;
    for (int i = 0; i < N; i++) {
        double pi = 0;
        for (int j = 0; j < M; j++) pi += bits[i * M + j];
        pi /= M;
        chi2 += (pi - 0.5) * (pi - 0.5);
    }
    chi2 *= 4.0 * M;
    double pv = chiPValue(chi2, static_cast<double>(N));
    return { "Block Frequency (M=128)", chi2, pv, pv >= 0.01 };
}

// Тест 3. Runs (NIST SP 800-22)
// Проверяет число чередований 0->1 и 1->0
TestResult runsTest(const std::vector<int>& bits) {
    int n = static_cast<int>(bits.size());
    double pi = 0;
    for (int b : bits) pi += b;
    pi /= n;
    if (std::abs(pi - 0.5) >= 2.0 / std::sqrt(static_cast<double>(n)))
        return { "Runs", 0, 0.0, false };
    int Vn = 1;
    for (int i = 1; i < n; i++) if (bits[i] != bits[i - 1]) Vn++;
    double num = std::abs(Vn - 2.0 * n * pi * (1.0 - pi));
    double den = 2.0 * std::sqrt(2.0 * n) * pi * (1.0 - pi);
    double pv = den > 1e-12 ? std::erfc(num / den) : 0.0;
    return { "Runs", static_cast<double>(Vn), pv, pv >= 0.01 };
}

// Тест 4. Serial (пары бит)
// Делим последовательность на неперекрывающиеся пары бит
// Пары 00, 01, 10, 11 должны встречаться с одинаковой частотой == N/4
// Хи-квадрат с 3 степенями свободы проверяет равномерность распределения пар
TestResult serialTest(const std::vector<int>& bits) {
    int n = static_cast<int>(bits.size());
    int pairs = n / 2; // число неперекрывающихся пар
    if (pairs == 0) return { "Serial (pairs)", 0, 0, false };

    // считаем частоты четырёх пар: 00=0, 01=1, 10=2, 11=3
    int freq[4] = { 0, 0, 0, 0 };
    for (int i = 0; i < pairs; i++) {
        int idx = bits[2 * i] * 2 + bits[2 * i + 1]; // 0..3
        freq[idx]++;
    }

    double expected = static_cast<double>(pairs) / 4.0;
    double chi2 = 0;
    for (int i = 0; i < 4; i++) {
        double d = freq[i] - expected;
        chi2 += d * d / expected;
    }
    // 4 категории - 1 = 3 степени свободы
    double pv = chiPValue(chi2, 3.0);
    return { "Serial (bit pairs)", chi2, pv, pv >= 0.01 };
}

// Тест 5. Serial Correlation (автокорреляция при lag = 1)
// H0: rho == 0 — значения независимы
TestResult autocorrelationTest(const std::vector<int>& data, int lag = 1) {
    int n = static_cast<int>(data.size());
    double mu = calcMean(data), num = 0, den = 0;
    for (int i = 0; i < n - lag; i++) num += (data[i] - mu) * (data[i + lag] - mu);
    for (int i = 0; i < n; i++)        den += (data[i] - mu) * (data[i] - mu);
    double rho = den > 1e-12 ? num / den : 0.0;
    double z = rho * std::sqrt(static_cast<double>(n - lag));
    double pv = std::erfc(std::abs(z) / std::sqrt(2.0));
    return { "Serial Correlation (lag=1)", rho, pv, pv >= 0.01 };
}

//  Генерация выборки
template<typename PRNG>
std::vector<int> generateSample(PRNG& gen, int size, int range) {
    std::vector<int> v(size);
    for (int& x : v) x = gen.nextInt(range);
    return v;
}

//  Замер времени генерации N чисел для наших генераторов
template<typename PRNG>
double measureTime(uint64_t seed, int count, int range) {
    PRNG gen(seed);
    volatile int sink = 0; // volatile предотвращает выброс цикла оптимизатором
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < count; i++) sink = gen.nextInt(range);
    auto t1 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

//  Замер времени генерации N чисел для std::rand
double measureTimeStd(int count, int range) {
    std::srand(42);
    volatile int sink = 0;
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < count; i++) sink = std::rand() % range;
    auto t1 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

//  Обработка одного генератора
template<typename PRNG>
void processGenerator(const std::string& gname,
    int samples, int sampleSz, int range,
    std::ofstream& fStats, std::ofstream& fNIST) {

    int chiPass = 0;
    std::vector<int> allInts;
    std::vector<int> allBits;

    for (int s = 0; s < samples; s++) {
        uint64_t seed = static_cast<uint64_t>(s * 1000 + 7919);
        PRNG gen(seed);
        std::vector<int> sample = generateSample(gen, sampleSz, range);

        double m = calcMean(sample);
        double sd = calcStdDev(sample, m);
        double cv = calcCV(sd, m);
        ChiResult chi = chiSquareTest(sample, range, 20);
        if (chi.uniform) chiPass++;

        fStats << gname << "," << (s + 1) << ","
            << std::fixed << std::setprecision(4)
            << m << "," << sd << "," << cv << ","
            << chi.chi2 << "," << chi.df << ","
            << chi.pValue << "," << (chi.uniform ? 1 : 0) << "\n";

        std::vector<int> b = toBits(sample);
        allBits.insert(allBits.end(), b.begin(), b.end());
        allInts.insert(allInts.end(), sample.begin(), sample.end());
    }

    std::vector<TestResult> nistRes = {
        frequencyTest(allBits),
        blockFrequencyTest(allBits, 128),
        runsTest(allBits),
        serialTest(allBits),
        autocorrelationTest(allInts, 1),
    };

    for (const auto& t : nistRes) {
        fNIST << gname << "," << t.name << ","
            << std::fixed << std::setprecision(6)
            << t.statistic << "," << t.pValue << ","
            << (t.passed ? 1 : 0) << "\n";
    }
}

//  main
int main() {
    constexpr int SAMPLES = 20;    // выборок на генератор 
    constexpr int SAMPLE_SZ = 1000;  // элементов в выборке
    constexpr int RANGE = 65536; // диапазон [0, RANGE)

    std::ofstream fStats("statistics.csv");
    if (!fStats) {
        std::cerr << "не удалось открыть statistics.csv\n";
        return 1;
    }
    fStats << "generator,sample,mean,stddev,cv,chi2,df,p_value,uniform\n";

    std::ofstream fNIST("nist_tests.csv");
    if (!fNIST) {
        std::cerr << "не удалось открыть nist_tests.csv\n";
        return 1;
    }
    fNIST << "generator,test_name,statistic,p_value,passed\n";

    processGenerator<CrossState>("CrossState", SAMPLES, SAMPLE_SZ, RANGE, fStats, fNIST);
    processGenerator<XorShiftACC>("XorShiftACC", SAMPLES, SAMPLE_SZ, RANGE, fStats, fNIST);
    processGenerator<QuadXOR>("QuadXOR", SAMPLES, SAMPLE_SZ, RANGE, fStats, fNIST);

    fStats.close();
    fNIST.close();

    //  Замер времени: от 1 000 до 1 000 000 элементов
    const std::vector<int> sizes = {
        1000, 2500, 5000, 10000, 25000,
        50000, 100000, 250000, 500000, 1000000
    };

    std::ofstream fTiming("timing.csv");
    if (!fTiming) {
        std::cerr << "Ошибка: не удалось открыть timing.csv\n";
        return 1;
    }
    fTiming << "size,CrossState,XorShiftACC,QuadXOR,StdRand\n";

    for (int N : sizes) {
        double tCS = measureTime<CrossState>(42ULL, N, RANGE);
        double tACC = measureTime<XorShiftACC>(42ULL, N, RANGE);
        double tQX = measureTime<QuadXOR>(42ULL, N, RANGE);
        double tStd = measureTimeStd(N, RANGE);

        fTiming << std::fixed << std::setprecision(4)
            << N << ","
            << tCS << "," << tACC << "," << tQX << "," << tStd << "\n";
    }
    fTiming.close();

    return 0;
}
