#include <cmath>
#include <complex>
#include <iostream>
#include <vector>
#include <stdexcept>

const double pi = 4 * atan(1);

void FFT(std::vector<double> &x, std::vector<std::complex<double>> &y, int n,
         int s = 0, int d = 0, int t = 1) {
    if (n == 1) { y[d] = x[s]; return; }
    FFT(x, y, n/2, s, d, t*2);
    FFT(x, y, n/2, s+t, d+n/2, t*2);
    std::complex<double> mi = 1;
    std::complex<double> w = std::pow(std::complex<double>(std::cos(2*pi/n), std::sin(2*pi/n)), -1);
    for (int k = d; k < d+n/2; k++) {
        auto c = y[k], c2 = mi*y[k+n/2];
        y[k] = c + c2; y[k+n/2] = c - c2; mi *= w;
    }
}

void invFFT(std::vector<std::complex<double>> &y, std::vector<std::complex<double>> &x,
            int n, int s = 0, int d = 0, int t = 1) {
    if (n == 1) { x[d] = y[s]; return; }
    invFFT(y, x, n/2, s, d, t*2);
    invFFT(y, x, n/2, s+t, d+n/2, t*2);
    std::complex<double> mi = 1;
    std::complex<double> w = std::complex<double>(std::cos(2*pi/n), std::sin(2*pi/n));
    for (int k = d; k < d+n/2; k++) {
        auto c = x[k], c2 = mi*x[k+n/2];
        x[k] = (c+c2)/2.0; x[k+n/2] = (c-c2)/2.0; mi *= w;
    }
}

std::vector<double> LossyCompress(std::vector<double> data, int new_size) {
    if (new_size < 2 || new_size > data.size()) throw std::range_error("Bad new size");
    if ((data.size() & (data.size()-1)) != 0) throw std::range_error("Data size must be a power of two");

    int n = data.size();
    std::vector<double> v1(n); std::vector<std::complex<double>> v2(n);
    for (int i=0; i<n/2; i++) v1[i] = data[i*2];
    for (int i=n/2; i<n; i++) v1[i] = data[2*(n-i)-1];
    FFT(v1, v2, n);

    std::vector<double> seq(new_size);
    for (int i=0; i<new_size-1; i++) {
        auto w = std::pow(std::complex<double>(std::cos(pi/n), std::sin(pi/n)), (-1.0*i)/2.0);
        seq[i] = (w*v2[i]).real();
    }
    seq[new_size-1] = n;
    return seq;
}

std::vector<double> LossyDecompress(std::vector<double> compressed) {
    int n = compressed.back();
    if (n < 1 || n < compressed.size()) throw std::logic_error("Bad compressed sequence");
    if ((n & (n-1)) != 0) throw std::range_error("Data size must be a power of two");

    std::vector<std::complex<double>> y(n), x(n);
    y[0] = compressed[0];
    for (int k=1; k<compressed.size()-1; k++) {
        auto w = std::pow(std::complex<double>(std::cos(pi/n), std::sin(pi/n)), k/2.0);
        y[k] = 2.0*w*compressed[k];
    }
    invFFT(y, x, n);

    std::vector<double> recon(n);
    for (int i=0; i<n; i++) recon[i] = (i%2==0) ? x[i/2].real() : x[n-(i+1)/2].real();
    return recon;
}

int main() {
    try {
        std::vector<double> start{1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16};
        auto compressed = LossyCompress(start, 5);
        auto decompressed = LossyDecompress(compressed);
        std::cout << "OK: compressed=";
        for (auto &c:compressed) std::cout << c << " ";
        std::cout << " | decompressed=";
        for (auto &d:decompressed) std::cout << std::round(d) << " ";
        std::cout << "\n";
    } catch (std::exception &e) {
        std::cout << "ne valja, exception: " << e.what() << "\n";
    }
    return 0;
}
