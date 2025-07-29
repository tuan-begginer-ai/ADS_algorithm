import numpy as np
import matplotlib.pyplot as plt
from typing import List, Union, Tuple
import pandas as pd

class ImpedanceCalculator:
    """
    Lớp tính toán trở kháng đầu vào (Zin) của mạng phối hợp thang LC phân tán.

    Phù hợp cho môi trường Google Colab với các tính năng:
    - Tính toán Zin tại một tần số hoặc dải tần số
    - Vẽ đồ thị Smith Chart và frequency response
    - Xuất kết quả dưới dạng DataFrame
    - Tối ưu hóa tham số
    """

    def __init__(self, ZL: float = 80.0, ZC: float = 25.0, Z_load: float = 50.0, C_dc: float = 100e-12):
        """
        Khởi tạo calculator với các tham số mặc định.

        Args:
            ZL (float): Trở kháng đặc tính của đường truyền nối tiếp (ohms)
            ZC (float): Trở kháng đặc tính của dây chập song song (ohms)
            Z_load (float): Trở kháng tải (ohms)
            C_dc (float): Điện dung của tụ chặn DC (Farad)
        """
        self.ZL = ZL
        self.ZC = ZC
        self.Z_load = Z_load
        self.C_dc = C_dc

        # Lưu trữ kết quả tính toán gần nhất
        self.last_result = None
        self.last_frequencies = None

    def calculate_zin_single(self, theta_L: List[float], theta_C: List[float],
                           freq_Hz: float) -> complex:
        """
        Tính Zin tại một tần số cụ thể.

        Args:
            theta_L: Danh sách các độ dài điện của các phần tử nối tiếp (độ)
            theta_C: Danh sách các độ dài điện của các phần tử song song (độ)
            freq_Hz: Tần số tính toán (Hz)

        Returns:
            complex: Giá trị trở kháng đầu vào phức
        """
        # Kiểm tra đầu vào
        if len(theta_L) != len(theta_C):
            raise ValueError("Số lượng phần tử nối tiếp và song song phải bằng nhau.")

        n = len(theta_L)
        omega = 2 * np.pi * freq_Hz

        # Khởi tạo ma trận ABCD tổng
        ABCD_total = np.eye(2, dtype=complex)

        # Vòng lặp qua n khối L-C
        for i in range(n):
            theta_l_rad = np.deg2rad(theta_L[i])
            theta_c_rad = np.deg2rad(theta_C[i])

            # Ma trận ABCD cho đường truyền nối tiếp
            cos_l = np.cos(theta_l_rad)
            sin_l = np.sin(theta_l_rad)
            ABCD_L = np.array([[cos_l, 1j * self.ZL * sin_l],
                              [1j * sin_l / self.ZL, cos_l]], dtype=complex)

            # Ma trận ABCD cho dây chập hở mạch song song
            tan_c = np.tan(theta_c_rad)
            ABCD_C = np.array([[1, 0],
                              [1j * tan_c / self.ZC, 1]], dtype=complex)

            # Ma trận ABCD của một khối L-C
            ABCD_section = ABCD_L @ ABCD_C
            ABCD_total = ABCD_total @ ABCD_section

        # Thêm ma trận của tụ chặn DC
        Z_cap = 1 / (1j * omega * self.C_dc)
        ABCD_dc_block = np.array([[1, Z_cap],
                                 [0, 1]], dtype=complex)
        ABCD_total = ABCD_total @ ABCD_dc_block

        # Tính toán Zin từ ma trận ABCD
        A, B = ABCD_total[0, 0], ABCD_total[0, 1]
        C, D = ABCD_total[1, 0], ABCD_total[1, 1]

        zin = (A * self.Z_load + B) / (C * self.Z_load + D)
        return zin

    def calculate_zin_sweep(self, theta_L: List[float], theta_C: List[float],
                          freq_start: float, freq_stop: float,
                          num_points: int = 1001) -> Tuple[np.ndarray, np.ndarray]:
        """
        Tính Zin trên một dải tần số.

        Args:
            theta_L: Danh sách các độ dài điện của các phần tử nối tiếp (độ)
            theta_C: Danh sách các độ dài điện của các phần tử song song (độ)
            freq_start: Tần số bắt đầu (Hz)
            freq_stop: Tần số kết thúc (Hz)
            num_points: Số điểm tần số

        Returns:
            Tuple[np.ndarray, np.ndarray]: (frequencies, zin_values)
        """
        frequencies = np.linspace(freq_start, freq_stop, num_points)
        zin_values = np.zeros(num_points, dtype=complex)

        for i, freq in enumerate(frequencies):
            zin_values[i] = self.calculate_zin_single(theta_L, theta_C, freq)

        # Lưu kết quả
        self.last_frequencies = frequencies
        self.last_result = zin_values

        return frequencies, zin_values

    def plot_impedance_vs_frequency(self, theta_L: List[float], theta_C: List[float],
                                  freq_start: float, freq_stop: float,
                                  num_points: int = 1001, figsize: Tuple[int, int] = (12, 8)):
        """
        Vẽ đồ thị Zin theo tần số.
        """
        frequencies, zin_values = self.calculate_zin_sweep(theta_L, theta_C,
                                                          freq_start, freq_stop, num_points)

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)

        freq_ghz = frequencies / 1e9

        # Magnitude
        ax1.plot(freq_ghz, np.abs(zin_values), 'b-', linewidth=2)
        ax1.set_xlabel('Tần số (GHz)')
        ax1.set_ylabel('|Zin| (Ω)')
        ax1.set_title('Độ lớn của Zin')
        ax1.grid(True, alpha=0.3)

        # Phase
        ax2.plot(freq_ghz, np.angle(zin_values, deg=True), 'r-', linewidth=2)
        ax2.set_xlabel('Tần số (GHz)')
        ax2.set_ylabel('Phase (độ)')
        ax2.set_title('Pha của Zin')
        ax2.grid(True, alpha=0.3)

        # Real part
        ax3.plot(freq_ghz, np.real(zin_values), 'g-', linewidth=2)
        ax3.set_xlabel('Tần số (GHz)')
        ax3.set_ylabel('Re(Zin) (Ω)')
        ax3.set_title('Phần thực của Zin')
        ax3.grid(True, alpha=0.3)

        # Imaginary part
        ax4.plot(freq_ghz, np.imag(zin_values), 'm-', linewidth=2)
        ax4.set_xlabel('Tần số (GHz)')
        ax4.set_ylabel('Im(Zin) (Ω)')
        ax4.set_title('Phần ảo của Zin')
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def plot_smith_chart(self, theta_L: List[float], theta_C: List[float],
                        freq_start: float, freq_stop: float,
                        num_points: int = 1001, figsize: Tuple[int, int] = (10, 10)):
        """
        Vẽ Smith Chart cho impedance.
        """
        frequencies, zin_values = self.calculate_zin_sweep(theta_L, theta_C,
                                                          freq_start, freq_stop, num_points)

        # Chuyển đổi sang reflection coefficient
        gamma = (zin_values - self.Z_load) / (zin_values + self.Z_load)

        fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(projection='polar'))

        # Vẽ đường cong impedance
        theta = np.angle(gamma)
        r = np.abs(gamma)

        # Chỉ vẽ các điểm có |Γ| <= 1 (trong Smith Chart)
        valid_idx = r <= 1

        ax.plot(theta[valid_idx], r[valid_idx], 'b-', linewidth=2, label='Zin locus')
        ax.set_ylim(0, 1)
        ax.set_title('Smith Chart', pad=20)
        ax.grid(True, alpha=0.3)
        ax.legend()

        plt.show()

    def export_to_dataframe(self, theta_L: List[float], theta_C: List[float],
                          freq_start: float, freq_stop: float,
                          num_points: int = 1001) -> pd.DataFrame:
        """
        Xuất kết quả tính toán ra DataFrame.
        """
        frequencies, zin_values = self.calculate_zin_sweep(theta_L, theta_C,
                                                          freq_start, freq_stop, num_points)

        df = pd.DataFrame({
            'Frequency_Hz': frequencies,
            'Frequency_GHz': frequencies / 1e9,
            'Zin_Real': np.real(zin_values),
            'Zin_Imag': np.imag(zin_values),
            'Zin_Magnitude': np.abs(zin_values),
            'Zin_Phase_deg': np.angle(zin_values, deg=True)
        })

        return df

    def calculate_vswr(self, theta_L: List[float], theta_C: List[float],
                      freq_Hz: float) -> float:
        """
        Tính VSWR (Voltage Standing Wave Ratio).
        """
        zin = self.calculate_zin_single(theta_L, theta_C, freq_Hz)
        gamma = (zin - self.Z_load) / (zin + self.Z_load)
        gamma_mag = np.abs(gamma)

        if gamma_mag >= 1:
            return float('inf')

        vswr = (1 + gamma_mag) / (1 - gamma_mag)
        return vswr

    def print_summary(self, theta_L: List[float], theta_C: List[float], freq_Hz: float):
        """
        In tóm tắt kết quả tính toán.
        """
        zin = self.calculate_zin_single(theta_L, theta_C, freq_Hz)
        vswr = self.calculate_vswr(theta_L, theta_C, freq_Hz)

        print("=" * 60)
        print("TÓMLƯỢT KẾT QUẢ TÍNH TOÁN IMPEDANCE")
        print("=" * 60)
        print(f"Tần số: {freq_Hz/1e9:.3f} GHz")
        print(f"Tham số mạch:")
        print(f"  - theta_L: {theta_L} (độ)")
        print(f"  - theta_C: {theta_C} (độ)")
        print(f"  - ZL = {self.ZL} Ω, ZC = {self.ZC} Ω")
        print(f"  - Z_load = {self.Z_load} Ω, C_dc = {self.C_dc*1e12:.1f} pF")
        print("-" * 60)
        print(f"Kết quả:")
        print(f"  Zin = {zin:.2f} Ω")
        print(f"      = {zin.real:.2f} + j({zin.imag:.2f}) Ω")
        print(f"  |Zin| = {abs(zin):.2f} Ω")
        print(f"  Phase = {np.angle(zin, deg=True):.2f}°")
        print(f"  VSWR = {vswr:.2f}")
        print("=" * 60)