"""
Hàm Mục Tiêu (Objective Function) cho Tối Ưu Hóa Smith Chart
===========================================================

Kết hợp ImpedanceCalculator và SmithChartAnalyzer để tạo hàm chi phí
cho bài toán tối ưu hóa thiết kế mạng phối hợp impedance.

Sử dụng trong môi trường Google Colab với các chương trình:
- Zin1.py (ImpedanceCalculator)
- update_class_nearest_point.py (SmithChartAnalyzer)

Author: Your Name
Version: 1.0.0
"""

import numpy as np
import pandas as pd
from typing import List, Union, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class SmithChartObjectiveFunction:
    """
    Lớp tạo hàm mục tiêu để tối ưu hóa thiết kế mạng phối hợp impedance
    sử dụng Smith Chart analysis.
    """
    
    def __init__(self, 
                 smith_analyzer: 'SmithChartAnalyzer',
                 impedance_calculator: 'ImpedanceCalculator' = None,
                 target_frequency: float = 2.4e9,
                 reference_impedance: float = 50.0):
        """
        Khởi tạo hàm mục tiêu
        
        Args:
            smith_analyzer: Instance của SmithChartAnalyzer đã được cấu hình
            impedance_calculator: Instance của ImpedanceCalculator (tùy chọn)
            target_frequency: Tần số mục tiêu (Hz), mặc định 2.4 GHz
            reference_impedance: Trở kháng tham chiếu (Ohm), mặc định 50Ω
        """
        self.smith_analyzer = smith_analyzer
        self.impedance_calculator = impedance_calculator
        self.target_frequency = target_frequency
        self.reference_impedance = reference_impedance
        
        # Khởi tạo ImpedanceCalculator mặc định nếu không được cung cấp
        if self.impedance_calculator is None:
            self.impedance_calculator = ImpedanceCalculator()
        
        # Kiểm tra xem SmithChartAnalyzer đã có dữ liệu và envelope chưa
        self._validate_smith_analyzer()
        
        print("🎯 KHỞI TẠO HÀM MỤC TIÊU THÀNH CÔNG")
        print(f"   • Tần số mục tiêu: {target_frequency/1e9:.3f} GHz")
        print(f"   • Trở kháng tham chiếu: {reference_impedance} Ω")
        print(f"   • Smith Analyzer: {'✅ Sẵn sàng' if self.smith_analyzer.interpolated_envelope is not None else '⚠️ Cần cấu hình'}")
    
    def _validate_smith_analyzer(self):
        """Kiểm tra và chuẩn bị SmithChartAnalyzer"""
        if self.smith_analyzer.data is None:
            print("⚠️ SmithChartAnalyzer chưa có dữ liệu")
            return False
            
        # Nếu chưa có envelope, thử tạo với dữ liệu mặc định
        if self.smith_analyzer.interpolated_envelope is None:
            try:
                # Lọc dữ liệu mặc định
                self.smith_analyzer.filter_data(pae_threshold=40, pout_threshold=40)
                
                # Tạo envelope
                self.smith_analyzer.create_envelope()
                self.smith_analyzer.interpolate_envelope()
                
                print("✅ Đã tự động tạo envelope từ dữ liệu")
            except Exception as e:
                print(f"❌ Không thể tạo envelope: {e}")
                return False
        
        return True
    
    def impedance_to_smith_coordinates(self, zin_complex: complex) -> Tuple[float, float]:
        """
        Chuyển đổi impedance phức sang tọa độ Smith Chart
        
        Args:
            zin_complex: Trở kháng phức (a + jb)
            
        Returns:
            Tuple[float, float]: Tọa độ (real, imag) trên Smith Chart
        """
        # Chuẩn hóa impedance
        z_normalized = zin_complex / self.reference_impedance
        
        # Chuyển đổi sang reflection coefficient
        gamma = (z_normalized - 1) / (z_normalized + 1)
        
        # Tọa độ Smith Chart
        real_coord = gamma.real
        imag_coord = gamma.imag
        
        return real_coord, imag_coord
    
    def calculate_cost_function(self, theta_params: List[float], 
                              frequency: Optional[float] = None,
                              verbose: bool = False) -> float:
        """
        Tính hàm chi phí Lambda(m) = d_in^2
        
        Args:
            theta_params: Danh sách 8 tham số [theta_L1, theta_L2, theta_L3, theta_L4,
                                              theta_C1, theta_C2, theta_C3, theta_C4]
            frequency: Tần số tính toán (Hz). Nếu None, dùng target_frequency
            verbose: In thông tin chi tiết
            
        Returns:
            float: Giá trị hàm chi phí (d_in^2)
        """
        try:
            # Sử dụng tần số được chỉ định hoặc tần số mục tiêu
            freq = frequency if frequency is not None else self.target_frequency
            
            # Kiểm tra đầu vào
            if len(theta_params) != 8:
                raise ValueError(f"Cần đúng 8 tham số theta, nhận được {len(theta_params)}")
            
            # Tách tham số thành theta_L và theta_C
            theta_L = theta_params[:4]  # 4 tham số đầu
            theta_C = theta_params[4:]  # 4 tham số cuối
            
            if verbose:
                print(f"\n🔍 TÍNH TOÁN HÀM CHI PHÍ:")
                print(f"   • Tần số: {freq/1e9:.3f} GHz")
                print(f"   • theta_L: {[f'{t:.2f}°' for t in theta_L]}")
                print(f"   • theta_C: {[f'{t:.2f}°' for t in theta_C]}")
            
            # Bước 1: Tính trở kháng đầu vào Z_in
            zin_complex = self.impedance_calculator.calculate_zin_single(
                theta_L=theta_L,
                theta_C=theta_C,
                freq_Hz=freq
            )
            
            if verbose:
                print(f"   • Z_in = {zin_complex:.3f} Ω")
                print(f"         = {zin_complex.real:.3f} + j({zin_complex.imag:.3f}) Ω")
            
            # Bước 2: Chuyển Z_in sang tọa độ Smith Chart
            smith_coords = self.impedance_to_smith_coordinates(zin_complex)
            test_point = [smith_coords[0], smith_coords[1]]
            
            if verbose:
                print(f"   • Tọa độ Smith: ({test_point[0]:.6f}, {test_point[1]:.6f})")
            
            # Bước 3: Tính khoảng cách đến đường bao
            nearest_point, min_distance, _, is_inside = self.smith_analyzer.calculate_distance_to_envelope(test_point)
            
            if verbose:
                status = "TRONG" if is_inside else "NGOÀI"
                print(f"   • Vị trí: {status} đường bao")
                print(f"   • Khoảng cách: {min_distance:.6f}")
                if not is_inside and nearest_point is not None:
                    print(f"   • Điểm gần nhất: ({nearest_point[0]:.6f}, {nearest_point[1]:.6f})")
            
            # Bước 4: Tính hàm chi phí Lambda(m) = d_in^2
            cost = min_distance ** 2
            
            if verbose:
                print(f"   • Hàm chi phí: λ = d²= {cost:.8f}")
            
            return cost
            
        except Exception as e:
            if verbose:
                print(f"❌ Lỗi tính toán: {e}")
            
            # Trả về penalty cao khi có lỗi
            return 1e6
    
    def batch_evaluate(self, theta_sets: List[List[float]], 
                      frequency: Optional[float] = None,
                      verbose: bool = False) -> List[float]:
        """
        Tính hàm chi phí cho nhiều bộ tham số
        
        Args:
            theta_sets: Danh sách các bộ tham số theta
            frequency: Tần số tính toán
            verbose: In thông tin chi tiết
            
        Returns:
            List[float]: Danh sách các giá trị hàm chi phí
        """
        costs = []
        
        print(f"🔄 TÍNH TOÁN HÀNG LOẠT {len(theta_sets)} BỘ THAM SỐ")
        
        for i, theta_params in enumerate(theta_sets):
            if verbose:
                print(f"\n--- Bộ tham số {i+1}/{len(theta_sets)} ---")
            
            cost = self.calculate_cost_function(theta_params, frequency, verbose)
            costs.append(cost)
            
            if not verbose:
                print(f"Bộ {i+1}: λ = {cost:.8f}")
        
        print(f"✅ Hoàn thành {len(costs)} tính toán")
        return costs
    
    def create_optimization_wrapper(self, frequency: Optional[float] = None):
        """
        Tạo wrapper function cho các thư viện tối ưu hóa
        
        Args:
            frequency: Tần số cố định cho tối ưu hóa
            
        Returns:
            function: Hàm wrapper nhận theta_params và trả về cost
        """
        def objective_wrapper(theta_params):
            """Wrapper function cho scipy.optimize hoặc các thư viện tối ưu khác"""
            return self.calculate_cost_function(theta_params, frequency, verbose=False)
        
        return objective_wrapper
    
    def analyze_sensitivity(self, base_theta: List[float], 
                          delta_percent: float = 5.0,
                          frequency: Optional[float] = None) -> pd.DataFrame:
        """
        Phân tích độ nhạy của hàm chi phí theo từng tham số
        
        Args:
            base_theta: Bộ tham số cơ sở
            delta_percent: Phần trăm thay đổi để test (%)
            frequency: Tần số phân tích
            
        Returns:
            pd.DataFrame: Bảng kết quả phân tích độ nhạy
        """
        print(f"🔍 PHÂN TÍCH ĐỘ NHẠY (±{delta_percent}%)")
        
        base_cost = self.calculate_cost_function(base_theta, frequency)
        results = []
        
        for i in range(8):
            # Tính delta
            delta = abs(base_theta[i]) * delta_percent / 100
            if delta == 0:  # Tránh trường hợp base_theta[i] = 0
                delta = 1.0
            
            # Test tăng
            theta_plus = base_theta.copy()
            theta_plus[i] += delta
            cost_plus = self.calculate_cost_function(theta_plus, frequency)
            
            # Test giảm
            theta_minus = base_theta.copy()
            theta_minus[i] -= delta
            cost_minus = self.calculate_cost_function(theta_minus, frequency)
            
            # Tính gradient
            gradient = (cost_plus - cost_minus) / (2 * delta)
            
            # Lưu kết quả
            param_name = f"theta_L{i+1}" if i < 4 else f"theta_C{i-3}"
            results.append({
                'Parameter': param_name,
                'Base_Value': base_theta[i],
                'Delta': delta,
                'Cost_Base': base_cost,
                'Cost_Plus': cost_plus,
                'Cost_Minus': cost_minus,
                'Gradient': gradient,
                'Sensitivity': abs(gradient)
            })
        
        df_sensitivity = pd.DataFrame(results)
        df_sensitivity = df_sensitivity.sort_values('Sensitivity', ascending=False)
        
        print("✅ Hoàn thành phân tích độ nhạy")
        print("\n📊 Top 3 tham số nhạy nhất:")
        for i in range(min(3, len(df_sensitivity))):
            row = df_sensitivity.iloc[i]
            print(f"   {i+1}. {row['Parameter']}: Gradient = {row['Gradient']:.2e}")
        
        return df_sensitivity
    
    def plot_cost_surface_2d(self, base_theta: List[float], 
                           param_indices: Tuple[int, int] = (0, 4),
                           param_ranges: Tuple[Tuple[float, float], Tuple[float, float]] = None,
                           resolution: int = 20,
                           frequency: Optional[float] = None):
        """
        Vẽ mặt cắt 2D của hàm chi phí theo 2 tham số
        
        Args:
            base_theta: Bộ tham số cơ sở
            param_indices: Indices của 2 tham số cần vẽ
            param_ranges: Phạm vi của 2 tham số [(min1,max1), (min2,max2)]
            resolution: Độ phân giải lưới
            frequency: Tần số tính toán
        """
        try:
            import matplotlib.pyplot as plt
            
            # Thiết lập phạm vi mặc định
            if param_ranges is None:
                range1 = (base_theta[param_indices[0]] - 30, base_theta[param_indices[0]] + 30)
                range2 = (base_theta[param_indices[1]] - 30, base_theta[param_indices[1]] + 30)
            else:
                range1, range2 = param_ranges
            
            # Tạo lưới
            param1_vals = np.linspace(range1[0], range1[1], resolution)
            param2_vals = np.linspace(range2[0], range2[1], resolution)
            P1, P2 = np.meshgrid(param1_vals, param2_vals)
            
            # Tính hàm chi phí
            costs = np.zeros_like(P1)
            total_points = resolution * resolution
            
            print(f"🎨 Đang tính toán {total_points} điểm cho mặt cắt 2D...")
            
            for i in range(resolution):
                for j in range(resolution):
                    theta_test = base_theta.copy()
                    theta_test[param_indices[0]] = P1[i, j]
                    theta_test[param_indices[1]] = P2[i, j]
                    
                    costs[i, j] = self.calculate_cost_function(theta_test, frequency)
                
                # Progress update
                if (i + 1) % 5 == 0:
                    progress = ((i + 1) * resolution) / total_points * 100
                    print(f"   Tiến độ: {progress:.1f}%")
            
            # Vẽ contour plot
            fig, ax = plt.subplots(figsize=(10, 8))
            
            contour = ax.contourf(P1, P2, costs, levels=20, cmap='viridis')
            contour_lines = ax.contour(P1, P2, costs, levels=10, colors='black', alpha=0.4, linewidths=0.5)
            ax.clabel(contour_lines, inline=True, fontsize=8)
            
            # Đánh dấu điểm cơ sở
            ax.plot(base_theta[param_indices[0]], base_theta[param_indices[1]], 
                   'ro', markersize=10, label='Base Point')
            
            param_names = ['theta_L1', 'theta_L2', 'theta_L3', 'theta_L4',
                          'theta_C1', 'theta_C2', 'theta_C3', 'theta_C4']
            
            ax.set_xlabel(f'{param_names[param_indices[0]]} (độ)')
            ax.set_ylabel(f'{param_names[param_indices[1]]} (độ)')
            ax.set_title(f'Cost Function Surface: {param_names[param_indices[0]]} vs {param_names[param_indices[1]]}')
            
            # Colorbar
            cbar = fig.colorbar(contour)
            cbar.set_label('Cost Function λ')
            
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
            
            print("✅ Đã vẽ mặt cắt 2D của hàm chi phí")
            
        except ImportError:
            print("❌ Cần cài đặt matplotlib để vẽ biểu đồ")
        except Exception as e:
            print(f"❌ Lỗi khi vẽ biểu đồ: {e}")

# Utility functions for Google Colab
def create_objective_function_demo(smith_data_source, 
                                 target_frequency: float = 2.4e9,
                                 sample_theta: List[float] = None):
    """
    Tạo demo hàm mục tiêu trong Google Colab
    
    Args:
        smith_data_source: Dữ liệu cho SmithChartAnalyzer
        target_frequency: Tần số mục tiêu (Hz)
        sample_theta: Bộ tham số mẫu để test
    """
    print("🚀 TẠO DEMO HÀM MỤC TIÊU")
    
    # Tạo SmithChartAnalyzer
    smith_analyzer = SmithChartAnalyzer(smith_data_source)
    smith_analyzer.filter_data(pae_threshold=40, pout_threshold=40, pin_value=30) #Thêm điều kiện Pin đầu vào mình mong muốn
    smith_analyzer.create_envelope()
    smith_analyzer.interpolate_envelope()
    
    # Tạo ImpedanceCalculator
    impedance_calc = ImpedanceCalculator()
    
    # Tạo hàm mục tiêu
    objective_func = SmithChartObjectiveFunction(
        smith_analyzer=smith_analyzer,
        impedance_calculator=impedance_calc,
        target_frequency=target_frequency
    )
    
    # Test với bộ tham số mẫu
    if sample_theta is None:
        sample_theta = [45.0, 30.0, 60.0, 45.0,  # theta_L
                       90.0, 45.0, 90.0, 60.0]   # theta_C
    
    print(f"\n🧪 TEST VỚI BỘ THAM SỐ MẪU:")
    cost = objective_func.calculate_cost_function(sample_theta, verbose=True)
    
    print(f"\n🎯 KẾT QUẢ CUỐI CÙNG:")
    print(f"   Hàm chi phí λ = {cost:.8f}")
    
    return objective_func

def quick_optimization_example():
    """
    Ví dụ nhanh về cách sử dụng với scipy.optimize
    """
    print("📚 VÍ DỤ SỬ DỤNG VỚI SCIPY.OPTIMIZE:")
    print("""
    # Import thư viện tối ưu
    from scipy.optimize import minimize
    
    # Tạo hàm mục tiêu (giả sử đã có objective_func)
    obj_wrapper = objective_func.create_optimization_wrapper()
    
    # Điểm khởi tạo
    initial_theta = [45.0, 30.0, 60.0, 45.0, 90.0, 45.0, 90.0, 60.0]
    
    # Ràng buộc (ví dụ: 0° ≤ theta ≤ 180°)
    bounds = [(0, 180)] * 8
    
    # Tối ưu hóa
    result = minimize(obj_wrapper, initial_theta, bounds=bounds, method='L-BFGS-B')
    
    print("Optimal theta:", result.x)
    print("Optimal cost:", result.fun)
    """)

