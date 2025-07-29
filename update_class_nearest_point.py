"""
Smith Chart Analyzer Library - Google Colab Version (Updated)
=============================================================

Thư viện phân tích Smith Chart được tối ưu cho Google Colab với:
- Tự động cài đặt dependencies
- Tương thích với Google Drive
- Hỗ trợ widgets tương tác
- Xuất kết quả trực tiếp
- Kiểm tra điểm test nằm trong/ngoài đường nội suy

Author: Your Name
Version: 1.1.0 (Google Colab - Updated)
"""

# Tự động cài đặt các thư viện cần thiết
import subprocess
import sys

def install_if_missing(package):
    """Cài đặt package nếu chưa có"""
    try:
        __import__(package)
    except ImportError:
        print(f"📦 Đang cài đặt {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Cài đặt dependencies
required_packages = ['plotly', 'scipy', 'ipywidgets', 'matplotlib']
for package in required_packages:
    install_if_missing(package)

# Import các thư viện
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from scipy.interpolate import interp1d, splprep, splev
from scipy.spatial.distance import cdist
from scipy.spatial import ConvexHull
import ipywidgets as widgets
from IPython.display import display, clear_output, HTML
from google.colab import drive
from matplotlib.path import Path
import os
import warnings
warnings.filterwarnings('ignore')

class SmithChartAnalyzer:
    """
    Lớp chính để phân tích Smith Chart - Tối ưu cho Google Colab
    Cập nhật với chức năng kiểm tra điểm test nằm trong/ngoài đường nội suy
    """

    def __init__(self, data_source=None, auto_mount_drive=True):
        """
        Khởi tạo analyzer

        Args:
            data_source: DataFrame hoặc đường dẫn file CSV
            auto_mount_drive: Tự động mount Google Drive
        """
        self.data = None
        self.filtered_data = None
        self.current_pin = None
        self.envelope_points = None
        self.interpolated_envelope = None
        self.drive_mounted = False

        # Tự động mount Google Drive nếu cần
        if auto_mount_drive and isinstance(data_source, str) and '/content/drive' in data_source:
             self.mount_drive()

        if data_source is not None:
            self.load_data(data_source)

    def mount_drive(self):
        """Mount Google Drive"""
        try:
            if not os.path.exists('/content/drive'):
                print("🔗 Đang kết nối Google Drive...")
                drive.mount('/content/drive')
                self.drive_mounted = True
                print("✅ Đã kết nối Google Drive thành công!")
            else:
                self.drive_mounted = True
                print("✅ Google Drive đã được kết nối!")
        except Exception as e:
            print(f"❌ Lỗi khi kết nối Google Drive: {e}")

    def load_data(self, data_source):
        """
        Tải dữ liệu từ CSV hoặc DataFrame

        Args:
            data_source: DataFrame hoặc đường dẫn file CSV
        """
        if isinstance(data_source, str):
            # Kiểm tra nếu file ở Google Drive
            if '/content/drive' in data_source and not self.drive_mounted:
                self.mount_drive()

            if not os.path.exists(data_source):
                raise FileNotFoundError(f"Không tìm thấy file: {data_source}")

            self.data = pd.read_csv(data_source)
            print(f"📁 Đã tải file: {data_source}")

        elif isinstance(data_source, pd.DataFrame):
            self.data = data_source.copy()
            print("📊 Đã tải DataFrame")
        else:
            raise ValueError("data_source phải là DataFrame hoặc đường dẫn file CSV")

        # Kiểm tra các cột cần thiết
        required_columns = ['Real', 'Image', 'Pin', 'PAE', 'Pout']
        missing_columns = [col for col in required_columns if col not in self.data.columns]

        if missing_columns:
            raise ValueError(f"Thiếu các cột: {missing_columns}")

        print(f"✅ Đã tải {len(self.data)} dòng dữ liệu")
        self.display_data_info()

    def display_data_info(self):
        """Hiển thị thông tin dữ liệu"""
        if self.data is None:
            return

        print("\n📊 THÔNG TIN DỮ LIỆU:")
        print(f"   • Tổng số dòng: {len(self.data):,}")
        print(f"   • Số cột: {len(self.data.columns)}")
        print(f"   • Phạm vi Pin: {self.data['Pin'].min():.1f} - {self.data['Pin'].max():.1f}")
        print(f"   • Phạm vi PAE: {self.data['PAE'].min():.1f} - {self.data['PAE'].max():.1f}")
        print(f"   • Phạm vi Pout: {self.data['Pout'].min():.1f} - {self.data['Pout'].max():.1f}")

        # Hiển thị mẫu dữ liệu
        display(HTML("<h4>🔍 Mẫu dữ liệu (5 dòng đầu):</h4>"))
        display(self.data.head())

    def filter_data(self, pae_threshold=40, pout_threshold=40, pin_value=None):
        """
        Lọc dữ liệu theo các điều kiện

        Args:
            pae_threshold: Ngưỡng PAE tối thiểu
            pout_threshold: Ngưỡng Pout tối thiểu
            pin_value: Giá trị Pin cụ thể (None để lấy tất cả)
        """
        if self.data is None:
            raise ValueError("Chưa tải dữ liệu. Gọi load_data() trước.")

        # Lọc cơ bản
        self.filtered_data = self.data[
            (self.data['PAE'] > pae_threshold) &
            (self.data['Pout'] > pout_threshold)
        ]

        # Lọc theo Pin nếu được chỉ định
        if pin_value is not None:
            self.filtered_data = self.filtered_data[self.filtered_data['Pin'] == pin_value]
            self.current_pin = pin_value

        print(f"✅ Đã lọc còn {len(self.filtered_data):,} dòng dữ liệu")
        if pin_value is not None:
            print(f"   • Pin = {pin_value}")
        print(f"   • PAE > {pae_threshold}")
        print(f"   • Pout > {pout_threshold}")

        return self.filtered_data

    def get_available_pins(self):
        """Lấy danh sách các giá trị Pin có sẵn"""
        if self.data is None:
            raise ValueError("Chưa tải dữ liệu")

        pins = sorted(self.data['Pin'].unique())
        print(f"📍 Có {len(pins)} giá trị Pin: {pins}")
        return pins

    def create_envelope(self, real_vals=None, imag_vals=None, method='convex_hull'):
        """
        Tạo đường bao từ các điểm dữ liệu
        """
        if real_vals is None or imag_vals is None:
            if self.filtered_data is None or len(self.filtered_data) == 0:
                raise ValueError("Không có dữ liệu đã lọc. Gọi filter_data() trước.")
            real_vals = self.filtered_data['Real']
            imag_vals = self.filtered_data['Image']

        if len(real_vals) < 3:
            print("❌ Cần ít nhất 3 điểm để tạo đường bao")
            return None

        points = np.column_stack([real_vals, imag_vals])

        try:
            if method == 'convex_hull':
                hull = ConvexHull(points)
                envelope_points = points[hull.vertices]
                envelope_points = np.vstack([envelope_points, envelope_points[0]])

            self.envelope_points = envelope_points
            print(f"✅ Đã tạo đường bao với {len(envelope_points)} điểm")
            return envelope_points
        except Exception as e:
            print(f"❌ Lỗi tạo đường bao: {e}")
            return None

    def interpolate_envelope(self, envelope_points=None, method='spline', num_points=200, smoothing_factor=0):
        """
        Nội suy đường bao thành đường cong mượt
        """
        if envelope_points is None:
            envelope_points = self.envelope_points

        if envelope_points is None or len(envelope_points) < 4:
            print("❌ Cần ít nhất 4 điểm để nội suy")
            return None

        x = envelope_points[:, 0]
        y = envelope_points[:, 1]

        try:
            if method == 'spline':
                tck, u = splprep([x, y], s=smoothing_factor, per=True)
                u_new = np.linspace(0, 1, num_points)
                x_new, y_new = splev(u_new, tck)
                interpolated = np.column_stack([x_new, y_new])

            self.interpolated_envelope = interpolated
            print(f"✅ Đã nội suy thành {num_points} điểm với độ mịn s={smoothing_factor}")
            return interpolated
        except Exception as e:
            print(f"❌ Lỗi nội suy: {e}")
            return None

    def is_point_inside_envelope(self, test_point, envelope_curve=None):
        """
        Kiểm tra xem điểm test có nằm trong đường nội suy hay không
        Sử dụng thuật toán Ray Casting
        """
        if envelope_curve is None:
            envelope_curve = self.interpolated_envelope

        if envelope_curve is None or len(envelope_curve) < 3:
            return False

        try:
            # Tạo đường path từ envelope curve
            path = Path(envelope_curve)

            # Kiểm tra điểm có nằm trong path hay không
            is_inside = path.contains_point(test_point)

            return is_inside
        except:
            return False

    def find_nearest_point_on_envelope(self, test_point, envelope_curve=None):
        """Tìm điểm gần nhất trên đường bao và khoảng cách ngắn nhất"""
        if envelope_curve is None:
            envelope_curve = self.interpolated_envelope

        if envelope_curve is None:
            return None, float('inf'), -1

        test_point = np.array(test_point).reshape(1, -1)
        distances = cdist(test_point, envelope_curve)[0]

        min_index = np.argmin(distances)
        min_distance = distances[min_index]
        nearest_point = envelope_curve[min_index]

        return nearest_point, min_distance, min_index

    def calculate_distance_to_envelope(self, test_point, envelope_curve=None):
        """
        Tính khoảng cách từ điểm test đến đường nội suy
        - Nếu điểm nằm trong đường nội suy: khoảng cách = 0
        - Nếu điểm nằm ngoài đường nội suy: tính khoảng cách ngắn nhất
        """
        if envelope_curve is None:
            envelope_curve = self.interpolated_envelope

        if envelope_curve is None:
            return None, float('inf'), -1, False

        # Kiểm tra điểm có nằm trong đường nội suy hay không
        is_inside = self.is_point_inside_envelope(test_point, envelope_curve)

        if is_inside:
            # Điểm nằm trong đường nội suy, khoảng cách = 0
            # Tìm điểm gần nhất để hiển thị
            nearest_point, _, index = self.find_nearest_point_on_envelope(test_point, envelope_curve)
            return nearest_point, 0.0, index, True
        else:
            # Điểm nằm ngoài đường nội suy, tính khoảng cách thực tế
            nearest_point, min_distance, index = self.find_nearest_point_on_envelope(test_point, envelope_curve)
            return nearest_point, min_distance, index, False

    def analyze_test_point(self, test_point, pin_value=None):
        """
        Phân tích một điểm test với chức năng kiểm tra vị trí
        """
        print(f"\n🔍 PHÂN TÍCH ĐIỂM TEST: {test_point}")

        if pin_value is not None:
            self.filter_data(pin_value=pin_value)

        # Tạo đường bao
        envelope_points = self.create_envelope()
        if envelope_points is None:
            return {"error": "Không thể tạo đường bao"}

        # Nội suy
        interpolated_envelope = self.interpolate_envelope()
        if interpolated_envelope is None:
            return {"error": "Không thể nội suy đường bao"}

        # Tính khoảng cách với kiểm tra vị trí điểm
        nearest_point, min_distance, index, is_inside = self.calculate_distance_to_envelope(test_point)

        result = {
            "test_point": test_point,
            "nearest_point": nearest_point.tolist() if nearest_point is not None else None,
            "min_distance": min_distance,
            "nearest_index": index,
            "is_inside": is_inside,
            "pin_value": self.current_pin,
            "num_data_points": len(self.filtered_data)
        }

        print(f"✅ KẾT QUẢ PHÂN TÍCH:")
        print(f"   • Điểm test: ({test_point[0]:.3f}, {test_point[1]:.3f})")

        if is_inside:
            print(f"   • Trạng thái: NẰM TRONG đường nội suy")
            print(f"   • Khoảng cách: 0 (điểm nằm bên trong)")
        else:
            print(f"   • Trạng thái: NẰM NGOÀI đường nội suy")
            print(f"   • Điểm gần nhất: ({nearest_point[0]:.3f}, {nearest_point[1]:.3f})")
            print(f"   • Khoảng cách: {min_distance:.6f}")

        print(f"   • Pin: {self.current_pin}")
        print(f"   • Số điểm dữ liệu: {len(self.filtered_data)}")

        return result

    def create_smith_chart_figure(self, test_point=None, show_grid=True, width=900, height=700):
        """
        Tạo biểu đồ Smith Chart với hiển thị cải tiến
        """
        fig = go.Figure()

        # Vẽ dữ liệu gốc
        if self.filtered_data is not None and len(self.filtered_data) > 0:
            fig.add_trace(go.Scatter(
                x=self.filtered_data['Real'],
                y=self.filtered_data['Image'],
                mode='markers',
                name='Data Points',
                marker=dict(size=6, color='blue', opacity=0.7),
                text=[f'Pin: {p:.1f}<br>PAE: {pae:.1f}<br>Pout: {pout:.1f}'
                      for p, pae, pout in zip(self.filtered_data['Pin'],
                                             self.filtered_data['PAE'],
                                             self.filtered_data['Pout'])],
                hovertemplate='Real: %{x:.3f}<br>Imag: %{y:.3f}<br>%{text}<extra></extra>'
            ))

        # Vẽ đường bao gốc
        if self.envelope_points is not None:
            fig.add_trace(go.Scatter(
                x=self.envelope_points[:, 0],
                y=self.envelope_points[:, 1],
                mode='lines+markers',
                name='Original Envelope',
                line=dict(color='red', width=2),
                marker=dict(size=4, color='red')
            ))

        # Vẽ đường bao nội suy
        if self.interpolated_envelope is not None:
            fig.add_trace(go.Scatter(
                x=self.interpolated_envelope[:, 0],
                y=self.interpolated_envelope[:, 1],
                mode='lines',
                name='Interpolated Envelope',
                line=dict(color='green', width=3, dash='dash')
            ))

        # Vẽ điểm test và điểm gần nhất
        if test_point is not None:
            # Tính khoảng cách với kiểm tra vị trí điểm
            nearest_point, min_distance, _, is_inside = self.calculate_distance_to_envelope(test_point)

            # Vẽ điểm test với màu khác nhau tùy thuộc vào vị trí
            test_color = 'red' if is_inside else 'orange'
            test_symbol = 'circle' if is_inside else 'star'
            test_name = 'Test Point (Inside)' if is_inside else 'Test Point (Outside)'

            fig.add_trace(go.Scatter(
                x=[test_point[0]],
                y=[test_point[1]],
                mode='markers',
                name=test_name,
                marker=dict(size=15, color=test_color, symbol=test_symbol)
            ))

            # Vẽ điểm gần nhất
            if nearest_point is not None:
                fig.add_trace(go.Scatter(
                    x=[nearest_point[0]],
                    y=[nearest_point[1]],
                    mode='markers',
                    name='Nearest Point',
                    marker=dict(size=12, color='purple', symbol='diamond')
                ))

                # Chỉ vẽ đường nối khi điểm ở ngoài
                if not is_inside:
                    fig.add_trace(go.Scatter(
                        x=[test_point[0], nearest_point[0]],
                        y=[test_point[1], nearest_point[1]],
                        mode='lines',
                        name=f'Distance: {min_distance:.6f}',
                        line=dict(color='purple', width=2, dash='dot')
                    ))

        # Thêm lưới Smith Chart
        if show_grid:
            self._add_smith_chart_grid(fig)

        # Cập nhật title
        title = f'Smith Chart'
        if self.current_pin is not None:
            title += f' - Pin = {self.current_pin:.1f}'
        if test_point is not None:
            nearest_point, min_distance, _, is_inside = self.calculate_distance_to_envelope(test_point)
            status_text = "INSIDE (Distance = 0)" if is_inside else f"OUTSIDE (Distance = {min_distance:.6f})"
            title += f' - Test Point: ({test_point[0]:.3f}, {test_point[1]:.3f}) | Status: {status_text}'

        fig.update_layout(
            title=title,
            xaxis_title='Resistance (Real)',
            yaxis_title='Reactance (Imaginary)',
            width=width,
            height=height,
            showlegend=True,
            xaxis=dict(scaleanchor="y", scaleratio=1, range=[-1.2, 1.2]),
            yaxis=dict(scaleanchor="x", scaleratio=1, range=[-1.2, 1.2])
        )

        return fig

    def _add_smith_chart_grid(self, fig):
        """Thêm lưới Smith Chart"""
        # Vẽ đường tròn đơn vị
        theta = np.linspace(0, 2*np.pi, 100)
        fig.add_trace(go.Scatter(
            x=np.cos(theta),
            y=np.sin(theta),
            mode='lines',
            name='Unit Circle',
            line=dict(color='black', width=1),
            showlegend=False
        ))

        # Vẽ các đường tròn resistance
        for r in [0.2, 0.5, 1.0, 2.0, 5.0]:
            center_x = r / (r + 1)
            radius = 1 / (r + 1)
            theta = np.linspace(0, 2*np.pi, 100)
            x = center_x + radius * np.cos(theta)
            y = radius * np.sin(theta)
            fig.add_trace(go.Scatter(
                x=x, y=y,
                mode='lines',
                line=dict(color='gray', width=0.5),
                showlegend=False
            ))

        # Vẽ các đường tròn reactance
        for x in [0.2, 0.5, 1.0, 2.0, 5.0]:
            center_y = 1/x
            radius = 1/x
            theta = np.linspace(0, np.pi, 50)
            x_vals = 1 + radius * np.cos(theta)
            y_vals = center_y + radius * np.sin(theta)
            fig.add_trace(go.Scatter(
                x=x_vals, y=y_vals,
                mode='lines',
                line=dict(color='gray', width=0.5),
                showlegend=False
            ))
            fig.add_trace(go.Scatter(
                x=x_vals, y=-y_vals,
                mode='lines',
                line=dict(color='gray', width=0.5),
                showlegend=False
            ))

    def create_interactive_widget(self):
        """
        Tạo widget tương tác cho Google Colab với cải tiến
        """
        if self.data is None:
            raise ValueError("Chưa tải dữ liệu")

        # Lấy phạm vi Pin
        pins = self.get_available_pins()

        # Tạo widgets
        pin_slider = widgets.IntSlider(
            value=int(pins[0]),
            min=int(min(pins)),
            max=int(max(pins)),
            step=1,
            description='Pin:',
            style={'description_width': 'initial'}
        )

        test_x_slider = widgets.FloatSlider(
            value=0.5,
            min=-1.0,
            max=1.0,
            step=0.01,
            description='Test X:',
            style={'description_width': 'initial'}
        )

        test_y_slider = widgets.FloatSlider(
            value=0.3,
            min=-1.0,
            max=1.0,
            step=0.01,
            description='Test Y:',
            style={'description_width': 'initial'}
        )

        # Output widgets
        output_plot = widgets.Output()
        output_info = widgets.Output()

        def update_plot(pin_val, test_x, test_y):
            with output_plot:
                clear_output(wait=True)

                # Phân tích
                result = self.analyze_test_point([test_x, test_y], pin_val)

                if "error" not in result:
                    # Tạo và hiển thị biểu đồ
                    fig = self.create_smith_chart_figure([test_x, test_y])
                    fig.show()

                    # Hiển thị thông tin
                    with output_info:
                        clear_output(wait=True)

                        status_text = "ĐIỂM NẰM TRONG ĐƯỜNG NỘI SUY" if result['is_inside'] else "ĐIỂM NẰM NGOÀI ĐƯỜNG NỘI SUY"
                        status_color = "#d4edda" if result['is_inside'] else "#f8d7da"

                        distance_text = "0 (điểm nằm bên trong)" if result['is_inside'] else f"{result['min_distance']:.6f}"

                        display(HTML(f"""
                        <div style="background-color: {status_color}; padding: 10px; border-radius: 5px; margin: 10px 0;">
                            <h4>📊 Kết quả phân tích</h4>
                            <p><strong>Pin:</strong> {result['pin_value']}</p>
                            <p><strong>Điểm test:</strong> ({result['test_point'][0]:.3f}, {result['test_point'][1]:.3f})</p>
                            <p><strong>Trạng thái:</strong> {status_text}</p>
                            <p><strong>Khoảng cách:</strong> {distance_text}</p>
                            {f"<p><strong>Điểm gần nhất:</strong> ({result['nearest_point'][0]:.3f}, {result['nearest_point'][1]:.3f})</p>" if not result['is_inside'] else ""}
                            <p><strong>Số điểm dữ liệu:</strong> {result['num_data_points']}</p>
                        </div>
                        """))

        # Tạo interactive
        interactive_plot = widgets.interactive(
            update_plot,
            pin_val=pin_slider,
            test_x=test_x_slider,
            test_y=test_y_slider
        )

        # Layout
        controls = widgets.VBox([
            widgets.HTML("<h3>🎛️ Điều khiển Smith Chart</h3>"),
            interactive_plot.children[0],  # pin_slider
            interactive_plot.children[1],  # test_x_slider
            interactive_plot.children[2],  # test_y_slider
            output_info
        ])

        main_layout = widgets.VBox([
            controls,
            widgets.HTML("<h3>📊 Smith Chart</h3>"),
            output_plot
        ])

        display(main_layout)

        # Vẽ đồ thị ban đầu
        update_plot(pin_slider.value, test_x_slider.value, test_y_slider.value)

        print("🎉 Widget tương tác đã sẵn sàng!")

    def batch_analyze(self, test_points, pin_values=None):
        """
        Phân tích hàng loạt nhiều điểm test
        """
        print(f"🔄 BẮT ĐẦU PHÂN TÍCH HÀNG LOẠT {len(test_points)} ĐIỂM")

        results = []

        if pin_values is None:
            pin_values = [None] * len(test_points)

        for i, (test_point, pin_value) in enumerate(zip(test_points, pin_values)):
            print(f"\n📍 Phân tích điểm {i+1}/{len(test_points)}: {test_point}")
            result = self.analyze_test_point(test_point, pin_value)
            results.append(result)

        print(f"\n✅ Hoàn thành phân tích {len(results)} điểm")
        return results

    def export_results(self, results, filename=None):
        """
        Xuất kết quả ra file CSV
        """
        if filename is None:
            filename = f'/content/smith_chart_results_{pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")}.csv'

        if not isinstance(results, list):
            results = [results]

        # Chuyển đổi kết quả thành DataFrame
        export_data = []
        for result in results:
            if "error" not in result:
                export_data.append({
                    'test_x': result['test_point'][0],
                    'test_y': result['test_point'][1],
                    'nearest_x': result['nearest_point'][0] if result['nearest_point'] else None,
                    'nearest_y': result['nearest_point'][1] if result['nearest_point'] else None,
                    'min_distance': result['min_distance'],
                    'is_inside': result['is_inside'],
                    'pin_value': result['pin_value'],
                    'num_data_points': result['num_data_points']
                })

        df_export = pd.DataFrame(export_data)
        df_export.to_csv(filename, index=False)

        print(f"✅ Đã xuất {len(export_data)} kết quả ra file: {filename}")

         # Hiển thị mẫu kết quả
        if len(df_export) > 0:
            display(HTML("<h4>📋 Mẫu kết quả xuất:</h4>"))
            display(df_export.head())

        return filename


# Utility functions cho Google Colab
def setup_colab_environment():
    """
    Thiết lập môi trường Google Colab
    """
    print("🚀 THIẾT LẬP MÔI TRƯỜNG GOOGLE COLAB")

    # Enable widgets
    from google.colab import output
    output.enable_custom_widget_manager()

    # Cấu hình Plotly
    import plotly.io as pio
    pio.renderers.default = "colab"

    print("✅ Môi trường đã sẵn sàng!")

def quick_analyze_colab(data_source, test_point, pin_value=None):
    """
    Phân tích nhanh cho Google Colab
    """
    print("⚡ PHÂN TÍCH NHANH")

    analyzer = SmithChartAnalyzer(data_source)

    if pin_value is None:
        available_pins = analyzer.get_available_pins()
        pin_value = available_pins[0] if available_pins else None

    result = analyzer.analyze_test_point(test_point, pin_value)

    # Hiển thị biểu đồ
    fig = analyzer.create_smith_chart_figure(test_point)
    fig.show()

    return result

def create_colab_demo(data_source):
    """
    Tạo demo tương tác cho Google Colab
    """
    print("🎪 TẠO DEMO TƯƠNG TÁC")

    # Thiết lập môi trường
    setup_colab_environment()

    # Tạo analyzer
    analyzer = SmithChartAnalyzer(data_source)

    # Tạo widget tương tác
    analyzer.create_interactive_widget()

    return analyzer

# Hàm tiện ích cho việc xử lý file
def upload_and_analyze():
    """
    Upload file và phân tích (sử dụng với Google Colab file upload)
    """
    from google.colab import files

    print("📤 Tải file CSV lên:")
    uploaded = files.upload()

    if uploaded:
        filename = list(uploaded.keys())[0]
        print(f"✅ Đã tải file: {filename}")

        # Tạo analyzer
        analyzer = SmithChartAnalyzer(filename)

        # Tạo demo
        analyzer.create_interactive_widget()

        return analyzer
    else:
        print("❌ Không có file nào được tải lên")
        return None

def connect_drive_and_analyze(drive_path):
    """
    Kết nối Google Drive và phân tích
    """
    print("🔗 Kết nối Google Drive và phân tích")

    # Tạo analyzer (sẽ tự động mount drive)
    analyzer = SmithChartAnalyzer(drive_path)

    # Tạo demo
    analyzer.create_interactive_widget()

    return analyzer