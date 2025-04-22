import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QHBoxLayout, QPushButton, QLabel, QFileDialog,
                           QComboBox, QGroupBox, QSpinBox, QSlider)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt
import cv2
import numpy as np

class AppProcesareImagini(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Aplicatie Procesare Imagini")
        self.setGeometry(100, 100, 1400, 800)
        
        self.original_image = None
        self.processed_image = None
        
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)
        
        control_group = QGroupBox("Control")
        control_layout = QVBoxLayout()  # Schimbat in QVBoxLayout
        
        top_controls = QHBoxLayout()
        
        self.load_button = QPushButton("incarca Imagine")
        self.load_button.setFixedHeight(40)
        self.load_button.clicked.connect(self.load_image)
        control_layout.addWidget(self.load_button)
        
        self.processing_combo = QComboBox()
        self.processing_combo.setFixedHeight(40)
        self.processing_combo.addItems([
            "Selecteaza functia",
            "Conversie in tonuri de gri",
            "Conversie in BGR",
            "Blur simplu",
            "Blur gaussian",
            "Bilateral filtering",
            "Denoise",
            "Detectie margini Canny",
            "Conturare",
            "Binarizare",
            "Egalizare histograma",
            "Ajustare luminozitate",
            "Ajustare contrast",
            "Inversare culori",
            "Rotire imagine",
            "Translatie imagine",
            "Dilatare + Eroziune",
            "Filtru median",
            "Filtru Sobel",
            "Transformare cartopolara",
            "Detectie colturi Harris"
        ])
        self.processing_combo.currentIndexChanged.connect(self.process_image)
        top_controls.addWidget(self.processing_combo)
        
        self.param_layout = QHBoxLayout()
        
        # Adauga layout-urile in control_layout
        control_layout.addLayout(top_controls)      # Adauga layout-ul de sus
        control_layout.addLayout(self.param_layout) # Adauga layout-ul de jos
        
        # Seteaza layout-ul pentru group
        control_group.setLayout(control_layout)
        layout.addWidget(control_group)
        
        # Container pentru imagini
        image_container = QWidget()
        image_layout = QHBoxLayout(image_container)
        
        original_group = QGroupBox("Imagine Originala")
        original_layout = QVBoxLayout()
        self.original_label = QLabel()
        self.original_label.setAlignment(Qt.AlignCenter)
        self.original_label.setMinimumSize(500, 500)
        self.original_label.setStyleSheet("QLabel { background-color: #f0f0f0; border: 1px solid #cccccc; }")
        original_layout.addWidget(self.original_label)
        original_group.setLayout(original_layout)
        image_layout.addWidget(original_group)
        
        # Widget pentru imaginea procesata
        processed_group = QGroupBox("Imagine Procesata")
        processed_layout = QVBoxLayout()
        self.processed_label = QLabel()
        self.processed_label.setAlignment(Qt.AlignCenter)
        self.processed_label.setMinimumSize(500, 500)
        self.processed_label.setStyleSheet("QLabel { background-color: #f0f0f0; border: 1px solid #cccccc; }")
        processed_layout.addWidget(self.processed_label)
        processed_group.setLayout(processed_layout)
        image_layout.addWidget(processed_group)
        
        layout.addWidget(image_container)
        
        self.setup_parameters()

    def load_image(self):
        """incarca o imagine din calculator"""
        file_name, _ = QFileDialog.getOpenFileName(
            self, 
            "Selecteaza Imaginea",
            "",
            "Image files (*.jpg *.jpeg *.png *.bmp *.gif *.tiff)"
        )
        
        if file_name:
            # Citeste imaginea folosind OpenCV
            self.original_image = cv2.imread(file_name)
            if self.original_image is None:
                return
            
            # Converteste din BGR in RGB
            self.original_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
            
            # Redimensioneaza imaginea daca este prea mare
            max_size = 500
            height, width = self.original_image.shape[:2]
            if height > max_size or width > max_size:
                scale = max_size / max(height, width)
                new_width = int(width * scale)
                new_height = int(height * scale)
                self.original_image = cv2.resize(self.original_image, (new_width, new_height))
            
            self.display_image(self.original_image, self.original_label)
            
            self.processed_image = self.original_image.copy()
            self.display_image(self.processed_image, self.processed_label)

    def display_image(self, image, label):
        """Afiseaza o imagine intr-un QLabel"""
        height, width, channel = image.shape
        bytes_per_line = 3 * width
        q_image = QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        label.setPixmap(pixmap)

    def setup_parameters(self):
        """Initializeaza parametrii pentru functii"""
        self.params = {}
        
        # Parametri pentru blur
        self.params['kernel_size'] = QSpinBox()
        self.params['kernel_size'].setRange(1, 31)
        self.params['kernel_size'].setSingleStep(2)
        self.params['kernel_size'].setValue(5)
        
        # Parametri pentru Canny
        self.params['threshold1'] = QSpinBox()
        self.params['threshold1'].setRange(0, 255)
        self.params['threshold1'].setValue(100)
        
        self.params['threshold2'] = QSpinBox()
        self.params['threshold2'].setRange(0, 255)
        self.params['threshold2'].setValue(200)
        
        # Parametri pentru luminozitate si contrast
        self.params['alpha'] = QSlider(Qt.Horizontal)
        self.params['alpha'].setRange(0, 300)
        self.params['alpha'].setValue(100)
        
        self.params['beta'] = QSlider(Qt.Horizontal)
        self.params['beta'].setRange(-127, 127)
        self.params['beta'].setValue(0)
        
        # Parametri pentru rotatie
        self.params['angle'] = QSpinBox()
        self.params['angle'].setRange(0, 360)
        self.params['angle'].setValue(90)
        
        # Parametri pentru translatie
        self.params['tx'] = QSpinBox()
        self.params['tx'].setRange(-500, 500)
        self.params['ty'] = QSpinBox()
        self.params['ty'].setRange(-500, 500)
        
        # Conectare la evenimente
        for param in self.params.values():
            if isinstance(param, (QSlider, QSpinBox)):
                param.valueChanged.connect(lambda: self.process_image(None))

    def clear_param_layout(self):
        """sterge parametrii din layout"""
        while self.param_layout.count():
            item = self.param_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.hide()

    def show_parameters(self, param_list):
        """Afiseaza parametrii specificati"""
        # Mai intai ascundem toti parametrii
        for param in self.params.values():
            if isinstance(param, (QSlider, QSpinBox)):
                param.setParent(None)
        
        self.clear_param_layout()
    
        # Afisam doar parametrii necesari
        for param_name in param_list:
            if param_name in self.params:
                label = QLabel(param_name.replace('_', ' ').title() + ':')
                self.param_layout.addWidget(label)
                self.param_layout.addWidget(self.params[param_name])
                self.params[param_name].show()
                
    def process_image(self, index=None):
        if self.original_image is None:
            return
            
        function_name = self.processing_combo.currentText()
        
        try:
            if function_name == "Conversie in tonuri de gri":
                self.processed_image = cv2.cvtColor(self.original_image, cv2.COLOR_RGB2GRAY) # din 3 canale -> 1 canal
                self.processed_image = cv2.cvtColor(self.processed_image, cv2.COLOR_GRAY2RGB) # sa fie compatibil cu QLabel
                self.show_parameters([])
            
            elif function_name == "Conversie in BGR":
                self.processed_image = cv2.cvtColor(self.original_image, cv2.COLOR_RGB2BGR)
           
            elif function_name == "Blur simplu":
                self.show_parameters(['kernel_size'])
                ksize = self.params['kernel_size'].value()
                if ksize % 2 == 0:
                    ksize += 1      # ksize trb. sa fie impar
                self.processed_image = cv2.blur(self.original_image, (ksize, ksize))
                
            elif function_name == "Blur gaussian":
                self.show_parameters(['kernel_size'])
                ksize = self.params['kernel_size'].value()
                if ksize % 2 == 0:
                    ksize += 1
                self.processed_image = cv2.GaussianBlur(self.original_image, (ksize, ksize), 0) # sigmaX=0
                
            elif function_name == "Bilateral filtering":
                self.show_parameters(['kernel_size'])
                ksize = self.params['kernel_size'].value()
                self.processed_image = cv2.bilateralFilter(self.original_image, ksize, 75, 75) # estompeaza, reduce zgomotul din img
                
            elif function_name == "Denoise":
                self.show_parameters([])
                self.processed_image = cv2.fastNlMeansDenoisingColored(self.original_image, None, 
                                                                       10, 10, 7, 21) # zgomot, zgomot/culoare, templateWindowSearch, searchWindow
                
            elif function_name == "Detectie margini Canny":
                self.show_parameters(['threshold1', 'threshold2'])
                gray = cv2.cvtColor(self.original_image, cv2.COLOR_RGB2GRAY)
                edges = cv2.Canny(gray, 
                                self.params['threshold1'].value(),
                                self.params['threshold2'].value())
                self.processed_image = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
                
            elif function_name == "Conturare":
                self.show_parameters(['threshold1'])
                gray = cv2.cvtColor(self.original_image, cv2.COLOR_RGB2GRAY)
                _, thresh = cv2.threshold(gray, self.params['threshold1'].value(), 255, cv2.THRESH_BINARY) # img binara
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                self.processed_image = self.original_image.copy()
                cv2.drawContours(self.processed_image, contours, -1, (0, 255, 0), 2) # cul. verde a conturului, toate contururile
                # self.params['threshold1'].value() -> toti pixelii mai mari de acesta vor deveni albi
                
            elif function_name == "Binarizare":
                self.show_parameters(['threshold1'])
                gray = cv2.cvtColor(self.original_image, cv2.COLOR_RGB2GRAY)
                _, binary = cv2.threshold(gray, self.params['threshold1'].value(), 255, cv2.THRESH_BINARY)
                self.processed_image = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)
                
            elif function_name == "Egalizare histograma":
                self.show_parameters([])
                gray = cv2.cvtColor(self.original_image, cv2.COLOR_RGB2GRAY)
                equalized = cv2.equalizeHist(gray)
                self.processed_image = cv2.cvtColor(equalized, cv2.COLOR_GRAY2RGB)
                
            elif function_name == "Ajustare luminozitate":
                self.show_parameters(['beta'])
                beta = self.params['beta'].value()
                self.processed_image = cv2.convertScaleAbs(self.original_image, alpha=1, beta=beta)
                
            elif function_name == "Ajustare contrast":
                self.show_parameters(['alpha'])
                alpha = self.params['alpha'].value() / 100.0
                self.processed_image = cv2.convertScaleAbs(self.original_image, alpha=alpha, beta=0)
                
            elif function_name == "Inversare culori":
                # valorile pixelilor sunt inversate folosind complementul
                self.show_parameters([])
                self.processed_image = cv2.bitwise_not(self.original_image)
                
            elif function_name == "Rotire imagine":
                self.show_parameters(['angle'])
                angle = self.params['angle'].value()
                center = (self.original_image.shape[1] // 2, self.original_image.shape[0] // 2)
                matrix = cv2.getRotationMatrix2D(center, angle, 1.0) # center e punct de ref. pt rotatie
                self.processed_image = cv2.warpAffine(self.original_image, matrix, 
                                                    (self.original_image.shape[1], self.original_image.shape[0]))
                
            elif function_name == "Translatie imagine":
                self.show_parameters(['tx', 'ty'])
                tx = self.params['tx'].value()
                ty = self.params['ty'].value()
                matrix = np.float32([[1, 0, tx], [0, 1, ty]])
                self.processed_image = cv2.warpAffine(self.original_image, matrix, 
                                                    (self.original_image.shape[1], self.original_image.shape[0]))
                # modifica coordonatele fiecarui pixel din img
            elif function_name == "Dilatare + Eroziune":
                self.show_parameters(['kernel_size'])
                ksize = self.params['kernel_size'].value()
                kernel = np.ones((ksize, ksize), np.uint8) # matrice de pixeli ksxks cu val. de 1
                dilated = cv2.dilate(self.original_image, kernel, iterations=1)
                self.processed_image = cv2.erode(dilated, kernel, iterations=1)
                
            elif function_name == "Filtru median":
                self.show_parameters(['kernel_size'])
                ksize = self.params['kernel_size'].value()
                if ksize % 2 == 0:
                    ksize += 1
                self.processed_image = cv2.medianBlur(self.original_image, ksize)
                
            elif function_name == "Filtru Sobel":
                self.show_parameters([])
                gray = cv2.cvtColor(self.original_image, cv2.COLOR_RGB2GRAY)
                sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3) # marginile orizontale (pot fi si negative)
                sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
                sobel = np.sqrt(sobelx**2 + sobely**2)
                sobel = np.uint8(sobel)
                self.processed_image = cv2.cvtColor(sobel, cv2.COLOR_GRAY2RGB)
                
            elif function_name == "Transformare cartopolara":
                self.show_parameters([])
                center = (self.original_image.shape[1] // 2, self.original_image.shape[0] // 2)
                maxRadius = np.sqrt((self.original_image.shape[0] / 2.0) ** 2 + 
                                 (self.original_image.shape[1] / 2.0) ** 2) # se calc. distanta de la centru img la colturi
                self.processed_image = cv2.linearPolar(self.original_image, center, maxRadius, 
                                                     cv2.WARP_FILL_OUTLIERS)
                
            elif function_name == "Detectie colturi Harris":
                self.show_parameters([])
                gray = cv2.cvtColor(self.original_image, cv2.COLOR_RGB2GRAY)
                gray = np.float32(gray)
                dst = cv2.cornerHarris(gray, 2, 3, 0.04) 
                dst = cv2.dilate(dst, None) # intinde pixelii de ex din jurul ochilor pt a fi mai vizibile
                self.processed_image = self.original_image.copy()
                self.processed_image[dst > 0.01 * dst.max()] = [0, 0, 255]
            
    
            if len(self.processed_image.shape) == 2:  # Daca imaginea e grayscale
                self.processed_image = cv2.cvtColor(self.processed_image, cv2.COLOR_GRAY2RGB)
                        
            self.display_image(self.processed_image, self.processed_label)
                        
        except Exception as e:
            print(f"Eroare la procesarea imaginii: {str(e)}")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyle('Fusion') 
    window = AppProcesareImagini()
    window.show()
    sys.exit(app.exec_())