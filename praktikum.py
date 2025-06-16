import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    confusion_matrix, classification_report,
    mean_absolute_error, mean_squared_error, r2_score
)
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Konfigurasi halaman
st.set_page_config(
    page_title="Aplikasi Data Mining",
    page_icon="📊",
    layout="wide"
)

# Judul aplikasi
st.title("🔍 Prediksi Diabetes")
st.markdown("---")

# Sidebar untuk navigasi
st.sidebar.header("📋 Menu Navigasi")
menu = st.sidebar.selectbox(
    "Pilih Menu:",
    ["Upload Dataset", "Analisis Data", "Pemodelan", "Visualisasi Hasil"]
)

# Inisialisasi session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'predictions' not in st.session_state:
    st.session_state.predictions = None
if 'model_type' not in st.session_state:
    st.session_state.model_type = None

# Fungsi untuk preprocessing data
def preprocess_data(data):
    """Preprocessing data untuk menangani missing values dan encoding"""
    data_processed = data.copy()
    
    # Handle missing values
    for col in data_processed.columns:
        if data_processed[col].dtype == 'object':
            data_processed[col].fillna(data_processed[col].mode()[0] if not data_processed[col].mode().empty else 'Unknown', inplace=True)
        else:
            data_processed[col].fillna(data_processed[col].mean(), inplace=True)
    
    # Encode categorical variables
    le = LabelEncoder()
    categorical_cols = data_processed.select_dtypes(include=['object']).columns
    
    for col in categorical_cols:
        data_processed[col] = le.fit_transform(data_processed[col])
    
    return data_processed

# Fungsi untuk menentukan tipe problem
def determine_problem_type(target_column, data):
    """Menentukan apakah problem adalah klasifikasi atau regresi"""
    if data[target_column].dtype == 'object' or data[target_column].nunique() < 10:
        return 'classification'
    else:
        return 'regression'

# Menu 1: Upload Dataset
if menu == "Upload Dataset":
    st.header("📂 Upload Dataset")
    
    uploaded_file = st.file_uploader(
        "Pilih file CSV untuk dianalisis:",
        type=['csv'],
        help="Upload file CSV yang berisi dataset untuk analisis"
    )
    
    if uploaded_file is not None:
        try:
            # Baca dataset
            data = pd.read_csv(uploaded_file)
            st.session_state.data = data
            
            st.success("✅ Dataset berhasil diupload!")
            
            # Tampilkan informasi dataset
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📊 Informasi Dataset")
                st.write(f"*Jumlah Baris:* {data.shape[0]}")
                st.write(f"*Jumlah Kolom:* {data.shape[1]}")
                st.write(f"*Ukuran Dataset:* {data.memory_usage(deep=True).sum() / 1024:.2f} KB")
            
            with col2:
                st.subheader("🔍 Tipe Data")
                st.dataframe(data.dtypes.to_frame('Tipe Data'))
            
            # Tampilkan preview data
            st.subheader("👀 Preview Data (5 baris pertama)")
            st.dataframe(data.head())
            
            # Statistik deskriptif
            st.subheader("📈 Statistik Deskriptif")
            st.dataframe(data.describe())
            
        except Exception as e:
            st.error(f"❌ Error membaca file: {str(e)}")
    
    else:
        st.info("📥 Upload file CSV")

# Menu 2: Analisis Data
elif menu == "Analisis Data":
    st.header("🔍 Analisis Data Eksploratif")
    
    if st.session_state.data is not None:
        data = st.session_state.data
        
        # Pilihan analisis
        analysis_type = st.selectbox(
            "Pilih Jenis Analisis:",
            ["Distribusi Data", "Korelasi", "Missing Values", "Outliers"]
        )
        
        if analysis_type == "Distribusi Data":
            st.subheader("📊 Distribusi Data")
            
            # Pilih kolom untuk visualisasi
            numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            
            if numeric_cols:
                selected_col = st.selectbox("Pilih kolom untuk visualisasi:", numeric_cols)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Histogram
                    fig, ax = plt.subplots(figsize=(8, 6))
                    ax.hist(data[selected_col].dropna(), bins=30, alpha=0.7, color='skyblue')
                    ax.set_title(f'Distribusi {selected_col}')
                    ax.set_xlabel(selected_col)
                    ax.set_ylabel('Frekuensi')
                    st.pyplot(fig)
                
                with col2:
                    # Box plot
                    fig, ax = plt.subplots(figsize=(8, 6))
                    ax.boxplot(data[selected_col].dropna())
                    ax.set_title(f'Box Plot {selected_col}')
                    ax.set_ylabel(selected_col)
                    st.pyplot(fig)
        
        elif analysis_type == "Korelasi":
            st.subheader("🔗 Matriks Korelasi")
            
            numeric_data = data.select_dtypes(include=[np.number])
            
            if not numeric_data.empty:
                corr_matrix = numeric_data.corr()
                
                fig, ax = plt.subplots(figsize=(12, 8))
                sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax)
                ax.set_title('Matriks Korelasi')
                st.pyplot(fig)
            else:
                st.warning("⚠️ Tidak ada kolom numerik untuk analisis korelasi")
        
        elif analysis_type == "Missing Values":
            st.subheader("❓ Analisis Missing Values")
            
            missing_data = data.isnull().sum()
            missing_percent = (missing_data / len(data)) * 100
            
            missing_df = pd.DataFrame({
                'Kolom': missing_data.index,
                'Missing Count': missing_data.values,
                'Missing Percentage': missing_percent.values
            })
            
            missing_df = missing_df[missing_df['Missing Count'] > 0]
            
            if not missing_df.empty:
                st.dataframe(missing_df)
                
                # Visualisasi missing values
                fig, ax = plt.subplots(figsize=(10, 6))
                missing_df.plot(x='Kolom', y='Missing Percentage', kind='bar', ax=ax)
                ax.set_title('Persentase Missing Values per Kolom')
                ax.set_ylabel('Persentase (%)')
                plt.xticks(rotation=45)
                st.pyplot(fig)
            else:
                st.success("✅ Tidak ada missing values dalam dataset!")
        
        elif analysis_type == "Outliers":
            st.subheader("🎯 Deteksi Outliers")
            
            numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            
            if numeric_cols:
                selected_col = st.selectbox("Pilih kolom untuk deteksi outliers:", numeric_cols)
                
                # Menggunakan IQR method
                Q1 = data[selected_col].quantile(0.25)
                Q3 = data[selected_col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = data[(data[selected_col] < lower_bound) | (data[selected_col] > upper_bound)]
                
                st.write(f"*Jumlah Outliers:* {len(outliers)}")
                st.write(f"*Persentase Outliers:* {(len(outliers)/len(data))*100:.2f}%")
                
                if len(outliers) > 0:
                    st.dataframe(outliers[[selected_col]])
    
    else:
        st.warning("⚠️ Silakan upload dataset terlebih dahulu di menu 'Upload Dataset'")

# Menu 3: Pemodelan
elif menu == "Pemodelan":
    st.header("🤖 Pemodelan Machine Learning")
    
    if st.session_state.data is not None:
        data = st.session_state.data
        
        # Preprocessing data
        data_processed = preprocess_data(data)
        
        # Pilih algoritma
        algorithm = st.selectbox(
            "Pilih Algoritma:",
            ["Linear Regression", "Logistic Regression", "Naive Bayes", 
             "Support Vector Machine", "K-Nearest Neighbors", 
             "Decision Tree", "K-Means Clustering"]
        )
        
        if algorithm != "K-Means Clustering":
            # Supervised Learning
            st.subheader("🎯 Pengaturan Target Variable")
            
            target_column = st.selectbox(
                "Pilih kolom target (label):",
                data_processed.columns.tolist()
            )
            
            if target_column:
                # Menentukan tipe problem
                problem_type = determine_problem_type(target_column, data_processed)
                
                st.info(f"Tipe Problem: *{problem_type.upper()}*")
                
                # Pengaturan train-test split
                test_size = st.slider("Ukuran Data Testing (%)", 10, 50, 20) / 100
                
                # Tombol untuk melatih model
                if st.button("🚀 Latih Model"):
                    try:
                        # Persiapan data
                        X = data_processed.drop(columns=[target_column])
                        y = data_processed[target_column]
                        
                        # Split data
                        X_train, X_test, y_train, y_test = train_test_split(
                            X, y, test_size=test_size, random_state=42
                        )
                        
                        # Scaling untuk beberapa algoritma
                        if algorithm in ["Support Vector Machine", "K-Nearest Neighbors", "Logistic Regression"]:
                            scaler = StandardScaler()
                            X_train_scaled = scaler.fit_transform(X_train)
                            X_test_scaled = scaler.transform(X_test)
                        else:
                            X_train_scaled = X_train
                            X_test_scaled = X_test
                        
                        # Pilih model berdasarkan algoritma dan tipe problem
                        if algorithm == "Linear Regression":
                            model = LinearRegression()
                        elif algorithm == "Logistic Regression":
                            model = LogisticRegression(random_state=42)
                        elif algorithm == "Naive Bayes":
                            model = GaussianNB()
                        elif algorithm == "Support Vector Machine":
                            if problem_type == "classification":
                                model = SVC(random_state=42)
                            else:
                                model = SVR()
                        elif algorithm == "K-Nearest Neighbors":
                            k_value = st.sidebar.slider("Nilai K untuk KNN:", 1, 20, 5)
                            if problem_type == "classification":
                                model = KNeighborsClassifier(n_neighbors=k_value)
                            else:
                                model = KNeighborsRegressor(n_neighbors=k_value)
                        elif algorithm == "Decision Tree":
                            if problem_type == "classification":
                                model = DecisionTreeClassifier(random_state=42)
                            else:
                                model = DecisionTreeRegressor(random_state=42)
                        
                        # Latih model
                        model.fit(X_train_scaled, y_train)
                        
                        # Prediksi
                        y_pred = model.predict(X_test_scaled)
                        
                        # Simpan hasil ke session state
                        st.session_state.model = model
                        st.session_state.predictions = {
                            'y_test': y_test,
                            'y_pred': y_pred,
                            'X_test': X_test,
                            'X_train': X_train,
                            'y_train': y_train
                        }
                        st.session_state.model_type = problem_type
                        st.session_state.algorithm = algorithm
                        
                        # Tampilkan hasil
                        st.success("✅ Model berhasil dilatih!")
                        
                        # Evaluasi model
                        if problem_type == "classification":
                            accuracy = accuracy_score(y_test, y_pred)
                            precision = precision_score(y_test, y_pred, average='weighted')
                            recall = recall_score(y_test, y_pred, average='weighted')
                            f1 = f1_score(y_test, y_pred, average='weighted')
                            
                            col1, col2, col3, col4 = st.columns(4)
                            col1.metric("Accuracy", f"{accuracy:.3f}")
                            col2.metric("Precision", f"{precision:.3f}")
                            col3.metric("Recall", f"{recall:.3f}")
                            col4.metric("F1-Score", f"{f1:.3f}")
                            
                        else:  # regression
                            mae = mean_absolute_error(y_test, y_pred)
                            mse = mean_squared_error(y_test, y_pred)
                            r2 = r2_score(y_test, y_pred)
                            
                            col1, col2, col3 = st.columns(3)
                            col1.metric("MAE", f"{mae:.3f}")
                            col2.metric("MSE", f"{mse:.3f}")
                            col3.metric("R² Score", f"{r2:.3f}")
                        
                    except Exception as e:
                        st.error(f"❌ Error dalam pelatihan model: {str(e)}")
        
        else:
            # K-Means Clustering
            st.subheader("🎯 K-Means Clustering")
            
            # Pilih jumlah cluster
            n_clusters = st.slider("Jumlah Cluster (K):", 2, 10, 3)
            
            # Pilih fitur untuk clustering
            numeric_cols = data_processed.select_dtypes(include=[np.number]).columns.tolist()
            selected_features = st.multiselect(
                "Pilih fitur untuk clustering:",
                numeric_cols,
                default=numeric_cols[:2] if len(numeric_cols) >= 2 else numeric_cols
            )
            
            if len(selected_features) >= 2:
                if st.button("🚀 Jalankan K-Means"):
                    try:
                        # Persiapan data
                        X = data_processed[selected_features]
                        
                        # Scaling
                        scaler = StandardScaler()
                        X_scaled = scaler.fit_transform(X)
                        
                        # K-Means
                        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                        clusters = kmeans.fit_predict(X_scaled)
                        
                        # Simpan hasil
                        st.session_state.model = kmeans
                        st.session_state.predictions = {
                            'clusters': clusters,
                            'X': X,
                            'X_scaled': X_scaled,
                            'selected_features': selected_features
                        }
                        st.session_state.model_type = "clustering"
                        st.session_state.algorithm = algorithm
                        
                        st.success("✅ K-Means clustering berhasil!")
                        
                        # Tampilkan hasil
                        st.write(f"*Inertia:* {kmeans.inertia_:.3f}")
                        st.write(f"*Jumlah Cluster:* {n_clusters}")
                        
                        # Distribusi cluster
                        unique, counts = np.unique(clusters, return_counts=True)
                        cluster_dist = pd.DataFrame({
                            'Cluster': unique,
                            'Jumlah Data': counts
                        })
                        st.dataframe(cluster_dist)
                        
                    except Exception as e:
                        st.error(f"❌ Error dalam clustering: {str(e)}")
            else:
                st.warning("⚠️ Pilih minimal 2 fitur untuk clustering")
    
    else:
        st.warning("⚠️ Silakan upload dataset terlebih dahulu di menu 'Upload Dataset'")

# Menu 4: Visualisasi Hasil
elif menu == "Visualisasi Hasil":
    st.header("📊 Visualisasi Hasil Model")
    
    if st.session_state.model is not None and st.session_state.predictions is not None:
        model_type = st.session_state.model_type
        algorithm = st.session_state.algorithm
        
        if model_type == "classification":
            # Visualisasi untuk klasifikasi
            y_test = st.session_state.predictions['y_test']
            y_pred = st.session_state.predictions['y_pred']
            
            # Confusion Matrix
            st.subheader("🎯 Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_title('Confusion Matrix')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            st.pyplot(fig)
            
            # Classification Report
            st.subheader("📋 Classification Report")
            report = classification_report(y_test, y_pred, output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df)
            
        elif model_type == "regression":
            # Visualisasi untuk regresi
            y_test = st.session_state.predictions['y_test']
            y_pred = st.session_state.predictions['y_pred']
            
            # Scatter plot: Actual vs Predicted
            st.subheader("📈 Actual vs Predicted Values")
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.scatter(y_test, y_pred, alpha=0.6)
            ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
            ax.set_xlabel('Actual Values')
            ax.set_ylabel('Predicted Values')
            ax.set_title('Actual vs Predicted Values')
            st.pyplot(fig)
            
            # Residual plot
            st.subheader("📊 Residual Plot")
            residuals = y_test - y_pred
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.scatter(y_pred, residuals, alpha=0.6)
            ax.axhline(y=0, color='r', linestyle='--')
            ax.set_xlabel('Predicted Values')
            ax.set_ylabel('Residuals')
            ax.set_title('Residual Plot')
            st.pyplot(fig)
            
        elif model_type == "clustering":
            # Visualisasi untuk clustering
            clusters = st.session_state.predictions['clusters']
            X = st.session_state.predictions['X']
            selected_features = st.session_state.predictions['selected_features']
            
            st.subheader("🎨 Visualisasi Cluster")
            
            if len(selected_features) >= 2:
                # Scatter plot dengan warna berbeda untuk setiap cluster
                fig, ax = plt.subplots(figsize=(10, 8))
                scatter = ax.scatter(X.iloc[:, 0], X.iloc[:, 1], c=clusters, cmap='viridis', alpha=0.6)
                ax.set_xlabel(selected_features[0])
                ax.set_ylabel(selected_features[1])
                ax.set_title('K-Means Clustering Results')
                plt.colorbar(scatter)
                st.pyplot(fig)
                
                # Tampilkan center cluster
                centers = st.session_state.model.cluster_centers_
                st.subheader("🎯 Cluster Centers")
                centers_df = pd.DataFrame(centers, columns=selected_features)
                st.dataframe(centers_df)
        
        # Visualisasi Decision Tree (jika applicable)
        if algorithm == "Decision Tree" and model_type != "clustering":
            st.subheader("🌳 Visualisasi Decision Tree")
            
            try:
                fig, ax = plt.subplots(figsize=(20, 10))
                plot_tree(st.session_state.model, 
                         feature_names=st.session_state.predictions['X_test'].columns,
                         filled=True, 
                         rounded=True,
                         ax=ax)
                ax.set_title('Decision Tree Visualization')
                st.pyplot(fig)
            except Exception as e:
                st.warning(f"⚠️ Tidak dapat menampilkan decision tree: {str(e)}")
        
        # Feature Importance (jika applicable)
        if hasattr(st.session_state.model, 'feature_importances_'):
            st.subheader("📊 Feature Importance")
            
            feature_names = st.session_state.predictions['X_test'].columns
            importances = st.session_state.model.feature_importances_
            
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importances
            }).sort_values('Importance', ascending=False)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.barh(importance_df['Feature'], importance_df['Importance'])
            ax.set_xlabel('Importance')
            ax.set_title('Feature Importance')
            st.pyplot(fig)
    
    else:
        st.warning("⚠️ Silakan latih model terlebih dahulu di menu 'Pemodelan'")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>🔬 Aplikasi Data Mining - Dibuat dengan Streamlit dan Scikit-learn</p>
        <p>📊 Mendukung berbagai algoritma machine learning untuk analisis data</p>
    </div>
    """, 
    unsafe_allow_html=True
)

# Sidebar informasi
st.sidebar.markdown("---")
st.sidebar.markdown("### 📖 Panduan Penggunaan")
st.sidebar.markdown("""
1. *Upload Dataset*: Upload file CSV Anda
2. *Analisis Data*: Eksplorasi data untuk memahami pola
3. *Pemodelan*: Pilih algoritma dan latih model
4. *Visualisasi*: Lihat hasil dan performa model
""")

st.sidebar.markdown("### 🔧 Algoritma yang Didukung")
st.sidebar.markdown("""
- *Supervised Learning*:
  - Linear/Logistic Regression
  - Naive Bayes
  - Support Vector Machine
  - K-Nearest Neighbors
  - Decision Tree

- *Unsupervised Learning*:
  - K-Means Clustering
""")

st.sidebar.markdown("### 📊 Fitur Visualisasi")
st.sidebar.markdown("""
- Confusion Matrix
- ROC Curve
- Feature Importance
- Cluster Visualization
- Residual Plots
- Decision Tree Diagram
""")