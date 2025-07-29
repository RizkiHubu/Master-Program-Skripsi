"""
Alat Pemilihan dan Evaluasi Model Prediksi Kelulusan Mahasiswa
-------------------------------------------------------------------------
Script ini memungkinkan user untuk memilih di antara berbagai model yang tersedia
dan melakukan evaluasi serta prediksi menggunakan model yang dipilih.

Model yang tersedia:
1. Gradient Boosting tanpa balancing
2. Gradient Boosting dengan balancing
3. XGBoost tanpa balancing
4. XGBoost dengan balancing
"""

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.impute import SimpleImputer
import os
import sys
from tabulate import tabulate
import warnings
warnings.filterwarnings("ignore")


# Definisi model yang tersedia
AVAILABLE_MODELS = {
    '1': {
        'name': 'Gradient Boosting tanpa balancing',
        'model_path': 'model_kelulusan_gradient_boosting_no_balancing.joblib',
        'le_path': 'label_encoder_gradient_boosting_no_balancing.joblib'
    },
    '2': {
        'name': 'Gradient Boosting dengan balancing',
        'model_path': 'model_kelulusan_gradient_boosting_balancing.joblib',
        'le_path': 'label_encoder_gradient_boosting_balancing.joblib'
    },
    '3': {
        'name': 'XGBoost tanpa balancing',
        'model_path': 'model_kelulusan_xgboost_no_balancing.joblib',
        'le_path': 'label_encoder_xgboost_no_balancing.joblib'
    },
    '4': {
        'name': 'XGBoost dengan balancing',
        'model_path': 'model_kelulusan_xgboost_balancing.joblib',
        'le_path': 'label_encoder_xgboost_balancing.joblib'
    },
    '5': {
        'name': 'KNN tanpa balancing',
        'model_path': 'model_kelulusan_knn.joblib',
        'le_path': 'label_encoder_knn.joblib'
    },
    '6': {
        'name': 'KNN dengan balancing',
        'model_path': 'model_kelulusan_knn_balancing.joblib',
        'le_path': 'label_encoder_knn_balancing.joblib'
    }
}


def clear_screen():
    """Membersihkan layar terminal"""
    os.system('cls' if os.name == 'nt' else 'clear')


def print_title(title):
    """Menampilkan judul dengan format yang menarik"""
    width = len(title) + 10
    print("=" * width)
    print(f"    {title}")
    print("=" * width)


def load_model(model_info):
    """
    Memuat model dan label encoder dari file
    
    Args:
        model_info: Dictionary berisi informasi model
        
    Returns:
        tuple: (model, label_encoder) atau None jika gagal
    """
    try:
        model = joblib.load(model_info['model_path'])
        label_encoder = joblib.load(model_info['le_path'])
        print(f"\nModel {model_info['name']} berhasil dimuat")
        return model, label_encoder
    except Exception as e:
        print(f"\nError saat memuat model: {e}")
        print(f"Pastikan file {model_info['model_path']} dan {model_info['le_path']} tersedia")
        return None, None


def select_model():
    """
    Menampilkan menu pemilihan model
    
    Returns:
        tuple: (model, label_encoder, model_info) atau (None, None, None) jika dibatalkan
    """
    while True:
        clear_screen()
        print_title("PEMILIHAN MODEL PREDIKSI KELULUSAN MAHASISWA")
        print("\nModel yang tersedia:")
        
        for key, model_info in AVAILABLE_MODELS.items():
            print(f"{key}. {model_info['name']}")
        
        print("0. Kembali ke menu utama")
        
        choice = input(f"\nPilih model (0-{len(AVAILABLE_MODELS)}): ")
        
        if choice == '0':
            return None, None, None
        
        if choice in AVAILABLE_MODELS:
            model, label_encoder = load_model(AVAILABLE_MODELS[choice])
            if model is not None:
                return model, label_encoder, AVAILABLE_MODELS[choice]
        else:
            print("Pilihan tidak valid. Tekan Enter untuk melanjutkan...")
            input()


def create_imputer():
    """
    Membuat imputer untuk menangani nilai kosong
    
    Returns:
        SimpleImputer: Imputer dengan strategi mean
    """
    return SimpleImputer(strategy='mean')


def evaluate_model_with_test_data(model, label_encoder, features_list):
    """
    Mengevaluasi model dengan data test dan menampilkan hasil evaluasi
    
    Args:
        model: Model yang akan dievaluasi
        label_encoder: Label encoder untuk target
        features_list: List nama fitur
        
    Returns:
        bool: True jika berhasil, False jika gagal
    """
    try:
        # Meminta path file test data
        test_data_path = input("\nMasukkan path file CSV data test: ")
        
        if not os.path.exists(test_data_path):
            print(f"File {test_data_path} tidak ditemukan!")
            return False
        
        # Load data test
        test_data = pd.read_csv(test_data_path)
        
        # Cek apakah semua fitur yang diperlukan ada di data test
        missing_features = [f for f in features_list if f not in test_data.columns]
        if missing_features:
            print(f"Fitur berikut tidak ditemukan di data test: {missing_features}")
            return False
        
        # Cek apakah kolom target ada di data test
        target_column = input("Masukkan nama kolom target di data test: ")
        if target_column not in test_data.columns:
            print(f"Kolom target '{target_column}' tidak ditemukan di data test!")
            return False
        
        # Memisahkan fitur dan target
        X_test = test_data[features_list]
        y_test = test_data[target_column]
        
        # Imputasi nilai kosong
        imputer = create_imputer()
        X_test_imputed = imputer.fit_transform(X_test)
        
        # Encode target dengan label encoder yang sama
        y_test_encoded = label_encoder.transform(y_test)
        
        # Prediksi
        y_pred = model.predict(X_test_imputed)
        
        # Evaluasi
        conf_matrix = confusion_matrix(y_test_encoded, y_pred)
        class_report = classification_report(y_test_encoded, y_pred, 
                                            target_names=label_encoder.classes_,
                                            output_dict=True)
        class_report_str = classification_report(y_test_encoded, y_pred, 
                                                target_names=label_encoder.classes_)
        
        # Hitung akurasi
        accuracy = accuracy_score(y_test_encoded, y_pred)
        
        # Tampilkan hasil evaluasi
        clear_screen()
        print_title("HASIL EVALUASI MODEL")
        print(f"\nModel: {model.__class__.__name__}")
        print(f"Akurasi: {accuracy:.4f}")
        
        # Tampilkan confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                    xticklabels=label_encoder.classes_,
                    yticklabels=label_encoder.classes_)
        plt.xlabel('Prediksi')
        plt.ylabel('Aktual')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.show()
        
        # Tampilkan classification report
        print("\nClassification Report:")
        print(class_report_str)
        
        # Visualisasi classification report
        plt.figure(figsize=(12, 8))
        
        # Extract precision, recall, dan f1-score per kelas
        metrics = ['precision', 'recall', 'f1-score']
        classes = label_encoder.classes_
        
        # Buat dataframe untuk plot
        report_df = pd.DataFrame()
        for cls in classes:
            for metric in metrics:
                report_df.at[cls, metric] = class_report[cls][metric]
        
        # Plot
        ax = report_df.plot(kind='bar', figsize=(10, 6))
        plt.title('Performance Metrics per Class')
        plt.ylabel('Score')
        plt.xlabel('Class')
        plt.xticks(rotation=45)
        plt.legend(loc='lower right')
        plt.tight_layout()
        plt.show()
        
        # Tampilkan feature importance jika ada
        if hasattr(model, 'feature_importances_'):
            visualize_feature_importance(model, features_list)
        
        return True
    
    except Exception as e:
        print(f"Error saat mengevaluasi model: {e}")
        return False


def visualize_feature_importance(model, features):
    """
    Memvisualisasikan feature importance dari model
    
    Args:
        model: Model terlatih
        features: Nama-nama fitur
    """
    if not hasattr(model, 'feature_importances_'):
        print("Model tidak memiliki atribut feature_importances_")
        return
    
    feature_importance = model.feature_importances_
    fi_df = pd.DataFrame({'Fitur': features, 'Importance': feature_importance})
    fi_df = fi_df.sort_values(by='Importance', ascending=False)
    
    # Print feature importance
    print("\nFeature Importance:")
    for i, row in fi_df.iterrows():
        print(f"{row['Fitur']}: {row['Importance']:.4f}")
    
    # Visualisasi feature importance
    plt.figure(figsize=(12, 8))
    ax = sns.barplot(x='Importance', y='Fitur', data=fi_df)
    
    # Tambahkan label nilai di setiap bar
    for i, v in enumerate(fi_df['Importance']):
        ax.text(v + 0.01, i, f"{v:.4f}", va='center')
    
    plt.title('Feature Importance')
    plt.tight_layout()
    plt.show()


def predict_from_user_input(model, label_encoder, features_list):
    """
    Memprediksi kelulusan berdasarkan input manual dari user
    
    Args:
        model: Model terlatih
        label_encoder: Label encoder untuk target
        features_list: List nama fitur
    """
    clear_screen()
    print_title("PREDIKSI MANUAL")
    
    sample_data = []
    
    # Meminta input untuk setiap fitur
    print("\nMasukkan SKS TOTAL:")
    sks_total = float(input("> "))
    sample_data.append(sks_total)
    
    # IP Semester 1-7
    for i in range(1, 8):
        print(f"Masukkan IP SEMESTER {i} (0-4, kosongkan jika tidak ada):")
        try:
            ip = float(input("> ") or "nan")
            if 0 <= ip <= 4:
                sample_data.append(ip)
            else:
                print("IP harus antara 0-4, menggunakan NaN")
                sample_data.append(np.nan)
        except ValueError:
            sample_data.append(np.nan)
    
    # SKS Tempuh Semester 1-7
    for i in range(1, 8):
        print(f"Masukkan SKS TEMPUH SEMESTER {i} (kosongkan jika tidak ada):")
        try:
            sks = float(input("> ") or "nan")
            sample_data.append(sks)
        except ValueError:
            sample_data.append(np.nan)
    
    # Jumlah matkul yang diulang
    print("Masukkan JUMLAH MATKUL YANG DI ULANG:")
    matkul_ulang = int(input("> "))
    sample_data.append(matkul_ulang)
    
    # Jumlah cuti
    print("Masukkan JUMLAH CUTI (SEMESTER):")
    cuti = int(input("> "))
    sample_data.append(cuti)
    
    # Buat DataFrame dari sample_data
    sample_df = pd.DataFrame([sample_data], columns=features_list)
    
    # Imputasi
    imputer = create_imputer()
    sample_data_imputed = imputer.fit_transform(sample_df)
    
    # Prediksi
    prediksi = model.predict(sample_data_imputed)
    prediksi_label = label_encoder.inverse_transform(prediksi)
    
    # Prediksi probabilitas
    prediksi_proba = model.predict_proba(sample_data_imputed)[0]
    class_names = label_encoder.classes_
    
    # Tampilkan hasil prediksi
    clear_screen()
    print_title("HASIL PREDIKSI")
    print(f"\nStatus kelulusan: {prediksi_label[0]}")
    print("\nProbabilitas per kelas:")
    
    # Buat tabel probabilitas
    proba_table = []
    for i, kelas in enumerate(class_names):
        proba_table.append([kelas, f"{prediksi_proba[i]:.4f}", f"{prediksi_proba[i]*100:.2f}%"])
    
    print(tabulate.tabulate(proba_table, headers=["Kelas", "Probabilitas", "Persentase"], tablefmt="pretty"))
    
    # Visualisasi probabilitas
    plt.figure(figsize=(10, 6))
    bars = plt.bar(class_names, prediksi_proba)
    
    # Tambahkan label persentase di atas bar
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.2%}', ha='center', va='bottom')
    
    plt.title('Probabilitas Status Kelulusan')
    plt.ylabel('Probabilitas')
    plt.xlabel('Status')
    plt.ylim(0, 1.1)
    plt.tight_layout()
    plt.show()
    
    input("\nTekan Enter untuk kembali ke menu...")


def predict_from_csv(model, label_encoder, features_list):
    """
    Memprediksi kelulusan untuk batch data dari file CSV
    
    Args:
        model: Model terlatih
        label_encoder: Label encoder untuk target
        features_list: List nama fitur
    """
    clear_screen()
    print_title("PREDIKSI BATCH DARI CSV")
    
    # Meminta path file CSV
    csv_path = input("\nMasukkan path file CSV: ")
    
    if not os.path.exists(csv_path):
        print(f"File {csv_path} tidak ditemukan!")
        input("\nTekan Enter untuk kembali ke menu...")
        return
    
    try:
        # Load data
        data = pd.read_csv(csv_path)
        
        # Cek apakah semua fitur yang diperlukan ada di data
        missing_features = [f for f in features_list if f not in data.columns]
        if missing_features:
            print(f"Fitur berikut tidak ditemukan di CSV: {missing_features}")
            input("\nTekan Enter untuk kembali ke menu...")
            return
        
        # Ambil fitur
        X = data[features_list]
        
        # Imputasi
        imputer = create_imputer()
        X_imputed = imputer.fit_transform(X)
        
        # Prediksi
        prediksi = model.predict(X_imputed)
        prediksi_label = label_encoder.inverse_transform(prediksi)
        
        # Prediksi probabilitas
        prediksi_proba = model.predict_proba(X_imputed)
        
        # Buat dataframe hasil
        hasil = data.copy()
        hasil['prediksi_status'] = prediksi_label
        
        # Tambahkan probabilitas untuk setiap kelas
        for i, kelas in enumerate(label_encoder.classes_):
            hasil[f'prob_{kelas}'] = prediksi_proba[:, i]
        
        # Tampilkan distribusi hasil prediksi
        clear_screen()
        print_title("HASIL PREDIKSI BATCH")
        print(f"\nTotal data: {len(hasil)}")
        
        # Hitung jumlah per kelas hasil prediksi
        pred_counts = hasil['prediksi_status'].value_counts()
        print("\nDistribusi hasil prediksi:")
        for status, count in pred_counts.items():
            print(f"{status}: {count} ({count/len(hasil)*100:.2f}%)")
        
        # Visualisasi distribusi hasil prediksi
        plt.figure(figsize=(10, 6))
        ax = sns.countplot(x='prediksi_status', data=hasil)
        
        # Tambahkan label jumlah di setiap bar
        for p in ax.patches:
            ax.annotate(f'{int(p.get_height())}', 
                        (p.get_x() + p.get_width() / 2., p.get_height()), 
                        ha = 'center', va = 'bottom')
        
        plt.title('Distribusi Hasil Prediksi')
        plt.xlabel('Status Kelulusan')
        plt.ylabel('Jumlah')
        plt.tight_layout()
        plt.show()
        
        # Tanyakan apakah ingin menyimpan hasil
        save_option = input("\nSimpan hasil prediksi ke CSV? (y/n): ")
        if save_option.lower() == 'y':
            output_path = input("Masukkan path file output: ")
            hasil.to_csv(output_path, index=False)
            print(f"Hasil prediksi disimpan ke {output_path}")
        
        input("\nTekan Enter untuk kembali ke menu...")
    
    except Exception as e:
        print(f"Error saat memprediksi dari CSV: {e}")
        input("\nTekan Enter untuk kembali ke menu...")


def compare_models_performance():
    """
    Membandingkan performa semua model yang tersedia
    """
    clear_screen()
    print_title("PERBANDINGAN PERFORMA MODEL")
    
    # Meminta path file test data untuk evaluasi
    test_data_path = input("\nMasukkan path file CSV data test: ")
    
    if not os.path.exists(test_data_path):
        print(f"File {test_data_path} tidak ditemukan!")
        input("\nTekan Enter untuk kembali ke menu...")
        return
    
    # Definisi fitur
    features = [
        'SKS TOTAL',
        'IP SEMESTER 1', 'IP SEMESTER 2', 'IP SEMESTER 3', 'IP SEMESTER 4',
        'IP SEMESTER 5',
        'SKS TEMPUH SEMESTER 1', 'SKS TEMPUH SEMESTER 2', 'SKS TEMPUH SEMESTER 3',
        'SKS TEMPUH SEMESTER 4', 'SKS TEMPUH SEMESTER 5',
        'JUMLAH MATKUL YANG DI ULANG', 'JUMLAH CUTI (SEMESTER)'
    ]
    
    # Meminta nama kolom target
    target_column = input("Masukkan nama kolom target di data test: ")
    
    try:
        # Load data
        test_data = pd.read_csv(test_data_path)
        
        if target_column not in test_data.columns:
            print(f"Kolom target '{target_column}' tidak ditemukan!")
            input("\nTekan Enter untuk kembali ke menu...")
            return
        
        # Cek apakah semua fitur yang diperlukan ada di data test
        missing_features = [f for f in features if f not in test_data.columns]
        if missing_features:
            print(f"Fitur berikut tidak ditemukan di data test: {missing_features}")
            input("\nTekan Enter untuk kembali ke menu...")
            return
        
        # Memisahkan fitur dan target
        X_test = test_data[features]
        y_test = test_data[target_column]
        
        # Imputasi
        imputer = create_imputer()
        X_test_imputed = imputer.fit_transform(X_test)
        
        # Hasil per model
        results = []
        
        # Evaluasi semua model
        for model_id, model_info in AVAILABLE_MODELS.items():
            try:
                model, label_encoder = load_model(model_info)
                
                if model is None or label_encoder is None:
                    print(f"Model {model_info['name']} tidak dapat dimuat. Melewati...")
                    continue
                
                # Encode target dengan label encoder model
                y_test_encoded = label_encoder.transform(y_test)
                
                # Prediksi
                y_pred = model.predict(X_test_imputed)
                
                # Hitung akurasi
                accuracy = accuracy_score(y_test_encoded, y_pred)
                
                # Dapatkan report klasifikasi
                class_report = classification_report(y_test_encoded, y_pred, 
                                                    target_names=label_encoder.classes_,
                                                    output_dict=True)
                
                # Simpan hasil
                results.append({
                    'model_id': model_id,
                    'name': model_info['name'],
                    'accuracy': accuracy,
                    'class_report': class_report,
                    'y_pred': y_pred,
                    'y_test': y_test_encoded,
                    'label_encoder': label_encoder
                })
                
                print(f"Model {model_info['name']} berhasil dievaluasi")
                
            except Exception as e:
                print(f"Error saat mengevaluasi model {model_info['name']}: {e}")
        
        if not results:
            print("Tidak ada model yang berhasil dievaluasi!")
            input("\nTekan Enter untuk kembali ke menu...")
            return
        
        # Sort berdasarkan akurasi
        results.sort(key=lambda x: x['accuracy'], reverse=True)
        
        # Tampilkan hasil
        clear_screen()
        print_title("HASIL PERBANDINGAN MODEL")
        
        # Tabel perbandingan akurasi
        table_data = []
        for result in results:
            table_data.append([
                result['name'],
                f"{result['accuracy']:.4f}",
                f"{result['class_report']['weighted avg']['precision']:.4f}",
                f"{result['class_report']['weighted avg']['recall']:.4f}",
                f"{result['class_report']['weighted avg']['f1-score']:.4f}"
            ])
        
        print("\nPerforma Model:")
        print(tabulate.tabulate(table_data, 
                      headers=["Model", "Accuracy", "Precision", "Recall", "F1-Score"], 
                      tablefmt="pretty"))
        
        # Plot perbandingan akurasi
        plt.figure(figsize=(12, 6))
        model_names = [result['name'] for result in results]
        accuracies = [result['accuracy'] for result in results]
        
        bars = plt.bar(model_names, accuracies)
        
        # Tambahkan label akurasi di atas bar
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.4f}', ha='center', va='bottom')
        
        plt.title('Perbandingan Akurasi Model')
        plt.ylabel('Akurasi')
        plt.xticks(rotation=45)
        plt.ylim(0, 1.1)
        plt.tight_layout()
        plt.show()
        
        # Tanyakan apakah ingin melihat detail per model
        detail_option = input("\nApakah Anda ingin melihat detail evaluasi model tertentu? (y/n): ")
        if detail_option.lower() == 'y':
            while True:
                print("\nModel yang tersedia:")
                for i, result in enumerate(results):
                    print(f"{i+1}. {result['name']}")
                print("0. Kembali ke menu")
                
                choice = input("\nPilih model (0-4): ")
                if choice == '0':
                    break
                
                try:
                    idx = int(choice) - 1
                    if 0 <= idx < len(results):
                        result = results[idx]
                        
                        # Tampilkan confusion matrix
                        plt.figure(figsize=(10, 8))
                        conf_matrix = confusion_matrix(result['y_test'], result['y_pred'])
                        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                                   xticklabels=result['label_encoder'].classes_,
                                   yticklabels=result['label_encoder'].classes_)
                        plt.xlabel('Prediksi')
                        plt.ylabel('Aktual')
                        plt.title(f'Confusion Matrix - {result["name"]}')
                        plt.tight_layout()
                        plt.show()
                        
                        # Tampilkan classification report
                        print("\nClassification Report:")
                        class_report_str = classification_report(result['y_test'], result['y_pred'],
                                                               target_names=result['label_encoder'].classes_)
                        print(class_report_str)
                    else:
                        print("Pilihan tidak valid!")
                except ValueError:
                    print("Pilihan tidak valid!")
        
        input("\nTekan Enter untuk kembali ke menu...")
    
    except Exception as e:
        print(f"Error saat membandingkan model: {e}")
        input("\nTekan Enter untuk kembali ke menu...")


def main():
    """
    Fungsi utama program
    """
    # Definisi fitur
    features = [
        'SKS TOTAL',
        'IP SEMESTER 1', 'IP SEMESTER 2', 'IP SEMESTER 3', 'IP SEMESTER 4',
        'IP SEMESTER 5',
        'SKS TEMPUH SEMESTER 1', 'SKS TEMPUH SEMESTER 2', 'SKS TEMPUH SEMESTER 3',
        'SKS TEMPUH SEMESTER 4', 'SKS TEMPUH SEMESTER 5', 
        'JUMLAH MATKUL YANG DI ULANG', 'JUMLAH CUTI (SEMESTER)'
    ]
    
    # Variabel untuk menyimpan model aktif
    active_model = None
    active_le = None
    active_model_info = None
    
    # Loop menu utama
    while True:
        clear_screen()
        print_title("SISTEM PREDIKSI KELULUSAN MAHASISWA")
        
        if active_model is not None:
            print(f"\nModel aktif: {active_model_info['name']}")
        else:
            print("\nBelum ada model yang dipilih")
        
        print("\nMenu:")
        print("1. Pilih model")
        print("2. Prediksi dengan input manual")
        print("3. Prediksi batch dari file CSV")
        print("4. Evaluasi model dengan data test")
        print("5. Bandingkan performa model")
        print("0. Keluar")
        
        choice = input("\nPilihan Anda (0-5): ")
        
        if choice == '0':
            break
        
        elif choice == '1':
            active_model, active_le, active_model_info = select_model()
        
        elif choice == '2':
            if active_model is None:
                print("\nBelum ada model yang dipilih! Silakan pilih model terlebih dahulu.")
                input("\nTekan Enter untuk melanjutkan...")
                continue
            predict_from_user_input(active_model, active_le, features)
        
        elif choice == '3':
            if active_model is None:
                print("\nBelum ada model yang dipilih! Silakan pilih model terlebih dahulu.")
                input("\nTekan Enter untuk melanjutkan...")
                continue
            predict_from_csv(active_model, active_le, features)
        
        elif choice == '4':
            if active_model is None:
                print("\nBelum ada model yang dipilih! Silakan pilih model terlebih dahulu.")
                input("\nTekan Enter untuk melanjutkan...")
                continue
            evaluate_model_with_test_data(active_model, active_le, features)
            input("\nTekan Enter untuk melanjutkan...")
        
        elif choice == '5':
            compare_models_performance()
        
        else:
            print("\nPilihan tidak valid!")
            input("\nTekan Enter untuk melanjutkan...")


if __name__ == "__main__":
    try:
        # Cek apakah tabulate tersedia
        import tabulate
    except ImportError:
        print("Modul 'tabulate' tidak ditemukan.")
        print("Silakan install terlebih dahulu dengan perintah:")
        print("pip install tabulate")
        sys.exit(1)
        
    main()