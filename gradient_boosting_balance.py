"""
Sistem Prediksi Kelulusan Mahasiswa menggunakan Gradient Boosting
-------------------------------------------------------------------------
Model ini memprediksi status kelulusan mahasiswa berdasarkan data akademik
seperti IP semester, SKS yang ditempuh, jumlah matkul yang diulang, dan
jumlah cuti.

Kode dibuat secara modular untuk memudahkan pemeliharaan dan pengembangan.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from imblearn.combine import SMOTETomek
import matplotlib.pyplot as plt
import seaborn as sns
import joblib


def load_data(filepath):
    """
    Memuat data dari file CSV
    
    Args:
        filepath: Path ke file CSV yang berisi data
        
    Returns:
        DataFrame: Data yang dimuat dari file CSV
    """
    return pd.read_csv(filepath)


def prepare_features_and_target(df, feature_list, target_column):
    """
    Mempersiapkan fitur dan target dari DataFrame
    
    Args:
        df: DataFrame sumber data
        feature_list: List nama kolom yang akan digunakan sebagai fitur
        target_column: Nama kolom yang akan digunakan sebagai target
        
    Returns:
        tuple: (X, y) dimana X adalah fitur dan y adalah target
    """
    X = df[feature_list]
    y = df[target_column]
    return X, y


def preprocess_data(X, y, test_size=0.2, random_state=42):
    """
    Melakukan preprocessing data termasuk imputasi nilai kosong,
    encoding label, dan split data
    
    Args:
        X: Fitur
        y: Target
        test_size: Proporsi data yang digunakan untuk test set
        random_state: Seed untuk reproducibility
        
    Returns:
        tuple: (X_train_raw, X_test, y_train_raw, y_test, imputer, le)
    """
    # Imputasi nilai kosong
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)
    
    # Encode label
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Split data
    X_train_raw, X_test, y_train_raw, y_test = train_test_split(
        X_imputed, y_encoded, 
        test_size=test_size, 
        random_state=random_state, 
        stratify=y_encoded
    )
    
    return X_train_raw, X_test, y_train_raw, y_test, imputer, le


def balance_data(X_train_raw, y_train_raw, random_state=42):
    """
    Melakukan balancing pada data training menggunakan SMOTETomek
    
    Args:
        X_train_raw: Fitur pada data training
        y_train_raw: Target pada data training
        random_state: Seed untuk reproducibility
        
    Returns:
        tuple: (X_train_balanced, y_train_balanced)
    """
    smt = SMOTETomek(random_state=random_state)
    X_train, y_train = smt.fit_resample(X_train_raw, y_train_raw)
    return X_train, y_train


def evaluate_model_with_cv(model, X, y, cv_folds=5, random_state=42):
    """
    Mengevaluasi model menggunakan k-fold cross validation
    
    Args:
        model: Model yang akan dievaluasi
        X: Fitur
        y: Target
        cv_folds: Jumlah fold dalam cross validation
        random_state: Seed untuk reproducibility
        
    Returns:
        tuple: (scores, mean_score) scores per fold dan rata-rata
    """
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    mean_score = np.mean(scores)
    
    print(f"Hasil {cv_folds}-Fold Cross Validation (pada data balanced):")
    print("Akurasi per fold:", scores)
    print("Rata-rata akurasi:", mean_score)
    
    return scores, mean_score


def train_model(model, X_train, y_train):
    """
    Melatih model dengan data training
    
    Args:
        model: Model yang akan dilatih
        X_train: Fitur training
        y_train: Target training
        
    Returns:
        Model yang telah dilatih
    """
    model.fit(X_train, y_train)
    return model


def evaluate_model_on_test(model, X_test, y_test, label_encoder):
    """
    Mengevaluasi model pada data test
    
    Args:
        model: Model yang akan dievaluasi
        X_test: Fitur test
        y_test: Target test
        label_encoder: LabelEncoder yang digunakan untuk target
        
    Returns:
        tuple: (y_pred, confusion_mat, classification_rep)
    """
    y_pred = model.predict(X_test)
    
    confusion_mat = confusion_matrix(y_test, y_pred)
    classification_rep = classification_report(y_test, y_pred, 
                                              target_names=label_encoder.classes_)
    
    print("\nConfusion Matrix pada data test (tidak dibalancing):")
    print(confusion_mat)
    print("\nClassification Report:")
    print(classification_rep)
    
    # Visualisasi confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_mat, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.show()
    
    return y_pred, confusion_mat, classification_rep

def save_model(model, model_path, label_encoder=None, le_path=None):
    """
    Menyimpan model dan label encoder ke file
    
    Args:
        model: Model yang akan disimpan
        model_path: Path untuk menyimpan model
        label_encoder: Label encoder yang akan disimpan (opsional)
        le_path: Path untuk menyimpan label encoder (opsional)
    """
    joblib.dump(model, model_path)
    
    if label_encoder is not None and le_path is not None:
        joblib.dump(label_encoder, le_path)
        print(f"Model disimpan di {model_path} dan label encoder disimpan di {le_path}")
    else:
        print(f"Model disimpan di {model_path}")


def predict_single_sample(model, imputer, label_encoder, sample_data, feature_names):
    """
    Memprediksi status kelulusan untuk satu sampel data
    
    Args:
        model: Model terlatih
        imputer: Imputer yang digunakan pada data training
        label_encoder: Label encoder untuk target
        sample_data: Data sampel untuk diprediksi
        feature_names: Nama-nama fitur
        
    Returns:
        str: Hasil prediksi status kelulusan
    """
    # Buat DataFrame dari sample_data dengan nama kolom sama
    sample_df = pd.DataFrame([sample_data], columns=feature_names)
    
    # Terapkan imputasi
    sample_data_imputed = imputer.transform(sample_df)
    
    # Prediksi
    prediksi = model.predict(sample_data_imputed)
    prediksi_label = label_encoder.inverse_transform(prediksi)
    
    print("Prediksi kelulusan untuk data input manual:", prediksi_label[0])
    return prediksi_label[0]


def main():
    """
    Fungsi utama untuk menjalankan seluruh proses
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
    
    # Load data
    df = load_data("data_clean.csv")
    
    # Persiapkan fitur dan target
    X, y = prepare_features_and_target(df, features, 'status_kelulusan')
    
    # Preprocessing data
    X_train_raw, X_test, y_train_raw, y_test, imputer, le = preprocess_data(X, y)
    
    # Balance data training
    X_train, y_train = balance_data(X_train_raw, y_train_raw)
    
    # Inisialisasi model
    model = GradientBoostingClassifier(random_state=42)
    
    # Training model
    trained_model = train_model(model, X_train, y_train)
    
    # Evaluasi pada data test
    y_pred, conf_matrix, class_report = evaluate_model_on_test(trained_model, X_test, y_test, le)
    
    # Simpan model
    save_model(trained_model, 'model_kelulusan_gradient_boosting_balancing.joblib', le, 'label_encoder_gradient_boosting_balancing.joblib')
    
    # Contoh prediksi manual
    sample_data = [
        150,                                  # SKS TOTAL
        3.1, 3.2, 3.3, 3.4, 3.5,   # IP SEMESTER 1-7
        20, 22, 20, 21, 20,         # SKS TEMPUH SEMESTER 1-7
        1, 0                                  # JUMLAH MATKUL YANG DI ULANG, CUTI
    ]
    
    hasil_prediksi = predict_single_sample(trained_model, imputer, le, sample_data, features)


if __name__ == "__main__":
    main()