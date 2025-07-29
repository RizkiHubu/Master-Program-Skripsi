"""
Sistem Prediksi Kelulusan Mahasiswa menggunakan K-Nearest Neighbors (KNN)
-------------------------------------------------------------------------
Model ini memprediksi status kelulusan mahasiswa berdasarkan data akademik
seperti IP semester, SKS yang ditempuh, jumlah matkul yang diulang, dan
jumlah cuti.

Kode dibuat secara modular untuk memudahkan pemeliharaan dan pengembangan.
"""

import pandas as pd
import numpy as np
from sklearn.inspection import permutation_importance
from sklearn.neighbors import KNeighborsClassifier
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
    """
    return pd.read_csv(filepath)


def prepare_features_and_target(df, feature_list, target_column):
    """
    Mempersiapkan fitur dan target dari DataFrame
    """
    X = df[feature_list]
    y = df[target_column]
    return X, y


def preprocess_data(X, y, test_size=0.2, random_state=42):
    """
    Melakukan preprocessing data termasuk imputasi nilai kosong,
    encoding label, dan split data
    """
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
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
    """
    smt = SMOTETomek(random_state=random_state)
    X_train, y_train = smt.fit_resample(X_train_raw, y_train_raw)
    return X_train, y_train


def evaluate_model_with_cv(model, X, y, cv_folds=5, random_state=42):
    """
    Mengevaluasi model menggunakan k-fold cross validation
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
    """
    model.fit(X_train, y_train)
    return model


def evaluate_model_on_test(model, X_test, y_test, label_encoder):
    """
    Mengevaluasi model pada data test
    """
    y_pred = model.predict(X_test)
    
    confusion_mat = confusion_matrix(y_test, y_pred)
    classification_rep = classification_report(y_test, y_pred, 
                                               target_names=label_encoder.classes_)
    
    print("\nConfusion Matrix pada data test (tidak dibalancing):")
    print(confusion_mat)
    print("\nClassification Report:")
    print(classification_rep)
    
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
    """
    sample_df = pd.DataFrame([sample_data], columns=feature_names)
    sample_data_imputed = imputer.transform(sample_df)
    prediksi = model.predict(sample_data_imputed)
    prediksi_label = label_encoder.inverse_transform(prediksi)
    
    print("Prediksi kelulusan untuk data input manual:", prediksi_label[0])
    return prediksi_label[0]


def analyze_class_distribution(y, le=None, title="Distribusi Kelas"):
    """
    Menganalisis distribusi kelas pada target
    
    Args:
        y: Target variable
        le: Label encoder (opsional, untuk menampilkan nama kelas)
        title: Judul untuk visualisasi distribusi kelas
        
    Returns:
        Series: Distribusi kelas
    """
    if hasattr(y, 'value_counts'):  # Jika y adalah Series
        distribution = y.value_counts()
    else:  # Jika y adalah array
        distribution = pd.Series(y).value_counts()
    
    # Tampilkan nama kelas jika le tersedia
    if le is not None:
        class_names = le.classes_
        distribution.index = [class_names[i] for i in distribution.index]
    
    print(f"{title}:")
    print(distribution)
    
    # Visualisasi distribusi
    plt.figure(figsize=(8, 5))
    distribution.plot(kind='bar')
    plt.title(title)
    plt.ylabel('Jumlah')
    plt.xlabel('Kelas')
    plt.tight_layout()
    plt.show()
    
    return distribution

def visualize_correlation_matrix(df, features):
    """
    Memvisualisasikan matriks korelasi antar atribut numerik pada dataset.

    Args:
        df: DataFrame sumber data
        features: List nama fitur numerik yang akan dihitung korelasinya
    """
    corr_matrix = df[features].corr()
    print("Matriks Korelasi antar Atribut:")
    print(corr_matrix)
    plt.figure(figsize=(14, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Matriks Korelasi antar Atribut')
    plt.tight_layout()
    plt.show()
    return corr_matrix

def main():
    """
    Fungsi utama untuk menjalankan seluruh proses
    """
    features = [
        'SKS TOTAL',
        'IP SEMESTER 1', 'IP SEMESTER 2', 'IP SEMESTER 3', 'IP SEMESTER 4',
        'IP SEMESTER 5',
        'SKS TEMPUH SEMESTER 1', 'SKS TEMPUH SEMESTER 2', 'SKS TEMPUH SEMESTER 3',
        'SKS TEMPUH SEMESTER 4', 'SKS TEMPUH SEMESTER 5','JUMLAH MATKUL YANG DI ULANG', 'JUMLAH CUTI (SEMESTER)'
    ]
    #Load data
    df = load_data("data_clean.csv")

    # Preprocessing data
    X, y = prepare_features_and_target(df, features, 'status_kelulusan')

    # Memisahkan data menjadi training dan test
    X_train_raw, X_test, y_train_raw, y_test, imputer, le = preprocess_data(X, y)
    
    # Analisis distribusi kelas sebelum balancing
    analyze_class_distribution(y_train_raw, le, title="Distribusi Kelas Sebelum Balancing")

    # Melakukan balancing pada data training
    X_train, y_train = balance_data(X_train_raw, y_train_raw)
    
    # Analisis distribusi kelas setelah balancing
    analyze_class_distribution(y_train, le, title="Distribusi Kelas Setelah Balancing")
    
    #Inisialisasi model KNN
    model = KNeighborsClassifier(n_neighbors=5, weights='uniform', metric='minkowski')
    
    # Melatih model
    trained_model = train_model(model, X_train, y_train)
    # Mengevaluasi model pada data test
    y_pred, conf_matrix, class_report = evaluate_model_on_test(trained_model, X_test, y_test, le)
    # Visualisasi matriks korelasi antar atribut
    visualize_correlation_matrix(df,features)
    
    save_model(trained_model, 'model_kelulusan_knn_balancing.joblib', le, 'label_encoder_knn_balancing.joblib')
    
    sample_data = [
        150, 
        3.1, 3.2, 3.3, 3.4, 3.5,
        20, 22, 20, 21, 20,
        1, 0
    ]
    hasil_prediksi = predict_single_sample(trained_model, imputer, le, sample_data, features)


if __name__ == "__main__":
    main()