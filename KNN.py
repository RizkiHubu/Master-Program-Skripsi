"""
Sistem Prediksi Kelulusan Mahasiswa menggunakan K-Nearest Neighbors (KNN)
-------------------------------------------------------------------------
Model ini memprediksi status kelulusan mahasiswa berdasarkan data akademik
seperti IP semester, SKS yang ditempuh, jumlah matkul yang diulang, dan
jumlah cuti.

Versi ini tidak menggunakan teknik balancing data (tanpa SMOTETomek).
"""

import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.inspection import permutation_importance  # Tambahkan import ini


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
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_imputed, y_encoded, 
        test_size=test_size, 
        random_state=random_state, 
        stratify=y_encoded
    )
    
    return X_train, X_test, y_train, y_test, imputer, le


def evaluate_model_with_cv(model, X, y, cv_folds=5, random_state=42):
    """
    Mengevaluasi model menggunakan k-fold cross validation
    """
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    mean_score = np.mean(scores)
    
    print(f"Hasil {cv_folds}-Fold Cross Validation:")
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
    
    print("\nConfusion Matrix pada data test:")
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


def optimize_hyperparameters(X_train, y_train, param_grid=None, cv_folds=5, random_state=42):
    """
    Melakukan optimasi hyperparameter menggunakan GridSearchCV
    """
    if param_grid is None:
        param_grid = {
            'n_neighbors': [3, 5, 7, 9],
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan', 'minkowski']
        }
    
    base_model = KNeighborsClassifier()
    
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=cv,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )
    
    print("Melakukan optimasi hyperparameter...")
    grid_search.fit(X_train, y_train)
    
    print("Parameter terbaik:", grid_search.best_params_)
    print("Akurasi terbaik:", grid_search.best_score_)
    
    return grid_search.best_params_, grid_search.best_estimator_


def analyze_class_distribution(y, le=None):
    """
    Menganalisis distribusi kelas pada target
    
    Args:
        y: Target variable
        le: Label encoder (opsional, untuk menampilkan nama kelas)
        
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
    
    print("Distribusi kelas:")
    print(distribution)
    
    # Visualisasi distribusi
    plt.figure(figsize=(8, 5))
    distribution.plot(kind='bar')
    plt.title('Distribusi Kelas')
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
        'SKS TEMPUH SEMESTER 4','SKS TEMPUH SEMESTER 5',
        'JUMLAH MATKUL YANG DI ULANG', 'JUMLAH CUTI (SEMESTER)'
    ]
    
    # Load data
    df = load_data("data_clean.csv")
    X, y = prepare_features_and_target(df, features, 'status_kelulusan')
    X_train, X_test, y_train, y_test, imputer, le = preprocess_data(X, y)
    
    # Analisis distribusi kelas
    analyze_class_distribution(y_train, le)
    
    # Inisialisasi model KNN
    model = KNeighborsClassifier(n_neighbors=5, weights='uniform', metric='minkowski')
    
    # Training model
    trained_model = train_model(model, X_train, y_train)
    y_pred, conf_matrix, class_report = evaluate_model_on_test(trained_model, X_test, y_test, le)

    # Visualisasi matriks korelasi antar atribut
    visualize_correlation_matrix(df, features)
    
    save_model(trained_model, 'model_kelulusan_knn.joblib', le, 'label_encoder_knn.joblib')


if __name__ == "__main__":
    main()