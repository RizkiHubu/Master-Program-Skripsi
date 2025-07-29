import pandas as pd

# Daftar nama file Excel
excel_files = [
    '2018.xls',
    '2019.xls',
    '2020.xls'
]

# Fungsi untuk memproses satu file
def process_file(excel_file):
    xls = pd.ExcelFile(excel_file)
    df = xls.parse(xls.sheet_names[0])
    df.columns = df.iloc[1]
    df = df.drop([0,1]).reset_index(drop=True)
    df.columns = df.columns.str.strip()
    
    selected_columns = [
        'NIM', 'ANGKATAN', 'IPK', 'IPK LULUS', 'SKS TEMPUH', 'SKS LULUS', 'SKS TOTAL',
        'IP SEMESTER 1', 'IP SEMESTER 2', 'IP SEMESTER 3', 'IP SEMESTER 4',
        'IP SEMESTER 5', 
        'SKS TEMPUH SEMESTER 1', 'SKS TEMPUH SEMESTER 2', 'SKS TEMPUH SEMESTER 3',
        'SKS TEMPUH SEMESTER 4', 'SKS TEMPUH SEMESTER 5',
        'JUMLAH MATKUL YANG DI ULANG', 'JUMLAH CUTI (SEMESTER)'
    ]

    # Cari kolom semester 8 ke atas
    semester_cols = [col for col in df.columns if 'SEMESTER' in col and any(str(i) in col for i in range(8, 15))]
    
    def determine_status(row):
        for col in semester_cols:
            if pd.notna(row[col]) and str(row[col]).strip() != '' and float(str(row[col]).strip() or 0) != 0:
                return 'Tidak Tepat Waktu'
        return 'Tepat Waktu'
    
    df['status_kelulusan'] = df.apply(determine_status, axis=1)
    
    data_selected = df[selected_columns + ['status_kelulusan']]
    return data_selected

# Proses semua file dan gabungkan
combined_data = pd.concat([process_file(f) for f in excel_files], ignore_index=True)

# 1. Konversi kolom numerik dan hapus baris dengan nilai 0 di semua kolom penting
numeric_cols = ['IPK', 'IPK LULUS', 'SKS TEMPUH', 'SKS LULUS', 'SKS TOTAL'] + \
               [f'IP SEMESTER {i}' for i in range(1,6)] + \
               [f'SKS TEMPUH SEMESTER {i}' for i in range(1,6)]

# Konversi ke numeric
for col in numeric_cols:
    combined_data[col] = pd.to_numeric(combined_data[col], errors='coerce')

# 2. Hapus baris dengan IPK = 0 atau kosong
combined_data = combined_data[(combined_data['IPK'].notna()) & (combined_data['IPK'] > 0)]

# 3. Hapus baris dengan semua nilai numerik = 0
mask = (combined_data[numeric_cols] == 0).all(axis=1)
combined_data = combined_data[~mask]

# 4. Hapus baris dengan SKS TOTAL = 0 
combined_data = combined_data[combined_data['SKS TOTAL'] > 0]

# Simpan ke CSV gabungan
output_csv = 'data_clean.csv'
combined_data.to_csv(output_csv, index=False)

print(f"Data bersih berhasil disimpan ke file {output_csv}")
print(f"Jumlah baris data yang tersimpan: {len(combined_data)}")