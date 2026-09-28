import pandas as pd

# Ganti dengan nama file dataset Anda yang sebenarnya
df = pd.read_csv('GojekAppReviewV4.0.0-V4.9.3_Cleaned.csv')

# Menampilkan seluruh nama kolom yang ada di dataset
print("Daftar Nama Kolom:")
print(df.columns)

# Menampilkan 2 baris pertama data untuk melihat isinya
print("\nContoh Data:")
print(df.head(2))