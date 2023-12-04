import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
sns.set(style='dark')

day = pd.read_csv("D:\Teknik Informatika\Dicoding\Data Science\Bike-sharing-analisis\data\day.csv")

# create_daily_sharing_df() digunakan untuk menyiapkan daily_sharing_df
def create_daily_sharing_df(df):
    daily_sharing_df = df.resample(rule='D', on='dteday').agg({
        "instant": "nunique",
        "cnt": "sum"
    })
    daily_sharing_df = daily_sharing_df.reset_index()
    daily_sharing_df.rename(columns={
        "instant": "cust_count",
        "cnt": "sharing_count"
    }, inplace=True)
    
    return daily_sharing_df

# create_sum_cust_df() bertanggung jawab untuk menyiapkan sum_cust_df
def create_sum_cust_df(df):
    sum_cust_df = df.groupby("instant").cnt.sum().sort_values(ascending=False).reset_index()
    return sum_cust_df

# create_byseason_df() digunakan untuk menyiapkan byseason_df
def create_byseason_df(df):
    byseason_df = df.groupby(by="season").instant.nunique().reset_index()
    byseason_df.rename(columns={
        "instant": "customer_count"
    }, inplace=True)
    
    return byseason_df

# create_byworkingday_df() digunakan untuk menyiapkan byworkingday_df
def create_byworkingday_df(df):
    byworkingday_df = df.groupby(by="workingday").instant.nunique().reset_index()
    byworkingday_df.rename(columns={
        "instant": "customer_count"
    }, inplace=True)
    
    return byworkingday_df

# create_rfm_df() bertanggung jawab untuk menghasilkan rfm_df
def create_rfm_df(df):
    rfm_df = df.groupby(by="instant", as_index=False).agg({
        "dteday": "max",        # mengambil tanggal order terakhir
        "cnt": ["sum", "nunique"],               # Total number of rentals
    })
    rfm_df.columns = ["instant", "last_rental_date", "frequency", "monetary"]
    
    rfm_df["last_rental_date"] = rfm_df["last_rental_date"].dt.date
    recent_date = pd.to_datetime(day["dteday"]).dt.date.max()
    rfm_df["recency"] = rfm_df["last_rental_date"].apply(lambda x: (recent_date - x).days)

    # Drop 'last_rental_date' column
    rfm_df.drop("last_rental_date", axis=1, inplace=True)
    
    return rfm_df

# load berkas main_data.csv sebagai sebuah DataFrame
main_df = pd.read_csv("D:\Teknik Informatika\Dicoding\Data Science\Bike-sharing-analisis\main_data.csv")

# mengurutkan DataFrame berdasarkan order_date
# memastikan kedua kolom tersebut bertipe datetime
datetime_columns = ["dteday"]
main_df.sort_values(by="dteday", inplace=True)
main_df.reset_index(inplace=True)

for column in datetime_columns:
    main_df[column] = pd.to_datetime(main_df[column])

# Membuat Komponen Filter
# membuat filter dengan widget date input serta menambahkan logo perusahaan pada sidebar
min_date = main_df["dteday"].min()
max_date = main_df["dteday"].max()

with st.sidebar:
    # Menambahkan logo perusahaan
    left_co, cent_co, last_co = st.columns(3)
    with cent_co:
        st.image(
            "dashboard/logo.png", width=100
        )
    
    # Mengambil start_date & end_date dari date_input
    start_date, end_date = st.date_input(
        label='Rentang Waktu',min_value=min_date,
        max_value=max_date,
        value=[min_date, max_date]
    )

# Data yang telah difilter ini selanjutnya akan disimpan dalam all_df  
all_df = main_df[(main_df["dteday"] >= str(start_date)) & 
                (main_df["dteday"] <= str(end_date))]

# Membuat Visualisasi
daily_sharing_df = create_daily_sharing_df(all_df)
sum_cust_df = create_sum_cust_df(all_df)
byseason_df = create_byseason_df(all_df)
byworkingday_df = create_byworkingday_df(all_df)
rfm_df = create_rfm_df(all_df)

# Melengkapi Dashboard dengan Berbagai Visualisasi Data
# menambahkan header pada dashboard
st.header('Bike Sharing Dashboard :sparkles:')

# menampilkan tiga informasi terkait daily sharing, yaitu jumlah customer dan jumlah sharing
st.subheader('Daily sharing')

col1, col2 = st.columns(2)

with col1:
    total_cust = daily_sharing_df.cust_count.sum()
    st.metric("Total Customer", value=total_cust)

with col2:
    total_sharing = daily_sharing_df.sharing_count.sum()
    st.metric("Total Sharing", value=total_sharing)

fig, ax = plt.subplots(figsize=(16, 8))
ax.plot(
    daily_sharing_df["dteday"],
    daily_sharing_df["sharing_count"],
    marker='o', 
    linewidth=2,
    color="#90CAF9"
)
ax.tick_params(axis='y', labelsize=20)
ax.tick_params(axis='x', labelsize=15)

st.pyplot(fig)

# Menampilkan pola perbedaan peminjaman sepeda antara hari libur dan hari kerja
st.subheader("Differences in Bike Sharing Patterns between Holidays and Weekdays")

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(35, 20))
colors = ["#90CAF9", "#D3D3D3", "#D3D3D3", "#D3D3D3", "#D3D3D3"]

sns.barplot(x='workingday', y='cnt', data=day)
ax.set_ylabel('Total Rental Bikes (cnt)', fontsize=30)
ax.set_xlabel('Working Day (0: Holiday, 1: Working Day)', fontsize=30)
ax.tick_params(axis='y', labelsize=30)
ax.tick_params(axis='x', labelsize=30)

st.pyplot(fig)

# Menampilkan Korelasi antara Variabel Cuaca dengan Jumlah Total Sepeda yang Disewa per Hari
st.subheader("Correlation between Weather Variables and Total Number of Bike Shared per Day")
correlation_matrix = day[['temp', 'atemp', 'hum', 'windspeed', 'cnt']].corr()
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(35, 15))
heatmap = sns.heatmap(correlation_matrix, annot=True, cmap='Pastel2', fmt='.4f', linewidths=.5, annot_kws={"size": 30})
ax.tick_params(axis='x', labelsize=30)
ax.tick_params(axis='y', labelsize=30)

st.pyplot(fig)

# memasukkan 2 buah visualisasi data ke dalam dashboard
st.subheader("Customer Demographics")

col1, col2 = st.columns(2)

with col1:
    fig, ax = plt.subplots(figsize=(20, 10))
    
    colors = ["#D3D3D3", "#90CAF9", "#A7F9EA", "#FDD2FC", "#FBF7D3"]

    sns.barplot(
        y="customer_count", 
        x="season",
        data=byseason_df.sort_values(by="customer_count", ascending=False),
        palette=colors,
        ax=ax
    )
    ax.set_title("Number of Customer by Season", loc="center", fontsize=50)
    ax.set_ylabel(None)
    ax.set_xlabel(None)
    ax.tick_params(axis='x', labelsize=35)
    ax.tick_params(axis='y', labelsize=30)
    st.pyplot(fig)

with col2:
    fig, ax = plt.subplots(figsize=(20, 10))
    
    colors = ["#D3D3D3", "#90CAF9", "#A7F9EA", "#FDD2FC", "#FBF7D3"]

    sns.barplot(
        y="customer_count", 
        x="workingday",
        data=byworkingday_df.sort_values(by="customer_count", ascending=False),
        palette=colors,
        ax=ax
    )
    ax.set_title("Number of Customer by Workingday", loc="center", fontsize=50)
    ax.set_ylabel(None)
    ax.set_xlabel(None)
    ax.tick_params(axis='x', labelsize=35)
    ax.tick_params(axis='y', labelsize=30)
    st.pyplot(fig)

# menampilkan nilai average atau rata-rata dari ketiga parameter tersebut menggunakan widget metric()
# menampilkan hasil visualisasi data dari latihan sebelumnya
st.subheader("Best Customer Based on RFM Parameters")

col1, col2, col3 = st.columns(3)

with col1:
    avg_recency = round(rfm_df.recency.mean(), 1)
    st.metric("Average Recency (days)", value=avg_recency)

with col2:
    avg_frequency = round(rfm_df.frequency.mean(), 2)
    st.metric("Average Frequency", value=avg_frequency)

with col3:
    avg_monetary = round(rfm_df.monetary.mean(), 3) 
    st.metric("Average Monetary", value=avg_monetary)

fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(35, 15))
colors = ["#90CAF9", "#90CAF9", "#90CAF9", "#90CAF9", "#90CAF9"]

sns.barplot(y="recency", x="instant", data=rfm_df.sort_values(by="recency", ascending=True).head(5), palette=colors, ax=ax[0])
ax[0].set_ylabel(None)
ax[0].set_xlabel("instant", fontsize=30)
ax[0].set_title("By Recency (days)", loc="center", fontsize=50)
ax[0].tick_params(axis='y', labelsize=30)
ax[0].tick_params(axis='x', labelsize=35)

sns.barplot(y="frequency", x="instant", data=rfm_df.sort_values(by="frequency", ascending=False).head(5), palette=colors, ax=ax[1])
ax[1].set_ylabel(None)
ax[1].set_xlabel("instant", fontsize=30)
ax[1].set_title("By Frequency", loc="center", fontsize=50)
ax[1].tick_params(axis='y', labelsize=30)
ax[1].tick_params(axis='x', labelsize=35)

sns.barplot(y="monetary", x="instant", data=rfm_df.sort_values(by="monetary", ascending=False).head(5), palette=colors, ax=ax[2])
ax[2].set_ylabel(None)
ax[2].set_xlabel("instant", fontsize=30)
ax[2].set_title("By Monetary", loc="center", fontsize=50)
ax[2].tick_params(axis='y', labelsize=30)
ax[2].tick_params(axis='x', labelsize=35)

st.pyplot(fig)

st.caption('Copyright (c) Bike Sharing 2023')