# ames-housing-ridge-lasso-regression
A comprehensive house price prediction project using Ridge and Lasso Regression, featuring data preprocessing, feature engineering, cross-validation, hyperparameter tuning, and model evaluation on the Ames Housing dataset.


# Ames Housing Price Prediction | Ridge & Lasso Regression

Ames Housing veri seti üzerinde konut satış fiyatlarını tahmin etmeyi amaçlayan, veri keşfinden nihai test analizine kadar **8 bölümlük** uçtan uca makine öğrenmesi projesi.

Bu çalışmada **DummyRegressor**, **LinearRegression**, **Ridge Regression** ve **Lasso Regression** karşılaştırılır. Özellikle veri sızıntısının önlenmesi, eğitim/test ayrımının sabitlenmesi, 5 katlı çapraz doğrulama ve test verisinin yalnızca nihai değerlendirmede kullanılması üzerinde durulmuştur.

## Veri seti

- **Kaynak:** Ames Housing; Dean De Cock (2011), Ames, Iowa, ABD, 2006–2010 satışları.
- **Ham veri:** 2.930 kayıt, 82 sütun (80 konut özelliği, `SalePrice` hedefi, `Order` ve `PID` kimlik alanları).
- **Hedef:** `SalePrice` — satış fiyatı (USD).
- **Ayrım:** %80 eğitim (2.344 kayıt), %20 test (586 kayıt), `random_state=42`.
- **Model girdileri:** `Order` ve `PID` çıkarıldıktan sonra 79 konut özelliği; eğitimde belirlenen iki aykırı satış çıkarıldığında model geliştirmede 2.342 satır.

> Veri setindeki `NA` ifadesi bazı alanlarda özellik bulunmadığını belirtir; her zaman kayıp değer anlamına gelmez. Bu nedenle CSV `keep_default_na=False, na_values=[""]` yaklaşımıyla okunur.

## Projenin aşamaları

| Bölüm | Notebook | İçerik |
|---|---|---|
| 1 | `Bolum1_Veriyi_Tanima.ipynb` | Veri keşfi, dağılımlar, korelasyonlar, eğitim/test ayrımı |
| 2 | `Bolum2_Veri_Hazirlama.ipynb` | Eksik değerler, aykırı gözlemler, kodlama ve veri hazırlama |
| 3 | `Bolum3_Ozellik_Uretimi_ve_Secimi.ipynb` | 13 yeni özellik, filtre tabanlı ve gömülü seçim yöntemleri |
| 4 | `Bolum4_Adil_Test_Duzeni.ipynb` | Sabit ayrım, 5 katlı CV, pipeline ve deney kayıtları |
| 5 | `Bolum5_Baseline_ve_Referans_Model.ipynb` | DummyRegressor ve LinearRegression referans modelleri |
| 6 | `Bolum6_Ridge_Lasso_Hiperparametre_Ayari.ipynb` | Ridge/Lasso alpha araması ve özellik seçiminin karşılaştırılması |
| 7 | `Bolum7_Test_Setinde_Degerlendirme.ipynb` | Dondurulmuş ayarlarla tek nihai test değerlendirmesi |
| 8 | `Bolum8_Sonuclarin_Yorumlanmasi.ipynb` | Katsayı analizi, hata analizi, sınırlılıklar ve öneriler |

## Yöntem

1. Veriyi sabit rastgelelikle eğitim ve test olarak ayır.
2. Keşif ve hazırlık kararlarını eğitim verisi üzerinde geliştir.
3. Yapısal eksikleri anlamlarına göre işle; sayısal/kategorik dönüşümleri `Pipeline` içinde uygula.
4. Yeni özellikler üret; farklı özellik seçimi yaklaşımlarını değerlendir.
5. Eğitim verisinde **5-fold cross-validation** ile modelleri ve hiperparametreleri karşılaştır.
6. Nihai ayarları test değerlendirmesinden önce sabitle.
7. Aynı **586 test konutunda** dört modeli değerlendir; sonuçları ve büyük hataları yorumla.

Logaritmik hedef dönüşümü uygulanır; sonuçlar hem dolar hem log ölçekte raporlanır. Değerlendirme metrikleri **MAE, RMSE, R² ve log ölçekli metriklerdir**.

## Sonuçlar

| Model | Nihai testten öne çıkan sonuç |
|---|---|
| DummyRegressor | Özellik kullanmayan temel referans |
| LinearRegression | Dolar RMSE ≈ **30.466 USD**, dolar R² ≈ **0,8842** |
| Ridge | MAE ≈ **13.970 USD**, log RMSE ≈ **0,11485**, dolar RMSE ≈ **34.941 USD** |
| Lasso | Dolar RMSE ≈ **34.825 USD** |

- Eğitim CV'sinde seçilen Ridge log RMSE ≈ **0,11484**, Lasso ≈ **0,11568**; bu küçük fark istatistiksel üstünlük kanıtı değildir.
- Ridge, nihai testte MAE ve log RMSE açısından güçlü sonuç verirken, LinearRegression dolar RMSE ve dolar R² açısından daha iyidir.
- Lasso, analiz edilen 227 katsayıdan **99'unu sıfıra çekmiştir**.
- Yeni özelliklerin performans üzerindeki etkisi sınırlı kalmıştır.
- Özellikle büyük ve alışılmadık bir test konutundaki tahmin hatası dolar RMSE sonucunu belirgin biçimde etkilemiştir. Bu gözlem test kümesinden çıkarılmamıştır.

**Önemli:** Farklı hata metrikleri farklı model sıralamaları oluşturur. Test sonuçlarına bakılarak yeniden hiperparametre seçimi yapılmamıştır.

## Kurulum ve çalıştırma

Python 3.10+ önerilir.

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
# source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook
```

1. `AmesHousing.csv` dosyasını notebook'larla **aynı dizinde** tutun. Orijinal dosya `AmesHousing(1).csv` adındaysa `AmesHousing.csv` olarak yeniden adlandırın.
2. Notebook'ları **Bölüm 1 → Bölüm 8** sırasıyla çalıştırın.
3. Özellikle **Bölüm 4–8**, önceki bölümlerin oluşturduğu deney kayıtlarını ve sonuç dosyalarını kullanır. Önceki bölümün çıktı klasörlerini silmeyin.
4. Var olan bir deney planı veya test kaydı değiştirilmişse, veri bütünlüğü kontrolleri çalışmayı durdurabilir. Ayrı, temiz bir çalışma klasöründe yeniden üretim yapmak daha güvenlidir.

## Tekrarlanabilirlik ve değerlendirme ilkeleri

- Sabit `random_state=42` ve %20 test ayrımı.
- Dönüşümler eğitim katlarında `fit` edilir; test üzerinde yeniden öğrenilmez.
- Veri ve deney kayıtlarında SHA-256 bütünlük kontrolleri.
- Hiperparametreler test değerlendirmesinden önce kaydedilir.
- Test kümesindeki 586 konutun tamamı değerlendirmede korunur.
- Nihai test sonuçları model seçmek veya yeniden ayar yapmak için kullanılmaz.

## Sınırlılıklar

Bu sonuçlar Ames, Iowa'daki 2006–2010 yılları arasındaki satışlar için geçerlidir. Rastgele test ayrımı zamansal genelleme testi değildir. Büyük/alışılmadık konutlarda önemli tahmin hataları oluşabilir. Düzenlileştirilmiş modellerin her metrikte üstün olması beklenmemelidir.

## Teknolojiler

Python · Jupyter Notebook · NumPy · Pandas · Matplotlib · Seaborn · scikit-learn

## Veri kaynağı ve atıf

De Cock, D. (2011). *Ames, Iowa: Alternative to the Boston Housing Data as an End of Semester Regression Project*. Journal of Statistics Education, 19(3).

Bu depo veri setinin kaynağını belirtir; yeniden dağıtım koşulları ve veri seti lisansı ayrıca kontrol edilmelidir. Proje kodu için açık kaynak lisansı, hak sahibi tarafından ayrıca seçilmedikçe varsayılmamalıdır.

