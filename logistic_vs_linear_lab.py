import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import confusion_matrix, accuracy_score
from matplotlib.colors import ListedColormap
from io import StringIO

# 1. Veri Hazırlama
csv_data = """User ID,Gender,Age,EstimatedSalary,Purchased
15624510,Male,19,19000,0
15810944,Male,35,20000,0
15668575,Female,26,43000,0
15603246,Female,27,57000,0
15804002,Male,19,76000,0
15728773,Male,27,58000,0
15598044,Female,27,84000,0
15694829,Female,32,150000,1
15600575,Male,25,33000,0
15727311,Female,35,65000,0
15570769,Female,26,80000,0
15606274,Female,26,52000,0
15746139,Male,20,86000,0
15704987,Male,32,18000,0
15628972,Male,18,82000,0
15697686,Male,29,80000,0
15733883,Male,47,25000,1
15617482,Male,45,26000,1
15704583,Male,46,28000,1
15621083,Female,48,29000,1
15649487,Male,45,22000,1
15736760,Female,47,49000,1
15714658,Male,48,41000,1
15599081,Female,45,22000,1
15705113,Male,46,23000,1
15631159,Male,47,20000,1
15792818,Male,49,28000,1
15633531,Female,47,30000,1
15744529,Male,29,43000,0
15669656,Male,31,18000,0"""

df = pd.read_csv(StringIO(csv_data))
X = df[['Age', 'EstimatedSalary']].values
y = df['Purchased'].values

# Veriyi bölme ve ölçeklendirme
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# 2. Modelleri Eğitme
log_classifier = LogisticRegression(random_state=0)
log_classifier.fit(X_train, y_train)

lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

# 3. Performans Sonuçları (Terminal)
y_pred_log = log_classifier.predict(X_test)
y_pred_lin = (lin_reg.predict(X_test) >= 0.5).astype(int)

print(f"Logistic Reg. Accuracy: {accuracy_score(y_test, y_pred_log)}")
print(f"Linear Reg. Accuracy: {accuracy_score(y_test, y_pred_lin)}")

# 4. Karşılaştırmalı Görselleştirme Fonksiyonu
def plot_final_comparison(X_set, y_set):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Izgara oluşturma
    X1, X2 = np.meshgrid(np.arange(X_set[:, 0].min() - 1, X_set[:, 0].max() + 1, 0.01),
                         np.arange(X_set[:, 1].min() - 1, X_set[:, 1].max() + 1, 0.01))
    grid_points = np.array([X1.ravel(), X2.ravel()]).T

    # SOL: Logistic Regression Olasılık Haritası
    # Sadece 0-1 arasında renk geçişi yapar
    Z1 = log_classifier.predict_proba(grid_points)[:, 1].reshape(X1.shape)
    cp1 = ax1.contourf(X1, X2, Z1, alpha=0.3, cmap='RdYlGn')
    ax1.set_title('Logistic Regression (Probability Mapping)')
    
    # SAĞ: Linear Regression Ham Çıktı Haritası
    # 0'ın altında ve 1'in üstünde değerler de olabilir
    Z2 = lin_reg.predict(grid_points).reshape(X1.shape)
    cp2 = ax2.contourf(X1, X2, Z2, alpha=0.3, cmap='RdYlGn')
    ax2.set_title('Linear Regression (Raw Numerical Values)')

    # Veri noktalarını ekleme
    for ax, cp in zip([ax1, ax2], [cp1, cp2]):
        for i, j in enumerate(np.unique(y_set)):
            ax.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                        color=ListedColormap(('red', 'green'))(i), 
                        edgecolor='black', label=f"Class {j}")
        ax.set_xlabel('Age (Scaled)')
        ax.legend()
        plt.colorbar(cp, ax=ax) # Renk skalasını gösterir

    ax1.set_ylabel('Salary (Scaled)')
    plt.tight_layout()
    plt.show()

# Grafiği çalıştır
plot_final_comparison(X_test, y_test)