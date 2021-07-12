from tslearn.clustering import TimeSeriesKMeans, KShape, KernelKMeans
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
import pandas as pd
from scipy.fftpack import rfft
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_samples, silhouette_score
df = pd.read_csv("egauge_maxtrix_201501_work.csv", header=None)
df = df[0:100]
df = df.dropna()
df = df.drop(columns=0)
df = df.set_index(1)
print(df)

X = rfft(df).transpose()
print(X)

X = TimeSeriesScalerMeanVariance().fit_transform(X)
print(X)
df = pd.DataFrame(X.reshape(df.shape), columns=df.columns, index=df.index)

model = TimeSeriesKMeans(n_clusters=5, metric="dtw", max_iter=5, n_init=2).fit(X)
X_train = X[0:10]
sz = X_train.shape[1]
y_pred = model.fit_predict(X_train)


for yi in range(5):
    plt.subplot(5, 5, 6 + yi)
    for xx in X_train[y_pred == yi]:
        plt.plot(xx.ravel(), "k-", alpha=.2)
    plt.plot(model.cluster_centers_[yi].ravel(), "r-")
    plt.xlim(0, sz)
    plt.ylim(-4, 4)
    plt.text(0.55, 0.85,'Cluster %d' % (yi + 1),
             transform=plt.gca().transAxes)
    if yi == 1:
        plt.title("Soft-DTW $k$-means")

plt.tight_layout()
plt.show()