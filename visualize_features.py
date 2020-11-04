import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import argparse
from multiprocessing import cpu_count
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import io
from PIL import Image

def plot_pca(embeddings, labels):    
    pca = PCA(2)
    embeddings[embeddings == 0] = 1e-10
    embeddings = pca.fit_transform(np.log(embeddings))
    return _plot(embeddings, labels)

def plot_tsne(embeddings, labels):
    tsne = TSNE(n_jobs=-1, verbose=2)
    embeddings = tsne.fit_transform(embeddings)
    return _plot(embeddings, labels)

def _plot(embeddings, labels):
    sns.set(font_scale=2, style='white')
    df = pd.DataFrame()
    df['x1'] = embeddings[:,0]
    df['x2'] = embeddings[:,1]
    df['Label'] = labels+1
    plt.figure(figsize=(20,10))
    sns.scatterplot(
        x="x1", y="x2",
        hue="Label",
        palette=sns.color_palette("hls", len(set(labels))),
        data=df,
        legend="full",
        alpha=0.3
    )
    buf = io.BytesIO()
    plt.savefig(buf, format='jpeg')
    buf.seek(0)
    image = Image.open(buf)
    return image