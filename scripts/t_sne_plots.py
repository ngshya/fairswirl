from numpy import load as np_load, delete
from pickle import load
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from pandas import DataFrame



def plot_tsne_legend(v, figsize=(10, 1)):
    if v == 's':
        p = "hls"
    else:
        p = "Set2"
    fig, ax = plt.subplots(figsize=figsize)
    for j in range(2):
        ax.scatter(
            [0], [0], 
            label=v+" = "+str(j), 
            color=sns.color_palette(p, 2)[j], 
        )
    ax.axis('off')
    leg = fig.legend(fontsize=30, loc="center", framealpha=1, ncol=2)
    leg.get_frame().set_linewidth(0.0)
    for handle in leg.legendHandles:
        handle.set_sizes([300.0])
    fig.savefig('output_imgs/tsne_legend_'+v+'.pdf', dpi=600, format="pdf", bbox_inches='tight')



def plot_tsne(title, x, y, df, hue, figsize=(8,8)):
    fig = plt.figure(figsize=figsize)
    fig.patch.set_facecolor('white')
    fig.patch.set_alpha(1.0)
    ax = plt.gca()
    ax.tick_params(top=False, bottom=False, left=False, right=False, labelleft=False, labelbottom=False)
    #ax.set_title(title)
    if hue == 's':
        p = "hls"
    else:
        p = "Set2"
    sns.scatterplot(
        x=x, y=y,
        hue=hue,
        data=df,
        palette=sns.color_palette(p, 2),
        legend="full",
        alpha=0.3
    )
    ax.get_legend().remove()
    fig.savefig('output_imgs/tsne_'+title.lower().replace(' ', '_')+'_'+hue+'.pdf', dpi=600, format="pdf", bbox_inches='tight')



def plot_tsne_dataset(dataset, split='L100_L100x20_U10000_V100_T10000_seed1102', sets='1'):
    embeddings = np_load("outputs/dsa_embeddings_"+dataset+"_"+sets+"_"+split+".npy")
    with open("datasets/"+dataset+"/"+dataset+"_"+split+".pickle", "rb") as f:
        dict_data = load(f)

    idx_del = []
    for j in range(dict_data["T"]["X"].shape[1]):
        if (sum(dict_data["T"]["X"][:, j] == dict_data["T"]["s"]) == dict_data["T"]["X"].shape[0]) \
        or (sum(dict_data["T"]["X"][:, j] == (1-dict_data["T"]["s"])) == dict_data["T"]["X"].shape[0]):
            idx_del.append(j)
    X_without_s = delete(dict_data["T"]["X"], idx_del, axis=1)

    tsne_2d_embeddings = TSNE(n_components=2, random_state=1102).fit_transform(embeddings)
    tsne_2d_original = TSNE(n_components=2, random_state=1102).fit_transform(dict_data["T"]["X"])
    tsne_2d_dummy = TSNE(n_components=2, random_state=1102).fit_transform(X_without_s)

    df = DataFrame({
        "x1_emb": tsne_2d_embeddings[:, 0], 
        "x2_emb": tsne_2d_embeddings[:, 1], 
        "x1_ori": tsne_2d_original[:, 0], 
        "x2_ori": tsne_2d_original[:, 1],     
        "x1_dum": tsne_2d_dummy[:, 0], 
        "x2_dum": tsne_2d_dummy[:, 1],     
        "s": dict_data["T"]["s"],
        "y": dict_data["T"]["y"],
    })

    df["case"] = "s" + df["s"].astype(str) + "_y" + df["y"].astype(str)

    plot_tsne(title=dataset.upper() + " - Original Dataset", x="x1_ori", y="x2_ori", df=df, hue='s', figsize=(8,8))
    plot_tsne(title=dataset.upper() + " - Dummy Debiasing", x="x1_dum", y="x2_dum", df=df, hue='s', figsize=(8,8))
    plot_tsne(title=dataset.upper() + " - FairSwiRL", x="x1_emb", y="x2_emb", df=df, hue='s', figsize=(8,8))

    plot_tsne(title=dataset.upper() + " - Original Dataset", x="x1_ori", y="x2_ori", df=df, hue='y', figsize=(8,8))
    plot_tsne(title=dataset.upper() + " - Dummy Debiasing", x="x1_dum", y="x2_dum", df=df, hue='y', figsize=(8,8))
    plot_tsne(title=dataset.upper() + " - FairSwiRL", x="x1_emb", y="x2_emb", df=df, hue='y', figsize=(8,8))

    
    
if __name__ == "__main__":
    plot_tsne_dataset(dataset="synthetic")
    plot_tsne_dataset(dataset="adult")
    plot_tsne_legend("s")
    plot_tsne_legend("y")
