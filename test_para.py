import stlearn as st
from pathlib import Path
import multiprocessing


def RapidPlotSettings(x):
    if x in list(data.var_names):
        st.pl.gene_plot(data, genes = x, threshold = 0, cmap = "autumn", data_alpha = 0.9, spot_size = 4, name = x, output = "output",show_plot=False)
        print("PLOT DONE!")
    else:
        print (x + " is not in list")

SpTxDirectories = ["../UQ/10X/BCBA/"]


BASE_PATH = Path("../UQ/10X/BCBA/")
data = st.Read10X(BASE_PATH)
data.var_names_make_unique()
st.pp.filter_genes(data,min_cells=3)
st.pp.normalize_total(data)
st.pp.log1p(data)
st.pp.scale(data)

TargetGenes = list(data.var_names)[:10]
        
if __name__ == '__main__':
    threads = []
    for i in TargetGenes:
        t = multiprocessing.Process(target=RapidPlotSettings, args=i)
        threads.append(t)
        t.start()


    for process in threads:
        process.join()