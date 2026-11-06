"""Plot published aggregate results without inferring missing experiments."""
import argparse,json
from pathlib import Path
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path,default=Path('results/main.json'))
    p.add_argument('--output',type=Path,default=Path('outputs/main-results.pdf'));a=p.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows=json.loads(a.data.read_text());methods=['student','teacher','k4_opd','k4_sft','full_opd','online_opd']
    colors=['#94999F','#CAB5D8','#5F9F9B','#E6AD75','#90B7D4','#D28F92']
    fig,axes=plt.subplots(1,len(rows),figsize=(15,3.3),layout='constrained')
    for ax,row in zip(axes,rows,strict=True):
        values=[row.get(k) for k in methods];valid=[(i,v) for i,v in enumerate(values) if v is not None]
        ax.bar([x[0] for x in valid],[x[1] for x in valid],color=[colors[x[0]] for x in valid])
        ax.set_xticks(range(len(methods)),['Base','Teacher','K4 OPD','K4 SFT','Full OPD','Online'],rotation=60,ha='right')
        ax.set_title(row['panel'],fontsize=10);ax.set_ylabel(row['metric']);ax.set_ylim(0,1)
        ax.spines[['top','right']].set_visible(False)
    a.output.parent.mkdir(parents=True,exist_ok=True);fig.savefig(a.output);plt.close(fig)
if __name__=='__main__':main()
