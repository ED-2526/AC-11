import json
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns

def visualitzar_f1_scores(filepath, output_dir=None):
    """
    Genera una gràfica de l'evolució del F1-score segons K per a cada mètode.
    
    Args:
        filepath: Ruta al fitxer JSON amb les estadístiques
        output_dir: Directori on guardar la imatge (opcional, per defecte el mateix que el JSON)
    """
    
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    plt.figure(figsize=(12, 7))
    
    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33', '#a65628', '#f781bf']
    estils = ['-', '-', '-', '--', '--', ':', ':', ':']
    marcadors = ['o', 's', '^', 'o', 's', 'o', 's', '^']
    
    for idx, (metode, resultats_k) in enumerate(data.items()):
        valors_k = sorted([int(k) for k in resultats_k.keys()])
        
        f1_scores = [resultats_k[str(k)]['f1'] for k in valors_k]
        
        plt.plot(valors_k, f1_scores, 
                 color=colors[idx % len(colors)],
                 linestyle=estils[idx % len(estils)],
                 marker=marcadors[idx % len(marcadors)],
                 markersize=5,
                 linewidth=2,
                 label=metode)
    
    nom_kernel = os.path.basename(filepath).replace('estadisticas_', '').replace('.json', '')
    plt.title(f"Evolució del F1-Score segons K (Kernel: {nom_kernel})", fontsize=14, fontweight='bold')
    plt.xlabel("Nombre de clusters (K)", fontsize=12)
    plt.ylabel("F1-Score (macro)", fontsize=12)
    
    plt.legend(title="Mètodes", loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=10)
    
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.ylim(0, 1)
    
    plt.tight_layout()
    
    if output_dir is None:
        output_dir = os.path.dirname(filepath)
    
    nom_sortida = f"f1_evolution_{nom_kernel}.png"
    path_sortida = os.path.join(output_dir, nom_sortida)
    
    plt.savefig(path_sortida, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Gràfica guardada a: {path_sortida}")
    return path_sortida


def visualitzar_tots_els_kernels(directori):
    """
    Genera gràfiques per a tots els fitxers d'estadístiques d'un directori.
    
    Args:
        directori: Ruta al directori que conté els JSON d'estadístiques
    """
    
    fitxers = [f for f in os.listdir(directori) if f.startswith('estadisticas_') and f.endswith('.json')]
    
    if not fitxers:
        print("No s'han trobat fitxers d'estadístiques al directori especificat.")
        return
    
    print(f"Trobats {len(fitxers)} fitxers d'estadístiques:")
    
    for fitxer in fitxers:
        filepath = os.path.join(directori, fitxer)
        print(f"\n  Processant: {fitxer}")
        visualitzar_f1_scores(filepath)

def generar_heatmap_millors_k(directori, output_path=None):
    """
    Genera un mapa de calor mostrant la millor K per a cada combinació mètode-kernel.
    El color indica el F1-score aconseguit.
    
    Args:
        directori: Ruta al directori que conté els JSON d'estadístiques
        output_path: Ruta on guardar la imatge (opcional)
    """
    
    fitxers = [f for f in os.listdir(directori) if f.startswith('estadisticas_') and f.endswith('.json')]
    
    if not fitxers:
        print("No s'han trobat fitxers d'estadístiques.")
        return
    
    millor_k = {}      # {kernel: {metode: k}}
    millor_f1 = {}     # {kernel: {metode: f1}}
    
    for fitxer in fitxers:
        kernel = fitxer.replace('estadisticas_', '').replace('.json', '')
        
        filepath = os.path.join(directori, fitxer)
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        millor_k[kernel] = {}
        millor_f1[kernel] = {}
        
        for metode, resultats_k in data.items():
            best_f1 = -1
            best_k = None
            
            for k_str, stats in resultats_k.items():
                f1 = stats['f1']
                if f1 > best_f1:
                    best_f1 = f1
                    best_k = int(k_str)
            
            millor_k[kernel][metode] = best_k
            millor_f1[kernel][metode] = best_f1
    
    kernels = sorted(millor_k.keys())
    metodes = sorted(list(millor_k[kernels[0]].keys()))
    
    matriu_k = np.zeros((len(metodes), len(kernels)))
    matriu_f1 = np.zeros((len(metodes), len(kernels)))
    
    for i, metode in enumerate(metodes):
        for j, kernel in enumerate(kernels):
            matriu_k[i, j] = millor_k[kernel].get(metode, 0)
            matriu_f1[i, j] = millor_f1[kernel].get(metode, 0)
    
    plt.figure(figsize=(12, 8))
    
    ax = sns.heatmap(
        matriu_f1,
        annot=matriu_k,  # Mostrar K com a text
        fmt='.0f',       # Format sense decimals per a K
        cmap='RdYlGn',   # Vermell (dolent) -> Groc -> Verd (bo)
        vmin=0,
        vmax=1,
        xticklabels=kernels,
        yticklabels=metodes,
        cbar_kws={'label': 'F1-Score'},
        annot_kws={'size': 10, 'weight': 'bold'}
    )
    
    plt.title("Millor K per a cada combinació Mètode-Kernel\n(Color = F1-Score, Valor = K òptima)", 
              fontsize=14, fontweight='bold', pad=20)
    plt.xlabel("Kernel SVM", fontsize=12)
    plt.ylabel("Mètode d'extracció", fontsize=12)
    
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    
    if output_path is None:
        output_path = os.path.join(directori, 'heatmap_millors_k.png')
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Heatmap guardat a: {output_path}")
    
    return output_path

if __name__ == '__main__':
    visualitzar_tots_els_kernels(os.path.dirname(os.path.abspath(__file__)))
    generar_heatmap_millors_k(os.path.dirname(os.path.abspath(__file__)))