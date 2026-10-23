"""
Módulo 12 — Validação e Teste do Modelo ConvLSTM na Grade de 5 km (Projeto APHIM)
Dissertação de Mestrado: Comparação com Modelo Espaço-Temporal Profundo (ConvLSTM).

Objetivos:
1. Validação Temporal (Dobras 2022 e 2023):
   - Treinar o ConvLSTM com PoissonNLLLoss sobre recortes aleatórios (96x96) das células do Pará.
   - Grade de busca: L=12, 1 camada, hidden em {16, 32}, lr em {1e-3, 3e-4} (4 combinações).
   - Cada combinação roda com 3 sementes (42, 43, 44), com previsão dada pela média das 3.
   - Parada antecipada por perda Poisson na validação (paciência 15 épocas, máx 200 épocas).
   - Métrica de escolha: CapturaÁrea@10% média das dobras 2022 e 2023 (mesmos 15 municípios observados em t).
   - Salvar convlstm_validacao.csv e convlstm_melhor_config.json.
2. Teste Único (Parte C):
   - Retreinar o melhor modelo em todo o conjunto de treino (alvos até 2023-12) com as 3 sementes.
   - Número de épocas fixo = média das épocas de parada das duas dobras.
   - Prever os 24 meses de teste (2024-01 a 2025-12).
   - Salvar convlstm_predicoes_teste.parquet e convlstm_teste_mensal.csv.
   - Comparar contra: G1, Histórico Fixo, Climatologia e Média Simples (G1 + ConvLSTM)/2.
   - Salvar convlstm_teste_resumo.csv (diferença média, IC 95% Bootstrap, Wilcoxon, Holm, vitórias).
   - Gerar mapa comparativo de Altamira (2024-09): convlstm_altamira_2024-09.png.
"""

import os
import sys
import json
import time
import random
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pyproj import Transformer
from scipy.stats import wilcoxon

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.utils.logger import setup_logger

logger = setup_logger("processamento_12_convlstm_validacao")

# ==============================================================================
# 1. ARQUITETURA CONVLSTM
# ==============================================================================

class ConvLSTMCell(nn.Module):
    """Célula individual ConvLSTM (Shi et al., 2015)."""
    def __init__(self, in_channels: int, hidden_dim: int, kernel_size: int = 3):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_channels + hidden_dim,
            4 * hidden_dim,
            kernel_size=kernel_size,
            padding=padding
        )

    def forward(self, x: torch.Tensor, h: torch.Tensor, c: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        combined = torch.cat([x, h], dim=1)
        conv_out = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.chunk(conv_out, 4, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

class ConvLSTMModel(nn.Module):
    """Modelo ConvLSTM de 1 ou mais camadas com saída Poisson Softplus."""
    def __init__(self, in_channels: int, hidden_dim: int, num_layers: int = 1):
        super().__init__()
        self.num_layers = num_layers
        self.cells = nn.ModuleList([
            ConvLSTMCell(in_channels if i == 0 else hidden_dim, hidden_dim)
            for i in range(num_layers)
        ])
        self.head = nn.Conv2d(hidden_dim, 1, kernel_size=1)

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        # x_seq: [B, L, C, H, W]
        B, L, C, H, W = x_seq.shape
        h_list = [torch.zeros(B, cell.hidden_dim, H, W, device=x_seq.device) for cell in self.cells]
        c_list = [torch.zeros(B, cell.hidden_dim, H, W, device=x_seq.device) for cell in self.cells]

        for t in range(L):
            xt = x_seq[:, t]
            for i, cell in enumerate(self.cells):
                h_prev = xt if i == 0 else h_list[i - 1]
                h_list[i], c_list[i] = cell(h_prev, h_list[i], c_list[i])

        out = F.softplus(self.head(h_list[-1]))
        return out.squeeze(1) # [B, H, W]

# ==============================================================================
# 2. FUNÇÕES AUXILIARES E MÉTRICA DE PRIORIZAÇÃO
# ==============================================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def holm_bonferroni(pvals: np.ndarray) -> np.ndarray:
    pvals = np.asarray(pvals, dtype=float)
    n = len(pvals)
    if n == 0:
        return np.array([], dtype=float)
    order = np.argsort(pvals)
    sorted_p = pvals[order]
    adj_p = np.empty(n, dtype=float)
    running_max = 0.0
    for k in range(n):
        val = sorted_p[k] * (n - k)
        running_max = max(running_max, val)
        adj_p[k] = min(1.0, running_max)
    out = np.empty(n, dtype=float)
    out[order] = adj_p
    return out

def bootstrap_diff(a: np.ndarray, b: np.ndarray, n_boot: int = 10000, seed: int = 42):
    rng = np.random.RandomState(seed)
    diffs = a - b
    mean_diff = float(np.mean(diffs))
    boot_means = [np.mean(rng.choice(diffs, size=len(diffs), replace=True)) for _ in range(n_boot)]
    ci_lower = float(np.percentile(boot_means, 2.5))
    ci_upper = float(np.percentile(boot_means, 97.5))
    stat, pval = wilcoxon(a, b)
    return mean_diff, ci_lower, ci_upper, float(pval)

def calcular_captura_area_mes(
    df_mes: pd.DataFrame,
    col_score: str,
    top15_muns: List[str],
    pct_area_k: float = 10.0
) -> float:
    """Calcula a CapturaÁrea@k% para um único mês dentro dos 15 municípios."""
    sub_top15 = df_mes[df_mes["cd_mun"].isin(top15_muns)]
    focos_capturados = 0.0
    focos_top15 = 0.0

    for _, grp in sub_top15.groupby("cd_mun"):
        f_mun = grp["y"].sum()
        focos_top15 += f_mun

        grp_sort = grp.sort_values(by=[col_score, "cell_id"], ascending=[False, True])
        cell_areas = grp_sort["area_km2"].values
        cum_areas = np.cumsum(cell_areas)
        total_mun_area = cum_areas[-1] if len(cum_areas) > 0 else 0.0
        target_area = (pct_area_k / 100.0) * total_mun_area

        mask = (cum_areas - cell_areas) < target_area
        focos_capturados += grp_sort.iloc[mask]["y"].sum()

    cap = (focos_capturados / focos_top15) if focos_top15 > 0 else 0.0
    return float(cap)

# ==============================================================================
# 3. TREINAMENTO COM PARADA ANTECIPADA
# ==============================================================================

def extrair_recortes(
    X_tensor: np.ndarray,
    Y_tensor: np.ndarray,
    mask_pa: np.ndarray,
    idx_meses: List[int],
    L: int,
    crops_per_month: int = 4,
    crop_size: int = 96
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Gera mini-lote de recortes aleatórios (com aumentos de dados) para treino."""
    H, W = mask_pa.shape
    x_crops = []
    y_crops = []
    m_crops = []

    for t_idx in idx_meses:
        seq_x = X_tensor[t_idx - L + 1 : t_idx + 1] # [L, C, H, W]
        target_y = Y_tensor[t_idx] # [H, W]

        attempts = 0
        valid_crops = 0
        while valid_crops < crops_per_month and attempts < 25:
            attempts += 1
            r = random.randint(0, H - crop_size)
            c = random.randint(0, W - crop_size)
            m_sub = mask_pa[r : r + crop_size, c : c + crop_size]
            if m_sub.sum() == 0:
                continue

            x_sub = seq_x[:, :, r : r + crop_size, c : c + crop_size].copy()
            y_sub = target_y[r : r + crop_size, c : c + crop_size].copy()
            m_sub = m_sub.copy()

            # Data Augmentation (flips)
            if random.random() < 0.5:
                x_sub = np.flip(x_sub, axis=-1).copy()
                y_sub = np.flip(y_sub, axis=-1).copy()
                m_sub = np.flip(m_sub, axis=-1).copy()
            if random.random() < 0.5:
                x_sub = np.flip(x_sub, axis=-2).copy()
                y_sub = np.flip(y_sub, axis=-2).copy()
                m_sub = np.flip(m_sub, axis=-2).copy()

            x_crops.append(x_sub)
            y_crops.append(y_sub)
            m_crops.append(m_sub)
            valid_crops += 1

    batch_x = torch.from_numpy(np.stack(x_crops)).float()
    batch_y = torch.from_numpy(np.stack(y_crops)).float()
    batch_m = torch.from_numpy(np.stack(m_crops)).bool()
    return batch_x, batch_y, batch_m

def treinar_convlstm(
    X_tensor: np.ndarray,
    Y_tensor: np.ndarray,
    mask_pa: np.ndarray,
    idx_treino: List[int],
    idx_val: List[int],
    L: int,
    hidden_dim: int,
    num_layers: int,
    lr: float,
    seed: int,
    device: torch.device,
    max_epochs: int = 200,
    patience: int = 15,
    epocas_fixas: int = None
) -> Tuple[ConvLSTMModel, int, float]:
    """
    Treina o modelo ConvLSTM.
    Se epocas_fixas for None: aplica parada antecipada na perda de validação.
    Se epocas_fixas for int: treina exatamente por esse número de épocas (para o teste).
    """
    set_seed(seed)
    in_channels = X_tensor.shape[1]
    model = ConvLSTMModel(in_channels=in_channels, hidden_dim=hidden_dim, num_layers=num_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.PoissonNLLLoss(log_input=False)

    mask_full_torch = torch.from_numpy(mask_pa).bool().to(device)

    # Preparar tensores de validação em GPU para avaliação rápida
    if idx_val is not None and len(idx_val) > 0:
        val_x_list = [
            torch.from_numpy(X_tensor[v - L + 1 : v + 1]).unsqueeze(0).float().to(device)
            for v in idx_val
        ]
        val_y_list = [
            torch.from_numpy(Y_tensor[v]).to(device)
            for v in idx_val
        ]

    best_val_loss = float("inf")
    best_weights = None
    best_epoch = 1
    patience_counter = 0

    target_epochs = epocas_fixas if epocas_fixas is not None else max_epochs

    for epoch in range(1, target_epochs + 1):
        model.train()
        # Embaralhar meses de treino
        random.shuffle(idx_treino)
        # Dividir em mini-lotes de 6 meses (24 crops por lote)
        chunk_size = 6
        for k in range(0, len(idx_treino), chunk_size):
            chunk_m = idx_treino[k : k + chunk_size]
            bx, by, bm = extrair_recortes(X_tensor, Y_tensor, mask_pa, chunk_m, L, crops_per_month=4)
            bx, by, bm = bx.to(device), by.to(device), bm.to(device)

            optimizer.zero_grad()
            preds = model(bx)
            loss = criterion(preds[bm], by[bm])
            loss.backward()
            optimizer.step()

        # Validação para parada antecipada
        if epocas_fixas is None:
            model.eval()
            val_losses = []
            with torch.no_grad():
                for vx, vy in zip(val_x_list, val_y_list):
                    v_pred = model(vx).squeeze(0)
                    v_loss = criterion(v_pred[mask_full_torch], vy[mask_full_torch])
                    val_losses.append(v_loss.item())

            mean_val_loss = float(np.mean(val_losses))
            if mean_val_loss < best_val_loss - 1e-4:
                best_val_loss = mean_val_loss
                best_epoch = epoch
                best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

    if epocas_fixas is None and best_weights is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_weights.items()})
        return model, best_epoch, best_val_loss
    else:
        return model, target_epochs, 0.0

def prever_meses(
    model: ConvLSTMModel,
    X_tensor: np.ndarray,
    idx_meses: List[int],
    L: int,
    device: torch.device
) -> Dict[int, np.ndarray]:
    """Gera predições no mapa completo para a lista de meses."""
    model.eval()
    preds_dict = {}
    with torch.no_grad():
        for m_idx in idx_meses:
            x_seq = torch.from_numpy(X_tensor[m_idx - L + 1 : m_idx + 1]).unsqueeze(0).float().to(device)
            out_map = model(x_seq).squeeze(0).cpu().numpy()
            preds_dict[m_idx] = out_map
    return preds_dict

# ==============================================================================
# 4. EXPERIMENTO PRINCIPAL (VALIDAÇÃO E TESTE)
# ==============================================================================

def main():
    logger.info("=== INICIANDO EXPERIMENTO CONVLSTM NA GRADE DE 5 KM (PARTES B E C) ===")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Dispositivo de execução: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    processed_dir = Path("data/processed")
    processed_deter_dir = Path("data/processed_deter")
    processed_convlstm_dir = Path("data/processed_convlstm")
    out_res_dir = processed_convlstm_dir / "resultados"
    out_res_dir.mkdir(parents=True, exist_ok=True)

    # 1. Carregar tensores da grade
    npz_path = processed_convlstm_dir / "tensores_5000m.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"Arquivo '{npz_path}' não encontrado. Execute 11_convlstm_tensores.py primeiro.")

    logger.info(f"Carregando tensores de '{npz_path}'...")
    data_npz = np.load(npz_path, allow_pickle=True)
    X = data_npz["X"] # [83, 10, 277, 287]
    Y = data_npz["Y"] # [83, 277, 287]
    meses_emissao = list(data_npz["meses_emissao"])
    meses_alvo = list(data_npz["meses_alvo"])
    grid_mask_pa = data_npz["grid_mask_pa"]
    cell_id_arr = data_npz["cell_id"]
    row_norm_arr = data_npz["row_norm"]
    col_norm_arr = data_npz["col_norm"]
    cd_mun_arr = data_npz["cd_mun"]

    # 2. Carregar metadados das células para métrica de priorização
    gpkg_path = processed_dir / "grade_5000m_celulas.gpkg"
    gdf_cells = gpd.read_file(gpkg_path)
    cell_area_map = dict(zip(gdf_cells["cell_id"].astype(np.int32), gdf_cells["area_km2_dentro_pa"].astype(float)))

    # Carregar base de modelagem para mapear focos observados municipais em t (top 15)
    grade_mod_path = processed_dir / "grade_5000m_modelagem.parquet"
    df_mod = pd.read_parquet(grade_mod_path, columns=["cell_id", "mes_emissao", "mes_alvo", "y", "focos_mun_t", "cd_mun"])
    df_mod["cd_mun"] = df_mod["cd_mun"].astype(str).str.zfill(7)
    df_mod["area_km2"] = df_mod["cell_id"].map(cell_area_map).astype(np.float32)

    # 3. Partições de Validação
    L = 12
    # Dobra 2022: Treino alvos 2020-01 a 2021-12 (emissões 2019-12 a 2021-11, idx 11 a 34)
    #             Validação alvos 2022-01 a 2022-12 (emissões 2021-12 a 2022-11, idx 35 a 46)
    idx_tr_22 = [i for i, m in enumerate(meses_alvo) if "2020-01" <= m <= "2021-12"]
    idx_val_22 = [i for i, m in enumerate(meses_alvo) if m.startswith("2022-")]

    # Dobra 2023: Treino alvos 2020-01 a 2022-12 (emissões 2019-12 a 2022-11, idx 11 a 46)
    #             Validação alvos 2023-01 a 2023-12 (emissões 2022-12 a 2023-11, idx 47 a 58)
    idx_tr_23 = [i for i, m in enumerate(meses_alvo) if "2020-01" <= m <= "2022-12"]
    idx_val_23 = [i for i, m in enumerate(meses_alvo) if m.startswith("2023-")]

    # Teste: Treino completo alvos 2020-01 a 2023-12 (emissões 2019-12 a 2023-11, idx 11 a 58)
    #        Teste alvos 2024-01 a 2025-12 (emissões 2023-12 a 2025-11, idx 59 a 82)
    idx_tr_teste = [i for i, m in enumerate(meses_alvo) if "2020-01" <= m <= "2023-12"]
    idx_teste = [i for i, m in enumerate(meses_alvo) if m >= "2024-01"]

    logger.info(f"Partição 2022: {len(idx_tr_22)} meses treino | {len(idx_val_22)} meses validação")
    logger.info(f"Partição 2023: {len(idx_tr_23)} meses treino | {len(idx_val_23)} meses validação")
    logger.info(f"Partição Teste: {len(idx_tr_teste)} meses treino | {len(idx_teste)} meses teste")

    # Mapeamento dos 15 municípios com mais focos observados em t para cada mês
    top15_dict = {}
    for m_alvo, grp in df_mod.groupby("mes_alvo"):
        mun_focos = grp.groupby("cd_mun")["focos_mun_t"].first()
        top15_dict[m_alvo] = mun_focos.sort_values(ascending=False).head(15).index.tolist()

    # 4. Grade de Hiperparâmetros
    out_val_csv = out_res_dir / "convlstm_validacao.csv"
    out_cfg_json = out_res_dir / "convlstm_melhor_config.json"

    # Criar dataframe base das células ativas para avaliação rápida
    df_cells_base = pd.DataFrame({
        "cell_id": cell_id_arr,
        "row_norm": row_norm_arr,
        "col_norm": col_norm_arr,
        "cd_mun": cd_mun_arr
    })
    df_cells_base["area_km2"] = df_cells_base["cell_id"].map(cell_area_map).astype(np.float32)

    if out_val_csv.exists() and out_cfg_json.exists() and "--recomputar" not in sys.argv:
        logger.info(f"Carregando resultados de validação pré-calculados de '{out_val_csv}'...")
        df_val_res = pd.read_csv(out_val_csv)
        with open(out_cfg_json, "r", encoding="utf-8") as f:
            melhor_config = json.load(f)
        melhor_score_val = float(melhor_config["cap_area_10_media"])
    else:
        combinacoes = [
            {"nome": "C1_h16_lr1e3", "L": 12, "camadas": 1, "hidden": 16, "lr": 1e-3},
            {"nome": "C2_h16_lr3e4", "L": 12, "camadas": 1, "hidden": 16, "lr": 3e-4},
            {"nome": "C3_h32_lr1e3", "L": 12, "camadas": 1, "hidden": 32, "lr": 1e-3},
            {"nome": "C4_h32_lr3e4", "L": 12, "camadas": 1, "hidden": 32, "lr": 3e-4},
        ]

        seeds = [42, 43, 44]
        resultados_validacao = []
        logger.info(f"Iniciando busca em grade ({len(combinacoes)} combinações x {len(seeds)} sementes)...")

        melhor_score_val = -1.0
        melhor_config = None

        for cfg in combinacoes:
            t0 = time.time()
            logger.info(f"\n--- Avaliando {cfg['nome']} (hidden={cfg['hidden']}, lr={cfg['lr']}) ---")

        # Avaliar Dobra 2022 com 3 sementes
        preds_val_22_seeds = {m_idx: np.zeros((277, 287), dtype=np.float32) for m_idx in idx_val_22}
        epocas_22_list = []

        for s in seeds:
            m22, ep22, _ = treinar_convlstm(
                X, Y, grid_mask_pa, idx_tr_22, idx_val_22,
                L=cfg["L"], hidden_dim=cfg["hidden"], num_layers=cfg["camadas"],
                lr=cfg["lr"], seed=s, device=device
            )
            epocas_22_list.append(ep22)
            preds_s = prever_meses(m22, X, idx_val_22, L=cfg["L"], device=device)
            for m_idx in idx_val_22:
                preds_val_22_seeds[m_idx] += preds_s[m_idx] / len(seeds)

        # Avaliar Dobra 2023 com 3 sementes
        preds_val_23_seeds = {m_idx: np.zeros((277, 287), dtype=np.float32) for m_idx in idx_val_23}
        epocas_23_list = []

        for s in seeds:
            m23, ep23, _ = treinar_convlstm(
                X, Y, grid_mask_pa, idx_tr_23, idx_val_23,
                L=cfg["L"], hidden_dim=cfg["hidden"], num_layers=cfg["camadas"],
                lr=cfg["lr"], seed=s, device=device
            )
            epocas_23_list.append(ep23)
            preds_s = prever_meses(m23, X, idx_val_23, L=cfg["L"], device=device)
            for m_idx in idx_val_23:
                preds_val_23_seeds[m_idx] += preds_s[m_idx] / len(seeds)

        # Calcular CapturaÁrea@10% média na Dobra 2022
        caps_22 = []
        for m_idx in idx_val_22:
            m_alvo = meses_alvo[m_idx]
            sub_mod = df_mod[df_mod["mes_alvo"] == m_alvo].set_index("cell_id")["y"]
            df_eval = df_cells_base.copy()
            pred_map = preds_val_22_seeds[m_idx]
            df_eval["pred"] = pred_map[df_eval["row_norm"].values, df_eval["col_norm"].values]
            df_eval["y"] = df_eval["cell_id"].map(sub_mod).fillna(0.0)
            c = calcular_captura_area_mes(df_eval, "pred", top15_dict[m_alvo], pct_area_k=10.0)
            caps_22.append(c)
        score_22 = float(np.mean(caps_22))

        # Calcular CapturaÁrea@10% média na Dobra 2023
        caps_23 = []
        for m_idx in idx_val_23:
            m_alvo = meses_alvo[m_idx]
            sub_mod = df_mod[df_mod["mes_alvo"] == m_alvo].set_index("cell_id")["y"]
            df_eval = df_cells_base.copy()
            pred_map = preds_val_23_seeds[m_idx]
            df_eval["pred"] = pred_map[df_eval["row_norm"].values, df_eval["col_norm"].values]
            df_eval["y"] = df_eval["cell_id"].map(sub_mod).fillna(0.0)
            c = calcular_captura_area_mes(df_eval, "pred", top15_dict[m_alvo], pct_area_k=10.0)
            caps_23.append(c)
        score_23 = float(np.mean(caps_23))

        score_medio = (score_22 + score_23) / 2.0
        tempo_s = round(time.time() - t0, 1)

        ep_media_22 = float(np.mean(epocas_22_list))
        ep_media_23 = float(np.mean(epocas_23_list))

        logger.info(
            f"-> {cfg['nome']}: Dobra 2022 = {score_22*100:.3f}% ({ep_media_22:.1f} épocas) | "
            f"Dobra 2023 = {score_23*100:.3f}% ({ep_media_23:.1f} épocas) | "
            f"Média = {score_medio*100:.3f}% | Tempo = {tempo_s}s"
        )

        registro = {
            "configuracao": cfg["nome"],
            "L": cfg["L"],
            "camadas": cfg["camadas"],
            "hidden": cfg["hidden"],
            "lr": cfg["lr"],
            "cap_area_10_2022": score_22,
            "cap_area_10_2023": score_23,
            "cap_area_10_media": score_medio,
            "epocas_2022_media": ep_media_22,
            "epocas_2023_media": ep_media_23,
            "tempo_s": tempo_s
        }
        resultados_validacao.append(registro)

        if score_medio > melhor_score_val:
            melhor_score_val = score_medio
            melhor_config = registro

        df_val_res = pd.DataFrame(resultados_validacao)
        df_val_res.to_csv(out_val_csv, index=False, encoding="utf-8")
        logger.info(f"Tabela de validação salva em '{out_val_csv}'.")

        with open(out_cfg_json, "w", encoding="utf-8") as f:
            json.dump(melhor_config, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80)
    print("      TABELA DE VALIDAÇÃO TEMPORAL - CONVLSTM NA GRADE DE 5 KM")
    print("=" * 80)
    print(df_val_res.to_string(index=False))

    # Carregar referência de validação do G1
    ref_g1_path = processed_deter_dir / "resultados" / "deter_validacao_grade.csv"
    df_ref_g1 = pd.read_csv(ref_g1_path)
    score_g1_val = float(df_ref_g1[df_ref_g1["configuracao"] == "G1"]["cap_area_10_media"].iloc[0])
    diff_val = melhor_score_val - score_g1_val

    print("\n" + "=" * 80)
    print(f"CHECAGEM DE SANIDADE E COMPARAÇÃO NA VALIDAÇÃO:")
    print(f"• G1 (XGBoost + DETER): {score_g1_val*100:.3f}%")
    print(f"• ConvLSTM (Melhor: {melhor_config['configuracao']}): {melhor_score_val*100:.3f}%")
    print(f"• Diferença (ConvLSTM - G1): {diff_val*100:+.3f} p.p. (limiar de vitória: +0.500 p.p.)")
    convlstm_venceu = diff_val >= 0.0050
    if convlstm_venceu:
        print("DECISÃO: ConvLSTM VENCEU NA VALIDAÇÃO (+0.500 p.p.)!")
    else:
        print("DECISÃO: G1 CONTINUA COMO VENCEDOR OFICIAL DA GRADE (ConvLSTM não atingiu +0.500 p.p.).")
    print("=" * 80)

    # ==========================================================================
    # 5. PARTE C: TESTE ÚNICO (AVALIAÇÃO DE BENCHMARK SOLICITADA PELA BANCA)
    # ==========================================================================
    logger.info("=== EXECUTANDO AVALIAÇÃO DE TESTE DO CONVLSTM (PARTE C) ===")
    
    out_parquet = processed_convlstm_dir / "convlstm_predicoes_teste.parquet"
    out_mensal_csv = out_res_dir / "convlstm_teste_mensal.csv"
    out_resumo_csv = out_res_dir / "convlstm_teste_resumo.csv"

    # Carregar previsões oficiais do Nível 1 para fixar os 15 municípios do topo
    l1_path = processed_dir / "resultados" / "features_v2_predicoes_teste.parquet"
    df_pred_l1 = pd.read_parquet(l1_path)
    df_pred_l1["cd_mun"] = df_pred_l1["cd_mun"].astype(str).str.zfill(7)

    # Carregar predições do G1 da Parte 0
    g1_pred_path = processed_convlstm_dir / "g1_predicoes_teste.parquet"
    df_pred_g1 = pd.read_parquet(g1_pred_path)

    if out_parquet.exists() and out_mensal_csv.exists() and out_resumo_csv.exists() and "--recomputar" not in sys.argv:
        logger.info(f"Carregando predições e resultados de teste já salvos de '{out_parquet}'...")
        df_pred_convlstm = pd.read_parquet(out_parquet)
        df_mensal = pd.read_csv(out_mensal_csv)
        df_resumo_teste = pd.read_csv(out_resumo_csv)

        df_teste_completo = df_pred_convlstm.merge(
            df_pred_g1[["cell_id", "mes_alvo", "pred_g1"]],
            on=["cell_id", "mes_alvo"]
        )
        df_teste_completo["pred_ensemble"] = (df_teste_completo["pred_g1"] + df_teste_completo["pred_convlstm"]) / 2.0
        df_teste_completo["area_km2"] = df_teste_completo["cell_id"].map(cell_area_map).astype(np.float32)
    else:
        # Número de épocas fixo para o retreino: média das épocas das duas dobras
        epocas_retreino = int(round((melhor_config["epocas_2022_media"] + melhor_config["epocas_2023_media"]) / 2.0))
        epocas_retreino = max(epocas_retreino, 10)
        logger.info(f"Retreinando melhor configuração ({melhor_config['configuracao']}) por {epocas_retreino} épocas...")

        preds_teste_seeds = {m_idx: np.zeros((277, 287), dtype=np.float32) for m_idx in idx_teste}

        for s in seeds:
            logger.info(f"Treinando semente {s} em todo o treino da grade...")
            m_teste, _, _ = treinar_convlstm(
                X, Y, grid_mask_pa, idx_tr_teste, None,
                L=melhor_config["L"], hidden_dim=melhor_config["hidden"], num_layers=melhor_config["camadas"],
                lr=melhor_config["lr"], seed=s, device=device,
                epocas_fixas=epocas_retreino
            )
            preds_s = prever_meses(m_teste, X, idx_teste, L=melhor_config["L"], device=device)
            for m_idx in idx_teste:
                preds_teste_seeds[m_idx] += preds_s[m_idx] / len(seeds)

        # Montar predições de teste para as células do Pará
        registros_pred_teste = []
        for m_idx in idx_teste:
            m_emissao = meses_emissao[m_idx]
            m_alvo = meses_alvo[m_idx]
            pred_map = preds_teste_seeds[m_idx]

            sub_mod = df_mod[df_mod["mes_alvo"] == m_alvo].set_index("cell_id")["y"]
            df_t = df_cells_base.copy()
            df_t["mes_emissao"] = m_emissao
            df_t["mes_alvo"] = m_alvo
            df_t["pred_convlstm"] = pred_map[df_t["row_norm"].values, df_t["col_norm"].values]
            df_t["y"] = df_t["cell_id"].map(sub_mod).fillna(0.0).astype(np.float32)

            registros_pred_teste.append(df_t[["cell_id", "cd_mun", "mes_emissao", "mes_alvo", "y", "pred_convlstm"]])

        df_pred_convlstm = pd.concat(registros_pred_teste, ignore_index=True)
        df_pred_convlstm.to_parquet(out_parquet, index=False)
        logger.info(f"Predições de teste do ConvLSTM salvas em '{out_parquet}'.")

        # Mesclar predições de G1 e ConvLSTM para avaliação pareada
        df_teste_completo = df_pred_convlstm.merge(
            df_pred_g1[["cell_id", "mes_alvo", "pred_g1"]],
            on=["cell_id", "mes_alvo"]
        )
        df_teste_completo["pred_ensemble"] = (df_teste_completo["pred_g1"] + df_teste_completo["pred_convlstm"]) / 2.0
        df_teste_completo["area_km2"] = df_teste_completo["cell_id"].map(cell_area_map).astype(np.float32)

        # Avaliar CapturaÁrea@5%, @10%, @20% por mês para ConvLSTM
        meses_teste_lista = sorted(df_teste_completo["mes_alvo"].unique())
        linhas_mensais = []

        for m_alvo in meses_teste_lista:
            sub_l1 = df_pred_l1[df_pred_l1["mes_alvo"] == m_alvo].sort_values(
                by=["pred_vencedor", "cd_mun"], ascending=[False, True]
            )
            top15_muns = sub_l1["cd_mun"].head(15).tolist()

            df_m = df_teste_completo[df_teste_completo["mes_alvo"] == m_alvo]

            cap5 = calcular_captura_area_mes(df_m, "pred_convlstm", top15_muns, pct_area_k=5.0)
            cap10 = calcular_captura_area_mes(df_m, "pred_convlstm", top15_muns, pct_area_k=10.0)
            cap20 = calcular_captura_area_mes(df_m, "pred_convlstm", top15_muns, pct_area_k=20.0)

            # Também calcular para G1 e Ensemble a @10%
            cap10_g1 = calcular_captura_area_mes(df_m, "pred_g1", top15_muns, pct_area_k=10.0)
            cap10_ens = calcular_captura_area_mes(df_m, "pred_ensemble", top15_muns, pct_area_k=10.0)

            linhas_mensais.append({
                "mes_alvo": m_alvo,
                "cap_area_5_convlstm": cap5,
                "cap_area_10_convlstm": cap10,
                "cap_area_20_convlstm": cap20,
                "cap_area_10_g1": cap10_g1,
                "cap_area_10_ensemble": cap10_ens
            })

        df_mensal = pd.DataFrame(linhas_mensais)
        df_mensal.to_csv(out_mensal_csv, index=False, encoding="utf-8")
        logger.info(f"Resultados mensais de teste salvos em '{out_mensal_csv}'.")

        # Comparação Estatística Pareada no Teste
        deter_mensal_path = processed_deter_dir / "resultados" / "deter_grade_mensal.csv"
        df_deter_m = pd.read_csv(deter_mensal_path)
        caps_fixo = df_deter_m[df_deter_m["metodo"] == "historico_fixo_celula"].sort_values("mes_alvo")["captura_area_10"].values
        caps_clim = df_deter_m[df_deter_m["metodo"] == "climatologia_celula"].sort_values("mes_alvo")["captura_area_10"].values

        caps_convlstm = df_mensal.sort_values("mes_alvo")["cap_area_10_convlstm"].values
        caps_g1 = df_mensal.sort_values("mes_alvo")["cap_area_10_g1"].values
        caps_ens = df_mensal.sort_values("mes_alvo")["cap_area_10_ensemble"].values

        comparadores = [
            ("G1_xgb_deter", caps_g1),
            ("historico_fixo_celula", caps_fixo),
            ("climatologia_celula", caps_clim),
            ("ensemble_g1_convlstm", caps_ens)
        ]

        resumo_linhas = []
        pvals_raw = []

        for nome_comp, vals_comp in comparadores:
            diff_med, ci_inf, ci_sup, p_val = bootstrap_diff(caps_convlstm, vals_comp)
            vitorias = int(np.sum(caps_convlstm > vals_comp))
            pvals_raw.append(p_val)

            resumo_linhas.append({
                "comparador": nome_comp,
                "cap_area_10_convlstm": float(np.mean(caps_convlstm)),
                "cap_area_10_comparador": float(np.mean(vals_comp)),
                "diff_media": diff_med,
                "ic95_inferior": ci_inf,
                "ic95_superior": ci_sup,
                "p_wilcoxon": p_val,
                "vitorias_convlstm": vitorias,
                "total_meses": 24
            })

        pvals_holm = holm_bonferroni(np.array(pvals_raw[:3])) # Holm nos 3 comparadores de hipótese
        for idx, ph in enumerate(pvals_holm):
            resumo_linhas[idx]["p_holm"] = ph
        resumo_linhas[3]["p_holm"] = np.nan # informativo apenas

        df_resumo_teste = pd.DataFrame(resumo_linhas)
        df_resumo_teste.to_csv(out_resumo_csv, index=False, encoding="utf-8")
        logger.info(f"Resumo estatístico do teste salvo em '{out_resumo_csv}'.")


    print("\n" + "=" * 80)
    print("      BENCHMARK ESTATÍSTICO NO TESTE - CONVLSTM NA GRADE DE 5 KM")
    print("=" * 80)
    print(df_resumo_teste.to_string(index=False))

    # ==========================================================================
    # 6. MAPA CARTOGRÁFICO DE ALTAMIRA (2024-09): G1 vs CONVLSTM
    # ==========================================================================
    logger.info("Gerando mapa cartográfico de Altamira (2024-09)...")
    cd_altamira = "1500602"
    mes_exemplo = "2024-09"

    mun_gpkg_path = processed_dir / "municipios_pa.gpkg"
    gdf_mun = gpd.read_file(mun_gpkg_path).to_crs(epsg=5880)
    gdf_mun["cd_mun"] = gdf_mun["cd_mun"].astype(str).str.zfill(7)
    geom_altamira = gdf_mun[gdf_mun["cd_mun"] == cd_altamira].copy()

    # Células de Altamira
    df_alt_m = df_teste_completo[
        (df_teste_completo["mes_alvo"] == mes_exemplo) &
        (df_teste_completo["cd_mun"] == cd_altamira)
    ].copy()

    gdf_alt_cells = gdf_cells[gdf_cells["cell_id"].isin(df_alt_m["cell_id"])].copy()
    gdf_alt_cells = gdf_alt_cells.merge(
        df_alt_m[["cell_id", "pred_g1", "pred_convlstm", "y", "area_km2"]],
        on="cell_id"
    )

    # Identificar top 10% de área para G1
    gdf_alt_cells = gdf_alt_cells.sort_values(by=["pred_g1", "cell_id"], ascending=[False, True]).reset_index(drop=True)
    cum_a = gdf_alt_cells["area_km2"].cumsum().values
    tot_a = cum_a[-1]
    gdf_alt_cells["top10_g1"] = (cum_a - gdf_alt_cells["area_km2"].values) < (0.10 * tot_a)

    # Identificar top 10% de área para ConvLSTM
    gdf_alt_cells = gdf_alt_cells.sort_values(by=["pred_convlstm", "cell_id"], ascending=[False, True]).reset_index(drop=True)
    cum_a_c = gdf_alt_cells["area_km2"].cumsum().values
    gdf_alt_cells["top10_convlstm"] = (cum_a_c - gdf_alt_cells["area_km2"].values) < (0.10 * tot_a)

    # Focos reais VIIRS
    focos_parquet_path = Path("data/interim/firms_focos_pa.parquet")
    df_focos = pd.read_parquet(focos_parquet_path, columns=["latitude", "longitude", "acq_date", "cd_mun"])
    df_focos["cd_mun"] = df_focos["cd_mun"].astype(str).str.zfill(7)
    df_focos_alt = df_focos[
        (df_focos["cd_mun"] == cd_altamira) &
        (df_focos["acq_date"].astype(str).str.startswith(mes_exemplo))
    ].copy()

    trans = Transformer.from_crs("EPSG:4326", "EPSG:5880", always_xy=True)
    xs, ys = trans.transform(df_focos_alt["longitude"].values, df_focos_alt["latitude"].values)
    gdf_focos_pts = gpd.GeoDataFrame(geometry=gpd.points_from_xy(xs, ys), crs="EPSG:5880")

    focos_obs = len(gdf_focos_pts)
    focos_cap_g1 = gdf_alt_cells[gdf_alt_cells["top10_g1"]]["y"].sum()
    pct_g1 = (focos_cap_g1 / focos_obs * 100.0) if focos_obs > 0 else 0.0

    focos_cap_conv = gdf_alt_cells[gdf_alt_cells["top10_convlstm"]]["y"].sum()
    pct_conv = (focos_cap_conv / focos_obs * 100.0) if focos_obs > 0 else 0.0

    # Plotagem lado a lado
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(1, 2, figsize=(18, 11), dpi=300)

    # Escala de cores comum
    vmax = max(float(np.percentile(gdf_alt_cells["pred_g1"], 99)), float(np.percentile(gdf_alt_cells["pred_convlstm"], 99)), 10.0)
    norm = mcolors.PowerNorm(gamma=0.5, vmin=0, vmax=vmax)

    # Painel 1: G1 (XGBoost DETER)
    ax1 = axes[0]
    geom_altamira.plot(ax=ax1, color="#f8f9fa", edgecolor="#343a40", linewidth=1.5, zorder=1)
    gdf_alt_cells.plot(
        column="pred_g1", ax=ax1, cmap="YlOrRd", norm=norm,
        edgecolor="#ced4da", linewidth=0.2, legend=True,
        legend_kwds={"label": "Focos Previstos (G1 XGBoost DETER)", "orientation": "horizontal", "shrink": 0.7, "pad": 0.03},
        zorder=2
    )
    gdf_alt_cells[gdf_alt_cells["top10_g1"]].plot(ax=ax1, facecolor="none", edgecolor="#004085", linewidth=1.1, zorder=3)
    if len(gdf_focos_pts) > 0:
        gdf_focos_pts.plot(ax=ax1, color="#212529", markersize=2.5, alpha=0.6, zorder=4)

    info_g1 = (
        f"G1 (XGBoost + DETER)\n"
        f"• Total focos observados: {focos_obs:,}\n"
        f"• Top 10% da área: {gdf_alt_cells['top10_g1'].sum()} células (borda azul)\n"
        f"• Focos capturados: {focos_cap_g1:,.0f} ({pct_g1:.1f}%)\n"
        f"• Pontos pretos: focos VIIRS"
    )
    ax1.text(0.03, 0.03, info_g1, transform=ax1.transAxes, fontsize=10, bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="#adb5bd", alpha=0.92), zorder=5)
    ax1.set_title("G1 — XGBoost Poisson + Alertas DETER", fontsize=13, fontweight="bold", pad=10)
    ax1.set_axis_off()

    # Painel 2: ConvLSTM
    ax2 = axes[1]
    geom_altamira.plot(ax=ax2, color="#f8f9fa", edgecolor="#343a40", linewidth=1.5, zorder=1)
    gdf_alt_cells.plot(
        column="pred_convlstm", ax=ax2, cmap="YlOrRd", norm=norm,
        edgecolor="#ced4da", linewidth=0.2, legend=True,
        legend_kwds={"label": "Focos Previstos (ConvLSTM)", "orientation": "horizontal", "shrink": 0.7, "pad": 0.03},
        zorder=2
    )
    gdf_alt_cells[gdf_alt_cells["top10_convlstm"]].plot(ax=ax2, facecolor="none", edgecolor="#004085", linewidth=1.1, zorder=3)
    if len(gdf_focos_pts) > 0:
        gdf_focos_pts.plot(ax=ax2, color="#212529", markersize=2.5, alpha=0.6, zorder=4)

    info_conv = (
        f"ConvLSTM Espaço-Temporal\n"
        f"• Total focos observados: {focos_obs:,}\n"
        f"• Top 10% da área: {gdf_alt_cells['top10_convlstm'].sum()} células (borda azul)\n"
        f"• Focos capturados: {focos_cap_conv:,.0f} ({pct_conv:.1f}%)\n"
        f"• Pontos pretos: focos VIIRS"
    )
    ax2.text(0.03, 0.03, info_conv, transform=ax2.transAxes, fontsize=10, bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="#adb5bd", alpha=0.92), zorder=5)
    ax2.set_title("ConvLSTM — Modelo Espaço-Temporal Profundo", fontsize=13, fontweight="bold", pad=10)
    ax2.set_axis_off()

    fig.suptitle("APHIM — Comparação de Priorização Espacial na Grade de 5 km\nAltamira (PA) — Setembro de 2024", fontsize=15, fontweight="bold", y=0.98)
    plt.tight_layout()

    out_png_path = out_res_dir / "convlstm_altamira_2024-09.png"
    plt.savefig(out_png_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Mapa comparativo de Altamira salvo em '{out_png_path}'.")

    logger.info("=== PROCESSO CONCLUÍDO COM SUCESSO! ===")

if __name__ == "__main__":
    main()
