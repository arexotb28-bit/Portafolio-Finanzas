from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import f as f_distribution
from scipy.stats import t as t_distribution


# ============================================================
# HERRAMIENTA FINANCIERA - AVANCE MODULOS 1, 2 Y 3 - FINANZAS I
# Un solo archivo:
# - Limpieza de base Refinitiv
# - Modulo 1: riesgo, rentabilidad y analisis comparativo
# - Modulo 2: optimizacion de portafolios, perfiles y aversion al riesgo
# - Modulo 3: CAPM, beta OLS, SML y alfa de Jensen
#
# Dependencias:
# python -m pip install pandas numpy openpyxl streamlit plotly
#
# Ejecucion:
# streamlit run herramienta_finanzas_completa.py
# ============================================================


DEFAULT_DATA_PATHS = [
    Path("base_diaria.xlsx"),
    Path("datos") / "base_diaria.xlsx",
    Path(__file__).resolve().parent / "base_diaria.xlsx",
    Path(__file__).resolve().parent / "datos" / "base_diaria.xlsx",
]

BENCHMARK_TICKER = "SPX"
TRADING_DAYS = 252


RISK_PROFILES = {
    "Conservador": {
        "gamma": 9.0,
        "max_vol": 0.12,
        "description": "Prioriza preservación de capital y baja volatilidad. Se apoya en alta aversión al riesgo y menor tolerancia a caídas.",
        "basis": "Adecuado cuando la capacidad de riesgo o el horizonte son bajos; en media-varianza penaliza con fuerza la varianza.",
    },
    "Moderado": {
        "gamma": 5.0,
        "max_vol": 0.18,
        "description": "Busca equilibrio entre crecimiento y estabilidad. Acepta volatilidad intermedia si mejora el retorno esperado.",
        "basis": "Perfil balanceado: combina tolerancia psicológica y capacidad financiera medias.",
    },
    "Crecimiento": {
        "gamma": 3.0,
        "max_vol": 0.25,
        "description": "Acepta fluctuaciones relevantes para capturar mayor retorno esperado de largo plazo.",
        "basis": "Perfil con mayor horizonte/capacidad de riesgo; la penalizacion por varianza es menor.",
    },
    "Agresivo": {
        "gamma": 1.5,
        "max_vol": 0.40,
        "description": "Maximiza crecimiento esperado y tolera caídas amplias en el camino.",
        "basis": "Perfil de baja aversión al riesgo: la función de utilidad permite mayor volatilidad por retorno adicional.",
    },
}


@dataclass(frozen=True)
class MarketData:
    prices: pd.DataFrame
    benchmark: pd.DataFrame
    risk_free: pd.DataFrame
    metadata: pd.DataFrame


def find_data_file() -> Path | None:
    """Busca el Excel en rutas relativas para mantener el proyecto reproducible."""
    for path in DEFAULT_DATA_PATHS:
        if path.exists():
            return path.resolve()
    return None


def file_fingerprint(path: Path) -> tuple[int, int]:
    """Devuelve tamaño y fecha de modificación; ambos invalidan el caché al guardar Excel."""
    stat = path.stat()
    return stat.st_size, stat.st_mtime_ns


def format_file_update(path: Path) -> str:
    """Convierte la fecha del archivo a una etiqueta legible para la barra lateral."""
    modified = datetime.fromtimestamp(path.stat().st_mtime).astimezone()
    return modified.strftime("%d/%m/%Y %H:%M:%S")


def normalize_name(value: object) -> str:
    """Normaliza etiquetas técnicas sin modificar los nombres visibles de los activos."""
    return str(value).strip().upper()


def load_asset_metadata(path_or_buffer) -> pd.DataFrame:
    """Lee ACTIVOS y devuelve ticker, RIC, grupo y tipo depurados."""
    raw = pd.read_excel(path_or_buffer, sheet_name="ACTIVOS", header=1, usecols="A:D")
    raw = raw.dropna(subset=["Ticker", "RIC"])
    raw.columns = ["Ticker", "RIC", "Grupo", "Tipo"]
    raw["Ticker"] = raw["Ticker"].map(normalize_name)
    raw["RIC"] = raw["RIC"].map(lambda x: str(x).strip())
    raw["Grupo"] = raw["Grupo"].fillna("Sin clasificar").astype(str).str.strip()
    raw["Tipo"] = raw["Tipo"].fillna("Sin clasificar").astype(str).str.strip()
    return raw.drop_duplicates("Ticker").reset_index(drop=True)


def read_refinitiv_prices(path_or_buffer, sheet_name: str) -> pd.DataFrame:
    """Limpia fechas, encabezados y precios de una hoja Refinitiv."""
    raw = pd.read_excel(path_or_buffer, sheet_name=sheet_name, header=8)
    raw = raw.dropna(how="all").dropna(axis=1, how="all")
    raw = raw.rename(columns={raw.columns[0]: "Date"})
    raw["Date"] = pd.to_datetime(raw["Date"], errors="coerce", dayfirst=True)
    raw = raw.dropna(subset=["Date"]).drop_duplicates("Date").sort_values("Date")
    raw = raw.set_index("Date")
    raw.columns = [str(c).strip() for c in raw.columns]
    for col in raw.columns:
        raw[col] = pd.to_numeric(raw[col], errors="coerce")
    return raw.replace(0, np.nan).ffill().dropna(axis=1, how="all")


def load_prices(path_or_buffer) -> pd.DataFrame:
    """Carga BASE FINAL, conserva tickers originales y devuelve precios por fecha."""
    metadata = load_asset_metadata(path_or_buffer)
    ric_to_ticker = dict(zip(metadata["RIC"], metadata["Ticker"]))
    prices = read_refinitiv_prices(path_or_buffer, "BASE FINAL")
    prices = prices.rename(columns=ric_to_ticker)
    prices.columns = [normalize_name(c) for c in prices.columns]
    return prices.loc[:, ~prices.columns.duplicated()]


def load_benchmark(path_or_buffer) -> pd.DataFrame:
    """Carga BENCHMARK y devuelve el cierre histórico identificado como SPX."""
    benchmark = read_refinitiv_prices(path_or_buffer, "BENCHMARK")
    first_col = benchmark.columns[0]
    return benchmark[[first_col]].rename(columns={first_col: BENCHMARK_TICKER})


def load_risk_free(path_or_buffer) -> pd.DataFrame:
    """Carga T-BILL y devuelve tasas anual y diaria alineables por fecha."""
    rf = pd.read_excel(
        path_or_buffer,
        sheet_name="T-BILL",
        skiprows=5,
        header=None,
        usecols="B:C",
        names=["Date", "rf_annual_pct"],
    )
    rf["Date"] = pd.to_datetime(rf["Date"], errors="coerce", dayfirst=True)
    rf["rf_annual_pct"] = pd.to_numeric(rf["rf_annual_pct"], errors="coerce")
    rf = rf.dropna(subset=["Date", "rf_annual_pct"])
    rf = rf.drop_duplicates("Date").sort_values("Date").set_index("Date")
    rf = rf[(rf["rf_annual_pct"] >= 0) & (rf["rf_annual_pct"] <= 25)]
    rf["rf_annual"] = rf["rf_annual_pct"] / 100
    rf["rf_daily"] = (1 + rf["rf_annual"]) ** (1 / TRADING_DAYS) - 1
    return rf


def load_market_data(path_or_buffer) -> MarketData:
    """Agrupa activos, precios, benchmark y T-Bill ya limpios para los módulos."""
    return MarketData(
        prices=load_prices(path_or_buffer),
        benchmark=load_benchmark(path_or_buffer),
        risk_free=load_risk_free(path_or_buffer),
        metadata=load_asset_metadata(path_or_buffer),
    )


def filter_dates(df: pd.DataFrame, start=None, end=None) -> pd.DataFrame:
    out = df.copy()
    if start is not None:
        out = out.loc[out.index >= pd.to_datetime(start)]
    if end is not None:
        out = out.loc[out.index <= pd.to_datetime(end)]
    return out


def resample_prices(prices: pd.DataFrame, frequency: str) -> tuple[pd.DataFrame, int]:
    frequency = frequency.lower()
    if frequency.startswith("di"):
        return prices.dropna(how="all"), 252
    if frequency.startswith("se"):
        return prices.resample("W-FRI").last().dropna(how="all"), 52
    if frequency.startswith("me"):
        return prices.resample("ME").last().dropna(how="all"), 12
    raise ValueError("Frecuencia inválida. Usa Diaria, Semanal o Mensual.")


def calculate_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)


def align_risk_free(rf: pd.DataFrame, index: pd.Index, annual_factor: int) -> pd.Series:
    annual = rf["rf_annual"].reindex(index, method="ffill")
    return (1 + annual) ** (1 / annual_factor) - 1


def annualize_geometric(returns: pd.Series, annual_factor: int) -> float:
    r = returns.dropna()
    if r.empty:
        return np.nan
    growth = (1 + r).prod()
    if growth <= 0:
        return np.nan
    return growth ** (annual_factor / len(r)) - 1


def max_drawdown_from_prices(prices: pd.Series) -> float:
    p = prices.dropna()
    if p.empty:
        return np.nan
    wealth = p / p.iloc[0]
    return (wealth / wealth.cummax() - 1).min()


def safe_divide(num: float, den: float) -> float:
    if den == 0 or pd.isna(den):
        return np.nan
    return num / den


def metadata_lookup(metadata: pd.DataFrame, ticker: str, field: str) -> str:
    ticker = normalize_name(ticker)
    row = metadata.loc[metadata["Ticker"] == ticker]
    if row.empty:
        return "Benchmark" if ticker == BENCHMARK_TICKER else "Sin clasificar"
    return str(row.iloc[0][field])


# ============================================================
# MODULO 1
# ============================================================


def beta_against_market(asset_returns: pd.Series, market_returns: pd.Series) -> float:
    data = pd.concat([asset_returns, market_returns], axis=1).dropna()
    if len(data) < 5:
        return np.nan
    return safe_divide(data.iloc[:, 0].cov(data.iloc[:, 1]), data.iloc[:, 1].var(ddof=1))


def downside_deviation(excess_returns: pd.Series, annual_factor: int) -> float:
    downside = excess_returns.dropna()
    downside = downside[downside < 0]
    if len(downside) < 2:
        return np.nan
    return downside.std(ddof=1) * np.sqrt(annual_factor)


def omega_ratio(excess_returns: pd.Series, threshold: float = 0.0) -> float:
    x = excess_returns.dropna() - threshold
    return safe_divide(x[x > 0].sum(), -x[x < 0].sum())


def metrics_table(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    rf_period: pd.Series,
    benchmark: str,
    metadata: pd.DataFrame,
    annual_factor: int,
) -> pd.DataFrame:
    rows = []
    market_returns = returns[benchmark]
    for asset in returns.columns:
        r = returns[asset].dropna()
        aligned = pd.concat([returns[asset].rename("asset"), rf_period.rename("rf")], axis=1).dropna()
        excess = aligned["asset"] - aligned["rf"]

        ret_ann = annualize_geometric(returns[asset], annual_factor)
        mean_period = r.mean() if not r.empty else np.nan
        var_period = r.var(ddof=1) if len(r) > 1 else np.nan
        std_period = r.std(ddof=1) if len(r) > 1 else np.nan
        mean_ann = r.mean() * annual_factor if not r.empty else np.nan
        vol_ann = r.std(ddof=1) * np.sqrt(annual_factor) if len(r) > 1 else np.nan
        var_ann = var_period * annual_factor if not pd.isna(var_period) else np.nan
        excess_ann = excess.mean() * annual_factor if not excess.empty else np.nan
        excess_vol_ann = excess.std(ddof=1) * np.sqrt(annual_factor) if len(excess) > 1 else np.nan
        dd = max_drawdown_from_prices(prices[asset]) if asset in prices else np.nan
        var_95 = r.quantile(0.05) if not r.empty else np.nan
        cvar_95 = r[r <= var_95].mean() if not r.empty else np.nan

        rows.append(
            {
                "Activo": asset,
                "Tipo": metadata_lookup(metadata, asset, "Tipo"),
                "Grupo": metadata_lookup(metadata, asset, "Grupo"),
                "Retorno anualizado": ret_ann,
                "Media periodo": mean_period,
                "Varianza periodo": var_period,
                "Desviacion periodo": std_period,
                "Retorno medio anualizado": mean_ann,
                "Varianza anualizada": var_ann,
                "Volatilidad anualizada": vol_ann,
                "Sharpe Ratio": safe_divide(excess_ann, excess_vol_ann),
                "Sortino Ratio": safe_divide(excess_ann, downside_deviation(excess, annual_factor)),
                "Max Drawdown": dd,
                "Calmar Ratio": safe_divide(ret_ann, abs(dd)),
                "Beta": 1.0 if asset == benchmark else beta_against_market(returns[asset], market_returns),
                "VaR 95% periodo": var_95,
                "CVaR 95% periodo": cvar_95,
                "Omega Ratio": omega_ratio(excess),
                "Observaciones": int(r.count()),
            }
        )
    return pd.DataFrame(rows).sort_values("Activo").reset_index(drop=True)


def run_modulo1(data: MarketData, start=None, end=None, frequency: str = "Diaria", benchmark: str = BENCHMARK_TICKER):
    benchmark = benchmark.strip().upper() or BENCHMARK_TICKER
    prices = filter_dates(data.prices, start, end)
    bench = filter_dates(data.benchmark, start, end)
    combined_prices = prices.join(bench, how="inner")
    combined_prices, annual_factor = resample_prices(combined_prices, frequency)

    if benchmark not in combined_prices.columns:
        raise ValueError(f"No se encontro el benchmark {benchmark}. Columnas disponibles: {list(combined_prices.columns)}")

    returns = calculate_returns(combined_prices).dropna(how="all")
    rf_period = align_risk_free(data.risk_free, returns.index, annual_factor)
    indicators = metrics_table(combined_prices, returns, rf_period, benchmark, data.metadata, annual_factor)

    return {
        "prices": combined_prices,
        "returns": returns,
        "rf_period": rf_period,
        "indicators": indicators,
        "metadata": data.metadata,
        "annual_factor": annual_factor,
        "benchmark": benchmark,
    }


# ============================================================
# MODULO 2
# ============================================================


def project_to_simplex(values: np.ndarray) -> np.ndarray:
    v = np.asarray(values, dtype=float)
    n = len(v)
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - 1
    ind = np.arange(1, n + 1)
    cond = u - cssv / ind > 0
    if not np.any(cond):
        return np.repeat(1 / n, n)
    theta = cssv[cond][-1] / ind[cond][-1]
    return np.maximum(v - theta, 0)


def portfolio_metrics(weights: np.ndarray, mu: pd.Series, cov: pd.DataFrame, rf_annual: float) -> tuple[float, float, float]:
    w = np.asarray(weights, dtype=float)
    ret = float(w @ mu.values)
    var = float(w.T @ cov.values @ w)
    vol = float(np.sqrt(max(var, 0)))
    sharpe = np.nan if vol <= 0 else (ret - rf_annual) / vol
    return ret, vol, sharpe


def optimize_min_variance(cov: pd.DataFrame, max_iter: int = 4000) -> np.ndarray:
    n = cov.shape[0]
    w = np.repeat(1 / n, n)
    sigma = cov.values
    step = 1 / (2 * max(np.linalg.norm(sigma, ord=2), 1e-8))
    for _ in range(max_iter):
        new_w = project_to_simplex(w - step * (2 * sigma @ w))
        if np.linalg.norm(new_w - w) < 1e-10:
            break
        w = new_w
    return w


def optimize_max_sharpe(mu: pd.Series, cov: pd.DataFrame, rf_annual: float, max_iter: int = 5000) -> np.ndarray:
    n = len(mu)
    w = np.repeat(1 / n, n)
    sigma = cov.values
    excess_mu = mu.values - rf_annual
    step = 0.05

    for i in range(max_iter):
        port_var = max(float(w.T @ sigma @ w), 1e-12)
        port_vol = np.sqrt(port_var)
        port_excess = float(w @ excess_mu)
        grad = excess_mu / port_vol - (port_excess * (sigma @ w)) / (port_vol**3)
        new_w = project_to_simplex(w + step * grad)
        old_sharpe = portfolio_metrics(w, mu, cov, rf_annual)[2]
        new_sharpe = portfolio_metrics(new_w, mu, cov, rf_annual)[2]
        if pd.isna(new_sharpe) or new_sharpe < old_sharpe:
            step *= 0.5
            if step < 1e-8:
                break
            continue
        if np.linalg.norm(new_w - w) < 1e-10:
            break
        w = new_w
        if i % 200 == 0:
            step = min(step * 1.05, 0.10)
    return w


def optimize_target_return(mu: pd.Series, cov: pd.DataFrame, target_return: float, max_iter: int = 3500) -> np.ndarray:
    n = len(mu)
    w = np.repeat(1 / n, n)
    sigma = cov.values
    mu_values = mu.values
    penalty = 200.0
    step = 1 / (2 * max(np.linalg.norm(sigma, ord=2), 1e-8) + penalty * max(float(mu_values @ mu_values), 1e-8))
    for _ in range(max_iter):
        gap = float(w @ mu_values - target_return)
        grad = 2 * sigma @ w + 2 * penalty * gap * mu_values
        new_w = project_to_simplex(w - step * grad)
        if np.linalg.norm(new_w - w) < 1e-10:
            break
        w = new_w
    return w


def optimize_mean_variance_utility(mu: pd.Series, cov: pd.DataFrame, risk_aversion: float, max_iter: int = 5000) -> np.ndarray:
    n = len(mu)
    w = np.repeat(1 / n, n)
    sigma = cov.values
    mu_values = mu.values
    risk_aversion = max(float(risk_aversion), 1e-6)
    step = 1 / max(risk_aversion * np.linalg.norm(sigma, ord=2), 1e-8)
    step = min(step, 0.25)
    for _ in range(max_iter):
        grad = mu_values - risk_aversion * (sigma @ w)
        new_w = project_to_simplex(w + step * grad)
        old_utility = float(w @ mu_values - 0.5 * risk_aversion * (w.T @ sigma @ w))
        new_utility = float(new_w @ mu_values - 0.5 * risk_aversion * (new_w.T @ sigma @ new_w))
        if new_utility + 1e-12 < old_utility:
            step *= 0.5
            if step < 1e-9:
                break
            continue
        if np.linalg.norm(new_w - w) < 1e-10:
            break
        w = new_w
    return w


def concentration_index(weights: np.ndarray) -> float:
    w = np.asarray(weights, dtype=float)
    return float(np.sum(w**2))


def simulate_portfolios(mu: pd.Series, cov: pd.DataFrame, rf_annual: float, n_portfolios: int = 12000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    assets = mu.index.tolist()
    rows = []
    for i in range(n_portfolios):
        w = rng.dirichlet(np.ones(len(assets)))
        ret, vol, sharpe = portfolio_metrics(w, mu, cov, rf_annual)
        row = {"Portafolio": i + 1, "Retorno anualizado": ret, "Volatilidad anualizada": vol, "Sharpe Ratio": sharpe}
        row.update({f"Peso_{asset}": weight for asset, weight in zip(assets, w)})
        rows.append(row)
    return pd.DataFrame(rows)


def efficient_frontier_from_simulation(simulated: pd.DataFrame, points: int = 80) -> pd.DataFrame:
    df = simulated.dropna(subset=["Retorno anualizado", "Volatilidad anualizada"]).copy()
    bins = np.linspace(df["Volatilidad anualizada"].min(), df["Volatilidad anualizada"].max(), points + 1)
    rows = []
    for low, high in zip(bins[:-1], bins[1:]):
        bucket = df[(df["Volatilidad anualizada"] >= low) & (df["Volatilidad anualizada"] < high)]
        if not bucket.empty:
            rows.append(bucket.loc[bucket["Retorno anualizado"].idxmax()])
    frontier = pd.DataFrame(rows).sort_values("Volatilidad anualizada")
    if frontier.empty:
        return frontier
    frontier = frontier[frontier["Retorno anualizado"].cummax() <= frontier["Retorno anualizado"] + 1e-12]
    return frontier.reset_index(drop=True)


def efficient_frontier_optimized(mu: pd.Series, cov: pd.DataFrame, rf_annual: float, points: int = 60) -> pd.DataFrame:
    min_w = optimize_min_variance(cov)
    min_ret, min_vol, min_sharpe = portfolio_metrics(min_w, mu, cov, rf_annual)
    max_ret = float(mu.max())
    targets = np.linspace(min_ret, max_ret, points)
    rows = []
    for target in targets:
        w = optimize_target_return(mu, cov, target)
        ret, vol, sharpe = portfolio_metrics(w, mu, cov, rf_annual)
        rows.append(
            {
                "Retorno objetivo": target,
                "Retorno anualizado": ret,
                "Volatilidad anualizada": vol,
                "Sharpe Ratio": sharpe,
                **{f"Peso_{asset}": weight for asset, weight in zip(mu.index, w)},
            }
        )
    frontier = pd.DataFrame(rows).drop_duplicates(subset=["Volatilidad anualizada", "Retorno anualizado"])
    return frontier.sort_values("Volatilidad anualizada").reset_index(drop=True)


def run_modulo2(
    returns: pd.DataFrame,
    rf_period: pd.Series,
    annual_factor: int,
    assets: list[str],
    n_portfolios: int = 12000,
    risk_aversion: float = 5.0,
):
    selected = returns[assets].dropna(how="any")
    if selected.shape[1] < 2:
        raise ValueError("Selecciona al menos dos activos con datos suficientes.")
    if len(selected) < 30:
        raise ValueError("Se necesitan al menos 30 observaciones para optimizar.")

    rf_aligned = rf_period.reindex(selected.index, method="ffill").dropna()
    selected = selected.loc[rf_aligned.index]
    rf_annual = float((1 + rf_aligned.mean()) ** annual_factor - 1)
    mu = selected.mean() * annual_factor
    cov = selected.cov() * annual_factor
    corr = selected.corr()

    simulated = simulate_portfolios(mu, cov, rf_annual, n_portfolios=n_portfolios)
    min_w = optimize_min_variance(cov)
    tan_w = optimize_max_sharpe(mu, cov, rf_annual)
    utility_w = optimize_mean_variance_utility(mu, cov, risk_aversion)
    min_ret, min_vol, min_sharpe = portfolio_metrics(min_w, mu, cov, rf_annual)
    tan_ret, tan_vol, tan_sharpe = portfolio_metrics(tan_w, mu, cov, rf_annual)
    utility_ret, utility_vol, utility_sharpe = portfolio_metrics(utility_w, mu, cov, rf_annual)
    frontier = efficient_frontier_optimized(mu, cov, rf_annual)
    if frontier.empty:
        frontier = efficient_frontier_from_simulation(simulated)

    max_vol = max(simulated["Volatilidad anualizada"].max(), tan_vol, min_vol)
    cml_vol = np.linspace(0, max_vol * 1.05, 100)
    cml = pd.DataFrame({"Volatilidad anualizada": cml_vol, "Retorno CML": rf_annual + tan_sharpe * cml_vol})
    summary = pd.DataFrame(
        [
            {
                "Portafolio": "Minima varianza",
                "Retorno anualizado": min_ret,
                "Volatilidad anualizada": min_vol,
                "Varianza anualizada": min_vol**2,
                "Sharpe Ratio": min_sharpe,
                "Concentracion HHI": concentration_index(min_w),
                "Utilidad media-varianza": min_ret - 0.5 * risk_aversion * min_vol**2,
            },
            {
                "Portafolio": "Tangente max Sharpe",
                "Retorno anualizado": tan_ret,
                "Volatilidad anualizada": tan_vol,
                "Varianza anualizada": tan_vol**2,
                "Sharpe Ratio": tan_sharpe,
                "Concentracion HHI": concentration_index(tan_w),
                "Utilidad media-varianza": tan_ret - 0.5 * risk_aversion * tan_vol**2,
            },
            {
                "Portafolio": "Recomendado por aversion",
                "Retorno anualizado": utility_ret,
                "Volatilidad anualizada": utility_vol,
                "Varianza anualizada": utility_vol**2,
                "Sharpe Ratio": utility_sharpe,
                "Concentracion HHI": concentration_index(utility_w),
                "Utilidad media-varianza": utility_ret - 0.5 * risk_aversion * utility_vol**2,
            },
        ]
    )
    weights = pd.DataFrame(
        {
            "Activo": selected.columns,
            "Peso minima varianza": min_w,
            "Peso tangente max Sharpe": tan_w,
            "Peso recomendado aversion": utility_w,
        }
    )
    weights = weights.sort_values("Peso recomendado aversion", ascending=False)

    risk_decomposition = []
    for label, w in [
        ("Minima varianza", min_w),
        ("Tangente max Sharpe", tan_w),
        ("Recomendado por aversion", utility_w),
    ]:
        variance = float(w.T @ cov.values @ w)
        marginal = cov.values @ w
        contribution = w * marginal / variance if variance > 0 else np.repeat(np.nan, len(w))
        for asset, weight, contrib in zip(selected.columns, w, contribution):
            risk_decomposition.append({"Portafolio": label, "Activo": asset, "Peso": weight, "Contribucion al riesgo": contrib})
    risk_decomposition = pd.DataFrame(risk_decomposition)

    return {
        "returns": selected,
        "mu": mu,
        "cov": cov,
        "corr": corr,
        "rf_annual": rf_annual,
        "simulated": simulated,
        "frontier": frontier,
        "cml": cml,
        "summary": summary,
        "weights": weights,
        "risk_decomposition": risk_decomposition,
        "risk_aversion": risk_aversion,
    }



# ============================================================
# MODULO 3
# ============================================================


def ols_against_benchmark(asset_returns: pd.Series, market_returns: pd.Series, rf_period: pd.Series, annual_factor: int) -> dict:
    data = pd.concat(
        [
            asset_returns.rename("asset"),
            market_returns.rename("market"),
            rf_period.rename("rf"),
        ],
        axis=1,
    ).dropna()

    empty_result = {
        "Beta OLS": np.nan,
        "Alpha por periodo": np.nan,
        "Alpha OLS anual": np.nan,
        "R2": np.nan,
        "R2 ajustado": np.nan,
        "regression_data": data,
        "Retorno historico anualizado": np.nan,
        "Retorno benchmark anualizado": np.nan,
        "RF anualizada": np.nan,
        "Prima de mercado anual": np.nan,
        "Retorno CAPM anual": np.nan,
        "Alpha Jensen anual": np.nan,
        "Error estandar alpha": np.nan,
        "Error estandar beta": np.nan,
        "t alpha": np.nan,
        "t beta": np.nan,
        "p-value alpha": np.nan,
        "p-value beta": np.nan,
        "IC 95% alpha inferior": np.nan,
        "IC 95% alpha superior": np.nan,
        "IC 95% beta inferior": np.nan,
        "IC 95% beta superior": np.nan,
        "SSE": np.nan,
        "Error estandar residual": np.nan,
        "Estadistico F": np.nan,
        "p-value F": np.nan,
        "Durbin-Watson": np.nan,
        "Estado inferencia": "Sin datos suficientes",
        "Observaciones": int(len(data)),
    }
    if len(data) < 10:
        return empty_result

    x = data["market"] - data["rf"]
    y = data["asset"] - data["rf"]
    x_var = x.var(ddof=1)
    if pd.isna(x_var) or x_var == 0:
        return empty_result

    beta = y.cov(x) / x_var
    alpha_period = y.mean() - beta * x.mean()
    y_hat = alpha_period + beta * x
    sse = float(((y - y_hat) ** 2).sum())
    sst = float(((y - y.mean()) ** 2).sum())
    r2 = np.nan if sst == 0 else 1 - sse / sst

    # Diagnósticos OLS: amplían la información estadística sin alterar beta, alfa ni R².
    n = len(data)
    degrees_of_freedom = n - 2
    x_matrix = np.column_stack([np.ones(n), x.to_numpy(dtype=float)])
    xtx_inverse = np.linalg.inv(x_matrix.T @ x_matrix)
    residuals = (y - y_hat).to_numpy(dtype=float)
    residual_variance = sse / degrees_of_freedom
    residual_standard_error = float(np.sqrt(residual_variance))
    coefficient_variance = residual_variance * xtx_inverse
    standard_errors = np.sqrt(np.maximum(np.diag(coefficient_variance), 0.0))
    se_alpha, se_beta = float(standard_errors[0]), float(standard_errors[1])
    t_alpha = safe_divide(alpha_period, se_alpha)
    t_beta = safe_divide(beta, se_beta)
    p_alpha = float(2 * t_distribution.sf(abs(t_alpha), degrees_of_freedom)) if not pd.isna(t_alpha) else np.nan
    p_beta = float(2 * t_distribution.sf(abs(t_beta), degrees_of_freedom)) if not pd.isna(t_beta) else np.nan
    critical_t = float(t_distribution.ppf(0.975, degrees_of_freedom))
    adjusted_r2 = 1 - (1 - r2) * (n - 1) / degrees_of_freedom if not pd.isna(r2) else np.nan
    explained_sum = max(sst - sse, 0.0)
    f_statistic = safe_divide(explained_sum, residual_variance)
    p_f = float(f_distribution.sf(f_statistic, 1, degrees_of_freedom)) if not pd.isna(f_statistic) else np.nan
    durbin_watson = safe_divide(float(np.sum(np.diff(residuals) ** 2)), sse)

    historical_return = annualize_geometric(data["asset"], annual_factor)
    benchmark_return = annualize_geometric(data["market"], annual_factor)
    rf_annual = float((1 + data["rf"].mean()) ** annual_factor - 1) if not data["rf"].dropna().empty else np.nan
    market_premium = benchmark_return - rf_annual if not (pd.isna(benchmark_return) or pd.isna(rf_annual)) else np.nan
    capm_return = rf_annual + beta * market_premium if not (pd.isna(rf_annual) or pd.isna(beta) or pd.isna(market_premium)) else np.nan
    alpha_jensen = historical_return - capm_return if not (pd.isna(historical_return) or pd.isna(capm_return)) else np.nan

    return {
        "Beta OLS": beta,
        "Alpha por periodo": alpha_period,
        "Alpha OLS anual": alpha_period * annual_factor,
        "R2": r2,
        "R2 ajustado": adjusted_r2,
        "regression_data": data,
        "Retorno historico anualizado": historical_return,
        "Retorno benchmark anualizado": benchmark_return,
        "RF anualizada": rf_annual,
        "Prima de mercado anual": market_premium,
        "Retorno CAPM anual": capm_return,
        "Alpha Jensen anual": alpha_jensen,
        "Error estandar alpha": se_alpha,
        "Error estandar beta": se_beta,
        "t alpha": t_alpha,
        "t beta": t_beta,
        "p-value alpha": p_alpha,
        "p-value beta": p_beta,
        "IC 95% alpha inferior": alpha_period - critical_t * se_alpha,
        "IC 95% alpha superior": alpha_period + critical_t * se_alpha,
        "IC 95% beta inferior": beta - critical_t * se_beta,
        "IC 95% beta superior": beta + critical_t * se_beta,
        "SSE": sse,
        "Error estandar residual": residual_standard_error,
        "Estadistico F": f_statistic,
        "p-value F": p_f,
        "Durbin-Watson": durbin_watson,
        "Estado inferencia": "Inferencia OLS disponible",
        "Observaciones": int(len(data)),
    }


def sml_position(alpha: float, tolerance: float = 0.0025) -> str:
    if pd.isna(alpha):
        return "Sin datos suficientes"
    if alpha > tolerance:
        return "Por encima de la SML"
    if alpha < -tolerance:
        return "Por debajo de la SML"
    return "Cerca de la SML"


def interpret_beta(beta: float) -> str:
    if pd.isna(beta):
        return "No hay observaciones suficientes para interpretar la beta."
    if beta < 0:
        return "Beta negativa: el activo muestra relación inversa con el benchmark en la muestra."
    if abs(beta - 1) <= 0.10:
        return "Beta cercana a 1: sensibilidad similar a la del benchmark."
    if beta > 1:
        return "Beta mayor que 1: amplifica los movimientos del benchmark; sube y cae con mayor sensibilidad."
    return "Beta entre 0 y 1: se mueve en la misma dirección del benchmark, pero con menor sensibilidad."


def interpret_r2(r2: float) -> str:
    if pd.isna(r2):
        return "No hay R2 confiable por falta de datos o varianza insuficiente del benchmark."
    if r2 < 0.30:
        level = "bajo"
    elif r2 < 0.60:
        level = "medio"
    else:
        level = "alto"
    return f"R² {level}: el benchmark explica aproximadamente {r2:.1%} de la variabilidad de los retornos excedentes del activo."


def interpret_jensen_alpha(alpha: float) -> str:
    if pd.isna(alpha):
        return "No hay alfa de Jensen suficiente para concluir."
    if alpha > 0.0025:
        return "Alfa de Jensen positivo: el activo supera el retorno exigido por CAPM para su beta histórica."
    if alpha < -0.0025:
        return "Alfa de Jensen negativo: el activo rindió menos que lo exigido por CAPM para su beta histórica."
    return "Alfa de Jensen cercano a cero: el activo está alineado con el retorno esperado por CAPM."


def run_modulo3(
    returns: pd.DataFrame,
    rf_period: pd.Series,
    indicators: pd.DataFrame,
    benchmark: str,
    annual_factor: int,
    assets: list[str],
) -> dict:
    benchmark = benchmark.strip().upper() or BENCHMARK_TICKER
    if benchmark not in returns.columns:
        raise ValueError(f"No se encontro el benchmark {benchmark} en los retornos disponibles.")

    candidate_assets = [asset for asset in assets if asset in returns.columns and asset != benchmark]
    market_returns = returns[benchmark]
    rows = []
    regressions = {}

    for asset in candidate_assets:
        result = ols_against_benchmark(returns[asset], market_returns, rf_period, annual_factor)
        regressions[asset] = result
        ind = indicators.loc[indicators["Activo"] == asset]
        ind_row = ind.iloc[0] if not ind.empty else pd.Series(dtype=float)
        alpha_jensen = result["Alpha Jensen anual"]
        rows.append(
            {
                "Activo": asset,
                "Tipo": ind_row.get("Tipo", "Sin clasificar"),
                "Grupo": ind_row.get("Grupo", "Sin clasificar"),
                "Beta OLS": result["Beta OLS"],
                "R2": result["R2"],
                "Retorno esperado CAPM": result["Retorno CAPM anual"],
                "Retorno historico anualizado": result["Retorno historico anualizado"],
                "Alpha Jensen anual": alpha_jensen,
                "Posicion respecto a SML": sml_position(alpha_jensen),
                "Volatilidad anualizada": ind_row.get("Volatilidad anualizada", np.nan),
                "Sharpe Ratio": ind_row.get("Sharpe Ratio", np.nan),
                "Max Drawdown": ind_row.get("Max Drawdown", np.nan),
                "VaR 95%": ind_row.get("VaR 95% periodo", np.nan),
                "CVaR 95%": ind_row.get("CVaR 95% periodo", np.nan),
                "Observaciones": result["Observaciones"],
            }
        )

    capm_table = pd.DataFrame(rows)
    if not capm_table.empty:
        capm_table = capm_table.sort_values(["Posicion respecto a SML", "Alpha Jensen anual", "Activo"], ascending=[True, False, True]).reset_index(drop=True)

    valid_params = [reg for reg in regressions.values() if not pd.isna(reg["RF anualizada"])]
    first_valid = valid_params[0] if valid_params else {}
    params = pd.DataFrame(
        [
            {
                "Benchmark": benchmark,
                "RF anual": first_valid.get("RF anualizada", np.nan),
                "Retorno anual benchmark": first_valid.get("Retorno benchmark anualizado", np.nan),
                "Prima de mercado anual": first_valid.get("Prima de mercado anual", np.nan),
                "Annual factor": annual_factor,
                "Activos analizados": len(candidate_assets),
            }
        ]
    )
    return {"capm_table": capm_table, "regressions": regressions, "params": params}


# ============================================================
# APP STREAMLIT
# ============================================================


def pct_cols(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col in out.columns:
            out[col] = out[col].map(lambda x: "" if pd.isna(x) else f"{x:.2%}")
    return out


def make_excel_download(result1, result2=None, result3=None) -> bytes:
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        result1["prices"].to_excel(writer, sheet_name="Precios")
        result1["returns"].to_excel(writer, sheet_name="Retornos")
        result1["rf_period"].to_frame("rf_period").to_excel(writer, sheet_name="RF")
        result1["indicators"].to_excel(writer, sheet_name="Indicadores_M1", index=False)
        if result2:
            result2["corr"].to_excel(writer, sheet_name="Correlaciones")
            result2["cov"].to_excel(writer, sheet_name="Covarianzas")
            result2["summary"].to_excel(writer, sheet_name="Portafolios_Optimos", index=False)
            result2["weights"].to_excel(writer, sheet_name="Pesos_Optimos", index=False)
            result2["risk_decomposition"].to_excel(writer, sheet_name="Riesgo_Portafolio", index=False)
            result2["frontier"].to_excel(writer, sheet_name="Frontera", index=False)
        if result3:
            result3["capm_table"].to_excel(writer, sheet_name="CAPM_Modulo3", index=False)
            result3["params"].to_excel(writer, sheet_name="Parametros_CAPM", index=False)
            # Exporta diagnósticos visibles sin incluir las series internas de la regresión.
            ols_rows = []
            for asset, regression in result3["regressions"].items():
                visible_stats = {key: value for key, value in regression.items() if key != "regression_data"}
                ols_rows.append({"Activo": asset, **visible_stats})
            pd.DataFrame(ols_rows).to_excel(writer, sheet_name="Estadisticas_OLS", index=False)
    return buffer.getvalue()


def save_clean_outputs(data: MarketData, indicators: pd.DataFrame, output_dir: Path | None = None) -> list[Path]:
    """Guarda tablas limpias en una carpeta relativa y devuelve las rutas creadas.

    La función no modifica la base original ni recalcula métricas: solo exporta
    los datos ya depurados y los indicadores que la aplicación muestra.
    """
    destination = output_dir or Path(__file__).resolve().parent / "outputs"
    destination.mkdir(parents=True, exist_ok=True)
    files = {
        "precios_limpios.xlsx": data.prices,
        "benchmark_limpio.xlsx": data.benchmark,
        "tbill_limpio.xlsx": data.risk_free,
        "indicadores.xlsx": indicators,
    }
    created = []
    for filename, table in files.items():
        path = destination / filename
        table.to_excel(path, index=True, engine="openpyxl")
        created.append(path)
    return created


def format_pct(value: float, digits: int = 2) -> str:
    """Devuelve porcentajes legibles para metricas y textos de interpretacion."""
    if pd.isna(value):
        return "Sin dato"
    return f"{value:.{digits}%}"


def format_num(value: float, digits: int = 3) -> str:
    """Devuelve numeros legibles y evita mostrar nan en la interfaz."""
    if pd.isna(value):
        return "Sin dato"
    return f"{value:.{digits}f}"


def run_app() -> None:
    import plotly.express as px
    import plotly.graph_objects as go
    import streamlit as st

    st.set_page_config(page_title="PortfolioLab", page_icon="PL", layout="wide")

    # ------------------------------------------------------------------
    # UI THEME: todo el diseno queda concentrado aqui para mantener un
    # archivo unico y facil de publicar en Streamlit Cloud.
    # ------------------------------------------------------------------
    PRIMARY = "#2563eb"
    SECONDARY = "#0d9488"
    NAVY = "#0f172a"
    RED = "#dc2626"
    GREEN = "#16a34a"
    AMBER = "#f59e0b"
    MUTED = "#64748b"
    BORDER = "#e5e7eb"

    st.markdown(
        f"""
        <style>
        .stApp {{
            background: #f8fafc;
            color: #0f172a;
        }}
        [data-testid="stSidebar"] {{
            background: #ffffff;
            border-right: 1px solid {BORDER};
        }}
        .hero-dark {{
            width: 100%;
            padding: 30px 34px;
            border-radius: 18px;
            margin-bottom: 18px;
            background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 100%);
            color: #ffffff;
            box-shadow: 0 16px 36px rgba(15, 23, 42, 0.18);
        }}
        .hero-dark h1 {{
            margin: 0;
            font-size: 38px;
            line-height: 1.05;
            font-weight: 800;
            letter-spacing: 0;
        }}
        .hero-dark p {{
            margin: 9px 0 0 0;
            color: #dbeafe;
            font-size: 17px;
        }}
        .kpi-card {{
            min-height: 132px;
            padding: 18px 18px 16px 18px;
            border-radius: 16px;
            background: #ffffff;
            border: 1px solid {BORDER};
            box-shadow: 0 12px 26px rgba(15, 23, 42, 0.055);
        }}
        .kpi-emoji {{
            font-size: 23px;
            margin-bottom: 6px;
        }}
        .kpi-label {{
            color: {MUTED};
            font-size: 12px;
            text-transform: uppercase;
            font-weight: 700;
            letter-spacing: .04em;
        }}
        .kpi-value {{
            color: {NAVY};
            font-size: 26px;
            line-height: 1.15;
            font-weight: 800;
            margin-top: 5px;
        }}
        .kpi-help {{
            color: {MUTED};
            font-size: 12px;
            margin-top: 7px;
            line-height: 1.35;
        }}
        .analyst-note {{
            padding: 16px 18px;
            border-radius: 14px;
            background: #fffbeb;
            border: 1px solid #fde68a;
            border-left: 6px solid {AMBER};
            color: #78350f;
            line-height: 1.45;
        }}
        .profile-box {{
            padding: 20px 22px;
            border-radius: 16px;
            background: linear-gradient(135deg, #eff6ff 0%, #f0fdfa 100%);
            border: 1px solid #bfdbfe;
        }}
        .gamma-pill {{
            padding: 18px;
            border-radius: 16px;
            background: #ffffff;
            border: 1px solid #bfdbfe;
            text-align: center;
            box-shadow: 0 10px 24px rgba(37, 99, 235, 0.10);
        }}
        .gamma-pill b {{
            font-size: 32px;
            color: {PRIMARY};
        }}
        .interpretation-box {{
            padding: 18px;
            border-radius: 16px;
            background: #ecfdf5;
            border: 1px solid #bbf7d0;
            border-left: 6px solid {GREEN};
            color: #14532d;
            min-height: 430px;
        }}
        .section-subtitle {{
            color: {MUTED};
            font-size: 14px;
            margin-top: -8px;
            margin-bottom: 12px;
        }}
        div[data-testid="stDataFrame"] {{
            border-radius: 12px;
            overflow: hidden;
        }}
        .author-card {{
            margin: -5px 0 20px 0;
            padding: 14px 18px;
            border-radius: 14px;
            background: #ffffff;
            border: 1px solid #bfdbfe;
            border-left: 6px solid {PRIMARY};
            color: {NAVY};
            font-weight: 650;
        }}
        .education-grid {{
            padding: 18px 20px;
            border-radius: 16px;
            background: #ffffff;
            border: 1px solid {BORDER};
            color: {MUTED};
            line-height: 1.55;
        }}
        @media (max-width: 900px) {{
            .hero-dark {{ padding: 24px 22px; }}
            .hero-dark h1 {{ font-size: 30px; }}
            .kpi-card {{ min-height: 116px; }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

    def chart_title(title: str, subtitle: str) -> str:
        """Titulo HTML estandarizado para todos los graficos Plotly."""
        return f"<b>{title}</b><br><span style='font-size:12px;color:gray;'>{subtitle}</span>"

    def kpi_card(icon: str, label: str, value: str, help_text: str) -> None:
        """Card KPI personalizada. Evita st.metric para lograr el look corporativo pedido."""
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-emoji">{icon}</div>
                <div class="kpi-label">{label}</div>
                <div class="kpi-value">{value}</div>
                <div class="kpi-help">{help_text}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    def polish(fig, *, show_legend: bool = True, height: int | None = None, legend_items: int = 0):
        """Aplica estilo y reserva espacio inferior para leyendas de varias filas."""
        legend_rows = max(1, int(np.ceil(legend_items / 4))) if legend_items else 1
        fig.update_layout(
            template="plotly_white",
            height=height,
            margin=dict(l=24, r=24, t=92, b=54 + 24 * legend_rows if show_legend else 42),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color=NAVY, family="Arial"),
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.16,
                xanchor="left",
                x=0,
                title_text="",
                font=dict(size=11),
            ) if show_legend else dict(visible=False),
            hoverlabel=dict(bgcolor="white", font_size=12),
        )
        fig.update_xaxes(showgrid=True, gridcolor="#eef2f7", zeroline=False)
        fig.update_yaxes(showgrid=True, gridcolor="#eef2f7", zeroline=False)
        return fig

    def rounded_bars(fig, radius: int = 8):
        """Redondea barras cuando la version de Plotly soporta marker_cornerradius."""
        try:
            fig.update_traces(marker_cornerradius=radius, selector=dict(type="bar"))
        except Exception:
            pass
        return fig

    # Carga de datos: primero intenta encontrar base_diaria.xlsx; si no existe,
    # permite subirla desde la barra lateral para que funcione en Streamlit Cloud.
    path = find_data_file()
    uploaded = None
    with st.sidebar:
        st.markdown("## PortfolioLab")
        page = st.radio(
            "Navegación",
            [
                "Cobertura",
                "Módulo 1 — Riesgo y Rentabilidad",
                "Módulo 2 — Optimización de Portafolios",
                "Módulo 3 — Modelos de Valoración de Activos",
            ],
            label_visibility="collapsed",
        )
        if st.button("Actualizar análisis", width="stretch"):
            st.cache_data.clear()
            st.rerun()
        st.divider()
        uploaded = st.file_uploader("Archivo base_diaria.xlsx", type=["xlsx"])

    data_source = uploaded if uploaded is not None else path
    if data_source is None:
        st.info("Carga `base_diaria.xlsx` desde la barra lateral para iniciar el análisis.")
        st.stop()

    @st.cache_data(show_spinner="Cargando y limpiando datos de Refinitiv...")
    def cached_market_data(source_id, source_version, source):
        """Carga la base; source_version cambia cuando varían tamaño o fecha del Excel."""
        return load_market_data(source)

    source_id = uploaded.name if uploaded is not None else str(path)
    if uploaded is not None:
        uploaded_bytes = uploaded.getvalue()
        source_version = (len(uploaded_bytes), hash(uploaded_bytes))
    else:
        source_version = file_fingerprint(path)
    try:
        with st.spinner("Preparando precios, benchmark y tasa libre de riesgo..."):
            data = cached_market_data(source_id, source_version, data_source)
    except Exception as exc:
        st.error("No se pudo cargar la base. Revisa que tenga las hojas ACTIVOS, BASE FINAL, BENCHMARK y T-BILL.")
        st.exception(exc)
        st.stop()

    # Refresco controlado: compara metadatos y confirma dos veces una huella estable.
    # Esto evita intentar leer el ZIP interno de Excel mientras aún se está guardando.
    if uploaded is None and path is not None and hasattr(st, "fragment"):
        st.session_state.setdefault("excel_fingerprint", source_version)

        @st.fragment(run_every="10s")
        def monitor_local_excel() -> None:
            """Invalida el caché solo cuando el Excel cambió, es válido y ya está estable."""
            if not zipfile.is_zipfile(path):
                return
            current_fingerprint = file_fingerprint(path)
            stored_fingerprint = st.session_state.get("excel_fingerprint")
            if current_fingerprint == stored_fingerprint:
                st.session_state.pop("pending_excel_fingerprint", None)
                return
            if current_fingerprint == st.session_state.get("pending_excel_fingerprint"):
                st.session_state["excel_fingerprint"] = current_fingerprint
                st.session_state.pop("pending_excel_fingerprint", None)
                cached_market_data.clear()
                st.rerun()
                return
            st.session_state["pending_excel_fingerprint"] = current_fingerprint

        monitor_local_excel()

    min_date = data.prices.index.min().date()
    max_date = data.prices.index.max().date()

    with st.sidebar:
        st.success("Archivo cargado")
        st.caption(str(data_source))
        if uploaded is None and path is not None:
            st.caption(f"Última actualización de datos: {format_file_update(path)}")
            st.caption("Detección automática activa; confirma que el Excel terminó de guardarse.")
        start = st.date_input("Fecha inicial", value=min_date, min_value=min_date, max_value=max_date)
        end = st.date_input("Fecha final", value=max_date, min_value=min_date, max_value=max_date)
        frequency = st.selectbox("Frecuencia", ["Diaria", "Semanal", "Mensual"], index=0)
        benchmark = st.text_input("Benchmark", value=BENCHMARK_TICKER).strip().upper() or BENCHMARK_TICKER
        n_portfolios = st.slider("Portafolios simulados", 2000, 30000, 12000, step=1000)
        profile_name = st.selectbox("Perfil de riesgo", list(RISK_PROFILES.keys()), index=1)
        profile = RISK_PROFILES[profile_name]
        risk_aversion = st.slider(
            "Grado de aversión al riesgo",
            min_value=0.5,
            max_value=12.0,
            value=float(profile["gamma"]),
            step=0.5,
            help="Mayor aversión penaliza más la varianza en la utilidad media-varianza.",
        )

    if start > end:
        st.error("La fecha inicial no puede ser posterior a la fecha final.")
        st.stop()

    try:
        with st.spinner("Calculando Módulo 1: riesgo y rentabilidad..."):
            result1 = run_modulo1(data, start=start, end=end, frequency=frequency, benchmark=benchmark)
    except Exception as exc:
        st.error("No se pudo procesar el Módulo 1 con los filtros seleccionados.")
        st.exception(exc)
        st.stop()

    prices = result1["prices"]
    returns = result1["returns"]
    indicators = result1["indicators"]
    rf_period = result1["rf_period"]
    annual_factor = result1["annual_factor"]
    asset_universe = [c for c in prices.columns if c != benchmark]
    default_assets = asset_universe[: min(10, len(asset_universe))]

    with st.sidebar:
        selected_assets = st.multiselect("Activos para analizar", asset_universe, default=default_assets)

    if not selected_assets:
        st.warning("Selecciona al menos un activo en la barra lateral.")
        st.stop()

    selected_with_benchmark = list(dict.fromkeys(selected_assets + [benchmark]))
    selected_indicators = indicators[indicators["Activo"].isin(selected_with_benchmark)].copy()
    focus_asset = st.sidebar.selectbox("Activo foco", selected_with_benchmark)

    # Caché de análisis: evita repetir optimización y OLS al navegar. La huella del
    # Excel y todos los parámetros forman parte de la clave, por lo que un cambio
    # de datos o filtros invalida estos resultados de manera reproducible.
    @st.cache_data(show_spinner=False)
    def cached_module2(source_version, returns, rf_period, annual_factor, assets, n_portfolios, risk_aversion):
        """Recibe datos y parámetros; devuelve el Módulo 2 calculado una sola vez por versión."""
        return run_modulo2(
            returns,
            rf_period,
            annual_factor,
            list(assets),
            n_portfolios=n_portfolios,
            risk_aversion=risk_aversion,
        )

    @st.cache_data(show_spinner=False)
    def cached_module3(source_version, returns, rf_period, indicators, benchmark, annual_factor, assets):
        """Recibe datos y parámetros; devuelve CAPM y OLS reutilizables durante la navegación."""
        return run_modulo3(returns, rf_period, indicators, benchmark, annual_factor, list(assets))

    # Calcula los módulos 2 y 3 para mantener disponible la exportación completa.
    result2 = None
    if len(selected_assets) >= 2:
        try:
            with st.spinner("Calculando frontera eficiente y portafolios óptimos..."):
                result2 = cached_module2(
                    source_version,
                    returns,
                    rf_period,
                    annual_factor,
                    tuple(selected_assets),
                    n_portfolios,
                    risk_aversion,
                )
        except Exception as exc:
            st.warning("El Módulo 2 no pudo calcularse con la selección actual.")
            st.exception(exc)

    result3 = None
    capm_assets = [asset for asset in selected_assets if asset != benchmark]
    if capm_assets:
        try:
            with st.spinner("Estimando CAPM, SML y regresiones OLS..."):
                result3 = cached_module3(
                    source_version,
                    returns,
                    rf_period,
                    indicators,
                    benchmark,
                    annual_factor,
                    tuple(capm_assets),
                )
        except Exception as exc:
            st.warning("El Módulo 3 no pudo calcularse con la selección actual.")
            st.exception(exc)

    if page == "Cobertura":
        # FILA 0: banner hero.
        st.markdown(
            """
            <div class="hero-dark">
                <h1>PortfolioLab</h1>
                <p>Analiza, optimiza y entiende tu portafolio de inversión.</p>
            </div>
            <div class="author-card">
                Herramienta elaborada por: Camila Arias, Arelys Paredes, Alejandro Fernández y Alejandro Urquidi.
            </div>
            """,
            unsafe_allow_html=True,
        )

        # FILA 1: tarjetas KPI HTML personalizadas.
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            kpi_card("📁", "Activos disponibles", f"{len(asset_universe)}", "Universo listo para análisis y selección.")
        with c2:
            kpi_card("📊", "Observaciones", f"{len(prices):,}", f"Precios filtrados entre {start} y {end}.")
        with c3:
            kpi_card("🛡️", "Benchmark", benchmark, "Índice de referencia usado en comparaciones y CAPM.")
        with c4:
            kpi_card("🎯", "Activos seleccionados", f"{len(selected_assets)}", "Canasta actual para riesgo, optimización y CAPM.")

        # FILA 2: graficos de cobertura.
        selected_meta = data.metadata[data.metadata["Ticker"].isin(selected_assets)]
        col_a, col_b = st.columns(2)
        with col_a:
            with st.container(border=True):
                type_count = data.metadata["Tipo"].value_counts().reset_index()
                type_count.columns = ["Tipo", "Cantidad"]
                fig = px.bar(
                    type_count,
                    x="Tipo",
                    y="Cantidad",
                    title=chart_title("Distribución por tipo", "Cantidad de instrumentos disponibles por clase."),
                    color_discrete_sequence=[PRIMARY],
                )
                fig.update_traces(marker_color=PRIMARY)
                fig.update_yaxes(title="Cantidad")
                fig.update_xaxes(title="")
                st.plotly_chart(polish(rounded_bars(fig), show_legend=False), width="stretch")
        with col_b:
            with st.container(border=True):
                group_count = selected_meta["Grupo"].value_counts().reset_index()
                group_count.columns = ["Grupo", "Cantidad"]
                group_count = group_count.sort_values("Cantidad", ascending=True)
                fig = px.bar(
                    group_count,
                    x="Cantidad",
                    y="Grupo",
                    orientation="h",
                    title=chart_title("Distribución por grupo", "Diversificación sectorial o geográfica de la selección actual."),
                    color_discrete_sequence=[SECONDARY],
                )
                fig.update_traces(marker_color=SECONDARY)
                fig.update_xaxes(title="Cantidad")
                fig.update_yaxes(title="")
                st.plotly_chart(polish(rounded_bars(fig), show_legend=False), width="stretch")

        # FILA 3: guia del analista.
        st.markdown(
            """
            <div class="analyst-note">
                <b>Guía del analista.</b> Diversificar por tipo de instrumento y por grupo sectorial o geográfico
                reduce la dependencia de una sola fuente de riesgo. SPX funciona como benchmark para comparar el
                comportamiento de los activos y estimar su sensibilidad al mercado. La diversificación no elimina
                las pérdidas, pero puede moderar riesgos específicos cuando las exposiciones no se mueven igual.
            </div>
            """,
            unsafe_allow_html=True,
        )

        # FILA 4: tabla con buscador.
        with st.container(border=True):
            st.markdown("**Tabla de activos disponibles**")
            search = st.text_input("Buscar activo en la tabla", "", placeholder="Ticker, RIC, grupo o tipo...")
            table = data.metadata.copy()
            if search:
                mask = table.astype(str).apply(lambda col: col.str.contains(search, case=False, na=False)).any(axis=1)
                table = table.loc[mask]
            st.dataframe(table, width="stretch", hide_index=True, height=430)

    elif page == "Módulo 1 — Riesgo y Rentabilidad":
        st.subheader("Módulo 1 — Riesgo y Rentabilidad")
        row = selected_indicators.loc[selected_indicators["Activo"] == focus_asset]
        row = row.iloc[0] if not row.empty else pd.Series(dtype=float)
        selected_only = selected_indicators[selected_indicators["Activo"] != benchmark].copy()
        best_sharpe = selected_only.loc[selected_only["Sharpe Ratio"].idxmax()] if not selected_only["Sharpe Ratio"].dropna().empty else pd.Series(dtype=float)
        worst_dd = selected_only.loc[selected_only["Max Drawdown"].idxmin()] if not selected_only["Max Drawdown"].dropna().empty else pd.Series(dtype=float)

        # FILA 1: cinco KPI cards ejecutivas.
        m = st.columns(5)
        with m[0]:
            kpi_card("📈", "Retorno geométrico", format_pct(row.get("Retorno anualizado", np.nan)), f"{focus_asset} | anualizado realizado")
        with m[1]:
            kpi_card("➗", "Retorno medio", format_pct(row.get("Media periodo", np.nan)), f"{focus_asset} | base por periodo")
        with m[2]:
            kpi_card("⚠️", "Volatilidad", format_pct(row.get("Volatilidad anualizada", np.nan)), f"{focus_asset} | desviación anualizada")
        with m[3]:
            kpi_card("🏆", "Mejor Sharpe", format_num(best_sharpe.get("Sharpe Ratio", np.nan)), f"{best_sharpe.get('Activo', 'Sin dato')} | retorno por unidad de riesgo")
        with m[4]:
            kpi_card("📉", "Peor drawdown", format_pct(worst_dd.get("Max Drawdown", np.nan)), f"{worst_dd.get('Activo', 'Sin dato')} | mayor caída histórica")

        price_view = prices[selected_with_benchmark].dropna(how="all")
        ret_view = returns[selected_with_benchmark].dropna(how="all")
        base100 = price_view / price_view.dropna().iloc[0] * 100
        drawdowns = price_view.apply(lambda s: s / s.cummax() - 1)

        # FILA 2: series temporales.
        col_a, col_b = st.columns(2)
        with col_a:
            with st.container(border=True):
                fig = px.line(
                    base100,
                    title=chart_title("Evolución base 100", "Crecimiento acumulado normalizado desde el inicio del periodo seleccionado."),
                    labels={"value": "Índice base 100", "Date": "Fecha", "variable": "Activo"},
                    color_discrete_sequence=px.colors.qualitative.Bold,
                )
                for trace in fig.data:
                    if trace.name == benchmark:
                        trace.update(line=dict(color=AMBER, width=4, dash="dash"))
                    else:
                        trace.update(line=dict(width=2))
                fig.update_layout(hovermode="x unified")
                dynamic_height = min(680, 420 + max(0, len(selected_with_benchmark) - 6) * 18)
                st.plotly_chart(
                    polish(fig, height=dynamic_height, legend_items=len(selected_with_benchmark)),
                    width="stretch",
                )
        with col_b:
            with st.container(border=True):
                fig = px.line(
                    drawdowns,
                    title=chart_title("Drawdown histórico", "Caídas porcentuales desde máximos acumulados."),
                    labels={"value": "Drawdown", "Date": "Fecha", "variable": "Activo"},
                    color_discrete_sequence=[RED, SECONDARY, NAVY],
                )
                fig.update_yaxes(tickformat=".1%")
                fig.update_layout(hovermode="x unified")
                st.plotly_chart(polish(fig, legend_items=len(selected_with_benchmark)), width="stretch")

        # FILA 3: distribucion y dispersion.
        col_c, col_d = st.columns(2)
        with col_c:
            with st.container(border=True):
                hist_asset = st.selectbox("Activo para histograma de retornos", selected_with_benchmark, index=selected_with_benchmark.index(focus_asset), key="hist_asset_m1")
                hist_data = ret_view[[hist_asset]].dropna().rename(columns={hist_asset: "Retorno"})
                fig = px.histogram(
                    hist_data,
                    x="Retorno",
                    nbins=50,
                    marginal="box",
                    title=chart_title("Histograma de retornos", f"Distribución empírica de retornos por periodo para {hist_asset}."),
                    color_discrete_sequence=[SECONDARY],
                )
                fig.update_traces(marker_color=SECONDARY)
                fig.update_xaxes(tickformat=".1%", title="Retorno por periodo")
                fig.update_yaxes(title="Frecuencia")
                st.plotly_chart(polish(rounded_bars(fig), show_legend=False), width="stretch")
        with col_d:
            with st.container(border=True):
                fig = px.scatter(
                    selected_indicators,
                    x="Volatilidad anualizada",
                    y="Retorno anualizado",
                    color_discrete_sequence=[PRIMARY],
                    text="Activo",
                    hover_data=["Sharpe Ratio", "Sortino Ratio", "Beta", "Max Drawdown", "Omega Ratio"],
                    title=chart_title("Riesgo vs Retorno", "X: volatilidad anualizada | Y: retorno geométrico realizado."),
                )
                fig.update_traces(
                    mode="markers+text",
                    textposition="top center",
                    textfont=dict(size=10),
                    marker=dict(size=12, color=PRIMARY, line=dict(width=1, color="white")),
                )
                fig.update_xaxes(tickformat=".1%", title="Volatilidad anualizada")
                fig.update_yaxes(tickformat=".1%", title="Retorno geométrico realizado")
                st.plotly_chart(polish(fig, show_legend=False), width="stretch")

        with st.container(border=True):
            st.markdown("**Tabla dinámica de indicadores**")
            st.dataframe(
                pct_cols(
                    selected_indicators,
                    ["Retorno anualizado", "Media periodo", "Desviacion periodo", "Retorno medio anualizado", "Volatilidad anualizada", "Varianza anualizada", "Max Drawdown", "VaR 95% periodo", "CVaR 95% periodo"],
                ),
                width="stretch",
                hide_index=True,
                height=350,
            )

        st.markdown(
            f"""
            <div class="education-grid">
                <b>Cómo leer estas métricas para {focus_asset}.</b><br>
                El <b>retorno</b> resume el crecimiento histórico y la <b>volatilidad</b> mide la dispersión de sus
                rendimientos. Sharpe compara retorno excedente con volatilidad total; Sortino considera solo la
                desviación desfavorable. El <b>máximo drawdown</b> es la mayor caída desde un máximo previo. La
                <b>beta</b> compara sensibilidad frente a {benchmark}. VaR 95% marca un umbral de pérdida histórica
                por periodo y CVaR 95% promedia las pérdidas que exceden ese umbral. Son estimaciones históricas,
                no garantías de resultados futuros.
            </div>
            """,
            unsafe_allow_html=True,
        )

    elif page == "Módulo 2 — Optimización de Portafolios":
        st.subheader("Módulo 2 — Optimización de Portafolios")
        if result2 is None:
            st.warning("Selecciona al menos dos activos con datos suficientes para optimizar.")
        else:
            summary = result2["summary"]
            recommended = summary.loc[summary["Portafolio"] == "Recomendado por aversion"].iloc[0]
            # FILA 0: encabezado dinamico de perfil.
            with st.container(border=True):
                p1, p2 = st.columns([0.78, 0.22])
                with p1:
                    st.markdown(
                        f"""
                        <div class="profile-box">
                            <h3 style="margin:0;color:{NAVY};">Perfil Seleccionado: {profile_name}</h3>
                            <p style="margin:8px 0 0 0;color:{MUTED};">{profile["description"]}</p>
                            <p style="margin:6px 0 0 0;color:{MUTED};"><b>Fundamento:</b> {profile["basis"]}</p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                with p2:
                    st.markdown(
                        f"""
                        <div class="gamma-pill">
                            <div style="color:{MUTED};font-size:12px;text-transform:uppercase;font-weight:700;">Aversión usada</div>
                            <b>{risk_aversion:.1f}</b>
                            <div style="color:{MUTED};font-size:12px;">Parámetro gamma</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            # FILA 1: KPI del portafolio recomendado.
            o1, o2, o3, o4, o5 = st.columns(5)
            with o1:
                kpi_card("📈", "Retorno recomendado", format_pct(recommended["Retorno anualizado"]), "Retorno esperado anual del portafolio recomendado.")
            with o2:
                kpi_card("⚠️", "Volatilidad", format_pct(recommended["Volatilidad anualizada"]), "Riesgo anualizado del portafolio recomendado.")
            with o3:
                kpi_card("⚖️", "Sharpe", format_num(recommended["Sharpe Ratio"]), "Compensación por unidad de volatilidad.")
            with o4:
                kpi_card("🧩", "Índice HHI", format_num(recommended["Concentracion HHI"]), "Concentración de pesos; un valor menor sugiere más diversificación.")
            with o5:
                kpi_card("🎯", "Utilidad neta", format_num(recommended["Utilidad media-varianza"]), "U = E(R) - 0.5 x Gamma x Varianza.")

            # FILA 2: frontera eficiente a ancho completo.
            with st.container(border=True):
                sim, front, cml = result2["simulated"], result2["frontier"], result2["cml"]
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=sim["Volatilidad anualizada"], y=sim["Retorno anualizado"], mode="markers", marker=dict(size=4, color=sim["Sharpe Ratio"], colorscale="Viridis", showscale=True, colorbar=dict(title="Sharpe")), name="Portafolios simulados", opacity=0.30))
                fig.add_trace(go.Scatter(x=front["Volatilidad anualizada"], y=front["Retorno anualizado"], mode="lines", name="Frontera eficiente", line=dict(color="#b42318", width=4)))
                fig.add_trace(go.Scatter(x=cml["Volatilidad anualizada"], y=cml["Retorno CML"], mode="lines", name="Capital Market Line", line=dict(color="#2563eb", dash="dash", width=3)))
                fig.add_trace(go.Scatter(x=[0], y=[result2["rf_annual"]], mode="markers+text", text=["Rf"], textposition="bottom right", marker=dict(size=11, color="#475569"), name="Tasa libre de riesgo"))
                for _, item in summary.iterrows():
                    fig.add_trace(go.Scatter(x=[item["Volatilidad anualizada"]], y=[item["Retorno anualizado"]], mode="markers+text", text=[item["Portafolio"]], textposition="top center", marker=dict(size=15, symbol="star", line=dict(width=1, color="white")), name=item["Portafolio"]))
                fig.update_layout(title=chart_title("La Frontera Eficiente", "Simulaciones, frontera optimizada, CML y portafolios clave."), xaxis_title="Volatilidad anualizada", yaxis_title="Retorno anualizado", xaxis_tickformat=".1%", yaxis_tickformat=".1%")
                st.plotly_chart(polish(fig, legend_items=len(fig.data)), width="stretch")

            # Correlaciones y contribución al riesgo usan las matrices ya calculadas por el módulo.
            corr_col, risk_col = st.columns(2)
            with corr_col:
                with st.container(border=True):
                    corr = result2["corr"]
                    if corr.empty or corr.notna().sum().sum() == 0:
                        st.info("Sin datos suficientes")
                    else:
                        heatmap = go.Figure(
                            data=go.Heatmap(
                                z=corr.to_numpy(),
                                x=corr.columns.tolist(),
                                y=corr.index.tolist(),
                                zmin=-1,
                                zmax=1,
                                colorscale="RdBu_r",
                                colorbar=dict(title="Correlación"),
                                hoverongaps=False,
                                hovertemplate="%{y} / %{x}<br>Correlación: %{z:.4f}<extra></extra>",
                            )
                        )
                        corr_height = min(760, max(430, 300 + 22 * len(corr.columns)))
                        heatmap.update_layout(
                            title=chart_title(
                                "Matriz de Correlaciones",
                                "Negativo: diversificación potencial · Cero: relación lineal baja · Positivo alto: movimientos similares.",
                            ),
                            xaxis=dict(side="bottom", tickangle=-45, automargin=True),
                            yaxis=dict(autorange="reversed", automargin=True),
                        )
                        st.plotly_chart(polish(heatmap, show_legend=False, height=corr_height), width="stretch")
                        st.caption("La correlación mide cómo se mueven dos activos y no implica causalidad. Puede cambiar durante periodos de crisis.")
                        if corr.isna().any().any():
                            st.warning("Sin datos suficientes en una o más parejas; las celdas faltantes se mantienen vacías.")
            with risk_col:
                with st.container(border=True):
                    risk_view = result2["risk_decomposition"]
                    recommended_risk = risk_view[risk_view["Portafolio"] == "Recomendado por aversion"].copy()
                    recommended_risk = recommended_risk.sort_values("Contribucion al riesgo", ascending=True)
                    fig = px.bar(
                        recommended_risk,
                        x="Contribucion al riesgo",
                        y="Activo",
                        orientation="h",
                        title=chart_title("Contribución al riesgo", "Participación de cada activo en el riesgo del portafolio recomendado."),
                        color_discrete_sequence=[SECONDARY],
                    )
                    fig.update_xaxes(tickformat=".1%", title="Contribución al riesgo")
                    fig.update_yaxes(title="")
                    fig.update_traces(hovertemplate="%{y}<br>Contribución: %{x:.3%}<extra></extra>")
                    st.plotly_chart(polish(rounded_bars(fig), show_legend=False, height=430), width="stretch")
                    st.markdown("**Tabla de contribución al riesgo**")
                    risk_table = recommended_risk[["Activo", "Peso", "Contribucion al riesgo"]].sort_values("Contribucion al riesgo", ascending=False)
                    st.dataframe(
                        pct_cols(risk_table, ["Peso", "Contribucion al riesgo"]),
                        width="stretch",
                        hide_index=True,
                        height=250,
                    )

            # FILA 3: desglose de asignacion.
            col_c, col_d = st.columns(2)
            with col_c:
                with st.container(border=True):
                    portfolio_options = {
                        "Portafolio recomendado por aversión": "Peso recomendado aversion",
                        "Portafolio tangente max Sharpe": "Peso tangente max Sharpe",
                        "Portafolio mínima varianza": "Peso minima varianza",
                    }
                    selected_portfolio_label = st.selectbox("Portafolio para desglose", list(portfolio_options.keys()))
                    selected_weight_col = portfolio_options[selected_portfolio_label]
                    weights_table = result2["weights"][["Activo", selected_weight_col]].copy()
                    weights_table = weights_table.merge(data.metadata[["Ticker", "Tipo", "Grupo"]], left_on="Activo", right_on="Ticker", how="left")
                    weights_table = weights_table.rename(columns={selected_weight_col: "Peso Exacto"})
                    weights_table = weights_table[["Activo", "Tipo", "Grupo", "Peso Exacto"]].sort_values("Peso Exacto", ascending=False)

                    # La agrupación es solo visual; la tabla conserva cada peso exacto.
                    donut_data = weights_table[weights_table["Peso Exacto"] > 0.0001].copy()
                    if len(donut_data) > 8:
                        small_mask = donut_data["Peso Exacto"] < 0.01
                        if small_mask.sum() > 1:
                            others = pd.DataFrame(
                                [{"Activo": "Otros", "Tipo": "Varios", "Grupo": "Varios", "Peso Exacto": small_mask["Peso Exacto"].sum()}]
                            )
                            donut_data = pd.concat([donut_data.loc[~small_mask], others], ignore_index=True)
                    fig = px.pie(
                        donut_data,
                        names="Activo",
                        values="Peso Exacto",
                        hole=0.58,
                        title=chart_title("Composición porcentual", f"Pesos del {selected_portfolio_label.lower()}."),
                        custom_data=["Tipo", "Grupo"],
                        color_discrete_sequence=["#2563eb", "#0d9488", "#f59e0b", "#dc2626", "#7c3aed", "#0891b2", "#65a30d", "#db2777", "#475569", "#ea580c", "#4f46e5", "#059669"],
                    )
                    fig.update_traces(
                        textinfo="percent+label",
                        marker=dict(line=dict(color="white", width=2)),
                        hovertemplate="%{label}<br>Peso: %{value:.3%}<br>Tipo: %{customdata[0]}<br>Grupo: %{customdata[1]}<extra></extra>",
                    )
                    st.plotly_chart(polish(fig, legend_items=len(donut_data)), width="stretch")
            with col_d:
                with st.container(border=True):
                    st.markdown("**Tabla completa de pesos**")
                    st.dataframe(pct_cols(weights_table, ["Peso Exacto"]), width="stretch", hide_index=True, height=420)

            with st.container(border=True):
                st.markdown("**Resumen técnico de portafolios óptimos**")
                st.dataframe(pct_cols(summary, ["Retorno anualizado", "Volatilidad anualizada", "Varianza anualizada"]), width="stretch", hide_index=True, height=190)

            st.markdown(
                f"""
                <div class="education-grid">
                    <b>Lectura del módulo.</b> La correlación muestra cómo se han movido conjuntamente los activos
                    seleccionados; combinar relaciones distintas puede mejorar la diversificación. La frontera eficiente
                    reúne portafolios con la menor varianza para cada retorno objetivo. La CML une la tasa libre de riesgo
                    con el portafolio tangente. Mínima varianza prioriza estabilidad; tangente maximiza Sharpe; el recomendado
                    equilibra retorno y riesgo con aversión {risk_aversion:.1f}. Los pesos indican asignación y el HHI resume
                    concentración: cuanto mayor es, más depende el portafolio de pocas posiciones.
                </div>
                """,
                unsafe_allow_html=True,
            )

    elif page == "Módulo 3 — Modelos de Valoración de Activos":
        st.subheader("Módulo 3 — Modelos de Valoración de Activos")
        if result3 is None or result3["capm_table"].empty:
            st.warning("Selecciona al menos un activo distinto del benchmark para calcular CAPM.")
        else:
            capm_table = result3["capm_table"]
            capm_params = result3["params"].iloc[0]
            selected_capm_asset = st.selectbox("Activo foco para CAPM", capm_table["Activo"].tolist(), key="capm_focus")
            selected_row = capm_table.loc[capm_table["Activo"] == selected_capm_asset].iloc[0]
            selected_reg = result3["regressions"][selected_capm_asset]

            # FILA 1: KPI superiores del modelo.
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                kpi_card("🛡️", "Tasa Libre de Riesgo", format_pct(capm_params["RF anual"]), "RF anual usada para CAPM y retornos excedentes.")
            with c2:
                kpi_card("📊", "Retorno Benchmark", format_pct(capm_params["Retorno anual benchmark"]), f"Retorno medio anualizado de {benchmark}.")
            with c3:
                kpi_card("💎", "Prima de mercado", format_pct(capm_params["Prima de mercado anual"]), "Rm - Rf exigido por el mercado.")
            with c4:
                kpi_card("🎯", "Activos analizados", f"{int(capm_params['Activos analizados'])}", "Cantidad de activos bajo CAPM.")

            # FILA 2: SML + interpretacion automatica.
            row_model_left, row_model_right = st.columns([0.70, 0.30])
            valid_capm = capm_table.dropna(subset=["Beta OLS", "Retorno esperado CAPM", "Retorno historico anualizado"]).copy()
            with row_model_left:
                with st.container(border=True):
                    if not valid_capm.empty:
                        beta_min = min(0.0, float(valid_capm["Beta OLS"].min()) - 0.20)
                        beta_max = float(valid_capm["Beta OLS"].max()) + 0.20
                        beta_axis = np.linspace(beta_min, beta_max if beta_max > beta_min else beta_min + 1.0, 100)
                        rf_ann = capm_params["RF anual"]
                        premium = capm_params["Prima de mercado anual"]
                        sml_y = rf_ann + premium * beta_axis
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=beta_axis, y=sml_y, mode="lines", name="SML / CAPM", line=dict(color=PRIMARY, width=3)))
                        for position, color in [("Por encima de la SML", GREEN), ("Cerca de la SML", SECONDARY), ("Por debajo de la SML", RED), ("Sin datos suficientes", MUTED)]:
                            subset = valid_capm[valid_capm["Posicion respecto a SML"] == position]
                            if subset.empty:
                                continue
                            text_positions = ["top center", "bottom center", "middle right", "middle left"]
                            fig.add_trace(
                                go.Scatter(
                                    x=subset["Beta OLS"],
                                    y=subset["Retorno historico anualizado"],
                                    mode="markers+text",
                                    text=subset["Activo"],
                                    textposition=[text_positions[i % len(text_positions)] for i in range(len(subset))],
                                    textfont=dict(size=10),
                                    marker=dict(size=12, color=color, line=dict(width=1, color="white")),
                                    name=position,
                                    hovertemplate="%{text}<br>Beta: %{x:.3f}<br>Retorno: %{y:.2%}<extra></extra>",
                                )
                            )
                        fig.update_layout(title=chart_title("Security Market Line (SML)", "Recta teórica CAPM y posición relativa de cada activo."), xaxis_title="Beta histórica", yaxis_title="Retorno anual", yaxis_tickformat=".1%")
                        st.plotly_chart(polish(fig, height=480, legend_items=len(fig.data)), width="stretch")
                    else:
                        st.info("No hay suficientes observaciones válidas para graficar la SML.")
            with row_model_right:
                beta_text = interpret_beta(selected_row["Beta OLS"])
                r2_text = interpret_r2(selected_row["R2"])
                alpha_text = interpret_jensen_alpha(selected_row["Alpha Jensen anual"])
                significance = "R² alto: la relación con el mercado es estadísticamente más informativa." if not pd.isna(selected_row["R2"]) and selected_row["R2"] >= 0.60 else "R² bajo o medio: interpreta beta y alfa junto con otros indicadores de riesgo."
                st.markdown(
                    f"""
                    <div class="interpretation-box">
                        <h4 style="margin-top:0;">Interpretación automática</h4>
                        <p><b>Activo foco:</b> {selected_capm_asset}</p>
                        <p><b>Beta:</b> {beta_text}</p>
                        <p><b>R2:</b> {r2_text}</p>
                        <p><b>Alfa de Jensen:</b> {alpha_text}</p>
                        <p><b>Significancia práctica:</b> {significance}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # FILA 3: regresion individual y KPIs econometricos.
            with st.container(border=True):
                st.markdown("**Análisis de regresión individual**")
                r1, r2, r3, r4, r5 = st.columns(5)
                with r1:
                    kpi_card("β", "Beta OLS", format_num(selected_row["Beta OLS"]), "Sensibilidad histórica frente al benchmark.")
                with r2:
                    kpi_card("R²", "Bondad de ajuste", format_num(selected_row["R2"]), "Proporción explicada por el mercado.")
                with r3:
                    kpi_card("📌", "Retorno CAPM", format_pct(selected_row["Retorno esperado CAPM"]), "Retorno exigido según beta.")
                with r4:
                    kpi_card("📈", "Retorno observado", format_pct(selected_row["Retorno historico anualizado"]), "Retorno histórico anualizado.")
                with r5:
                    kpi_card("α", "Alfa Jensen", format_pct(selected_row["Alpha Jensen anual"]), "Exceso frente al retorno CAPM.")

                reg_data = selected_reg["regression_data"].copy()
                if len(reg_data) >= 10 and not pd.isna(selected_reg["Beta OLS"]):
                    reg_data["Exceso benchmark"] = reg_data["market"] - reg_data["rf"]
                    reg_data["Exceso activo"] = reg_data["asset"] - reg_data["rf"]
                    line_x = np.linspace(reg_data["Exceso benchmark"].min(), reg_data["Exceso benchmark"].max(), 100)
                    line_y = selected_reg["Alpha por periodo"] + selected_reg["Beta OLS"] * line_x
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=reg_data["Exceso benchmark"], y=reg_data["Exceso activo"], mode="markers", name="Observaciones", marker=dict(size=5, color=PRIMARY), opacity=0.45))
                    fig.add_trace(go.Scatter(x=line_x, y=line_y, mode="lines", name="Recta OLS", line=dict(color=RED, width=3)))
                    fig.update_layout(title=chart_title(f"Regresión OLS: {selected_capm_asset} vs {benchmark}", "Excesos de retorno del activo contra excesos del mercado."), xaxis_title="Rm - Rf", yaxis_title="Ri - Rf", xaxis_tickformat=".1%", yaxis_tickformat=".1%")
                    st.plotly_chart(polish(fig, height=480, legend_items=2), width="stretch")
                else:
                    st.info("Este activo no tiene suficientes observaciones para mostrar la regresión OLS.")

            with st.container(border=True):
                st.markdown("**Estadísticas OLS completas del activo seleccionado**")
                st.caption(str(selected_reg["Estado inferencia"]))
                ols_statistics = pd.DataFrame(
                    [
                        ("Observaciones", f"{int(selected_reg['Observaciones'])}"),
                        ("Alfa diario OLS", format_pct(selected_reg["Alpha por periodo"], 4)),
                        ("Alfa OLS anualizado", format_pct(selected_reg["Alpha OLS anual"], 3)),
                        ("Beta OLS", format_num(selected_reg["Beta OLS"], 4)),
                        ("R²", format_num(selected_reg["R2"], 4)),
                        ("R² ajustado", format_num(selected_reg["R2 ajustado"], 4)),
                        ("Error estándar del alfa", format_num(selected_reg["Error estandar alpha"], 6)),
                        ("Error estándar de beta", format_num(selected_reg["Error estandar beta"], 6)),
                        ("Estadístico t del alfa", format_num(selected_reg["t alpha"], 4)),
                        ("Estadístico t de beta", format_num(selected_reg["t beta"], 4)),
                        ("p-value del alfa", format_num(selected_reg["p-value alpha"], 6)),
                        ("p-value de beta", format_num(selected_reg["p-value beta"], 6)),
                        (
                            "Intervalo de confianza 95% del alfa",
                            f"[{format_num(selected_reg['IC 95% alpha inferior'], 6)}, {format_num(selected_reg['IC 95% alpha superior'], 6)}]",
                        ),
                        (
                            "Intervalo de confianza 95% de beta",
                            f"[{format_num(selected_reg['IC 95% beta inferior'], 4)}, {format_num(selected_reg['IC 95% beta superior'], 4)}]",
                        ),
                        ("SSE", format_num(selected_reg["SSE"], 6)),
                        ("Error estándar residual", format_num(selected_reg["Error estandar residual"], 6)),
                        ("Estadístico F", format_num(selected_reg["Estadistico F"], 4)),
                        ("p-value F", format_num(selected_reg["p-value F"], 6)),
                        ("Durbin-Watson", format_num(selected_reg["Durbin-Watson"], 4)),
                    ],
                    columns=["Estadística", "Valor"],
                )
                st.dataframe(ols_statistics, width="stretch", hide_index=True, height=700)

            st.markdown(
                f"""
                <div class="education-grid">
                    <b>Cómo interpretar CAPM y OLS para {selected_capm_asset}.</b><br>
                    Beta mide la sensibilidad histórica de los retornos excedentes frente a {benchmark}. La regresión OLS
                    estima esa relación y R² indica qué proporción de la variación observada explica el benchmark. CAPM usa
                    beta para obtener un retorno requerido; la SML lo representa gráficamente. El alfa de Jensen compara
                    el retorno histórico con ese retorno requerido. La significancia estadística se evalúa con p-values e
                    intervalos de confianza: un p-value bajo aporta evidencia muestral, pero no demuestra causalidad ni
                    garantiza que la relación persista.
                </div>
                """,
                unsafe_allow_html=True,
            )

            with st.container(border=True):
                st.markdown("**Tabla comparativa CAPM vs histórico**")
                st.dataframe(
                    pct_cols(capm_table, ["Retorno esperado CAPM", "Retorno historico anualizado", "Alpha Jensen anual", "Volatilidad anualizada", "Max Drawdown", "VaR 95%", "CVaR 95%"]),
                    width="stretch",
                    hide_index=True,
                    height=300,
                )

    with st.sidebar:
        st.divider()
        if st.button(
            "Guardar datos limpios",
            help="Crea archivos Excel en la carpeta relativa outputs/ sin modificar la base original.",
            width="stretch",
        ):
            try:
                created_files = save_clean_outputs(data, result1["indicators"])
                st.success(f"Se guardaron {len(created_files)} archivos en outputs/.")
            except OSError as exc:
                st.error(f"No fue posible escribir en outputs/: {exc}")
        st.download_button(
            "Exportar resultados en Excel",
            data=make_excel_download(result1, result2, result3),
            file_name="PortfolioLab_resultados.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            width="stretch",
        )

    st.caption("PortfolioLab listo para ejecutarse localmente o publicarse en Streamlit Cloud.")


if __name__ == "__main__":
    run_app()
