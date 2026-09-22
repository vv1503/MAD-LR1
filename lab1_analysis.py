# -*- coding: utf-8 -*-
"""
Лабораторная работа №1.
Описательная статистика. Корреляционный анализ.
Датасет: Apartment for Rent Classified (UCI, id=555)
"""

from __future__ import annotations

import math
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

warnings.filterwarnings("ignore", category=FutureWarning)

BASE = Path(__file__).resolve().parent
FIG = BASE / "figures"
RES = BASE / "results"
FIG.mkdir(exist_ok=True)
RES.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams["figure.dpi"] = 120
plt.rcParams["font.size"] = 10
# Поддержка кириллицы на Windows
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Segoe UI", "Arial", "DejaVu Sans", "Tahoma"]
plt.rcParams["axes.unicode_minus"] = False

# Русские подписи переменных на графиках
VAR_RU = {
    "price": "Цена аренды, USD/мес.",
    "bedrooms": "Число спален",
    "square_feet": "Площадь, кв. фут",
    "pets_yes": "Можно с животными (да=1)",
    "pets_label": "Можно с животными",
}


def sturges_bins(series: pd.Series) -> tuple[int, float]:
    """Число интервалов и длина интервала по формуле Стерджесса."""
    n = len(series.dropna())
    k = max(1, int(round(1 + 3.322 * math.log10(n))))
    width = (series.max() - series.min()) / k
    return k, width


def corr_significance(r: float, n: int, alpha: float = 0.05) -> dict:
    """t-критерий значимости коэффициента корреляции."""
    if n < 3 or abs(r) >= 1:
        return {"t": np.nan, "t_crit": np.nan, "p": np.nan, "significant": False}
    t_stat = r * math.sqrt((n - 2) / (1 - r**2))
    df = n - 2
    t_crit = stats.t.ppf(1 - alpha / 2, df)
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df))
    return {
        "t": t_stat,
        "t_crit": t_crit,
        "p": p_value,
        "significant": abs(t_stat) > t_crit,
    }


def cheddock(abs_r: float) -> str:
    if abs_r < 0.1:
        return "связь практически отсутствует"
    if abs_r < 0.3:
        return "слабая"
    if abs_r < 0.5:
        return "умеренная"
    if abs_r < 0.7:
        return "заметная"
    if abs_r < 0.9:
        return "высокая"
    return "весьма высокая"


def load_and_prepare() -> pd.DataFrame:
    raw_path = BASE / "data_raw.csv"
    print("Чтение data_raw.csv ...")
    df = pd.read_csv(
        raw_path,
        sep=";",
        encoding="utf-8",
        engine="python",
        on_bad_lines="skip",
    )
    cols = ["price", "bedrooms", "square_feet", "pets_allowed"]
    df = df[cols].copy()

    for c in ["price", "bedrooms", "square_feet"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Бинарная категориальная: можно ли с животными (да/нет)
    # Пустые значения трактуем как «нет» (животные не указаны / не разрешены)
    pets = df["pets_allowed"].astype(str).str.strip()
    df["pets_yes"] = pets.notna() & (~pets.isin(["", "nan", "None", "NaN"]))
    # Более строго: pets_allowed содержит Cats/Dogs
    df["pets_yes"] = pets.str.contains("Cat|Dog", case=False, na=False)
    df["pets_label"] = np.where(df["pets_yes"], "да", "нет")

    df = df.dropna(subset=["price", "bedrooms", "square_feet"])
    # Отсекаем явные выбросы/ошибки ввода для устойчивости анализа
    df = df[(df["price"] > 0) & (df["price"] <= 10000)]
    df = df[(df["square_feet"] > 100) & (df["square_feet"] <= 5000)]
    df = df[(df["bedrooms"] >= 0) & (df["bedrooms"] <= 8)]
    df = df.reset_index(drop=True)

    out = BASE / "data_clean.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"Очищенная выборка: {df.shape[0]} наблюдений -> {out.name}")
    return df


def point1_describe(df: pd.DataFrame) -> str:
    text = []
    text.append("Пункт 1. Описание данных и переменных")
    text.append(
        "Источник: Apartment for Rent Classified, UCI ML Repository, "
        "https://doi.org/10.24432/C5X623 (id=555)."
    )
    text.append(
        "Датасет объявлений об аренде квартир в США. Подходит для задач "
        "регрессии (прогноз непрерывной цены), классификации и кластеризации."
    )
    text.append("")
    text.append("Выбранные переменные (4 шт.):")
    text.append(
        "  • price (зависимая, Y) — стоимость аренды, USD/месяц, количественная;"
    )
    text.append(
        "  • bedrooms (независимая, X1) — число спален, количественная;"
    )
    text.append(
        "  • square_feet (независимая, X2) — площадь, кв. фут, количественная;"
    )
    text.append(
        "  • pets_label (независимая, X3) — можно ли с животными (да/нет), "
        "категориальная бинарная (из pets_allowed)."
    )
    text.append("")
    text.append(
        "Регрессионная постановка: price ~ bedrooms + square_feet + pets_yes. "
        "Модель регрессионная — да, целевая переменная непрерывная."
    )
    text.append(f"Объём рабочей выборки после очистки: n = {len(df)}.")
    report = "\n".join(text)
    (RES / "01_description.txt").write_text(report, encoding="utf-8")
    print(report)
    return report


def point2_descriptive(df: pd.DataFrame) -> pd.DataFrame:
    """Дескриптивный анализ в стиле «Описательная статистика» Excel."""
    print("\nПункт 2. Дескриптивный анализ и нормальность")

    # Таблица как в Excel: показатели — строки, переменные — столбцы
    excel_rows = [
        "Среднее",
        "Стандартная ошибка",
        "Медиана",
        "Мода",
        "Стандартное отклонение",
        "Дисперсия выборки",
        "Эксцесс",
        "Асимметричность",
        "Интервал (размах)",
        "Минимум",
        "Максимум",
        "Сумма",
        "Счёт",
    ]
    excel_tbl = pd.DataFrame(index=excel_rows)

    summary_rows = []
    for col in ["price", "bedrooms", "square_feet"]:
        s = df[col]
        n = len(s)
        mean = s.mean()
        std = s.std(ddof=1)
        se = std / math.sqrt(n)
        mode_val = s.mode().iloc[0] if not s.mode().empty else np.nan
        skew = s.skew()
        kurt = s.kurtosis()  # избыточный эксцесс, как в Excel

        excel_tbl[VAR_RU[col]] = [
            mean,
            se,
            s.median(),
            mode_val,
            std,
            s.var(ddof=1),
            kurt,
            skew,
            s.max() - s.min(),
            s.min(),
            s.max(),
            s.sum(),
            n,
        ]

        sample = s.sample(min(5000, n), random_state=42)
        sh_stat, sh_p = stats.shapiro(sample)
        _, k2_p = stats.normaltest(s)
        # По асимметрии/эксцессу: у нормального ≈ 0
        by_shape = abs(skew) < 0.5 and abs(kurt) < 1.0
        summary_rows.append(
            {
                "переменная": col,
                "название": VAR_RU[col],
                "асимметрия": skew,
                "эксцесс": kurt,
                "shapiro_p": sh_p,
                "normaltest_p": k2_p,
                "по_форме_(|A|<0.5,|E|<1)": "похоже на норм." if by_shape else "не похоже на норм.",
                "нормальность_(тест)": "да" if k2_p >= 0.05 else "нет",
            }
        )

    excel_tbl.to_csv(RES / "02_descriptive_excel.csv", encoding="utf-8-sig")
    excel_tbl.to_excel(RES / "02_descriptive_excel.xlsx")
    print("\nОписательная статистика (как в Excel):")
    print(excel_tbl.round(4).to_string())

    desc = pd.DataFrame(summary_rows)
    desc.to_csv(RES / "02_normality.csv", index=False, encoding="utf-8-sig")
    print("\nОценка нормальности:")
    print(desc.to_string(index=False, float_format=lambda x: f"{x:.4g}"))

    # Категориальная переменная — частоты (аналог сводной в Excel)
    cats = df["pets_label"].value_counts().rename_axis("Можно с животными").reset_index(name="Частота")
    cats["Доля, %"] = (cats["Частота"] / cats["Частота"].sum() * 100).round(2)
    cats.to_csv(RES / "02_pets_freq.csv", index=False, encoding="utf-8-sig")
    print("\nЧастоты (можно с животными):\n", cats.to_string(index=False))

    # Удаляем старый график ящиков, если остался
    old_box = FIG / "02_boxplots.png"
    if old_box.exists():
        old_box.unlink()

    return desc


def point3_histograms(df: pd.DataFrame) -> None:
    print("\nПункт 3. Гистограммы (формула Стерджесса)")
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    lines = []
    for ax, col in zip(axes, ["price", "bedrooms", "square_feet"]):
        k, width = sturges_bins(df[col])
        # Для bedrooms — дискретные целые бины
        if col == "bedrooms":
            bins = np.arange(df[col].min(), df[col].max() + 2) - 0.5
            k_eff = int(df[col].nunique())
            width_eff = 1.0
        else:
            bins = k
            k_eff, width_eff = k, width
        ax.hist(df[col], bins=bins, color="steelblue", edgecolor="white", alpha=0.85)
        ax.set_title(f"{VAR_RU[col]}\nk = {k_eff}, длина интервала h ≈ {width_eff:.2f}")
        ax.set_xlabel(VAR_RU[col])
        ax.set_ylabel("Частота")
        lines.append(f"{col}: n={len(df)}, k={k_eff}, длина интервала h={width_eff:.4f}")
    fig.suptitle("Гистограммы по формуле Стерджесса (пункт 3)")
    fig.tight_layout()
    fig.savefig(FIG / "03_histograms.png", bbox_inches="tight")
    plt.close(fig)
    (RES / "03_sturges.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


def point4_contingency(df: pd.DataFrame) -> None:
    print("\nПункт 4. Таблицы сопряжённости")
    # Категории: pets (да/нет) и уровень цены (ниже/выше медианы)
    work = df.copy()
    med = work["price"].median()
    work["price_cat"] = np.where(work["price"] <= med, "цена ≤ медианы", "цена > медианы")
    # Также bedrooms как категории: студия/1/2/3+
    work["bed_cat"] = pd.cut(
        work["bedrooms"],
        bins=[-0.1, 0.5, 1.5, 2.5, 100],
        labels=["0", "1", "2", "3+"],
    )

    ct1 = pd.crosstab(work["pets_label"], work["price_cat"], margins=True)
    ct1_pct = pd.crosstab(work["pets_label"], work["price_cat"], normalize="index") * 100
    chi2, p, dof, expected = stats.chi2_contingency(pd.crosstab(work["pets_label"], work["price_cat"]))

    ct2 = pd.crosstab(work["bed_cat"], work["pets_label"], margins=True)
    chi2b, pb, dofb, _ = stats.chi2_contingency(pd.crosstab(work["bed_cat"], work["pets_label"]))

    ct1.to_csv(RES / "04_crosstab_pets_price.csv", encoding="utf-8-sig")
    ct2.to_csv(RES / "04_crosstab_bed_pets.csv", encoding="utf-8-sig")

    comment = (
        f"Таблица 1: pets_label × price_cat\n{ct1}\n\n"
        f"Доли по строкам (%):\n{ct1_pct.round(2)}\n\n"
        f"χ² = {chi2:.4f}, df = {dof}, p-value = {p:.4g}\n"
        f"Вывод: {'зависимость есть' if p < 0.05 else 'независимость не отвергается'} "
        f"(α=0.05).\n\n"
        f"Таблица 2: bed_cat × pets_label\n{ct2}\n\n"
        f"χ² = {chi2b:.4f}, df = {dofb}, p-value = {pb:.4g}\n"
        f"Вывод: {'зависимость есть' if pb < 0.05 else 'независимость не отвергается'} "
        f"(α=0.05)."
    )
    (RES / "04_contingency_comment.txt").write_text(comment, encoding="utf-8")
    print(comment)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ct_pets_price = pd.crosstab(work["pets_label"], work["price_cat"])
    ct_pets_price.index.name = "Можно с животными"
    ct_pets_price.columns.name = "Уровень цены"
    sns.heatmap(ct_pets_price, annot=True, fmt="d", cmap="Blues", ax=axes[0])
    axes[0].set_title("Животные × уровень цены")
    axes[0].set_xlabel("Уровень цены")
    axes[0].set_ylabel("Можно с животными")

    ct_bed_pets = pd.crosstab(work["bed_cat"], work["pets_label"])
    ct_bed_pets.index.name = "Число спален"
    ct_bed_pets.columns.name = "Можно с животными"
    sns.heatmap(ct_bed_pets, annot=True, fmt="d", cmap="Greens", ax=axes[1])
    axes[1].set_title("Число спален × разрешение животных")
    axes[1].set_xlabel("Можно с животными")
    axes[1].set_ylabel("Число спален")
    fig.suptitle("Таблицы сопряжённости (пункт 4)")
    fig.tight_layout()
    fig.savefig(FIG / "04_contingency.png", bbox_inches="tight")
    plt.close(fig)


def point5_7_correlation(df: pd.DataFrame, desc: pd.DataFrame) -> pd.DataFrame:
    print("\nПункты 5–7. Корреляционный анализ, значимость, сила связи")
    # Выборки не нормальны -> Спирмен (+ Пирсон для сравнения)
    num = df[["price", "bedrooms", "square_feet"]]
    # Бисериальная/точечно-бисериальная для pets_yes vs price: Пирсон с 0/1
    work = num.copy()
    work["pets_yes"] = df["pets_yes"].astype(int)

    pearson = work.corr(method="pearson")
    spearman = work.corr(method="spearman")
    pearson.to_csv(RES / "05_pearson.csv", encoding="utf-8-sig")
    spearman.to_csv(RES / "05_spearman.csv", encoding="utf-8-sig")

    pairs = [
        ("price", "bedrooms"),
        ("price", "square_feet"),
        ("price", "pets_yes"),
        ("bedrooms", "square_feet"),
        ("bedrooms", "pets_yes"),
        ("square_feet", "pets_yes"),
    ]
    n = len(work)
    rows = []
    for a, b in pairs:
        r_p = work[a].corr(work[b], method="pearson")
        r_s = work[a].corr(work[b], method="spearman")
        sig_p = corr_significance(r_p, n)
        sig_s = corr_significance(r_s, n)
        # Основной — Спирмен (ненормальность)
        rows.append(
            {
                "пара": f"{a}–{b}",
                "Pearson_r": r_p,
                "Spearman_rho": r_s,
                "t_Spearman": sig_s["t"],
                "t_crit": sig_s["t_crit"],
                "p_Spearman": sig_s["p"],
                "значим_Spearman": sig_s["significant"],
                "сила_по_Чеддоку": cheddock(abs(r_s)),
            }
        )

    corr_tbl = pd.DataFrame(rows)
    corr_tbl.to_csv(RES / "05_07_corr_significance.csv", index=False, encoding="utf-8-sig")
    print(corr_tbl.to_string(index=False, float_format=lambda x: f"{x:.4g}"))

    # Heatmaps с русскими именами
    rename_ru = {k: VAR_RU[k] for k in work.columns}
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.heatmap(
        pearson.rename(index=rename_ru, columns=rename_ru),
        annot=True, fmt=".3f", cmap="RdBu_r", center=0, ax=axes[0],
    )
    axes[0].set_title("Коэффициент Пирсона")
    sns.heatmap(
        spearman.rename(index=rename_ru, columns=rename_ru),
        annot=True, fmt=".3f", cmap="RdBu_r", center=0, ax=axes[1],
    )
    axes[1].set_title("Коэффициент Спирмена (основной)")
    fig.suptitle("Корреляционные матрицы (пункты 5–7)")
    fig.tight_layout()
    fig.savefig(FIG / "05_corr_heatmaps.png", bbox_inches="tight")
    plt.close(fig)

    # Scatter
    sample = df.sample(min(3000, len(df)), random_state=0)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    sns.scatterplot(data=sample, x="bedrooms", y="price", ax=axes[0], alpha=0.3, s=12)
    axes[0].set_title("Цена аренды и число спален")
    axes[0].set_xlabel(VAR_RU["bedrooms"])
    axes[0].set_ylabel(VAR_RU["price"])
    sns.scatterplot(data=sample, x="square_feet", y="price", ax=axes[1], alpha=0.3, s=12)
    axes[1].set_title("Цена аренды и площадь")
    axes[1].set_xlabel(VAR_RU["square_feet"])
    axes[1].set_ylabel(VAR_RU["price"])
    fig.suptitle("Диаграммы рассеяния (пункты 5–7)")
    fig.tight_layout()
    fig.savefig(FIG / "05_scatter.png", bbox_inches="tight")
    plt.close(fig)

    strong = corr_tbl[corr_tbl["Spearman_rho"].abs() >= 0.5]
    weak = corr_tbl[corr_tbl["Spearman_rho"].abs() < 0.3]
    note = (
        "Сильно/заметно коррелированные (|ρ|≥0.5):\n"
        + strong[["пара", "Spearman_rho", "сила_по_Чеддоку"]].to_string(index=False)
        + "\n\nСлабо коррелированные (|ρ|<0.3):\n"
        + weak[["пара", "Spearman_rho", "сила_по_Чеддоку"]].to_string(index=False)
        + "\n\nПримечание: распределения не нормальны (см. п.2), поэтому "
        "основной коэффициент — Спирмена; Пирсон приведён для сравнения. "
        "Для pets_yes (0/1) коэффициент Пирсона = точечно-бисериальная корреляция."
    )
    (RES / "07_strong_weak.txt").write_text(note, encoding="utf-8")
    print(note)
    return corr_tbl


def point_regression_bonus(df: pd.DataFrame) -> None:
    """Доп.: множественная линейная регрессия (датасет регрессионный)."""
    print("\nДополнение. Множественная линейная регрессия price ~ X")
    X = df[["bedrooms", "square_feet"]].copy()
    X["pets_yes"] = df["pets_yes"].astype(int)
    y = df["price"]
    model = LinearRegression()
    model.fit(X, y)
    pred = model.predict(X)
    r2 = r2_score(y, pred)
    rmse = mean_squared_error(y, pred) ** 0.5
    coefs = pd.Series(model.coef_, index=X.columns)
    out = (
        f"Intercept = {model.intercept_:.4f}\n"
        f"Коэффициенты:\n{coefs.to_string()}\n"
        f"R² = {r2:.4f}\nRMSE = {rmse:.2f}\n"
        "Интерпретация: модель регрессионная (непрерывный Y=price). "
        "Площадь (square_feet) имеет устойчивый положительный вклад. "
        "Коэффициент bedrooms может стать отрицательным из-за сильной "
        "связи bedrooms↔square_feet (мультиколлинеарность): эффект «комнат» "
        "частично уже учтён площадью. pets_yes отражает средний сдвиг цены "
        "при разрешении животных (в данной выборке — небольшой отрицательный)."
    )
    (RES / "bonus_regression.txt").write_text(out, encoding="utf-8")
    print(out)

    fig, ax = plt.subplots(figsize=(5.5, 5))
    idx = y.sample(min(4000, len(y)), random_state=1).index
    ax.scatter(y.loc[idx], pd.Series(pred, index=y.index).loc[idx], alpha=0.25, s=10)
    lims = [min(y.min(), pred.min()), max(y.max(), pred.max())]
    ax.plot(lims, lims, "r--", lw=1, label="Идеальное совпадение")
    ax.set_xlabel("Фактическая цена аренды")
    ax.set_ylabel("Прогноз цены по модели")
    ax.set_title(f"Множественная линейная регрессия (R² = {r2:.3f})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "bonus_regression.png", bbox_inches="tight")
    plt.close(fig)


def point9_spurious() -> None:
    print("\nПункт 9. Ложная корреляция (Tyler Vigen)")
    # Классический пример: сыр vs смерти от запутывания в простынях, 2000–2009
    # Источник: tylervigen.com / Error Statistics Philosophy
    years = np.arange(1, 11)  # отсчёты 1..10 вместо годов (как разрешено в задании)
    cheese = np.array([29.8, 30.1, 30.5, 30.6, 31.3, 31.7, 32.6, 33.1, 32.7, 32.8])
    bed_deaths = np.array([327, 456, 509, 497, 596, 573, 661, 741, 809, 717])
    year_labels = list(range(2000, 2010))

    spurious = pd.DataFrame(
        {
            "year": year_labels,
            "t": years,
            "cheese_per_capita": cheese,
            "bedsheet_deaths": bed_deaths,
        }
    )
    spurious.to_csv(RES / "09_spurious_data.csv", index=False, encoding="utf-8-sig")

    r_raw = np.corrcoef(cheese, bed_deaths)[0, 1]
    sig_raw = corr_significance(r_raw, len(cheese))

    # 9.1–9.2 диаграмма + линии тренда (линейная регрессия)
    fig, ax1 = plt.subplots(figsize=(10, 5.5))
    ax2 = ax1.twinx()
    ax1.plot(year_labels, cheese, "o-", color="C0", label="Потребление сыра")
    ax2.plot(year_labels, bed_deaths, "s-", color="C1", label="Смерти (простыни)")
    # тренды
    m1 = LinearRegression().fit(years.reshape(-1, 1), cheese)
    m2 = LinearRegression().fit(years.reshape(-1, 1), bed_deaths)
    cheese_hat = m1.predict(years.reshape(-1, 1))
    deaths_hat = m2.predict(years.reshape(-1, 1))
    ax1.plot(year_labels, cheese_hat, "--", color="C0", alpha=0.7, label="Тренд (сыр)")
    ax2.plot(year_labels, deaths_hat, "--", color="C1", alpha=0.7, label="Тренд (смерти)")
    ax1.set_xlabel("Год")
    ax1.set_ylabel("Потребление сыра, фунты на человека", color="C0")
    ax2.set_ylabel("Число смертей от запутывания в простынях", color="C1")
    eq1 = f"сыр = {m1.coef_[0]:.3f}·t + {m1.intercept_:.3f}"
    eq2 = f"смерти = {m2.coef_[0]:.2f}·t + {m2.intercept_:.2f}"
    ax1.set_title(
        "Ложная корреляция (пункт 9): сыр и смерти от простыней\n"
        f"r = {r_raw:.3f}\n{eq1};  {eq2}"
    )
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG / "09_spurious_trends.png", bbox_inches="tight")
    plt.close(fig)

    # 9.3–9.5 остатки и стандартизированные остатки
    res_c = cheese - cheese_hat
    res_d = bed_deaths - deaths_hat
    std_c = (res_c - res_c.mean()) / res_c.std(ddof=1)
    std_d = (res_d - res_d.mean()) / res_d.std(ddof=1)

    resid_df = spurious.copy()
    resid_df["cheese_hat"] = cheese_hat
    resid_df["deaths_hat"] = deaths_hat
    resid_df["resid_cheese"] = res_c
    resid_df["resid_deaths"] = res_d
    resid_df["std_resid_cheese"] = std_c
    resid_df["std_resid_deaths"] = std_d
    resid_df.to_csv(RES / "09_residuals.csv", index=False, encoding="utf-8-sig")

    # 9.6 график остатков + гистограмма станд. остатков
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes[0, 0].plot(year_labels, res_c, "o-", color="C0")
    axes[0, 0].axhline(0, color="gray", ls="--")
    axes[0, 0].set_title("Остатки ряда «потребление сыра»\n(факт − тренд)")
    axes[0, 0].set_xlabel("Год")
    axes[0, 0].set_ylabel("Остаток")

    axes[0, 1].plot(year_labels, res_d, "o-", color="C1")
    axes[0, 1].axhline(0, color="gray", ls="--")
    axes[0, 1].set_title("Остатки ряда «смерти от простыней»\n(факт − тренд)")
    axes[0, 1].set_xlabel("Год")
    axes[0, 1].set_ylabel("Остаток")

    axes[1, 0].hist(std_c, bins=5, color="C0", edgecolor="white")
    axes[1, 0].set_title("Стандартизированные остатки\n(потребление сыра)")
    axes[1, 0].set_xlabel("Z-оценка остатка")
    axes[1, 0].set_ylabel("Частота")

    axes[1, 1].hist(std_d, bins=5, color="C1", edgecolor="white")
    axes[1, 1].set_title("Стандартизированные остатки\n(смерти от простыней)")
    axes[1, 1].set_xlabel("Z-оценка остатка")
    axes[1, 1].set_ylabel("Частота")

    fig.suptitle(
        "Пункт 9.6: остатки после удаления тренда\n"
        "(нужны, чтобы проверить: связь исчезла или осталась)"
    )
    fig.tight_layout()
    fig.savefig(FIG / "09_residuals.png", bbox_inches="tight")
    plt.close(fig)

    # 9.7–9.8 корреляция остатков и значимость
    r_res = np.corrcoef(res_c, res_d)[0, 1]
    sig_res = corr_significance(r_res, len(res_c))

    interpretation = f"""
Пункт 9. Ложная корреляция
Источник примера: http://www.tylervigen.com/spurious-correlations
Ряды (2000–2009):
  X — потребление сыра на душу населения (США), фунты;
  Y — число смертей от запутывания в простынях.

9.1–9.2. Оба ряда имеют выраженный восходящий тренд; уравнения:
  {eq1}
  {eq2}

9.3–9.5. Остатки = факт − прогноз по линейному тренду; стандартизация — z-оценки остатков.

Корреляция исходных рядов: r = {r_raw:.4f}
  t = {sig_raw['t']:.4f}, t_crit(α=0.05, df=8) = {sig_raw['t_crit']:.4f}, p = {sig_raw['p']:.4g}
  => коэффициент {'значим' if sig_raw['significant'] else 'не значим'}.

Корреляция остатков (после удаления тренда): r = {r_res:.4f}
  t = {sig_res['t']:.4f}, t_crit = {sig_res['t_crit']:.4f}, p = {sig_res['p']:.4g}
  => коэффициент {'значим' if sig_res['significant'] else 'не значим'}.

9.9. Вывод о причине ложной корреляции:
Оба показателя растут во времени (общий тренд / «третья переменная» — время,
рост населения, изменение учёта и т.п.). После удаления тренда связь
ослабевает или теряет значимость — классический пример spurious correlation:
высокая корреляция уровней не означает причинно-следственную связь.
""".strip()
    (RES / "09_interpretation.txt").write_text(interpretation, encoding="utf-8")
    print(interpretation)


def write_report(df: pd.DataFrame, desc: pd.DataFrame, corr_tbl: pd.DataFrame) -> None:
    """Краткая сводка без перезаписи подробного ОТЧЕТ.md."""
    stub = (
        f"Отчёт по ЛР1 готов. n={len(df)}. "
        f"Детали по пунктам — в папке results/ и файле ОТЧЕТ.md.\n"
        f"Датасет подходит для регрессии (Y=price).\n"
    )
    (RES / "00_summary.txt").write_text(stub, encoding="utf-8")
    print("\nОтчёт: ОТЧЕТ.md (+ results/00_summary.txt)")


def main() -> None:
    df = load_and_prepare()
    point1_describe(df)
    desc = point2_descriptive(df)
    point3_histograms(df)
    point4_contingency(df)
    corr_tbl = point5_7_correlation(df, desc)
    point_regression_bonus(df)
    point9_spurious()
    write_report(df, desc, corr_tbl)
    print("\nГотово. Все пункты 1–9 выполнены.")


if __name__ == "__main__":
    main()
