# -*- coding: utf-8 -*-
"""
Iris 鸢尾花数据集：分类与可视化实验

用法:
    python iris_classification.py

输出:
    results/ 目录下生成 4 张 EDA 图 + 4 张混淆矩阵图 + model_comparison.csv
    控制台打印数据概览、分类报告与模型对比
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # 无图形界面环境（服务器/CI）也能运行

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

# 参与对比的 4 个经典模型（均为 sklearn 内置，无额外依赖）
MODELS = {
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Logistic Regression": LogisticRegression(max_iter=500, random_state=42),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
    "SVM (RBF)": SVC(kernel="rbf", C=1.0, gamma="scale", random_state=42),
}


def load_data():
    """加载鸢尾花数据集并转为 DataFrame"""
    data = load_iris()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["species"] = pd.Categorical.from_codes(data.target, data.target_names)
    return df, data


def plot_distributions(df):
    """特征分布直方图 + KDE"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, col in zip(axes.ravel(), df.columns[:-1]):
        sns.histplot(data=df, x=col, hue="species", kde=True, ax=ax)
        ax.set_title(f"Distribution of {col}")
    fig.suptitle("Feature Distributions by Species", fontsize=14)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "1_feature_distributions.png", dpi=150)
    plt.close(fig)


def plot_pairplot(df):
    """散点矩阵：观察特征两两关系与类别可分性"""
    g = sns.pairplot(df, hue="species", diag_kind="kde", height=2.2)
    g.figure.suptitle("Pairplot of Iris Features", y=1.02)
    g.savefig(RESULTS_DIR / "2_pairplot.png", dpi=150)
    plt.close("all")


def plot_boxplots(df):
    """每个特征按物种的箱线图"""
    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    for ax, col in zip(axes, df.columns[:-1]):
        sns.boxplot(data=df, x="species", y=col, ax=ax)
        ax.set_title(col)
    fig.suptitle("Boxplots by Species", fontsize=14)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "3_boxplots.png", dpi=150)
    plt.close(fig)


def plot_correlation(df):
    """特征相关性热力图"""
    corr = df.drop(columns="species").corr()
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    ax.set_title("Feature Correlation Matrix")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "4_correlation.png", dpi=150)
    plt.close(fig)


def evaluate_models(X_train, X_test, y_train, y_test, target_names):
    """训练 4 个模型，打印分类报告并返回对比汇总"""
    summary = []
    for name, model in MODELS.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        cv = cross_val_score(model, X_train, y_train, cv=5).mean()
        summary.append(
            {
                "Model": name,
                "Test Accuracy": round(acc, 4),
                "CV Accuracy (5-fold)": round(cv, 4),
            }
        )
        print(f"\n{'=' * 52}\n{name}\n{'=' * 52}")
        print(classification_report(y_test, y_pred, target_names=target_names))
    return summary


def plot_confusion_matrices(X_test, y_test, target_names):
    """每个模型的混淆矩阵"""
    for name, model in MODELS.items():
        disp = ConfusionMatrixDisplay.from_estimator(
            model, X_test, y_test, display_labels=target_names, cmap="Blues"
        )
        disp.figure_.suptitle(f"Confusion Matrix — {name}")
        safe = name.replace(" ", "_").replace("(", "").replace(")", "")
        disp.figure_.savefig(RESULTS_DIR / f"5_confusion_{safe}.png", dpi=150)
        plt.close("all")


def main():
    df, data = load_data()
    target_names = data.target_names

    # ---- 1. 数据概览 ----
    print("=== 数据概览（前 5 行）===")
    print(df.head())
    print("\n=== 统计描述 ===")
    print(df.describe().round(3))
    print("\n=== 类别分布 ===")
    print(df["species"].value_counts())

    # ---- 2. 数据可视化 ----
    print("\n生成 EDA 图表 ...")
    plot_distributions(df)
    plot_pairplot(df)
    plot_boxplots(df)
    plot_correlation(df)

    # ---- 3. 建模与评估 ----
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # KNN / 逻辑回归 / SVM 对特征尺度敏感，先标准化
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    summary = evaluate_models(X_train, X_test, y_train, y_test, target_names)
    plot_confusion_matrices(X_test, y_test, target_names)

    # ---- 4. 结果汇总 ----
    result_df = pd.DataFrame(summary).sort_values("Test Accuracy", ascending=False)
    print("\n=== 模型对比（按测试集准确率排序）===")
    print(result_df.to_string(index=False))
    result_df.to_csv(RESULTS_DIR / "model_comparison.csv", index=False)

    best = result_df.iloc[0]
    print(
        f"\n最佳模型：{best['Model']} "
        f"（测试准确率 {best['Test Accuracy']}，5 折 CV {best['CV Accuracy (5-fold)']}）"
    )
    print(f"\n图表与结果已保存到 {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
