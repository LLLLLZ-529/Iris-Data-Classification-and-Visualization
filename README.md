# 🌸 Iris Data Classification and Visualization

鸢尾花（Iris）数据集分类与可视化实验：数据探索 → 特征可视化 → 经典机器学习模型对比评估。

## ✨ 功能

- 📊 **数据可视化**：特征分布直方图、散点矩阵、箱线图、相关性热力图（自动保存到 `results/`）
- 🤖 **分类实验**：KNN / 逻辑回归 / 决策树 / SVM(RBF) 四模型对比
- 📈 **结果评估**：准确率、5 折交叉验证、混淆矩阵、分类报告

## 🚀 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 运行实验
python iris_classification.py

# 3. 查看结果
# 控制台输出模型对比；图表与 CSV 保存在 results/
```

## 📁 项目结构

```
Iris-Data-Classification-and-Visualization/
├── iris_classification.py   # 主实验脚本（EDA + 建模 + 评估）
├── requirements.txt         # 依赖清单
├── .gitignore               # 忽略 __pycache__ / results 等
├── results/                 # 运行后生成：图表 + model_comparison.csv
└── README.md
```

## 📊 运行结果（实测）

```
=== 模型对比（按测试集准确率排序）===
              Model  Test Accuracy  CV Accuracy (5-fold)
          SVM (RBF)         0.9333                0.9714
                KNN         0.9111                0.9524
Logistic Regression         0.9111                0.9810
      Decision Tree         0.9111                0.9429

最佳模型：SVM (RBF) （测试准确率 0.9333，5 折 CV 0.9714）
```

> 实测环境：scikit-learn 内置 Iris（150 条 × 4 特征，3 类），test_size=0.3、random_state=42、分层抽样，特征标准化后训练。

## 🔬 后续可扩展

- [ ] 网格搜索调参（`GridSearchCV`）
- [ ] 特征选择/降维（PCA）对比实验
- [ ] 混淆矩阵与分类报告的详细解读
- [ ] 补充 Notebook 版实验记录

## 📄 许可

未指定开源许可（默认保留所有权利）。
