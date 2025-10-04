import marimo

__generated_with = "0.16.5"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Scikit-learn 完全指南

    ## 🎯 什么是Scikit-learn？

    **Scikit-learn** 是Python中最流行的机器学习库，提供了简单高效的数据挖掘和数据分析工具。它建立在NumPy、SciPy和matplotlib之上。

    ### 核心特点

    - **简单一致的API**：所有算法都遵循相同的接口模式
    - **丰富的算法库**：涵盖分类、回归、聚类、降维等
    - **优秀的文档**：详细的用户指南和API文档
    - **活跃的社区**：持续更新和维护
    - **生产就绪**：经过充分测试，可用于生产环境

    ### 主要功能模块

    1. **监督学习**：分类和回归算法
    2. **无监督学习**：聚类、降维、异常检测
    3. **模型选择**：交叉验证、网格搜索、评估指标
    4. **数据预处理**：特征缩放、编码、转换
    5. **特征工程**：特征选择、特征提取
    6. **集成方法**：Bagging、Boosting、Stacking
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 📦 安装和导入

    首先让我们导入必要的库：
    """
    )
    return


@app.cell
def _():
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn import __version__ as sklearn_version

    print(f"✅ Scikit-learn 版本: {sklearn_version}")
    print(f"✅ NumPy 版本: {np.__version__}")
    print(f"✅ Pandas 版本: {pd.__version__}")

    # 设置随机种子以保证结果可复现
    np.random.seed(42)

    # 设置绘图样式
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    return np, pd, plt, seaborn, sklearn_version, sns


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🏗️ Scikit-learn的核心API设计

    Scikit-learn的所有算法都遵循统一的API模式：

    ### 1. Estimator（估计器）
    所有机器学习算法都实现了`fit()`方法：
    ```python
    estimator.fit(X_train, y_train)
    ```

    ### 2. Predictor（预测器）
    监督学习算法实现了`predict()`方法：
    ```python
    predictions = estimator.predict(X_test)
    ```

    ### 3. Transformer（转换器）
    数据预处理类实现了`transform()`方法：
    ```python
    X_transformed = transformer.transform(X)
    ```

    ### 4. Pipeline（管道）
    可以将多个步骤串联起来：
    ```python
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression())
    ])
    ```
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 📊 数据集准备

    让我们使用scikit-learn内置的数据集来演示各种功能：
    """
    )
    return


@app.cell
def _(np, pd):
    from sklearn.datasets import load_iris, load_diabetes, make_classification, make_regression

    # 1. 加载鸢尾花数据集（分类任务）
    iris = load_iris()
    iris_df = pd.DataFrame(
        data=iris.data,
        columns=iris.feature_names
    )
    iris_df['target'] = iris.target
    iris_df['species'] = iris_df['target'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

    print("🌸 鸢尾花数据集（分类）:")
    print(f"  样本数: {len(iris_df)}")
    print(f"  特征数: {len(iris.feature_names)}")
    print(f"  类别数: {len(iris.target_names)}")
    print(f"  类别: {iris.target_names}")

    # 2. 加载糖尿病数据集（回归任务）
    diabetes = load_diabetes()
    diabetes_df = pd.DataFrame(
        data=diabetes.data,
        columns=diabetes.feature_names
    )
    diabetes_df['target'] = diabetes.target

    print("\n💉 糖尿病数据集（回归）:")
    print(f"  样本数: {len(diabetes_df)}")
    print(f"  特征数: {len(diabetes.feature_names)}")

    # 3. 生成自定义分类数据集
    X_class, y_class = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_classes=2,
        random_state=42
    )

    print("\n🎲 自定义分类数据集:")
    print(f"  样本数: {len(X_class)}")
    print(f"  特征数: {X_class.shape[1]}")
    print(f"  类别分布: {np.bincount(y_class)}")

    return X_class, diabetes, diabetes_df, iris, iris_df, load_diabetes, load_iris, make_classification, make_regression, y_class


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🔄 数据预处理

    数据预处理是机器学习流程中的关键步骤。Scikit-learn提供了丰富的预处理工具：
    """
    )
    return


@app.cell
def _(X_class, np, y_class):
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
    from sklearn.impute import SimpleImputer

    print("🔧 数据预处理演示:")
    print("=" * 50)

    # 1. 数据分割
    X_train, X_test, y_train, y_test = train_test_split(
        X_class, y_class, test_size=0.2, random_state=42, stratify=y_class
    )
    print(f"\n1️⃣ 数据分割:")
    print(f"  训练集: {X_train.shape}")
    print(f"  测试集: {X_test.shape}")

    # 2. 特征缩放 - StandardScaler（标准化）
    scaler_standard = StandardScaler()
    X_train_scaled = scaler_standard.fit_transform(X_train)
    X_test_scaled = scaler_standard.transform(X_test)

    print(f"\n2️⃣ 标准化（StandardScaler）:")
    print(f"  原始数据均值: {X_train[:, 0].mean():.4f}")
    print(f"  原始数据标准差: {X_train[:, 0].std():.4f}")
    print(f"  标准化后均值: {X_train_scaled[:, 0].mean():.4f}")
    print(f"  标准化后标准差: {X_train_scaled[:, 0].std():.4f}")

    # 3. 特征缩放 - MinMaxScaler（归一化）
    scaler_minmax = MinMaxScaler()
    X_train_normalized = scaler_minmax.fit_transform(X_train)

    print(f"\n3️⃣ 归一化（MinMaxScaler）:")
    print(f"  原始数据范围: [{X_train[:, 0].min():.4f}, {X_train[:, 0].max():.4f}]")
    print(f"  归一化后范围: [{X_train_normalized[:, 0].min():.4f}, {X_train_normalized[:, 0].max():.4f}]")

    # 4. 缺失值处理
    X_with_missing = X_train.copy()
    X_with_missing[0:10, 0] = np.nan

    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X_with_missing)

    print(f"\n4️⃣ 缺失值处理:")
    print(f"  缺失值数量: {np.isnan(X_with_missing).sum()}")
    print(f"  填充后缺失值: {np.isnan(X_imputed).sum()}")

    return LabelEncoder, MinMaxScaler, OneHotEncoder, SimpleImputer, StandardScaler, X_imputed, X_test, X_test_scaled, X_train, X_train_normalized, X_train_scaled, X_with_missing, imputer, scaler_minmax, scaler_standard, train_test_split, y_test, y_train


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🎯 监督学习：分类算法

    让我们探索scikit-learn中最常用的分类算法：
    """
    )
    return


@app.cell
def _(X_test_scaled, X_train_scaled, y_test, y_train):
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.svm import SVC
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

    print("🎯 分类算法对比:")
    print("=" * 70)

    classifiers = {
        "逻辑回归": LogisticRegression(random_state=42, max_iter=1000),
        "决策树": DecisionTreeClassifier(random_state=42, max_depth=5),
        "随机森林": RandomForestClassifier(n_estimators=100, random_state=42),
        "梯度提升": GradientBoostingClassifier(n_estimators=100, random_state=42),
        "支持向量机": SVC(kernel='rbf', random_state=42),
        "K近邻": KNeighborsClassifier(n_neighbors=5),
        "朴素贝叶斯": GaussianNB()
    }

    results = []

    for name, clf in classifiers.items():
        # 训练模型
        clf.fit(X_train_scaled, y_train)

        # 预测
        y_pred = clf.predict(X_test_scaled)

        # 评估
        accuracy = accuracy_score(y_test, y_pred)

        results.append({
            "算法": name,
            "准确率": f"{accuracy:.4f}",
            "模型对象": clf
        })

        print(f"{name:12s} - 准确率: {accuracy:.4f}")

    print("\n✅ 所有分类器训练完成")

    return DecisionTreeClassifier, GaussianNB, GradientBoostingClassifier, KNeighborsClassifier, LogisticRegression, RandomForestClassifier, SVC, accuracy, accuracy_score, classification_report, clf, classifiers, confusion_matrix, name, results, y_pred


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 📈 监督学习：回归算法

    回归算法用于预测连续值：
    """
    )
    return


@app.cell
def _(diabetes):
    from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.svm import SVR
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

    # 准备回归数据
    X_reg = diabetes.data
    y_reg = diabetes.target

    from sklearn.model_selection import train_test_split as tts
    X_reg_train, X_reg_test, y_reg_train, y_reg_test = tts(
        X_reg, y_reg, test_size=0.2, random_state=42
    )

    print("📈 回归算法对比:")
    print("=" * 80)

    regressors = {
        "线性回归": LinearRegression(),
        "岭回归": Ridge(alpha=1.0),
        "Lasso回归": Lasso(alpha=1.0),
        "弹性网络": ElasticNet(alpha=1.0, l1_ratio=0.5),
        "决策树回归": DecisionTreeRegressor(random_state=42, max_depth=5),
        "随机森林回归": RandomForestRegressor(n_estimators=100, random_state=42),
        "梯度提升回归": GradientBoostingRegressor(n_estimators=100, random_state=42),
    }

    regression_results = []

    for reg_name, regressor in regressors.items():
        # 训练
        regressor.fit(X_reg_train, y_reg_train)

        # 预测
        y_reg_pred = regressor.predict(X_reg_test)

        # 评估
        mse = mean_squared_error(y_reg_test, y_reg_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_reg_test, y_reg_pred)
        r2 = r2_score(y_reg_test, y_reg_pred)

        regression_results.append({
            "算法": reg_name,
            "RMSE": f"{rmse:.2f}",
            "MAE": f"{mae:.2f}",
            "R²": f"{r2:.4f}"
        })

        print(f"{reg_name:15s} - RMSE: {rmse:6.2f}, MAE: {mae:6.2f}, R²: {r2:.4f}")

    print("\n✅ 所有回归器训练完成")

    return DecisionTreeRegressor, ElasticNet, GradientBoostingRegressor, Lasso, LinearRegression, RandomForestRegressor, Ridge, SVR, X_reg, X_reg_test, X_reg_train, mae, mean_absolute_error, mean_squared_error, mse, r2, r2_score, reg_name, regressor, regressors, regression_results, rmse, tts, y_reg, y_reg_pred, y_reg_test, y_reg_train


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🔍 模型评估与选择

    Scikit-learn提供了丰富的模型评估工具：
    """
    )
    return


@app.cell
def _(X_train_scaled, y_train):
    from sklearn.model_selection import cross_val_score, cross_validate, KFold, StratifiedKFold
    from sklearn.linear_model import LogisticRegression as LR

    print("🔍 交叉验证演示:")
    print("=" * 50)

    model_cv = LR(random_state=42, max_iter=1000)

    # 1. 简单交叉验证
    cv_scores = cross_val_score(model_cv, X_train_scaled, y_train, cv=5)

    print(f"\n1️⃣ 5折交叉验证:")
    print(f"  各折得分: {cv_scores}")
    print(f"  平均得分: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

    # 2. 详细交叉验证
    scoring = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
    cv_results = cross_validate(
        model_cv, X_train_scaled, y_train,
        cv=5,
        scoring=scoring,
        return_train_score=True
    )

    print(f"\n2️⃣ 多指标交叉验证:")
    for metric in scoring:
        test_scores = cv_results[f'test_{metric}']
        print(f"  {metric:20s}: {test_scores.mean():.4f} (+/- {test_scores.std() * 2:.4f})")

    # 3. 分层K折交叉验证
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    stratified_scores = cross_val_score(model_cv, X_train_scaled, y_train, cv=skf)

    print(f"\n3️⃣ 分层K折交叉验证:")
    print(f"  平均得分: {stratified_scores.mean():.4f} (+/- {stratified_scores.std() * 2:.4f})")

    return KFold, LR, StratifiedKFold, cross_val_score, cross_validate, cv_results, cv_scores, metric, model_cv, scoring, skf, stratified_scores, test_scores


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🎛️ 超参数调优

    使用网格搜索和随机搜索来找到最佳超参数：
    """
    )
    return


@app.cell
def _(X_train_scaled, y_train):
    from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
    from sklearn.ensemble import RandomForestClassifier as RFC

    print("🎛️ 超参数调优演示:")
    print("=" * 50)

    # 1. 网格搜索
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    rf_model = RFC(random_state=42)

    grid_search = GridSearchCV(
        rf_model,
        param_grid,
        cv=3,
        scoring='accuracy',
        n_jobs=-1,
        verbose=0
    )

    print("\n1️⃣ 网格搜索（GridSearchCV）:")
    print(f"  参数组合数: {len(param_grid['n_estimators']) * len(param_grid['max_depth']) * len(param_grid['min_samples_split']) * len(param_grid['min_samples_leaf'])}")
    print("  搜索中...")

    grid_search.fit(X_train_scaled, y_train)

    print(f"  最佳参数: {grid_search.best_params_}")
    print(f"  最佳得分: {grid_search.best_score_:.4f}")

    # 2. 随机搜索
    from scipy.stats import randint

    param_distributions = {
        'n_estimators': randint(50, 300),
        'max_depth': [3, 5, 7, 10, None],
        'min_samples_split': randint(2, 20),
        'min_samples_leaf': randint(1, 10)
    }

    random_search = RandomizedSearchCV(
        rf_model,
        param_distributions,
        n_iter=20,
        cv=3,
        scoring='accuracy',
        random_state=42,
        n_jobs=-1,
        verbose=0
    )

    print("\n2️⃣ 随机搜索（RandomizedSearchCV）:")
    print(f"  随机尝试次数: 20")
    print("  搜索中...")

    random_search.fit(X_train_scaled, y_train)

    print(f"  最佳参数: {random_search.best_params_}")
    print(f"  最佳得分: {random_search.best_score_:.4f}")

    return GridSearchCV, RFC, RandomizedSearchCV, grid_search, param_distributions, param_grid, randint, random_search, rf_model


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🔗 Pipeline（管道）

    Pipeline可以将多个处理步骤串联起来，简化工作流程：
    """
    )
    return


@app.cell
def _(X_class, y_class):
    from sklearn.pipeline import Pipeline, make_pipeline
    from sklearn.preprocessing import StandardScaler as SS
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression as LReg

    print("🔗 Pipeline演示:")
    print("=" * 50)

    # 1. 使用Pipeline类
    pipeline1 = Pipeline([
        ('scaler', SS()),
        ('pca', PCA(n_components=10)),
        ('classifier', LReg(random_state=42, max_iter=1000))
    ])

    print("\n1️⃣ Pipeline构建:")
    print(f"  步骤: {[name for name, _ in pipeline1.steps]}")

    # 分割数据
    from sklearn.model_selection import train_test_split as tts2
    X_pipe_train, X_pipe_test, y_pipe_train, y_pipe_test = tts2(
        X_class, y_class, test_size=0.2, random_state=42
    )

    # 训练pipeline
    pipeline1.fit(X_pipe_train, y_pipe_train)

    # 预测
    pipe_score = pipeline1.score(X_pipe_test, y_pipe_test)

    print(f"  训练完成")
    print(f"  测试集准确率: {pipe_score:.4f}")

    # 2. 使用make_pipeline（自动命名）
    pipeline2 = make_pipeline(
        SS(),
        PCA(n_components=10),
        LReg(random_state=42, max_iter=1000)
    )

    print("\n2️⃣ make_pipeline（自动命名）:")
    print(f"  步骤: {[name for name, _ in pipeline2.steps]}")

    pipeline2.fit(X_pipe_train, y_pipe_train)
    pipe2_score = pipeline2.score(X_pipe_test, y_pipe_test)

    print(f"  测试集准确率: {pipe2_score:.4f}")

    return LReg, PCA, Pipeline, SS, X_pipe_test, X_pipe_train, make_pipeline, pipe2_score, pipe_score, pipeline1, pipeline2, tts2, y_pipe_test, y_pipe_train


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🎨 特征工程

    特征选择和特征提取是提升模型性能的关键：
    """
    )
    return


@app.cell
def _(X_class, y_class):
    from sklearn.feature_selection import SelectKBest, f_classif, RFE, SelectFromModel
    from sklearn.ensemble import RandomForestClassifier as RFCls
    from sklearn.decomposition import PCA as PCAComp

    print("🎨 特征工程演示:")
    print("=" * 50)

    # 1. 单变量特征选择
    selector_univariate = SelectKBest(score_func=f_classif, k=10)
    X_selected_univariate = selector_univariate.fit_transform(X_class, y_class)

    print(f"\n1️⃣ 单变量特征选择（SelectKBest）:")
    print(f"  原始特征数: {X_class.shape[1]}")
    print(f"  选择特征数: {X_selected_univariate.shape[1]}")
    print(f"  特征得分前5: {sorted(selector_univariate.scores_, reverse=True)[:5]}")

    # 2. 递归特征消除
    estimator_rfe = RFCls(n_estimators=50, random_state=42)
    selector_rfe = RFE(estimator=estimator_rfe, n_features_to_select=10, step=1)
    X_selected_rfe = selector_rfe.fit_transform(X_class, y_class)

    print(f"\n2️⃣ 递归特征消除（RFE）:")
    print(f"  原始特征数: {X_class.shape[1]}")
    print(f"  选择特征数: {X_selected_rfe.shape[1]}")
    print(f"  选中特征索引: {np.where(selector_rfe.support_)[0][:10]}")

    # 3. 基于模型的特征选择
    estimator_model = RFCls(n_estimators=100, random_state=42)
    estimator_model.fit(X_class, y_class)

    selector_model = SelectFromModel(estimator_model, prefit=True, threshold='median')
    X_selected_model = selector_model.transform(X_class)

    print(f"\n3️⃣ 基于模型的特征选择（SelectFromModel）:")
    print(f"  原始特征数: {X_class.shape[1]}")
    print(f"  选择特征数: {X_selected_model.shape[1]}")

    # 4. PCA降维
    pca_reducer = PCAComp(n_components=10)
    X_pca = pca_reducer.fit_transform(X_class)

    print(f"\n4️⃣ PCA降维:")
    print(f"  原始特征数: {X_class.shape[1]}")
    print(f"  降维后特征数: {X_pca.shape[1]}")
    print(f"  解释方差比: {pca_reducer.explained_variance_ratio_[:5]}")
    print(f"  累计解释方差: {pca_reducer.explained_variance_ratio_.sum():.4f}")

    return PCAComp, RFCls, RFE, SelectFromModel, SelectKBest, X_pca, X_selected_model, X_selected_rfe, X_selected_univariate, estimator_model, estimator_rfe, f_classif, pca_reducer, selector_model, selector_rfe, selector_univariate


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🤝 无监督学习：聚类

    聚类算法用于发现数据中的自然分组：
    """
    )
    return


@app.cell
def _(iris):
    from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
    from sklearn.metrics import silhouette_score, davies_bouldin_score

    print("🤝 聚类算法演示:")
    print("=" * 50)

    X_cluster = iris.data

    # 1. K-Means聚类
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    kmeans_labels = kmeans.fit_predict(X_cluster)

    kmeans_silhouette = silhouette_score(X_cluster, kmeans_labels)
    kmeans_db = davies_bouldin_score(X_cluster, kmeans_labels)

    print(f"\n1️⃣ K-Means聚类:")
    print(f"  聚类数: 3")
    print(f"  轮廓系数: {kmeans_silhouette:.4f} (越接近1越好)")
    print(f"  Davies-Bouldin指数: {kmeans_db:.4f} (越小越好)")
    print(f"  各簇样本数: {np.bincount(kmeans_labels)}")

    # 2. DBSCAN聚类
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    dbscan_labels = dbscan.fit_predict(X_cluster)

    n_clusters_dbscan = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
    n_noise = list(dbscan_labels).count(-1)

    print(f"\n2️⃣ DBSCAN聚类:")
    print(f"  发现簇数: {n_clusters_dbscan}")
    print(f"  噪声点数: {n_noise}")

    if n_clusters_dbscan > 1:
        dbscan_silhouette = silhouette_score(X_cluster[dbscan_labels != -1], dbscan_labels[dbscan_labels != -1])
        print(f"  轮廓系数: {dbscan_silhouette:.4f}")

    # 3. 层次聚类
    hierarchical = AgglomerativeClustering(n_clusters=3)
    hierarchical_labels = hierarchical.fit_predict(X_cluster)

    hierarchical_silhouette = silhouette_score(X_cluster, hierarchical_labels)
    hierarchical_db = davies_bouldin_score(X_cluster, hierarchical_labels)

    print(f"\n3️⃣ 层次聚类（AgglomerativeClustering）:")
    print(f"  聚类数: 3")
    print(f"  轮廓系数: {hierarchical_silhouette:.4f}")
    print(f"  Davies-Bouldin指数: {hierarchical_db:.4f}")
    print(f"  各簇样本数: {np.bincount(hierarchical_labels)}")

    return AgglomerativeClustering, DBSCAN, KMeans, X_cluster, davies_bouldin_score, dbscan, dbscan_labels, dbscan_silhouette, hierarchical, hierarchical_db, hierarchical_labels, hierarchical_silhouette, kmeans, kmeans_db, kmeans_labels, kmeans_silhouette, n_clusters_dbscan, n_noise, silhouette_score


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🎭 集成学习

    集成方法通过组合多个模型来提升性能：
    """
    )
    return


@app.cell
def _(X_train_scaled, y_train):
    from sklearn.ensemble import VotingClassifier, BaggingClassifier, AdaBoostClassifier, StackingClassifier
    from sklearn.linear_model import LogisticRegression as LogReg
    from sklearn.tree import DecisionTreeClassifier as DTC
    from sklearn.svm import SVC as SVCls

    print("🎭 集成学习演示:")
    print("=" * 50)

    # 1. 投票分类器
    voting_clf = VotingClassifier(
        estimators=[
            ('lr', LogReg(random_state=42, max_iter=1000)),
            ('dt', DTC(random_state=42, max_depth=5)),
            ('svc', SVCls(random_state=42, probability=True))
        ],
        voting='soft'
    )

    voting_clf.fit(X_train_scaled, y_train)
    voting_score = voting_clf.score(X_train_scaled, y_train)

    print(f"\n1️⃣ 投票分类器（VotingClassifier）:")
    print(f"  基学习器数量: {len(voting_clf.estimators_)}")
    print(f"  训练集准确率: {voting_score:.4f}")

    # 2. Bagging
    bagging_clf = BaggingClassifier(
        estimator=DTC(random_state=42),
        n_estimators=50,
        random_state=42,
        max_samples=0.8,
        max_features=0.8
    )

    bagging_clf.fit(X_train_scaled, y_train)
    bagging_score = bagging_clf.score(X_train_scaled, y_train)

    print(f"\n2️⃣ Bagging分类器:")
    print(f"  基学习器数量: {bagging_clf.n_estimators}")
    print(f"  训练集准确率: {bagging_score:.4f}")

    # 3. AdaBoost
    adaboost_clf = AdaBoostClassifier(
        estimator=DTC(max_depth=1, random_state=42),
        n_estimators=50,
        random_state=42
    )

    adaboost_clf.fit(X_train_scaled, y_train)
    adaboost_score = adaboost_clf.score(X_train_scaled, y_train)

    print(f"\n3️⃣ AdaBoost分类器:")
    print(f"  基学习器数量: {adaboost_clf.n_estimators}")
    print(f"  训练集准确率: {adaboost_score:.4f}")

    # 4. Stacking
    stacking_clf = StackingClassifier(
        estimators=[
            ('lr', LogReg(random_state=42, max_iter=1000)),
            ('dt', DTC(random_state=42, max_depth=5)),
        ],
        final_estimator=LogReg(random_state=42, max_iter=1000),
        cv=3
    )

    stacking_clf.fit(X_train_scaled, y_train)
    stacking_score = stacking_clf.score(X_train_scaled, y_train)

    print(f"\n4️⃣ Stacking分类器:")
    print(f"  基学习器数量: {len(stacking_clf.estimators_)}")
    print(f"  元学习器: {type(stacking_clf.final_estimator_).__name__}")
    print(f"  训练集准确率: {stacking_score:.4f}")

    return AdaBoostClassifier, BaggingClassifier, DTC, LogReg, SVCls, StackingClassifier, VotingClassifier, adaboost_clf, adaboost_score, bagging_clf, bagging_score, stacking_clf, stacking_score, voting_clf, voting_score


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 📊 模型持久化

    训练好的模型可以保存和加载：
    """
    )
    return


@app.cell
def _():
    import joblib
    import pickle

    print("📊 模型持久化演示:")
    print("=" * 50)

    # 创建并训练一个简单模型
    from sklearn.linear_model import LogisticRegression as LRModel
    from sklearn.datasets import make_classification as mc

    X_persist, y_persist = mc(n_samples=100, n_features=10, random_state=42)
    model_persist = LRModel(random_state=42, max_iter=1000)
    model_persist.fit(X_persist, y_persist)

    # 1. 使用joblib保存（推荐）
    joblib.dump(model_persist, 'model_joblib.pkl')
    print("\n1️⃣ 使用joblib保存:")
    print("  ✅ 模型已保存为 model_joblib.pkl")

    # 加载模型
    loaded_model_joblib = joblib.load('model_joblib.pkl')
    joblib_score = loaded_model_joblib.score(X_persist, y_persist)
    print(f"  ✅ 模型已加载，准确率: {joblib_score:.4f}")

    # 2. 使用pickle保存
    with open('model_pickle.pkl', 'wb') as f:
        pickle.dump(model_persist, f)

    print("\n2️⃣ 使用pickle保存:")
    print("  ✅ 模型已保存为 model_pickle.pkl")

    # 加载模型
    with open('model_pickle.pkl', 'rb') as f:
        loaded_model_pickle = pickle.load(f)

    pickle_score = loaded_model_pickle.score(X_persist, y_persist)
    print(f"  ✅ 模型已加载，准确率: {pickle_score:.4f}")

    print("\n💡 推荐使用joblib，因为它对大型numpy数组更高效")

    return LRModel, X_persist, joblib, joblib_score, loaded_model_joblib, loaded_model_pickle, mc, model_persist, pickle, pickle_score, y_persist


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🎯 最佳实践总结

    使用scikit-learn时的关键要点：
    """
    )
    return


@app.cell
def _(mo, pd):
    best_practices = {
        "数据准备": [
            "✅ 始终分割训练集和测试集",
            "✅ 使用stratify参数保持类别分布",
            "✅ 在训练集上fit，在测试集上transform",
            "✅ 处理缺失值和异常值",
            "✅ 对特征进行适当的缩放"
        ],

        "模型训练": [
            "✅ 从简单模型开始（如逻辑回归）",
            "✅ 使用交叉验证评估模型",
            "✅ 设置random_state保证可复现性",
            "✅ 使用Pipeline组织工作流",
            "✅ 监控训练时间和资源消耗"
        ],

        "超参数调优": [
            "✅ 先用RandomizedSearchCV快速探索",
            "✅ 再用GridSearchCV精细调优",
            "✅ 使用合适的评分指标",
            "✅ 注意过拟合风险",
            "✅ 考虑计算成本和时间"
        ],

        "模型评估": [
            "✅ 使用多个评估指标",
            "✅ 绘制混淆矩阵和ROC曲线",
            "✅ 分析特征重要性",
            "✅ 检查模型在不同数据子集上的表现",
            "✅ 进行错误分析"
        ],

        "生产部署": [
            "✅ 使用joblib保存模型",
            "✅ 保存预处理器和模型版本信息",
            "✅ 监控模型性能",
            "✅ 定期重新训练模型",
            "✅ 实施A/B测试"
        ]
    }

    practices_df = pd.DataFrame([
        {"类别": category, "实践": practice}
        for category, practices in best_practices.items()
        for practice in practices
    ])

    mo.md(f"""
    ### 📋 Scikit-learn最佳实践

    {practices_df.to_markdown(index=False)}

    ### 🔑 核心原则

    1. **一致的API**: 所有估计器都遵循fit/predict/transform模式
    2. **可组合性**: 使用Pipeline和FeatureUnion组合多个步骤
    3. **可检查性**: 模型参数和属性都可以访问
    4. **合理的默认值**: 大多数参数都有合理的默认设置
    5. **可复现性**: 设置random_state确保结果可复现

    ### 📚 常用模块速查

    | 模块 | 功能 | 常用类 |
    |------|------|--------|
    | `sklearn.preprocessing` | 数据预处理 | StandardScaler, MinMaxScaler, LabelEncoder |
    | `sklearn.model_selection` | 模型选择 | train_test_split, cross_val_score, GridSearchCV |
    | `sklearn.linear_model` | 线性模型 | LogisticRegression, LinearRegression, Ridge, Lasso |
    | `sklearn.tree` | 决策树 | DecisionTreeClassifier, DecisionTreeRegressor |
    | `sklearn.ensemble` | 集成方法 | RandomForest, GradientBoosting, AdaBoost |
    | `sklearn.svm` | 支持向量机 | SVC, SVR |
    | `sklearn.neighbors` | 近邻算法 | KNeighborsClassifier, KNeighborsRegressor |
    | `sklearn.cluster` | 聚类 | KMeans, DBSCAN, AgglomerativeClustering |
    | `sklearn.decomposition` | 降维 | PCA, TruncatedSVD, NMF |
    | `sklearn.metrics` | 评估指标 | accuracy_score, f1_score, roc_auc_score |
    | `sklearn.pipeline` | 管道 | Pipeline, make_pipeline, FeatureUnion |
    | `sklearn.feature_selection` | 特征选择 | SelectKBest, RFE, SelectFromModel |
    """)

    return best_practices, practices, practices_df


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🚀 实战案例：完整的机器学习流程

    让我们通过一个完整的案例来演示scikit-learn的典型使用流程：
    """
    )
    return


@app.cell
def _(np):
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split as final_split
    from sklearn.preprocessing import StandardScaler as FinalScaler
    from sklearn.decomposition import PCA as FinalPCA
    from sklearn.ensemble import RandomForestClassifier as FinalRF
    from sklearn.model_selection import cross_val_score as final_cv
    from sklearn.metrics import classification_report as final_report, confusion_matrix as final_cm
    from sklearn.pipeline import Pipeline as FinalPipeline

    print("🚀 完整机器学习流程演示：乳腺癌诊断")
    print("=" * 70)

    # 1. 加载数据
    cancer_data = load_breast_cancer()
    X_cancer = cancer_data.data
    y_cancer = cancer_data.target

    print(f"\n📊 步骤1: 数据加载")
    print(f"  样本数: {X_cancer.shape[0]}")
    print(f"  特征数: {X_cancer.shape[1]}")
    print(f"  类别: {cancer_data.target_names}")
    print(f"  类别分布: {np.bincount(y_cancer)}")

    # 2. 数据分割
    X_cancer_train, X_cancer_test, y_cancer_train, y_cancer_test = final_split(
        X_cancer, y_cancer, test_size=0.2, random_state=42, stratify=y_cancer
    )

    print(f"\n✂️ 步骤2: 数据分割")
    print(f"  训练集: {X_cancer_train.shape}")
    print(f"  测试集: {X_cancer_test.shape}")

    # 3. 构建Pipeline
    cancer_pipeline = FinalPipeline([
        ('scaler', FinalScaler()),
        ('pca', FinalPCA(n_components=0.95)),
        ('classifier', FinalRF(n_estimators=100, random_state=42))
    ])

    print(f"\n🔗 步骤3: 构建Pipeline")
    print(f"  步骤: {[name for name, _ in cancer_pipeline.steps]}")

    # 4. 交叉验证
    cv_scores_cancer = final_cv(cancer_pipeline, X_cancer_train, y_cancer_train, cv=5)

    print(f"\n🔍 步骤4: 交叉验证")
    print(f"  5折交叉验证得分: {cv_scores_cancer}")
    print(f"  平均得分: {cv_scores_cancer.mean():.4f} (+/- {cv_scores_cancer.std() * 2:.4f})")

    # 5. 训练模型
    cancer_pipeline.fit(X_cancer_train, y_cancer_train)

    print(f"\n🎓 步骤5: 模型训练")
    print(f"  ✅ 训练完成")
    print(f"  PCA保留成分数: {cancer_pipeline.named_steps['pca'].n_components_}")
    print(f"  解释方差比: {cancer_pipeline.named_steps['pca'].explained_variance_ratio_.sum():.4f}")

    # 6. 模型评估
    y_cancer_pred = cancer_pipeline.predict(X_cancer_test)
    test_score_cancer = cancer_pipeline.score(X_cancer_test, y_cancer_test)

    print(f"\n📈 步骤6: 模型评估")
    print(f"  测试集准确率: {test_score_cancer:.4f}")

    print(f"\n  分类报告:")
    print(final_report(y_cancer_test, y_cancer_pred, target_names=cancer_data.target_names))

    print(f"  混淆矩阵:")
    print(final_cm(y_cancer_test, y_cancer_pred))

    # 7. 特征重要性
    feature_importance_cancer = cancer_pipeline.named_steps['classifier'].feature_importances_
    print(f"\n🎯 步骤7: 特征重要性分析")
    print(f"  前5个最重要的PCA成分: {sorted(feature_importance_cancer, reverse=True)[:5]}")

    # 8. 保存模型
    import joblib as jl
    jl.dump(cancer_pipeline, 'cancer_diagnosis_model.pkl')

    print(f"\n💾 步骤8: 模型保存")
    print(f"  ✅ 模型已保存为 cancer_diagnosis_model.pkl")

    print(f"\n🎉 完整流程演示完成！")

    return FinalPCA, FinalPipeline, FinalRF, FinalScaler, X_cancer, X_cancer_test, X_cancer_train, cancer_data, cancer_pipeline, cv_scores_cancer, feature_importance_cancer, final_cm, final_cv, final_report, final_split, jl, load_breast_cancer, test_score_cancer, y_cancer, y_cancer_pred, y_cancer_test, y_cancer_train


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🎓 总结

    通过这个交互式指南，我们全面探索了scikit-learn的核心功能：

    ### 🎯 关键学习成果

    1. **理解了scikit-learn的设计哲学**：统一的API、可组合性、可检查性
    2. **掌握了数据预处理技术**：缩放、编码、缺失值处理
    3. **学习了监督学习算法**：分类和回归的各种方法
    4. **实践了模型评估与选择**：交叉验证、超参数调优
    5. **应用了Pipeline工作流**：简化和标准化机器学习流程
    6. **探索了特征工程**：特征选择和降维技术
    7. **了解了无监督学习**：聚类算法
    8. **掌握了集成方法**：提升模型性能的技术
    9. **学会了模型持久化**：保存和加载训练好的模型
    10. **完成了端到端案例**：从数据加载到模型部署的完整流程

    ### 🚀 Scikit-learn的核心优势

    - **简单易用**：一致的API设计，易于学习和使用
    - **功能全面**：涵盖机器学习的各个方面
    - **性能优秀**：底层使用Cython优化，运行高效
    - **文档完善**：详细的文档和丰富的示例
    - **社区活跃**：持续更新和维护

    ### 📚 进一步学习资源

    1. **官方文档**: https://scikit-learn.org/stable/
    2. **用户指南**: https://scikit-learn.org/stable/user_guide.html
    3. **API参考**: https://scikit-learn.org/stable/modules/classes.html
    4. **示例库**: https://scikit-learn.org/stable/auto_examples/index.html
    5. **教程**: https://scikit-learn.org/stable/tutorial/index.html

    ### 🎯 下一步行动

    1. 在实际项目中应用scikit-learn
    2. 尝试不同的算法和参数组合
    3. 参与Kaggle竞赛实践
    4. 阅读scikit-learn源码深入理解
    5. 为开源社区做贡献

    Scikit-learn是Python机器学习生态系统的基石，掌握它将为你的数据科学之旅打下坚实的基础！🎉
    """
    )
    return


@app.cell
def _():
    from datetime import timedelta
    return (timedelta,)


if __name__ == "__main__":
    app.run()

