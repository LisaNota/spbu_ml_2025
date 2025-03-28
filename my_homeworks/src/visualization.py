import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# HW 2

def plot_classification_metrics(model, X_test, y_test, model_name: str = None):
    """
    Функция принимает обученную модель, тестовые данные и истинные метки, 
    затем вычисляет предсказания, метрики и строит четыре графика:
      1. ROC-кривая с AUC.
      2. Столбчатую диаграмму основных метрик.
      3. Матрицу ошибок с черным текстом.
      4. Таблицу классификационного отчёта.
    """
    y_pred = model.predict(X_test)

    # Расчет ROC-кривой и ROC AUC
    y_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    roc_auc = roc_auc_score(y_test, y_proba)

    # Вычисление основных метрик
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Получаем классификационный отчёт и матрицу ошибок
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)

    fig, axs = plt.subplots(2, 2, figsize=(16, 12))

    # 1. ROC-кривая
    axs[0, 0].plot(fpr, tpr, color='blue', linewidth=2,
                   label=f'ROC (AUC = {roc_auc:.2f})')
    axs[0, 0].plot([0, 1], [0, 1], color='gray',
                   linestyle='--', label='Random Classifier')
    axs[0, 0].set_xlabel('False Positive Rate', fontsize=12)
    axs[0, 0].set_ylabel('True Positive Rate', fontsize=12)
    axs[0, 0].set_title('ROC Curve', fontsize=14)
    axs[0, 0].legend(loc="lower right", fontsize=12)
    axs[0, 0].grid(True)

    # 2. Столбчатая диаграмма с основными метриками
    metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    metrics_values = [acc, prec, rec, f1]
    axs[0, 1].bar(metrics_names, metrics_values, color='skyblue')
    axs[0, 1].set_ylim(0, 1)
    axs[0, 1].set_title('Evaluation Metrics', fontsize=14)
    for i, value in enumerate(metrics_values):
        axs[0, 1].text(i, value + 0.02,
                       f"{value:.2f}", ha='center', va='bottom', fontsize=12)

    # 3. Матрица ошибок с черным текстом
    im = axs[1, 0].imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    axs[1, 0].set_title('Confusion Matrix', fontsize=14)
    axs[1, 0].set_xticks([0, 1])
    axs[1, 0].set_yticks([0, 1])
    axs[1, 0].set_xlabel('Predicted Label', fontsize=12)
    axs[1, 0].set_ylabel('True Label', fontsize=12)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            axs[1, 0].text(j, i, format(cm[i, j], 'd'),
                           ha="center", va="center",
                           color="black", fontsize=12)
    fig.colorbar(im, ax=axs[1, 0])

    # 4. Таблица классификационного отчёта
    axs[1, 1].axis('off')
    axs[1, 1].set_title('Classification Report', fontsize=14)
    table_data = []
    row_labels = []
    for key in ['0', '1']:
        row_labels.append(f'Class {key}')
        metrics = report[key]
        row_data = [f"{metrics['precision']:.2f}", f"{metrics['recall']:.2f}",
                    f"{metrics['f1-score']:.2f}", f"{metrics['support']}"]
        table_data.append(row_data)
    col_labels = ["Precision", "Recall", "F1-Score", "Support"]
    table = axs[1, 1].table(cellText=table_data, rowLabels=row_labels, colLabels=col_labels,
                            cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 2)

    # Заголовок всей фигуры
    fig.suptitle(f'{model_name}: метрики и результаты',
                 fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def visualize_pca_3d(X_pca, y, pca, title="3D PCA Visualization"):
    """
    Функция для интерактивной 3D-визуализации результатов PCA с 3 компонентами.
    Позволяет вращать, масштабировать и детально исследовать график.

    Параметры:
    - X_pca: массив после преобразования PCA (форма: [n_samples, 3])
    - y: метки классов (числовые или строковые)
    - pca: объект PCA, чтобы извлечь explained_variance_ratio_
    - title: заголовок графика
    """
    # Вычисляем процент объяснённой дисперсии для каждой компоненты
    exp_var = pca.explained_variance_ratio_ * 100
    full_title = f"{title}\nExplained Variance: PC1: {exp_var[0]:.1f}%, PC2: {exp_var[1]:.1f}%, PC3: {exp_var[2]:.1f}%"

    # Создаем DataFrame для визуализации
    df = pd.DataFrame(X_pca, columns=["PC1", "PC2", "PC3"])
    df['label'] = y

    # Создаем интерактивный 3D scatter plot
    fig = px.scatter_3d(df, x='PC1', y='PC2', z='PC3', color='label',
                        title=full_title,
                        labels={"PC1": "PC1", "PC2": "PC2", "PC3": "PC3"},
                        opacity=0.7)
    fig.update_traces(marker=dict(size=5))
    fig.show()
