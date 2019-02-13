# Метрики качества классификации

Accuracy - Доля правильных ответов

|       | y = 1 | y = 0 |
|-------|-------|-------|
|a(x)=1 | TP    | FP    |
|a(x)=0 | FN    | TN    |

### Точность
Precision = TP / (TP + FP)
### Полнота
Recall = TP / (TP + FN)
### F-мера
F = 2 * Precision * Recall / (Precision + Recall)

## Метрики для вероятностных классификаторов

### AUC-PRC
Площадь под Precision-Recall кривой
### AUC-ROC 
По оси X: FPR = FP / (FP + TN)
По оси Y: TPR = TP / (TP + FN)