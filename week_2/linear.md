# Линейные методы классификации

Линейная модель - взвешенная сумма всех признаков

### Обучение

### Стохастический градиентный спуск
```
Инициализировать веса и оценку функционала
Пока значения фунционала не сойдутся:
    Выбрать случайный объект
    Вычислить ошибку
    Сделать градиеннтный шаг
    Оценить функционал
```
### Stochastic Average Gradient
#### Достоинства

* Применим к любым моделям и функциям потерь
* Возможно обучение на части выборки

#### Недостатки

* Застревание в локальных экстремумах
* Расходимость или медленная сходимость
* Сложно подобрать эвристики
* Мультиколлинеарность -> регуляризация, удаление признаков