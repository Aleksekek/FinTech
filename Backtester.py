from typing import Dict, List
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
import quantstats as qs


class DataSplitter:

    @staticmethod
    def split_data(df, train_ratio=0.7, test_ratio=0.2, val_ratio=0.1):

        # Проверка, что сумма соотношений равна 1
        total = train_ratio + test_ratio + val_ratio
        if not np.isclose(total, 1.0):
            raise ValueError("Сумма соотношений должна быть равна 1")

        # Разделение данных
        n = len(df)
        train_end = int(n * train_ratio)
        test_end = train_end + int(n * test_ratio)

        train_data = df.iloc[:train_end]
        test_data = df.iloc[train_end:test_end]
        val_data = df.iloc[test_end:]
        
        return train_data, test_data, val_data


class Backtester:
    """
    Класс для тестирования торговых стратегий
    
    Функционал:
    - Симуляция торговли с учетом комиссий и проскальзывания
    - Расчет ключевых метрик эффективности
    - Сравнение нескольких стратегий
    - Генерация отчетов
    """
    
    def __init__(self, signals: pd.DataFrame, prices: pd.DataFrame,
                 initial_capital: float = 10000.0, commission: float = 0.001):
        """
        Parameters:
            signals: DataFrame с торговыми сигналами
            prices: DataFrame с ценами активов
            initial_capital: Начальный капитал
            commission: Комиссия за сделку (% от объема)
        """
        self.signals = signals
        self.prices = prices
        self.initial_capital = initial_capital
        self.commission = commission
        
    def run_backtest(self, strategy_name: str) -> Dict:
        """
        Запуск бэктеста для выбранной стратегии
        
        Returns:
            Словарь с результатами:
            - returns: Доходность по периодам
            - stats: Рассчитанные метрики
            - trades: История сделок
        """
        # Реализация логики исполнения сделок
        # Расчет метрик производительности
        
    def generate_report(self, results: Dict, benchmark: str = 'SPY') -> None:
        """Генерация отчета с использованием quantstats"""
        qs.reports.full(
            results['returns'], 
            benchmark=benchmark,
            output='output.html')
        

class StrategyOptimizer:
    """
    Класс для оптимизации параметров стратегий
    
    Функционал:
    - Кросс-валидация временных рядов
    - Поиск оптимальных параметров стратегии
    - Анализ кривой обучения
    - Визуализация результатов оптимизации
    """
    
    def __init__(self, data: pd.DataFrame, n_splits: int = 5):
        self.data = data
        self.tscv = TimeSeriesSplit(n_splits=n_splits)
        
    def optimize_parameters(self, strategy_class, param_grid: Dict) -> Dict:
        """Оптимизация гиперпараметров с использованием GridSearch"""
        # Реализация поиска по сетке параметров