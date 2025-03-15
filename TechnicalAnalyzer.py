import pandas as pd
import numpy as np
import pandas_ta as ta
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from typing import List, Dict, Optional
import re

class TechnicalAnalyzer:
    """
    Расширенный класс для технического анализа мультиактивных портфелей
    
    Особенности:
    - Поддержка одновременной работы с несколькими активами
    - Автоматическое управление метриками и сигналами
    - Интерактивная визуализация с настройкой слоев
    - Гибкая система комбинирования сигналов
    """
    
    def __init__(self, ohlcv_data: pd.DataFrame):
        """
        Parameters:
            ohlcv_data: DataFrame с мультииндексом (Ticker, Metric) в колонках
                        Ожидаемые метрики: Open, High, Low, Close, Volume
        """
        self.ohlcv = ohlcv_data
        self.indicators = pd.DataFrame(index=ohlcv_data.index)
        self.signals = pd.DataFrame(index=ohlcv_data.index)
        self.tickers = ohlcv_data.columns.get_level_values(0).unique().tolist()
        
    def _validate_ticker(self, ticker: str) -> None:
        """Проверка наличия тикера в данных"""
        if ticker not in self.tickers:
            raise ValueError(f"Тикер {ticker} отсутствует в данных. Доступные тикеры: {self.tickers}")

    def _add_indicator(self, ticker: str, name: str, values: pd.Series) -> None:
        """Универсальный метод добавления индикаторов"""
        col_name = f"{ticker}_{name}"
        self.indicators[col_name] = values

    def calculate_sma(self, ticker: str, window: int) -> None:
        """
        Простое скользящее среднее
        
        Parameters:
            ticker: Тикер актива
            window: Период расчета
        """
        self._validate_ticker(ticker)
        close = self.ohlcv[(ticker, 'Close')]
        sma = ta.sma(close, length=window)
        self._add_indicator(ticker, f"SMA_{window}", sma)

    def calculate_ema(self, ticker: str, window: int) -> None:
        """
        Экспоненциальное скользящее среднее
        
        Parameters:
            ticker: Тикер актива
            window: Период расчета
        """
        self._validate_ticker(ticker)
        close = self.ohlcv[(ticker, 'Close')]
        ema = ta.ema(close, length=window)
        self._add_indicator(ticker, f"EMA_{window}", ema)

    def calculate_rsi(self, ticker: str, window: int = 14) -> None:
        """
        Индекс относительной силы (RSI)
        
        Parameters:
            ticker: Тикер актива
            window: Период расчета (по умолчанию 14)
        """
        self._validate_ticker(ticker)
        close = self.ohlcv[(ticker, 'Close')]
        rsi = ta.rsi(close, length=window)
        self._add_indicator(ticker, f"RSI_{window}", rsi)

    def calculate_macd(self, ticker: str, fast: int = 12, slow: int = 26, signal: int = 9) -> None:
        """
        Схождение/расхождение скользящих средних (MACD)
        
        Parameters:
            ticker: Тикер актива
            fast: Период быстрой MA
            slow: Период медленной MA
            signal: Период сигнальной линии
        """
        self._validate_ticker(ticker)
        close = self.ohlcv[(ticker, 'Close')]
        macd = ta.macd(close, fast=fast, slow=slow, signal=signal)
        for col in macd.columns:
            self._add_indicator(ticker, f"MACD_{col}", macd[col])

    def calculate_bollinger_bands(self, ticker: str, window: int = 20, std: int = 2) -> None:
        """
        Полосы Боллинджера
        
        Parameters:
            ticker: Тикер актива
            window: Период расчета
            std: Количество стандартных отклонений
        """
        self._validate_ticker(ticker)
        close = self.ohlcv[(ticker, 'Close')]
        bb = ta.bbands(close, length=window, std=std)
        for col in bb.columns:
            # Было: self._add_indicator(ticker, f"BB_{col}", bb[col])
            if col.split('_')[0][-1] == 'L':
                name = 'LOWER'
            elif col.split('_')[0][-1] == 'M':
                name = 'MIDDLE'
            elif col.split('_')[0][-1] == 'U':
                name = 'UPPER'
            else:
                name = col.split('_')[0]
            self._add_indicator(ticker, f"BB_{name}", bb[col])  # Генерирует BB_UPPER, BB_MIDDLE, BB_LOWER

    def calculate_obv(self, ticker: str) -> None:
        """
        Индекс баланса объема (On-Balance Volume)
        
        Parameters:
            ticker: Тикер актива
        """
        self._validate_ticker(ticker)
        close = self.ohlcv[(ticker, 'Close')]
        volume = self.ohlcv[(ticker, 'Volume')]
        obv = ta.obv(close, volume)
        self._add_indicator(ticker, "OBV", obv)

    def generate_ma_crossover_signal(self, ticker: str, fast: int = 50, slow: int = 200) -> None:
        """
        Генерация сигналов по пересечению скользящих средних
        
        Parameters:
            ticker: Тикер актива
            fast: Период быстрой MA
            slow: Период медленной MA
            
        Сигналы:
            1 - Покупка (Fast MA > Slow MA)
            -1 - Продажа (Fast MA < Slow MA)
        """
        self.calculate_sma(ticker, fast)
        self.calculate_sma(ticker, slow)
        
        fast_col = f"{ticker}_SMA_{fast}"
        slow_col = f"{ticker}_SMA_{slow}"
        
        signal = np.where(
            self.indicators[fast_col] > self.indicators[slow_col], 1, -1
        )
        self.signals[f"{ticker}_MA_Crossover"] = signal

    def generate_rsi_signal(self, ticker: str, window: int = 14, 
                           overbought: float = 70.0, oversold: float = 30.0) -> None:
        """
        Генерация сигналов по RSI
        
        Parameters:
            ticker: Тикер актива
            window: Период расчета RSI
            overbought: Уровень перекупленности
            oversold: Уровень перепроданности
            
        Сигналы:
            1 - Покупка (RSI пересекает oversold снизу вверх)
            -1 - Продажа (RSI пересекает overbought сверху вниз)
        """
        self.calculate_rsi(ticker, window)
        rsi_col = f"{ticker}_RSI_{window}"
        
        rsi = self.indicators[rsi_col]
        buy_condition = (rsi > oversold) & (rsi.shift(1) <= oversold)
        sell_condition = (rsi < overbought) & (rsi.shift(1) >= overbought)
        
        signal = np.select([buy_condition, sell_condition], [1, -1], default=0)
        self.signals[f"{ticker}_RSI_{window}"] = signal

    def plot_dashboard(self, ticker: str, periods: int = 100) -> None:
        self._validate_ticker(ticker)
        
        # Подготовка данных
        data = self.ohlcv[ticker].iloc[-periods:]
        indicators = self.indicators.filter(regex=f"^{ticker}_").iloc[-periods:]
        
        # Создание субплотов
        fig = make_subplots(
            rows=4, 
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.06,
            row_heights=[0.5, 0.2, 0.2, 0.2],
            specs=[[{}], [{}], [{}], [{}]]
        )
        
        # Построение графиков
        self._plot_price_panel(fig, data, indicators, ticker, 1)
        self._plot_macd_panel(fig, indicators, ticker, 2)
        self._plot_rsi_panel(fig, indicators, ticker, 3)
        self._plot_volume_panel(fig, data, indicators, ticker, 4)
        self._add_legend_headers(fig)
        
        # Финальная настройка
        self._configure_layout(fig, ticker)
        fig.show()

    def _add_legend_headers(self, fig):
        """Заголовки групп в легенде с правильной иерархией"""
        headers = [
            {'name': "Цены и тренд", 'rank': 0},
            {'name': "Скользящие средние", 'rank': 100},
            {'name': "Осцилляторы", 'rank': 200},
            {'name': "MACD", 'rank': 300},
            {'name': "Объемы", 'rank': 400}
        ]
        
        for header in headers:
            # Добавляем видимый заголовок группы
            fig.add_trace(go.Scatter(
                x=[None],
                y=[None],
                mode="lines",
                name=f"<b>{header['name']}</b>",
                line=dict(color="rgba(0,0,0,0)"),
                showlegend=True,
                legendrank=header['rank'] - 1,
                #legendgrouptitle=dict(
                #    text=header['name'],
                #    font=dict(size=12, color='#FFFFFF', family='Arial')
                #),
                hoverinfo='none'
            ))

    def _plot_price_panel(self, fig, data, indicators, ticker, row):
        # Свечной график
        fig.add_trace(go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name="Цены",
            showlegend=True,
            legendgroup="Цены и тренд",
            legendrank=0,  # Первый элемент после заголовка
            increasing_line_color='#2ECC71',
            decreasing_line_color='#E74C3C'
        ), row=row, col=1)
        
        # Bollinger Bands
        bb_columns = {
            'upper': f"{ticker}_BB_UPPER",
            'middle': f"{ticker}_BB_MIDDLE", 
            'lower': f"{ticker}_BB_LOWER"
        }
        
        if all(col in indicators.columns for col in bb_columns.values()):
            for i, key in enumerate(['upper', 'middle', 'lower'], start=1):
                fig.add_trace(go.Scatter(
                    x=indicators.index,
                    y=indicators[bb_columns[key]],
                    name=f"BB {key.title()}",
                    line=dict(
                        color='#3498DB' if key == 'middle' else '#95A5A6',
                        width=1.5 if key == 'middle' else 1,
                        dash='dot' if key == 'middle' else None
                    ),
                    fill='tonexty' if key == 'lower' else None,
                    fillcolor='rgba(52, 152, 219, 0.1)' if key == 'lower' else None,
                    showlegend=True,
                    legendgroup="Цены и тренд",
                    legendrank=i,  # 1, 2, 3
                ), row=row, col=1)
        
        # Moving Averages
        ma_columns = [col for col in indicators.columns if '_SMA_' in col or '_EMA_' in col]
        for i, col in enumerate(ma_columns, start=1):
            fig.add_trace(go.Scatter(
                x=indicators.index,
                y=indicators[col],
                name=f"{col.split('_')[-2]} {col.split('_')[-1]}",
                line=dict(width=1.5),
                visible='legendonly',
                showlegend=True,
                legendgroup="Скользящие средние",
                legendrank=100 + i  # 101, 102...
            ), row=row, col=1)

    def _plot_macd_panel(self, fig, indicators, ticker, row):
        macd_columns = {
            'hist': f"{ticker}_MACD_MACDh_12_26_9",
            'macd': f"{ticker}_MACD_MACD_12_26_9",
            'signal': f"{ticker}_MACD_MACDs_12_26_9"
        }
        
        if all(col in indicators.columns for col in macd_columns.values()):
            # Гистограмма
            colors = np.where(indicators[macd_columns['hist']] > 0, '#2ECC71', '#E74C3C')
            fig.add_trace(go.Bar(
                x=indicators.index,
                y=indicators[macd_columns['hist']],
                name="MACD Hist",
                marker_color=colors,
                opacity=0.6,
                showlegend=True,
                legendrank=301  # 300-399 для MACD
            ), row=row, col=1)
            
            # Линии
            fig.add_trace(go.Scatter(
                x=indicators.index,
                y=indicators[macd_columns['macd']],
                name="MACD Line",
                line=dict(color='#3498DB', width=1.5),
                showlegend=True,
                legendrank=302
            ), row=row, col=1)
            
            fig.add_trace(go.Scatter(
                x=indicators.index,
                y=indicators[macd_columns['signal']],
                name="Signal Line",
                line=dict(color='#9B59B6', width=1.5),
                showlegend=True,
                legendrank=303
            ), row=row, col=1)

    def _plot_rsi_panel(self, fig, indicators, ticker, row):
        rsi_col = f"{ticker}_RSI_14"
        if rsi_col in indicators.columns:
            fig.add_trace(go.Scatter(
                x=indicators.index,
                y=indicators[rsi_col],
                name="RSI 14",
                line=dict(color='#F39C12', width=1.5),
                showlegend=True,
                legendrank=201  # 200-299 для осцилляторов
            ), row=row, col=1)
            
            # Уровни RSI
            for level, color in [(70, '#E74C3C'), (30, '#2ECC71')]:
                fig.add_hline(
                    y=level,
                    line_dash="dot",
                    line_color=color,
                    opacity=0.7,
                    annotation_text=str(level),
                    annotation_position="bottom right" if level == 70 else "top right",
                    row=row,
                    col=1
                )

    def _plot_volume_panel(self, fig, data, indicators, ticker, row):
        fig.add_trace(go.Bar(
            x=data.index,
            y=data['Volume'],
            name="Volume",
            marker_color='#3498DB',
            opacity=0.5,
            showlegend=True,
            legendrank=401  # 400-499 для объемов
        ), row=row, col=1)
        
        obv_col = f"{ticker}_OBV"
        if obv_col in indicators.columns:
            fig.add_trace(go.Scatter(
                x=indicators.index,
                y=indicators[obv_col],
                name="OBV",
                line=dict(color='#95A5A6', width=1.5),
                visible='legendonly',
                showlegend=True,
                legendrank=402
            ), row=row, col=1)

    def _configure_layout(self, fig, ticker):
        fig.update_layout(
            title=dict(
                text=f"<b>{ticker} Технический анализ</b>",
                x=0.05,
                font=dict(size=20, color='#FFFFFF')
            ),
            height=1200,
            template="plotly_dark",
            margin=dict(t=80, b=20, l=80, r=200),
            legend=dict(
                orientation="v",
                yanchor="top",
                xanchor="left",
                x=1.02,
                y=0.95,
                font=dict(size=10),
                groupclick="toggleitem"  # Важно для индивидуального управления
            ),
            hovermode="x unified"
        )
        
        fig.update_yaxes(title_text="Цена", row=1)
        fig.update_yaxes(title_text="MACD", row=2)
        fig.update_yaxes(title_text="RSI", row=3, range=[0, 100])
        fig.update_yaxes(title_text="Объем", row=4)
        fig.update_xaxes(rangeslider=dict(visible=False))