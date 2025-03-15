import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots


class DataVisualizer:
    """
    Класс для комплексной визуализации финансовых данных с мультииндексной структурой.
    
    Обеспечивает:
    - Построение графиков цен и объемов
    - Анализ рыночных показателей
    - Обнаружение и визуализацию аномалий
    - Автоматическую адаптацию визуализаций под данные

    Параметры:
        full_df (pd.DataFrame): Исходный DataFrame с мультииндексом колонок (Ticker, Metric)
        category (str): Категория активов ('crypto' или другие) для стилизации графиков

    Исключения:
        ValueError: При несоответствии структуры входных данных требованиям
    """

    def __init__(self, full_df, category='crypto'):
        """
        Инициализация визуализатора с автоматической проверкой структуры данных.
        
        Параметры:
            full_df (pd.DataFrame): Данные с мультииндексом (Ticker, Metric) в колонках
            category (str): Категория активов для текстовых подписей
        """
        self.full_df = full_df
        self.category = category
        self._validate_data_structure()


    def _validate_data_structure(self):
        """
        Валидация структуры входных данных.
        
        Проверяет:
        - Наличие мультииндекса в колонках
        - Присутствие обязательных метрик для каждого тикера
        
        Исключения:
            ValueError: При нарушении структуры данных или отсутствии метрик
        """
        if not isinstance(self.full_df.columns, pd.MultiIndex):
            raise ValueError("DataFrame должен иметь мультииндексные колонки (Ticker, Metric)")
            
        required_metrics = {'Open', 'High', 'Low', 'Close', 'Volume'}
        for ticker in self.full_df.columns.get_level_values(0).unique():
            available_metrics = set(self.full_df[ticker].columns)
            if not required_metrics.issubset(available_metrics):
                missing = required_metrics - available_metrics
                raise ValueError(f"Тикер {ticker} отсутствуют метрики: {missing}")


    def _get_close_prices(self) -> pd.DataFrame:
        """
        Извлечение цен закрытия из мультииндексного DataFrame.
        
        Возвращает:
            pd.DataFrame: DataFrame с ценами закрытия для каждого тикера
        """
        return self.full_df.xs('Close', level=1, axis=1).copy()


    def _get_volumes(self) -> pd.DataFrame:
        """
        Извлечение объемов торгов из мультииндексного DataFrame.
        
        Возвращает:
            pd.DataFrame: DataFrame с объемами торгов для каждого тикера
        """
        return self.full_df.xs('Volume', level=1, axis=1).copy()


    def plot_close_prices(self):
        """
        Визуализация динамики цен закрытия в виде линейных графиков.
        
        Особенности:
        - Автоматическое определение типа актива по категории
        - Адаптивные подписи осей
        - Интерактивные подсказки с точными значениями
        - Единая временная шкала для сравнения активов
        """
        close_df = self._get_close_prices()
        
        if close_df.empty:
            print("Нет данных для отображения")
            return

        fig = go.Figure()
        asset_type = 'криптовалют' if self.category == 'crypto' else 'акций'
        
        for ticker in close_df.columns:
            fig.add_trace(go.Scatter(
                x=close_df.index,
                y=close_df[ticker],
                name=ticker,
                mode='lines',
                hovertemplate="%{x}<br>Цена: %{y:.2f} USD"
            ))

        fig.update_layout(
            title=f'Цены закрытия {asset_type}',
            template='seaborn',
            xaxis_title='Дата и время',
            yaxis_title='Цена (USD)',
            hovermode='x unified',
            height=600
        )
        fig.show(render='svg')


    def plot_candlestick_with_volume(self, ticker: str, periods: int = 30):
        """
        Построение свечного графика с объемами для указанного тикера.
        
        Параметры:
            ticker (str): Тикер для визуализации
            periods (int): Количество последних периодов для отображения

        Особенности:
        - Автоматическое определение таймфрейма
        - Цветовая дифференциация объемов
        - Оптимальное форматирование дат
        - Исключение выходных дней для дневных графиков
        """
        try:
            data = self.full_df.xs(ticker, level=0, axis=1)
            if len(data) < 2:
                print(f"Недостаточно данных для {ticker}")
                return
                
            data = data.iloc[-periods:]
            
            # Определение таймфрейма
            time_diff = data.index.to_series().diff().min()
            timeframe = 'Часовые' if time_diff < pd.Timedelta(hours=23) else 'Дневные'
            period_label = f'{periods} часов' if timeframe == 'Часовые' else f'{periods} дней'

            fig = make_subplots(
                rows=2, cols=1, 
                shared_xaxes=True,
                vertical_spacing=0.03,
                row_heights=[0.8, 0.2],
                subplot_titles=(f"{ticker} {timeframe} график", "Объемы торгов")
            )
            
            # Свечи
            fig.add_trace(go.Candlestick(
                x=data.index,
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'],
                name='Цена',
                increasing_line_color='#2ECC71',
                decreasing_line_color='#E74C3C'
            ), row=1, col=1)
            
            # Объемы с цветовой шкалой
            max_vol = data['Volume'].max()
            fig.add_trace(go.Bar(
                x=data.index,
                y=data['Volume'],
                name='Объем',
                marker=dict(
                    color=data['Volume'],
                    colorscale='Blues',
                    cmin=0,
                    cmax=max_vol,
                    opacity=0.9
                )
            ), row=2, col=1)
            
            # Форматирование осей
            tickformat = '%d.%m.%Y %H:%M' if timeframe == 'Часовые' else '%d.%m.%Y'
            fig.update_xaxes(
                rangebreaks=[dict(bounds=["sat", "mon"])] if timeframe == 'Дневные' else None,
                tickformat=tickformat,
                row=1, col=1
            )
            
            fig.update_layout(
                title=f"{ticker} • {timeframe} данные • {period_label}",
                template='seaborn',
                height=700,
                showlegend=False,
                margin=dict(t=80, b=20),
                xaxis_rangeslider_visible=False
            )
            fig.show()
            
        except KeyError:
            print(f"Данные для тикера {ticker} не найдены")
        except Exception as e:
            print(f"Ошибка визуализации: {str(e)}")


    def plot_market_overview(self, period: str = '7D'):
        """
        Генерация сводной таблицы рыночных показателей.
        
        Параметры:
            period (str): Период для расчета изменения цен (формат pandas offset)

        Особенности:
        - Динамический расчет размеров таблицы
        - Цветовое кодирование изменений цен
        - Автоматическое форматирование чисел
        - Адаптивные подписи единиц измерения
        """
        close_prices = self._get_close_prices()
        volumes = self._get_volumes()
        
        metrics = pd.DataFrame({
            'Тикер': close_prices.columns,
            'Последняя цена': close_prices.iloc[-1].values,
            f'Δ% ({period})': close_prices.pct_change(freq=period).iloc[-1].values * 100,
            'Волатильность (30 периодов)': close_prices.pct_change().rolling(30).std().iloc[-1].values * 100,
            'Ср. объем (30 периодов)': volumes.iloc[-30:].mean().values
        }).sort_values('Последняя цена', ascending=False)

        # Динамические настройки размера
        num_rows = len(metrics)
        row_height = 35
        header_height = 80
        max_height = 1000
        HEIGH_COEF = 2
        table_height = min(header_height + (num_rows+HEIGH_COEF) * row_height, max_height)

        changes = metrics[f'Δ% ({period})']
        max_change = max(abs(changes.max()), abs(changes.min())) or 1
        color_intensity = 0.3 * (abs(changes) / max_change).clip(0, 1)
        
        metrics_formatted = {
            'Тикер': metrics['Тикер'],
            'Последняя цена': metrics['Последняя цена'].apply(lambda x: f"${x:,.2f}"),
            f'Δ% ({period})': metrics[f'Δ% ({period})'].apply(lambda x: f"{x:+,.1f}%"),
            'Волатильность (30 периодов)': metrics['Волатильность (30 периодов)'].apply(lambda x: f"{x:.1f}%"),
            'Ср. объем (30 периодов)': metrics['Ср. объем (30 периодов)'].apply(
                lambda x: f"{x/1e6:.1f}M" if x > 1e6 else f"{x/1e3:.0f}K")
        }
        
        fig = go.Figure(data=[go.Table(
            columnwidth=[2, 1.5, 1.5, 1.5, 1.5],
            header=dict(
                values=list(metrics_formatted.keys()),
                fill_color='#2c3e50',
                font=dict(color='white', size=12),
                height=35
            ),
            cells=dict(
                values=list(metrics_formatted.values()),
                fill_color=[
                    'white',
                    '#f8f9f9',
                    [f'rgba(46, 204, 113, {0.3 + i})' if chg >=0 
                     else f'rgba(231, 76, 60, {0.3 + i})' 
                     for chg, i in zip(changes, color_intensity)],
                    '#f8f9f9',
                    '#f8f9f9'
                ],
                font=dict(color='#2c3e50', size=12),
                height=35
            )
        )])
        
        fig.update_layout(
            title=dict(text=f"Рыночные показатели • Период: {period}", x=0.05, font=dict(size=16)),
            margin=dict(t=60, l=10, r=10, b=55),
            height=table_height,
            annotations=[dict(
                x=0.5,
                y=-0.17,
                showarrow=False,
                text="▲ Цветовая интенсивность соответствует силе изменения",
                xref="paper",
                yref="paper",
                font=dict(color='#7f8c8d', size=10)
            )]
        )
        fig.show()


    def detect_outliers(self, threshold: float = 3.0) -> dict:
        """
        Обнаружение выбросов методом z-score для всех тикеров.
        
        Параметры:
            threshold (float): Пороговое значение z-score (по умолчанию 3.0)
            
        Возвращает:
            dict: Словарь с результатами для каждого тикера:
                {
                    'count': количество выбросов,
                    'dates': даты выбросов,
                    'values': значения цен
                }
        """
        outliers = {}
        close_df = self._get_close_prices()
        
        for ticker in close_df.columns:
            data = close_df[ticker].dropna()
            z_scores = (data - data.mean()) / data.std()
            mask = abs(z_scores) > threshold
            
            outliers[ticker] = {
                'count': mask.sum(),
                'dates': data.index[mask].strftime('%Y-%m-%d').tolist(),
                'values': data[mask].tolist()
            }
        return outliers


    def plot_outliers(self, ticker: str, threshold: float = 3.0):
        """
        Визуализация выбросов на графике цен закрытия.
        
        Параметры:
            ticker (str): Название тикера для анализа
            threshold (float): Пороговое значение z-score (по умолчанию 3.0)

        Особенности:
        - Совмещение линейного графика с маркерами выбросов
        - Автоматическое определение аномальных точек
        - Интерактивные элементы управления масштабом
        """
        try:
            data = self._get_close_prices()[ticker]
            z_scores = (data - data.mean()) / data.std()
            outliers = data[abs(z_scores) > threshold]

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=data.index,
                y=data,
                mode='lines',
                name='Цена закрытия',
                line=dict(color='#2980B9')
            ))
            fig.add_trace(go.Scatter(
                x=outliers.index,
                y=outliers,
                mode='markers',
                marker=dict(color='#E74C3C', size=8),
                name='Выбросы'
            ))
            fig.update_layout(
                title=f'Выбросы цен для {ticker} (Z-score > {threshold})',
                template='seaborn',
                height=500,
                xaxis_title='Дата',
                yaxis_title='Цена (USD)'
            )
            fig.show()
            
        except KeyError:
            print(f"Тикер {ticker} не найден в данных")


    def plot_price_distribution(self, ticker: str, bins: int = None):
        """
        Визуализация распределения цен закрытия с использованием гистограммы и горизонтального боксплота.
        
        Параметры:
            ticker (str): Тикер для анализа распределения цен
            bins (int, опционально): Количество интервалов для гистограммы. Если None - автоматический расчет

        Особенности:
            - Совмещение вертикальной гистограммы и горизонтального боксплота
            - Автоматический расчет оптимальных бинов гистограммы
            - Цветовая синхронизация с общей стилистикой класса
            - Интерактивные подсказки с квантильной информацией
            - Адаптивная подпись осей в зависимости от категории
        """
        try:
            close_prices = self._get_close_prices()[ticker].dropna()
            if len(close_prices) < 10:
                print(f"Недостаточно данных для {ticker} (минимум 10 точек)")
                return

            fig = make_subplots(
                rows=2, cols=1,
                vertical_spacing=0.05,
                row_heights=[0.8, 0.2],
                shared_xaxes=True
            )

            # Гистограмма
            fig.add_trace(go.Histogram(
                x=close_prices,
                nbinsx=bins,
                name='Распределение',
                marker_color='#3498DB',
                opacity=0.7,
                showlegend=False,
                hovertemplate="Диапазон: %{x}<br>Частота: %{y}",
                histnorm='probability density' if bins else None
            ), row=1, col=1)

            # Боксплот
            fig.add_trace(go.Box(
                x=close_prices,
                name=ticker,
                marker_color='#2ECC71',
                boxpoints=False,
                orientation='h',
                showlegend=False,
                hoverinfo='x'
            ), row=2, col=1)

            # Форматирование подписей
            currency = 'USD' if self.category == 'crypto' else ''
            y_title = 'Плотность вероятности' if bins else 'Частота'

            fig.update_layout(
                title=f'Распределение цен {ticker}',
                template='seaborn',
                height=600,
                xaxis_title=f'Цена закрытия, {currency}',
                yaxis_title=y_title,
                hovermode='x unified',
                margin=dict(t=60, b=40)
            )
            
            fig.update_yaxes(showticklabels=False, row=2, col=1)
            fig.update_xaxes(row=2, col=1, title=f'Цена, {currency}')

            fig.show()

        except KeyError:
            print(f"Тикер {ticker} не найден в данных")
        except Exception as e:
            print(f"Ошибка визуализации распределения: {str(e)}")