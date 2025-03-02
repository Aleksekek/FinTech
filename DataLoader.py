import yfinance as yf
import pandas as pd
import os
import hashlib
import re
from typing import List, Optional
import pytz
from datetime import datetime


class DataLoader:
    """
    Класс для загрузки и управления финансовыми данными из Yahoo Finance.
    
    Обеспечивает загрузку, кэширование и обновление данных с учетом временных зон,
    автоматической обработки дубликатов и оптимизированного хранения.

    Параметры:
        tickers (List[str]): Список тикеров для загрузки
        category (str): Категория данных (используется для именования файлов)
        interval (str): Интервал данных (по умолчанию '1d')
        start_date (str): Начальная дата данных в формате 'YYYY-MM-DD'
        end_date (Optional[str]): Конечная дата данных (None для текущей даты)
        user_timezone (str): Временная зона пользователя (по умолчанию 'Europe/Moscow')

    Особенности:
        - Автоматическая проверка актуальности данных
        - Конвертация временных зон в указанную пользователем
        - Генерация уникальных имен файлов для кэширования
        - Оптимизированное обновление без полной перезагрузки
    """
    
    VALID_INTERVALS = ['1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo']

    def __init__(
        self,
        tickers: List[str],
        category: str,
        interval: str = '1d',
        start_date: str = '2024-01-01',
        end_date: Optional[str] = None,
        user_timezone: str = 'Europe/Moscow'
    ):
        self.tickers = sorted(list(set(tickers)))
        self.category = category
        self.interval = interval
        self.start_date = start_date
        self.end_date = end_date
        self.user_timezone = user_timezone
        self.filename = self._generate_filename()
        self._validate_params()


    def _validate_params(self) -> None:
        """
        Проверяет валидность входных параметров.

        Выполняет проверки:
            - Наличие хотя бы одного тикера в списке
            - Корректность указанного интервала
            - Валидность временной зоны

        Исключения:
            ValueError: При несоответствии параметров требованиям
        """
        if not self.tickers:
            raise ValueError("Список тикеров не может быть пустым")
        if self.interval not in self.VALID_INTERVALS:
            raise ValueError(f"Недопустимый интервал. Допустимые значения: {self.VALID_INTERVALS}")
        try:
            pytz.timezone(self.user_timezone)
        except pytz.UnknownTimeZoneError:
            raise ValueError(f"Неизвестная временная зона: {self.user_timezone}")


    def _generate_filename(self) -> str:
        """
        Генерирует уникальное имя файла для кэширования данных.

        Формат имени:
            [категория]_[хэш тикеров]_[интервал]_[начальная дата]_[конечная дата].csv

        Возвращает:
            str: Уникальное имя файла для сохранения данных
        """
        safe_category = re.sub(r'[^a-zA-Z0-9_]', '', self.category)
        tickers_hash = hashlib.sha256(','.join(self.tickers).encode()).hexdigest()[:12]
        end_date_str = self.end_date if self.end_date else \
            datetime.now().replace(minute=0, second=0, microsecond=0).strftime('%Y-%m-%d-%H') if 'h' in self.interval \
            else datetime.now().replace(minute=0, second=0, microsecond=0).strftime('%Y-%m-%d')
        return f"{safe_category}_{tickers_hash}_{self.interval}_{self.start_date}_{end_date_str}.csv"


    def _download_new_data(self) -> pd.DataFrame:
        """
        Загружает новые данные с Yahoo Finance.

        Возвращает:
            pd.DataFrame: Загруженные данные с:
                - UTC временной зоной в индексе
                - Удаленными полностью пустыми колонками

        Исключения:
            Выводит сообщение об ошибке при проблемах с загрузкой
        """
        try:
            df = yf.download(
                self.tickers,
                start=self.start_date,
                end=self.end_date,
                interval=self.interval,
                group_by='ticker',
                progress=False,
                auto_adjust=True,
                threads=True
            )
            
            if not df.empty:
                df = df.dropna(axis=1, how='all')
                if df.index.tz is None:
                    df.index = df.index.tz_localize('UTC')
                else:
                    df.index = df.index.tz_convert('UTC')
                return df
                
        except Exception as e:
            print(f"Ошибка загрузки: {e}")
        
        return pd.DataFrame()


    def _handle_existing_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Обрабатывает данные, загруженные из файла.

        Параметры:
            df (pd.DataFrame): DataFrame с прочитанными из файла данными

        Возвращает:
            pd.DataFrame: Данные с:
                - Правильным типом индекса
                - UTC временной зоной
        """
        df.index = pd.to_datetime(df.index)
        if df.index.tz is None:
            df.index = df.index.tz_localize('UTC')
        else:
            df.index = df.index.tz_convert('UTC')
        return df


    def load_data(self, force_update: bool = False) -> pd.DataFrame:
        """
        Основной метод для получения данных.

        Параметры:
            force_update (bool): Принудительная перезагрузка данных (по умолчанию False)

        Возвращает:
            pd.DataFrame: Данные в указанной пользователем временной зоне

        Логика работы:
            1. Проверяет наличие локального файла
            2. При отсутствии или force_update=True загружает новые данные
            3. При наличии файла проверяет актуальность данных
            4. При необходимости дозагружает обновления

        Исключения:
            ValueError: При невозможности загрузить данные
        """
        if os.path.exists(self.filename) and not force_update:
            try:
                df = pd.read_csv(
                    self.filename,
                    header=[0, 1],
                    index_col=0,
                    parse_dates=True
                )
                
                if not df.empty:
                    df = self._handle_existing_data(df)
                    
                    if self._check_data_freshness(df.index[-1]):
                        return self._convert_to_user_tz(df)
                        
                    updated_df = self._update_data(df)
                    updated_df.to_csv(self.filename)
                    return self._convert_to_user_tz(updated_df)
                    
            except Exception as e:
                print(f"Ошибка чтения: {e}")
        
        new_df = self._download_new_data()
        if not new_df.empty:
            print(self.filename)
            new_df.to_csv(self.filename)
            return self._convert_to_user_tz(new_df)
            
        raise ValueError("Не удалось загрузить данные")


    def _check_data_freshness(self, last_date: pd.Timestamp) -> bool:
        """
        Проверяет актуальность имеющихся данных.

        Параметры:
            last_date (pd.Timestamp): Время последней доступной точки данных

        Возвращает:
            bool: True если данные актуальны, False если требуется обновление
        """
        now_utc = pd.Timestamp.now(tz='UTC')
        age = now_utc - last_date
        
        thresholds = {
            '1d': pd.Timedelta(hours=36),
            '1h': pd.Timedelta(hours=3),
            '1m': pd.Timedelta(minutes=5)
        }
        
        return age <= thresholds.get(self.interval[:2], pd.Timedelta(days=365))


    def _convert_to_user_tz(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Конвертирует временную зону индекса DataFrame.

        Параметры:
            df (pd.DataFrame): Исходные данные с UTC временем

        Возвращает:
            pd.DataFrame: Данные с конвертированной временной зоной
        """
        return df.tz_convert(self.user_timezone) if not df.empty else df


    def _update_data(self, existing_df: pd.DataFrame) -> pd.DataFrame:
        """
        Обновляет существующие данные новыми значениями.

        Параметры:
            existing_df (pd.DataFrame): Текущие данные

        Возвращает:
            pd.DataFrame: Объединенные данные после удаления дубликатов
        """
        last_point = existing_df.index[-1] + pd.Timedelta(milliseconds=1)
        updater = DataLoader(
            tickers=self.tickers,
            category=self.category,
            interval=self.interval,
            start_date=last_point.strftime('%Y-%m-%d'),
            end_date=self.end_date,
            user_timezone=self.user_timezone
        )
        new_data = updater._download_new_data()
        
        combined_df = pd.concat([existing_df, new_data])
        mask = ~combined_df.index.duplicated(keep='last')
        return combined_df.loc[mask]


    def __repr__(self):
        """
        Возвращает строковое представление объекта.

        Возвращает:
            str: Информация об основных параметрах загрузчика
        """
        return f"DataLoader(tickers={self.tickers}, interval={self.interval}, tz={self.user_timezone})"