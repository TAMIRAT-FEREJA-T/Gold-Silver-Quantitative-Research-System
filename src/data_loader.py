import pandas as pd
import logging
from typing import Optional
from src.mt5_client import MT5Client
from src.config import Config

logger = logging.getLogger(__name__)


class DataLoader:
    def __init__(self, client: MT5Client, config: Config):
        self.client = client
        self.config = config

    def load_gold_silver(self) -> tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        gold_symbol = self.client.gold_symbol
        silver_symbol = self.client.silver_symbol

        if not gold_symbol or not silver_symbol:
            logger.error("Symbols not discovered")
            return None, None

        if not self.client.select_symbol(gold_symbol):
            return None, None
        if not self.client.select_symbol(silver_symbol):
            return None, None

        tf = self.client.timeframe_to_mt5(self.config.mt5.timeframe)
        bars = self.config.data.bars

        logger.info(f"Downloading {bars} {self.config.mt5.timeframe} candles for {gold_symbol} and {silver_symbol}")

        gold_df = self.client.get_rates(gold_symbol, tf, bars)
        silver_df = self.client.get_rates(silver_symbol, tf, bars)

        if gold_df.empty or silver_df.empty:
            logger.error("Failed to retrieve data")
            return None, None

        return gold_df, silver_df

    def load_ticks(self, symbol: str, from_time: pd.Timestamp, count: int) -> Optional[pd.DataFrame]:
        return self.client.get_ticks(symbol, from_time.to_pydatetime(), count)