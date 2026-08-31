import MetaTrader5 as mt5
from dataclasses import dataclass
from typing import Optional, List, Tuple
from datetime import datetime
import pandas as pd
import logging
from src.config import MT5Config

logger = logging.getLogger(__name__)


@dataclass
class SymbolInfo:
    name: str
    description: str
    point: float
    digits: int
    spread: int
    trade_mode: int
    volume_min: float
    volume_max: float
    volume_step: float


class MT5Client:
    def __init__(self, config: MT5Config):
        self.config = config
        self._initialized = False
        self.gold_symbol: Optional[str] = None
        self.silver_symbol: Optional[str] = None

    def initialize(self) -> bool:
        if self._initialized:
            return True

        logger.info("Initializing MT5 connection")
        if not mt5.initialize():
            logger.error(f"MT5 initialization failed: {mt5.last_error()}")
            return False

        self._initialized = True
        logger.info("MT5 connection established")
        return True

    def shutdown(self):
        if self._initialized:
            mt5.shutdown()
            self._initialized = False
            logger.info("MT5 connection closed")

    def __enter__(self):
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()

    def discover_symbols(self) -> Tuple[Optional[str], Optional[str]]:
        if not self._initialized:
            raise RuntimeError("MT5 not initialized")

        all_symbols = mt5.symbols_get()
        if all_symbols is None:
            logger.error(f"Failed to get symbols: {mt5.last_error()}")
            return None, None

        symbol_names = [s.name for s in all_symbols]
        logger.debug(f"Available symbols: {symbol_names}")

        gold = self._find_symbol(symbol_names, self.config.gold_symbols, "Gold")
        silver = self._find_symbol(symbol_names, self.config.silver_symbols, "Silver")

        if gold:
            logger.info(f"Gold symbol: {gold}")
            self.gold_symbol = gold
        else:
            logger.error("Gold symbol not found. Add broker symbol to config.yaml")

        if silver:
            logger.info(f"Silver symbol: {silver}")
            self.silver_symbol = silver
        else:
            logger.error("Silver symbol not found. Add broker symbol to config.yaml")

        return gold, silver

    def _find_symbol(self, available: List[str], candidates: List[str], asset_name: str) -> Optional[str]:
        available_lower = {s.lower(): s for s in available}

        for candidate in candidates:
            if candidate in available:
                logger.debug(f"Exact match for {asset_name}: {candidate}")
                return candidate

            candidate_lower = candidate.lower()
            if candidate_lower in available_lower:
                match = available_lower[candidate_lower]
                logger.debug(f"Case-insensitive match for {asset_name}: {match}")
                return match

        for candidate in candidates:
            candidate_lower = candidate.lower()
            for avail in available:
                if candidate_lower in avail.lower() or avail.lower() in candidate_lower:
                    logger.warning(f"Partial match for {asset_name}: {avail} (candidate: {candidate})")
                    return avail

        return None

    def select_symbol(self, symbol: str) -> bool:
        if not mt5.symbol_select(symbol, True):
            logger.error(f"Failed to select symbol {symbol}: {mt5.last_error()}")
            return False
        return True

    def get_symbol_info(self, symbol: str) -> Optional[SymbolInfo]:
        info = mt5.symbol_info(symbol)
        if info is None:
            return None

        return SymbolInfo(
            name=info.name,
            description=info.description,
            point=info.point,
            digits=info.digits,
            spread=info.spread,
            trade_mode=info.trade_mode,
            volume_min=info.volume_min,
            volume_max=info.volume_max,
            volume_step=info.volume_step
        )

    def get_rates(self, symbol: str, timeframe: int, bars: int) -> pd.DataFrame:
        if not self._initialized:
            raise RuntimeError("MT5 not initialized")

        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, bars)
        if rates is None:
            logger.error(f"Failed to get rates for {symbol}: {mt5.last_error()}")
            return pd.DataFrame()

        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
        df.set_index('time', inplace=True)
        df.sort_index(inplace=True)
        df = df[~df.index.duplicated(keep='first')]

        logger.info(f"Retrieved {len(df)} bars for {symbol}")
        return df

    def get_ticks(self, symbol: str, from_time: datetime, count: int) -> pd.DataFrame:
        if not self._initialized:
            raise RuntimeError("MT5 not initialized")

        ticks = mt5.copy_ticks_from(symbol, from_time, count, mt5.COPY_TICKS_ALL)
        if ticks is None:
            logger.error(f"Failed to get ticks for {symbol}: {mt5.last_error()}")
            return pd.DataFrame()

        df = pd.DataFrame(ticks)
        df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
        df.set_index('time', inplace=True)
        df.sort_index(inplace=True)
        df = df[~df.index.duplicated(keep='first')]

        logger.info(f"Retrieved {len(df)} ticks for {symbol}")
        return df

    def timeframe_to_mt5(self, tf: str) -> int:
        mapping = {
            'M1': mt5.TIMEFRAME_M1,
            'M2': mt5.TIMEFRAME_M2,
            'M3': mt5.TIMEFRAME_M3,
            'M4': mt5.TIMEFRAME_M4,
            'M5': mt5.TIMEFRAME_M5,
            'M6': mt5.TIMEFRAME_M6,
            'M10': mt5.TIMEFRAME_M10,
            'M12': mt5.TIMEFRAME_M12,
            'M15': mt5.TIMEFRAME_M15,
            'M20': mt5.TIMEFRAME_M20,
            'M30': mt5.TIMEFRAME_M30,
            'H1': mt5.TIMEFRAME_H1,
            'H2': mt5.TIMEFRAME_H2,
            'H3': mt5.TIMEFRAME_H3,
            'H4': mt5.TIMEFRAME_H4,
            'H6': mt5.TIMEFRAME_H6,
            'H8': mt5.TIMEFRAME_H8,
            'H12': mt5.TIMEFRAME_H12,
            'D1': mt5.TIMEFRAME_D1,
            'W1': mt5.TIMEFRAME_W1,
            'MN1': mt5.TIMEFRAME_MN1,
        }
        if tf not in mapping:
            raise ValueError(f"Unknown timeframe: {tf}")
        return mapping[tf]