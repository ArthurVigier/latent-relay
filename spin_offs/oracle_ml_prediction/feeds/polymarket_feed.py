"""
PolymarketOracleFeed
====================

Live market data ingestion, feature extraction, and time-series preparation
for oracle prediction models.

Uses Polymarket's public REST APIs (Gamma Discovery, CLOB price data).
No authentication required.

Key capabilities:
  - Search markets by topic/keyword
  - Extract outcomePrices as probability time series
  - Build structured feature vectors (vol, spread, liquidity, time-to-close)
  - Compute textual features for JEPA world model input
  - Track price history via conditionId
"""

from __future__ import annotations

import json
import time
import logging
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime, timezone

try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False

try:
    import urllib.request
    import urllib.parse
    import urllib.error
    _HAS_URLLIB = True
except ImportError:
    _HAS_URLLIB = False

log = logging.getLogger("oracle.feed")

# ── Polymarket API endpoints ────────────────────────────────────────────────
GAMMA_API = "https://gamma-api.polymarket.com"
CLOB_API = "https://clob.polymarket.com"
DATA_API = "https://data-api.polymarket.com"


@dataclass
class MarketSnapshot:
    """A single market data point for oracle ingestion."""
    condition_id: str
    question: str
    yes_price: float  # 0.0 - 1.0
    no_price: float   # 0.0 - 1.0
    volume: float     # USDC
    spread: float
    liquidity: float
    closed: bool
    timestamp: str
    slug: str = ""
    # Derived oracle features
    implied_probability: float = 0.0
    uncertainty: float = 0.0  # entropy of the distribution
    category: str = ""
    tags: list[str] = field(default_factory=list)
    
    def to_feature_vector(self) -> list[float]:
        """Convert to normalized feature vector for model input."""
        return [
            self.yes_price,
            self.no_price,
            min(self.volume / 1_000_000, 1.0),      # volume in $M, capped at 1.0
            self.spread,
            min(self.liquidity / 100_000, 1.0),      # liquidity capped
            float(self.closed),
            self.implied_probability,
            self.uncertainty,
        ]
    
    def to_text_prompt(self) -> str:
        """Convert to text description for JEPA world model input."""
        status = "CLOSED" if self.closed else "OPEN"
        return (
            f"Market: '{self.question}' [{status}]\n"
            f"Yes: {self.yes_price*100:.1f}% | No: {self.no_price*100:.1f}%\n"
            f"Volume: ${self.volume:,.0f} | Spread: {self.spread:.4f}\n"
            f"Implied prob: {self.implied_probability*100:.1f}% | "
            f"Uncertainty: {self.uncertainty:.4f}"
        )
    
    def to_dict(self) -> dict:
        return {
            "condition_id": self.condition_id,
            "question": self.question,
            "yes_price": self.yes_price,
            "no_price": self.no_price,
            "volume": self.volume,
            "spread": self.spread,
            "liquidity": self.liquidity,
            "closed": self.closed,
            "timestamp": self.timestamp,
            "slug": self.slug,
            "implied_probability": self.implied_probability,
            "uncertainty": self.uncertainty,
            "category": self.category,
            "tags": self.tags,
            "feature_vector": self.to_feature_vector(),
        }


@dataclass
class TopicFeed:
    """Aggregated data feed for a prediction topic."""
    topic: str
    markets: list[MarketSnapshot]
    aggregate_yes: float = 0.0     # mean implied probability
    aggregate_volume: float = 0.0   # total volume
    trend_direction: float = 0.0    # -1 to +1
    consensus_strength: float = 0.0 # 0 (split) to 1 (unanimous)


class PolymarketOracleFeed:
    """
    Oracle data feed for prediction markets.
    
    Ingests live market data, extracts features, and prepares
    inputs for the JEPA world model and multi-agent verification.
    """
    
    def __init__(self, request_timeout: int = 15):
        self.timeout = request_timeout
        self._history_cache: dict[str, list[dict]] = {}
    
    def _get(self, url: str) -> dict | list:
        """GET request with error handling."""
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                             "ERIS-Oracle/0.1.0"
            }
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read().decode())
    
    def search_markets(self, query: str, limit: int = 10) -> list[MarketSnapshot]:
        """Search for markets by keyword and return parsed snapshots."""
        log.info(f"Searching Polymarket: '{query}' (limit={limit})")
        q = urllib.parse.quote(query)
        data = self._get(f"{GAMMA_API}/public-search?q={q}&limit={limit}")
        
        markets: list[MarketSnapshot] = []
        events = data.get("events", [])
        
        for evt in events:
            for mkt in evt.get("markets", []):
                snapshot = self._parse_market(mkt)
                if snapshot:
                    markets.append(snapshot)
        
        log.info(f"Found {len(markets)} markets for '{query}'")
        return markets
    
    def get_trending(self, limit: int = 20) -> list[MarketSnapshot]:
        """Get trending markets by volume."""
        log.info(f"Fetching top {limit} trending markets")
        events = self._get(
            f"{GAMMA_API}/events?limit={limit}&active=true"
            f"&closed=false&order=volume&ascending=false"
        )
        
        markets: list[MarketSnapshot] = []
        for evt in events:
            for mkt in evt.get("markets", []):
                snapshot = self._parse_market(mkt)
                if snapshot:
                    markets.append(snapshot)
        
        log.info(f"Found {len(markets)} trending market snapshots")
        return markets
    
    def get_market_detail(self, slug: str) -> Optional[MarketSnapshot]:
        """Get detailed data for a specific market."""
        data = self._get(f"{GAMMA_API}/markets?slug={urllib.parse.quote(slug)}")
        if not data:
            return None
        return self._parse_market(data[0])
    
    def get_price_history(self, condition_id: str, 
                          interval: str = "1day",
                          fidelity: int = 50) -> list[dict]:
        """Get price history for a market condition."""
        if condition_id in self._history_cache:
            return self._history_cache[condition_id]
        
        try:
            data = self._get(
                f"{CLOB_API}/prices-history?market={condition_id}"
                f"&interval={interval}&fidelity={fidelity}"
            )
            history = data.get("history", [])
            self._history_cache[condition_id] = history
            return history
        except Exception as e:
            log.warning(f"Failed to fetch history for {condition_id}: {e}")
            return []
    
    def get_orderbook(self, token_id: str) -> dict:
        """Get current orderbook for a market token."""
        try:
            data = self._get(f"{CLOB_API}/book?token_id={token_id}")
            return {
                "bids": data.get("bids", []),
                "asks": data.get("asks", []),
                "last_trade": data.get("last_trade_price", None),
                "tick_size": data.get("tick_size", None),
            }
        except Exception as e:
            log.warning(f"Failed to fetch orderbook for {token_id}: {e}")
            return {"bids": [], "asks": [], "last_trade": None}
    
    def build_topic_feed(self, query: str) -> TopicFeed:
        """Build an aggregated feed for a prediction topic."""
        markets = self.search_markets(query)
        if not markets:
            return TopicFeed(topic=query, markets=[])
        
        active = [m for m in markets if not m.closed]
        if not active:
            return TopicFeed(topic=query, markets=markets)
        
        # Aggregate statistics
        total_yes = sum(m.yes_price for m in active)
        n = len(active)
        mean_prob = total_yes / n if n > 0 else 0
        variance = sum((m.yes_price - mean_prob)**2 for m in active) / n if n > 1 else 0
        
        return TopicFeed(
            topic=query,
            markets=markets,
            aggregate_yes=mean_prob,
            aggregate_volume=sum(m.volume for m in markets),
            trend_direction=0.0,  # Computed from price history
            consensus_strength=1.0 - min(variance ** 0.5, 1.0),
        )
    
    def _parse_market(self, mkt: dict) -> Optional[MarketSnapshot]:
        """Parse a market dict from Gamma API into a MarketSnapshot."""
        try:
            # Parse double-encoded JSON fields
            outcome_prices = mkt.get("outcomePrices", "[]")
            if isinstance(outcome_prices, str):
                outcome_prices = json.loads(outcome_prices)
            
            outcomes = mkt.get("outcomes", "[]")
            if isinstance(outcomes, str):
                outcomes = json.loads(outcomes)
            
            if len(outcome_prices) < 2:
                return None
            
            yes_price = float(outcome_prices[0])
            no_price = float(outcome_prices[1])
            volume = float(mkt.get("volume", 0))
            closed = mkt.get("closed", False)
            condition_id = mkt.get("conditionId", "")
            
            # Fetch CLOB data for spread
            spread = 0.0
            liquidity = 0.0
            clob_tokens = mkt.get("clobTokenIds", "[]")
            if isinstance(clob_tokens, str):
                clob_tokens = json.loads(clob_tokens)
            
            if clob_tokens and len(clob_tokens) >= 2:
                try:
                    spread_data = self._get(
                        f"{CLOB_API}/spread?token_id={clob_tokens[0]}"
                    )
                    spread = float(spread_data.get("spread", 0))
                except:
                    spread = no_price - yes_price
            
            # Compute derived features
            implied = yes_price  # Yes price is the implied probability
            entropy = self._binary_entropy(yes_price)
            
            # Extract tags from market description/metadata
            tags = []
            desc = mkt.get("description", "")
            for keyword in ["politics", "crypto", "sports", "finance", "tech",
                          "ai", "election", "regulation", "climate"]:
                if keyword in desc.lower():
                    tags.append(keyword)
            
            return MarketSnapshot(
                condition_id=condition_id,
                question=mkt.get("question", ""),
                yes_price=yes_price,
                no_price=no_price,
                volume=volume,
                spread=spread,
                liquidity=liquidity,
                closed=closed,
                timestamp=mkt.get("endDate", "") or datetime.now(timezone.utc).isoformat(),
                slug=mkt.get("slug", ""),
                implied_probability=implied,
                uncertainty=entropy,
                category="",
                tags=tags,
            )
        except Exception as e:
            log.warning(f"Failed to parse market: {e}")
            return None
    
    @staticmethod
    def _binary_entropy(p: float) -> float:
        """Compute binary entropy: -p*log2(p) - (1-p)*log2(1-p)."""
        if p <= 0 or p >= 1:
            return 0.0
        try:
            import math
            return -(p * math.log2(p) + (1 - p) * math.log2(1 - p))
        except (ValueError, ZeroDivisionError):
            return 0.0


# ── Convenience API ─────────────────────────────────────────────────────────

def fetch_polymarket_query(query: str, limit: int = 10) -> list[MarketSnapshot]:
    """Quick search: returns parsed market snapshots."""
    feed = PolymarketOracleFeed()
    return feed.search_markets(query, limit=limit)


def fetch_polymarket_trending(limit: int = 20) -> list[MarketSnapshot]:
    """Quick fetch of trending markets."""
    feed = PolymarketOracleFeed()
    return feed.get_trending(limit=limit)


def build_oracle_feed(topic: str) -> TopicFeed:
    """Build aggregated feed for a prediction topic."""
    feed = PolymarketOracleFeed()
    return feed.build_topic_feed(topic)
