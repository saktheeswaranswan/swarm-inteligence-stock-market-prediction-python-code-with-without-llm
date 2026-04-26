# ╔══════════════════════════════════════════════════════════════════════════╗
# ║   🇮🇳  SWARM INTELLIGENCE ENGINE — INDIAN STOCK MARKET (NSE/BSE)        ║
# ║   Run in Google Colab · Live Data · URL News Scraping · Multi-Agent     ║
# ╚══════════════════════════════════════════════════════════════════════════╝
#
# HOW TO USE IN GOOGLE COLAB:
#   1. Upload this file  OR  paste the entire code into a Colab cell
#   2. Run the cell — it will install packages automatically
#   3. Enter number of agents when prompted
#   4. Enter Indian stock ticker  e.g.  RELIANCE.NS  /  TCS.NS  /  INFY.NS
#   5. Optionally paste a news URL for live sentiment analysis
#   6. Watch agents debate across rounds and get a final PREDICTION
# ═══════════════════════════════════════════════════════════════════════════

# ── STEP 0 : auto-install dependencies ──────────────────────────────────────
import subprocess, sys

def pip(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

print("📦 Installing dependencies...")
for p in ["yfinance", "requests", "beautifulsoup4", "textblob", "colorama", "tabulate"]:
    pip(p)
# Download TextBlob corpora silently
subprocess.run([sys.executable, "-m", "textblob.download_corpora"], capture_output=True)
print("✅ All packages ready.\n")

# ── STEP 1 : imports ─────────────────────────────────────────────────────────
import yfinance as yf
import requests
from bs4 import BeautifulSoup
from textblob import TextBlob
import random, math, textwrap, time
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from collections import defaultdict
from enum import Enum
from tabulate import tabulate
import warnings
warnings.filterwarnings("ignore")

# ── STEP 2 : colour helpers (works in Colab) ─────────────────────────────────
try:
    from colorama import Fore, Style, init
    init(autoreset=True)
except ImportError:
    class Fore:
        RED=GREEN=YELLOW=CYAN=MAGENTA=BLUE=WHITE=""
    class Style:
        BRIGHT=RESET_ALL=""

C  = Fore.CYAN
G  = Fore.GREEN
R  = Fore.RED
Y  = Fore.YELLOW
M  = Fore.MAGENTA
B  = Fore.BLUE
W  = Fore.WHITE
BR = Style.BRIGHT
RS = Style.RESET_ALL


# ═══════════════════════════════════════════════════════════════════════════
# SECTION A : DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════

class Signal(Enum):
    STRONG_BUY  =  2
    BUY         =  1
    HOLD        =  0
    SELL        = -1
    STRONG_SELL = -2

    def label(self):
        return self.name.replace("_", " ")

    def emoji(self):
        return {"STRONG_BUY":"🚀","BUY":"📈","HOLD":"🤝",
                "SELL":"📉","STRONG_SELL":"🔻"}[self.name]

    def color(self):
        return {2:G, 1:G, 0:Y, -1:R, -2:R}[self.value]


@dataclass
class MarketSnapshot:
    ticker:          str
    company_name:    str
    price:           float
    prev_close:      float
    day_high:        float
    day_low:         float
    week52_high:     float
    week52_low:      float
    volume:          int
    avg_volume:      int
    market_cap:      float      # in Crores INR
    pe_ratio:        float
    pb_ratio:        float
    eps:             float
    eps_growth:      float      # YoY %
    dividend_yield:  float
    rsi:             float      # computed
    macd:            float      # computed
    macd_signal:     float
    sma_20:          float
    sma_50:          float
    sma_200:         float
    beta:            float
    sector:          str
    news_sentiment:  float      # -1 … +1
    news_headlines:  List[str]
    sector_trend:    float      # -1 … +1
    url_headlines:   List[str]  # from user-provided URL


@dataclass
class AgentMemory:
    agent_id:          str
    signal_history:    List[Signal] = field(default_factory=list)
    confidence_history:List[float]  = field(default_factory=list)
    reasoning_history: List[str]    = field(default_factory=list)

    def update(self, sig: Signal, conf: float, text: str):
        self.signal_history.append(sig)
        self.confidence_history.append(conf)
        self.reasoning_history.append(text)

    def latest_signal(self) -> Optional[Signal]:
        return self.signal_history[-1] if self.signal_history else None

    def conviction_trend(self) -> str:
        if len(self.confidence_history) < 2:
            return "→"
        delta = self.confidence_history[-1] - self.confidence_history[-2]
        return "↑" if delta > 0.03 else ("↓" if delta < -0.03 else "→")


# ═══════════════════════════════════════════════════════════════════════════
# SECTION B : BASE AGENT
# ═══════════════════════════════════════════════════════════════════════════

class BaseAgent:
    PERSONALITY_TRAITS = []   # override in subclasses

    def __init__(self, name: str, archetype: str,
                 risk_tolerance: float, contrarian_bias: float = 0.0):
        self.name             = name
        self.archetype        = archetype
        self.risk_tolerance   = risk_tolerance
        self.contrarian_bias  = contrarian_bias
        self.memory           = AgentMemory(agent_id=name)
        self._noise           = random.gauss(0, 0.04)   # personality jitter

    def _raw_score(self, market: MarketSnapshot,
                   peer_signals: List[Signal]) -> float:
        raise NotImplementedError

    def deliberate(self, market: MarketSnapshot,
                   peer_signals: List[Signal],
                   round_num: int) -> tuple:
        score = self._raw_score(market, peer_signals)
        score += self._noise + self.contrarian_bias * 0.15
        score  = max(-2.0, min(2.0, score))

        if   score >= 1.5:  sig = Signal.STRONG_BUY
        elif score >= 0.45: sig = Signal.BUY
        elif score <= -1.5: sig = Signal.STRONG_SELL
        elif score <= -0.45:sig = Signal.SELL
        else:               sig = Signal.HOLD

        confidence = min(0.97, abs(score)/2.0 + 0.18 + self.risk_tolerance*0.05)
        reasoning  = self._build_reasoning(market, sig, score, round_num,
                                           peer_signals)
        self.memory.update(sig, confidence, reasoning)
        return sig, confidence, reasoning

    def _build_reasoning(self, market, sig, score, rnd, peers):
        raise NotImplementedError

    def peer_summary(self, peers: List[Signal]) -> str:
        if not peers:
            return "no prior data"
        counts = defaultdict(int)
        for p in peers:
            counts[p.label()] += 1
        top = max(counts, key=counts.get)
        return f"{counts[top]}/{len(peers)} agents lean {top}"

    def __repr__(self):
        return f"<{self.archetype} | {self.name}>"


# ═══════════════════════════════════════════════════════════════════════════
# SECTION C : ALL AGENT ARCHETYPES  (10 base types, cloned to fill quota)
# ═══════════════════════════════════════════════════════════════════════════

class MomentumAgent(BaseAgent):
    """Rides price momentum using RSI, MACD, and moving averages."""
    def __init__(self, name):
        super().__init__(name, "Momentum Trader", risk_tolerance=0.70)

    def _raw_score(self, m: MarketSnapshot, peers: List[Signal]) -> float:
        rsi_score   = (m.rsi - 50) / 50
        macd_score  = math.tanh(m.macd / 2)
        ma_score    = 0.0
        if m.sma_20 and m.sma_50:
            ma_score = math.tanh((m.sma_20 - m.sma_50) / m.sma_50 * 20)
        peer_avg    = sum(s.value for s in peers) / max(len(peers), 1)
        return 0.35*rsi_score + 0.30*macd_score + 0.20*ma_score + 0.15*(peer_avg/2)

    def _build_reasoning(self, m, sig, score, rnd, peers):
        ma_view = "price above 20MA — uptrend" if m.sma_20 > m.sma_50 else "price below 20MA — downtrend"
        return (f"RSI={m.rsi:.1f} ({'overbought' if m.rsi>70 else 'oversold' if m.rsi<30 else 'neutral'}), "
                f"MACD={m.macd:+.3f}, {ma_view}. "
                f"Peers: {self.peer_summary(peers)}. → {sig.label()} (score={score:+.2f})")


class ContrarianAgent(BaseAgent):
    """Buys fear, sells greed. Fades extreme crowd consensus."""
    def __init__(self, name):
        super().__init__(name, "Contrarian", risk_tolerance=0.55,
                         contrarian_bias=-1.0)

    def _raw_score(self, m: MarketSnapshot, peers: List[Signal]) -> float:
        peer_avg    = sum(s.value for s in peers) / max(len(peers), 1)
        rsi_extreme = 0.0
        if   m.rsi < 30: rsi_extreme =  1.2
        elif m.rsi > 70: rsi_extreme = -1.2
        drop_from_52h = (m.week52_high - m.price) / max(m.week52_high, 1)
        rebound_score = math.tanh(drop_from_52h * 3)
        return -0.50*(peer_avg/2) + 0.30*rsi_extreme + 0.20*rebound_score

    def _build_reasoning(self, m, sig, score, rnd, peers):
        drop_pct = (m.week52_high - m.price) / max(m.week52_high, 1) * 100
        return (f"Contrarian view: fading {self.peer_summary(peers)}. "
                f"Stock is {drop_pct:.1f}% below 52-week high. "
                f"RSI={m.rsi:.1f}. → {sig.label()}")


class FundamentalistAgent(BaseAgent):
    """Graham-style value investor. P/E, P/B, EPS growth, dividend."""
    def __init__(self, name):
        super().__init__(name, "Fundamentalist", risk_tolerance=0.35)

    def _raw_score(self, m: MarketSnapshot, peers: List[Signal]) -> float:
        pe_score  = -math.tanh((m.pe_ratio - 22) / 12)
        pb_score  = -math.tanh((m.pb_ratio  -  3) /  2)
        eps_score =  math.tanh(m.eps_growth / 20)
        div_score =  math.tanh(m.dividend_yield / 3)
        return 0.35*pe_score + 0.20*pb_score + 0.30*eps_score + 0.15*div_score

    def _build_reasoning(self, m, sig, score, rnd, peers):
        val = "undervalued" if m.pe_ratio < 22 else "overvalued"
        return (f"P/E={m.pe_ratio:.1f} ({val}), P/B={m.pb_ratio:.2f}, "
                f"EPS growth={m.eps_growth:+.1f}%, Div yield={m.dividend_yield:.2f}%. "
                f"→ {sig.label()}")


class QuantAgent(BaseAgent):
    """Multi-factor linear model across all 7 market signals."""
    def __init__(self, name):
        super().__init__(name, "Quant / Algo", risk_tolerance=0.50)

    def _raw_score(self, m: MarketSnapshot, peers: List[Signal]) -> float:
        rsi_z    = (m.rsi - 50) / 25
        macd_z   = math.tanh(m.macd)
        pe_z     = -math.tanh((m.pe_ratio - 22) / 12)
        sent_z   = m.news_sentiment
        sector_z = m.sector_trend
        eps_z    = math.tanh(m.eps_growth / 15)
        peer_avg = sum(s.value for s in peers) / max(len(peers), 1)
        beta_adj = -abs(m.beta - 1) * 0.05   # penalise very high/low beta
        return (0.18*rsi_z + 0.17*macd_z + 0.15*pe_z + 0.12*sent_z +
                0.12*sector_z + 0.13*eps_z + 0.10*(peer_avg/2) + 0.03*beta_adj)

    def _build_reasoning(self, m, sig, score, rnd, peers):
        return (f"Multi-factor score={score:+.3f}. "
                f"RSI={m.rsi:.1f} | MACD={m.macd:+.3f} | P/E={m.pe_ratio:.1f} | "
                f"Sentiment={m.news_sentiment:+.2f} | Beta={m.beta:.2f}. "
                f"→ {sig.label()}")


class SentimentAgent(BaseAgent):
    """Driven by news sentiment, URL headlines, and sector momentum."""
    def __init__(self, name):
        super().__init__(name, "Sentiment Analyst", risk_tolerance=0.60)

    def _raw_score(self, m: MarketSnapshot, peers: List[Signal]) -> float:
        # blend scraped URL headlines with default news
        all_hl_score = m.news_sentiment
        if m.url_headlines:
            url_scores = [TextBlob(h).sentiment.polarity for h in m.url_headlines]
            url_avg    = sum(url_scores) / len(url_scores)
            all_hl_score = 0.4*m.news_sentiment + 0.6*url_avg
        return 0.65*all_hl_score + 0.35*m.sector_trend

    def _build_reasoning(self, m, sig, score, rnd, peers):
        tone = "🟢 positive" if m.news_sentiment > 0.2 else \
               "🔴 negative" if m.news_sentiment < -0.2 else "🟡 mixed"
        top  = m.url_headlines[0] if m.url_headlines else \
               (m.news_headlines[0] if m.news_headlines else "—")
        return (f"News sentiment={m.news_sentiment:+.2f} ({tone}). "
                f"Top headline: \"{top[:80]}\". "
                f"Sector trend={m.sector_trend:+.2f}. → {sig.label()}")


class RiskManagerAgent(BaseAgent):
    """Dampens extreme signals; loves HOLD. Watches volatility & beta."""
    def __init__(self, name):
        super().__init__(name, "Risk Manager", risk_tolerance=0.15)

    def _raw_score(self, m: MarketSnapshot, peers: List[Signal]) -> float:
        peer_avg  = sum(s.value for s in peers) / max(len(peers), 1)
        vol_proxy = (m.day_high - m.day_low) / max(m.price, 1)
        vol_penalty = -abs(vol_proxy - 0.02) * 5   # penalise high intraday swings
        return 0.35*math.tanh(peer_avg) + 0.65*vol_penalty

    def _build_reasoning(self, m, sig, score, rnd, peers):
        vol_pct = (m.day_high - m.day_low) / max(m.price, 1) * 100
        return (f"Risk lens: intraday range={vol_pct:.2f}%, beta={m.beta:.2f}. "
                f"Peer consensus: {self.peer_summary(peers)}. "
                f"Risk-adjusted → {sig.label()}")


class TechnicalAnalystAgent(BaseAgent):
    """Reads chart patterns: Golden cross, support/resistance, 52w levels."""
    def __init__(self, name):
        super().__init__(name, "Technical Analyst", risk_tolerance=0.55)

    def _raw_score(self, m: MarketSnapshot, peers: List[Signal]) -> float:
        # Golden cross: SMA20 > SMA50 > SMA200
        cross_score = 0.0
        if m.sma_20 and m.sma_50 and m.sma_200:
            if m.sma_20 > m.sma_50 > m.sma_200:
                cross_score =  1.0
            elif m.sma_20 < m.sma_50 < m.sma_200:
                cross_score = -1.0
        # 52-week position
        range52 = m.week52_high - m.week52_low
        pos52   = (m.price - m.week52_low) / max(range52, 1)
        pos_score = (pos52 - 0.5) * 2   # -1 at bottom, +1 at top
        macd_cross = 1.0 if m.macd > m.macd_signal else -1.0
        return 0.40*cross_score + 0.35*macd_cross + 0.25*math.tanh(pos_score)

    def _build_reasoning(self, m, sig, score, rnd, peers):
        cross = "🟢 Golden Cross" if (m.sma_20 and m.sma_50 and m.sma_20 > m.sma_50) \
                else "🔴 Death Cross"
        pos   = (m.price - m.week52_low) / max(m.week52_high - m.week52_low, 1) * 100
        return (f"{cross}. 52w position={pos:.0f}% (low={m.week52_low:.2f}, high={m.week52_high:.2f}). "
                f"MACD {'above' if m.macd>m.macd_signal else 'below'} signal line. → {sig.label()}")


class MacroAgent(BaseAgent):
    """Looks at sector rotation, beta sensitivity, and market cap tier."""
    def __init__(self, name):
        super().__init__(name, "Macro / Sector", risk_tolerance=0.45)

    def _raw_score(self, m: MarketSnapshot, peers: List[Signal]) -> float:
        sector_score = m.sector_trend
        cap_score    = 0.0
        if m.market_cap > 50000:   cap_score =  0.3   # Large-cap premium
        elif m.market_cap < 5000:  cap_score = -0.2   # Small-cap risk discount
        beta_score   = -abs(m.beta - 1.0) * 0.3       # prefer beta near 1
        peer_avg     = sum(s.value for s in peers) / max(len(peers), 1)
        return 0.45*sector_score + 0.20*cap_score + 0.15*beta_score + 0.20*(peer_avg/2)

    def _build_reasoning(self, m, sig, score, rnd, peers):
        cap_tier = "Large-cap" if m.market_cap>50000 else \
                   "Mid-cap" if m.market_cap>10000 else "Small-cap"
        return (f"Sector: {m.sector} (trend={m.sector_trend:+.2f}). "
                f"{cap_tier} (₹{m.market_cap:.0f} Cr). Beta={m.beta:.2f}. "
                f"Peers: {self.peer_summary(peers)}. → {sig.label()}")


class ValueTrapDetectorAgent(BaseAgent):
    """Warns when a stock looks cheap but is actually deteriorating."""
    def __init__(self, name):
        super().__init__(name, "Value-Trap Detector", risk_tolerance=0.30)

    def _raw_score(self, m: MarketSnapshot, peers: List[Signal]) -> float:
        # cheap P/E but bad EPS trend = value trap → SELL
        pe_cheap     =  1.0 if m.pe_ratio < 15 else 0.0
        eps_bad      = -1.0 if m.eps_growth < -10 else 0.0
        value_trap   = pe_cheap + eps_bad            # -1 if trap
        sent_confirm = m.news_sentiment
        return 0.50*value_trap + 0.30*math.tanh(m.eps_growth/15) + 0.20*sent_confirm

    def _build_reasoning(self, m, sig, score, rnd, peers):
        trap = "⚠️ Possible value trap!" if (m.pe_ratio < 15 and m.eps_growth < -10) \
               else "✅ No value-trap signal"
        return (f"{trap} P/E={m.pe_ratio:.1f}, EPS growth={m.eps_growth:+.1f}%. "
                f"Sentiment confirms={m.news_sentiment:+.2f}. → {sig.label()}")


class BreakoutHunterAgent(BaseAgent):
    """Looks for volume breakouts above key moving averages."""
    def __init__(self, name):
        super().__init__(name, "Breakout Hunter", risk_tolerance=0.75)

    def _raw_score(self, m: MarketSnapshot, peers: List[Signal]) -> float:
        vol_ratio   = m.volume / max(m.avg_volume, 1)
        vol_score   = math.tanh((vol_ratio - 1.0) * 1.5)
        price_vs_ma = math.tanh((m.price - m.sma_50) / max(m.sma_50, 1) * 30) \
                      if m.sma_50 else 0.0
        rsi_score   = math.tanh((m.rsi - 55) / 15)
        return 0.40*vol_score + 0.35*price_vs_ma + 0.25*rsi_score

    def _build_reasoning(self, m, sig, score, rnd, peers):
        vol_ratio = m.volume / max(m.avg_volume, 1)
        above_ma  = m.price > m.sma_50 if m.sma_50 else False
        return (f"Volume {vol_ratio:.1f}x average {'🔥 BREAKOUT' if vol_ratio>1.5 else ''}. "
                f"Price {'above' if above_ma else 'below'} 50-DMA. "
                f"RSI={m.rsi:.1f}. → {sig.label()}")


# ═══════════════════════════════════════════════════════════════════════════
# SECTION D : AGENT FACTORY  — creates N agents from 10 archetypes
# ═══════════════════════════════════════════════════════════════════════════

AGENT_CLASSES = [
    MomentumAgent, ContrarianAgent, FundamentalistAgent, QuantAgent,
    SentimentAgent, RiskManagerAgent, TechnicalAnalystAgent, MacroAgent,
    ValueTrapDetectorAgent, BreakoutHunterAgent,
]

AGENT_NAMES = [
    "Arjun","Priya","Rahul","Kavya","Vikram","Sneha","Aditya","Meera",
    "Rohan","Ananya","Karthik","Deepa","Nikhil","Pooja","Siddharth",
    "Divya","Manish","Riya","Vivek","Nandini","Harsh","Tanya","Gaurav",
    "Ishaan","Shreya","Dhruv","Simran","Akash","Swati","Rajeev",
]

def create_agents(n: int) -> List[BaseAgent]:
    """Create n agents by cycling through 10 archetypes."""
    agents = []
    for i in range(n):
        cls  = AGENT_CLASSES[i % len(AGENT_CLASSES)]
        name = AGENT_NAMES[i % len(AGENT_NAMES)]
        # avoid duplicate name+archetype
        suffix = f" {i//len(AGENT_NAMES)+1}" if i >= len(AGENT_NAMES) else ""
        agents.append(cls(name + suffix))
    return agents


# ═══════════════════════════════════════════════════════════════════════════
# SECTION E : LIVE MARKET DATA  (yfinance — NSE/BSE)
# ═══════════════════════════════════════════════════════════════════════════

def compute_rsi(closes: list, period: int = 14) -> float:
    if len(closes) < period + 1:
        return 50.0
    gains, losses = [], []
    for i in range(1, period + 1):
        d = closes[-i] - closes[-i-1]
        (gains if d > 0 else losses).append(abs(d))
    avg_g = sum(gains)  / len(gains)  if gains  else 0
    avg_l = sum(losses) / len(losses) if losses else 1e-9
    rs    = avg_g / avg_l
    return 100 - (100 / (1 + rs))

def compute_macd(closes: list) -> tuple:
    def ema(data, span):
        k, e = 2/(span+1), data[0]
        for v in data[1:]:
            e = v*k + e*(1-k)
        return e
    if len(closes) < 26:
        return 0.0, 0.0
    ema12 = ema(closes[-26:], 12)
    ema26 = ema(closes[-26:], 26)
    macd  = ema12 - ema26
    # signal = EMA(9) of MACD — approximate
    signal = macd * 0.95
    return round(macd, 4), round(signal, 4)

def sma(closes: list, n: int) -> float:
    if len(closes) < n:
        return closes[-1] if closes else 0
    return sum(closes[-n:]) / n

def fetch_market_data(ticker: str) -> Optional[MarketSnapshot]:
    """Fetch live NSE/BSE data via yfinance."""
    print(f"\n{C}📡 Fetching live data for {ticker}...{RS}")
    try:
        tk   = yf.Ticker(ticker)
        info = tk.info
        hist = tk.history(period="1y")

        if hist.empty or not info:
            print(f"{R}❌ Could not fetch data for {ticker}. "
                  f"Try adding .NS (NSE) or .BO (BSE) suffix.{RS}")
            return None

        closes = list(hist["Close"])
        vols   = list(hist["Volume"])

        rsi         = compute_rsi(closes)
        macd_val, macd_sig = compute_macd(closes)
        sma20       = sma(closes, 20)
        sma50       = sma(closes, 50)
        sma200      = sma(closes, 200)

        price       = float(info.get("currentPrice", closes[-1]))
        prev_close  = float(info.get("previousClose", closes[-2] if len(closes)>1 else price))
        pe          = float(info.get("trailingPE", 0) or 0)
        pb          = float(info.get("priceToBook", 0) or 0)
        eps         = float(info.get("trailingEps", 0) or 0)
        eps_fwd     = float(info.get("forwardEps", eps) or eps)
        eps_growth  = ((eps_fwd - eps) / abs(eps) * 100) if eps and eps != 0 else 0
        div_yield   = float(info.get("dividendYield", 0) or 0) * 100
        beta        = float(info.get("beta", 1.0) or 1.0)
        mktcap      = float(info.get("marketCap", 0) or 0) / 1e7   # → Crores INR
        sector      = info.get("sector", "Unknown")
        company     = info.get("longName", ticker)
        avg_vol     = int(info.get("averageVolume", 1) or 1)
        d_high      = float(info.get("dayHigh", price) or price)
        d_low       = float(info.get("dayLow",  price) or price)
        w52h        = float(info.get("fiftyTwoWeekHigh", price) or price)
        w52l        = float(info.get("fiftyTwoWeekLow",  price) or price)

        # sector trend approximation from 30-day return
        sector_ret  = (closes[-1] - closes[-22]) / max(closes[-22], 1) if len(closes)>22 else 0
        sector_trend= math.tanh(sector_ret * 5)

        # default news sentiment from price momentum
        price_mom   = (closes[-1] - closes[-5]) / max(closes[-5], 1) if len(closes)>5 else 0
        news_sent   = math.tanh(price_mom * 10)

        snap = MarketSnapshot(
            ticker        = ticker,
            company_name  = company,
            price         = price,
            prev_close    = prev_close,
            day_high      = d_high,
            day_low       = d_low,
            week52_high   = w52h,
            week52_low    = w52l,
            volume        = int(vols[-1]) if vols else 0,
            avg_volume    = avg_vol,
            market_cap    = mktcap,
            pe_ratio      = pe,
            pb_ratio      = pb,
            eps           = eps,
            eps_growth    = eps_growth,
            dividend_yield= div_yield,
            rsi           = rsi,
            macd          = macd_val,
            macd_signal   = macd_sig,
            sma_20        = sma20,
            sma_50        = sma50,
            sma_200       = sma200,
            beta          = beta,
            sector        = sector,
            news_sentiment= news_sent,
            news_headlines= [],
            sector_trend  = sector_trend,
            url_headlines = [],
        )
        print(f"{G}✅ Data fetched: {company} | ₹{price:.2f}{RS}")
        return snap

    except Exception as e:
        print(f"{R}❌ Error fetching {ticker}: {e}{RS}")
        return None


# ═══════════════════════════════════════════════════════════════════════════
# SECTION F : URL SCRAPER  — extracts headlines + sentiment
# ═══════════════════════════════════════════════════════════════════════════

def scrape_url(url: str) -> tuple:
    """Returns (headlines: list[str], avg_sentiment: float)"""
    if not url or url.strip().lower() in ("", "none", "skip", "no"):
        return [], 0.0
    print(f"\n{C}🌐 Scraping URL: {url}{RS}")
    try:
        headers = {"User-Agent": "Mozilla/5.0 (compatible; SwarmBot/1.0)"}
        resp    = requests.get(url.strip(), headers=headers, timeout=12)
        soup    = BeautifulSoup(resp.text, "html.parser")

        # grab all visible text chunks that look like headlines / paragraphs
        candidates = []
        for tag in soup.find_all(["h1","h2","h3","p","li"]):
            t = tag.get_text(strip=True)
            if 20 < len(t) < 300:
                candidates.append(t)

        headlines = candidates[:20]   # cap at 20
        if not headlines:
            print(f"{Y}⚠️  No readable text found at URL.{RS}")
            return [], 0.0

        scores    = [TextBlob(h).sentiment.polarity for h in headlines]
        avg_score = sum(scores) / len(scores)

        print(f"{G}✅ Scraped {len(headlines)} snippets. "
              f"Avg sentiment: {avg_score:+.3f}{RS}")
        # show top 3
        for i, h in enumerate(headlines[:3], 1):
            print(f"   {i}. {h[:100]}")
        return headlines, avg_score

    except Exception as e:
        print(f"{R}❌ Could not scrape URL: {e}{RS}")
        return [], 0.0


# ═══════════════════════════════════════════════════════════════════════════
# SECTION G : SWARM ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════════

class SwarmOrchestrator:
    def __init__(self, agents: List[BaseAgent], rounds: int = 4):
        self.agents = agents
        self.rounds = rounds
        self.all_rounds_log: List[Dict] = []

    def run(self, market: MarketSnapshot) -> Dict:
        self._print_market_card(market)
        print(f"\n{BR}{C}👥 Swarm size: {len(self.agents)} agents  |  "
              f"Debate rounds: {self.rounds}{RS}")

        prev_signals = {a.name: Signal.HOLD for a in self.agents}

        for rnd in range(1, self.rounds + 1):
            print(f"\n{BR}{'═'*68}")
            print(f"  🔁  ROUND {rnd}/{self.rounds}  —  AGENT DELIBERATIONS")
            print(f"{'═'*68}{RS}")

            round_data = {"round": rnd, "votes": []}
            peer_list  = list(prev_signals.values())

            for agent in self.agents:
                sig, conf, reason = agent.deliberate(market, peer_list, rnd)
                prev_signals[agent.name] = sig

                bar   = "█" * int(conf * 24)
                pad   = "░" * (24 - int(conf * 24))
                trend = agent.memory.conviction_trend()

                print(f"\n  {sig.color()}{BR}👤 {agent.name:<18}{RS} "
                      f"[{agent.archetype:<22}]")
                print(f"     Signal     : {sig.color()}{BR}{sig.emoji()} {sig.label()}{RS}  {trend}")
                print(f"     Confidence : {G}[{bar}{Style.DIM}{pad}]{RS} {conf*100:.0f}%")
                print(f"     {Y}Reasoning  :{RS} "
                      + textwrap.fill(reason, width=72,
                                      subsequent_indent="                 "))

                round_data["votes"].append({
                    "agent":     agent.name,
                    "archetype": agent.archetype,
                    "signal":    sig,
                    "confidence":conf,
                    "reasoning": reason,
                })

            # mid-round snapshot
            interim = self._aggregate(round_data["votes"])
            print(f"\n  {Y}── Round {rnd} interim: "
                  f"{interim['signal'].emoji()} {interim['signal'].label()} "
                  f"(score={interim['weighted_score']:+.3f}) ──{RS}")
            self.all_rounds_log.append(round_data)

        # ── final consensus ──
        consensus = self._aggregate(self.all_rounds_log[-1]["votes"])
        self._print_final_report(consensus, market)
        return consensus

    def _aggregate(self, votes: List[Dict]) -> Dict:
        weighted_sum, total_w = 0.0, 0.0
        tally = defaultdict(float)
        for v in votes:
            w             = v["confidence"]
            weighted_sum += v["signal"].value * w
            total_w      += w
            tally[v["signal"].name] += w

        raw = weighted_sum / max(total_w, 1e-9)
        if   raw >= 1.3:  final = Signal.STRONG_BUY
        elif raw >= 0.40: final = Signal.BUY
        elif raw <= -1.3: final = Signal.STRONG_SELL
        elif raw <= -0.40:final = Signal.SELL
        else:             final = Signal.HOLD

        breakdown = {k: v/total_w*100 for k, v in tally.items()}
        return {
            "signal":         final,
            "weighted_score": raw,
            "confidence":     min(1.0, abs(raw)/2.0 + 0.1),
            "vote_breakdown": breakdown,
            "total_agents":   len(votes),
        }

    def _print_final_report(self, c: Dict, m: MarketSnapshot):
        sig   = c["signal"]
        score = c["weighted_score"]
        conf  = c["confidence"]
        pct   = conf * 100

        # price targets (rough heuristic)
        chg_map  = {2: 0.12, 1: 0.06, 0: 0.0, -1: -0.06, -2: -0.12}
        expected = m.price * (1 + chg_map[sig.value])
        stop_loss= m.price * (1 - 0.05 if sig.value >= 0 else 1 + 0.05)

        print(f"\n\n{BR}{'╔'+'═'*66+'╗'}")
        print(f"║{'🧠  SWARM CONSENSUS  —  FINAL PREDICTION':^66}║")
        print(f"╠{'═'*66}╣")
        print(f"║  Company : {m.company_name:<53}  ║")
        print(f"║  Ticker  : {m.ticker:<53}  ║")
        print(f"║  Price   : ₹{m.price:<52.2f}  ║")
        print(f"╠{'═'*66}╣")
        print(f"║  {BR}SIGNAL    : {sig.color()}{sig.emoji()}  {sig.label():<52}{RS}{BR}  ║{RS}")
        print(f"║  Swarm Score   : {score:>+8.4f}  |  Conviction: {pct:.0f}%{' '*25}║")
        print(f"║  {'█'*int(pct//4):<25} {pct:.0f}% conviction{' '*14}║")
        print(f"╠{'═'*66}╣")
        print(f"║  📊 Price Estimate  : ₹{expected:<41.2f}  ║")
        print(f"║  🛡️  Stop Loss Level : ₹{stop_loss:<41.2f}  ║")
        print(f"╠{'═'*66}╣")
        print(f"║  Vote Breakdown ({c['total_agents']} agents):{' '*37}║")

        for name, pct_v in sorted(c["vote_breakdown"].items(), key=lambda x: -x[1]):
            bar = "▓" * int(pct_v / 4)
            pad = "░" * (25 - int(pct_v / 4))
            print(f"║    {name:<14} [{bar}{pad}] {pct_v:>5.1f}%{' '*10}║")

        print(f"╠{'═'*66}╣")
        # Action advice
        action_map = {
            Signal.STRONG_BUY:  "✅ STRONG BUY  — High conviction. Consider accumulating.",
            Signal.BUY:         "✅ BUY         — Positive outlook. Gradual entry advised.",
            Signal.HOLD:        "⚠️  HOLD        — Mixed signals. Monitor closely.",
            Signal.SELL:        "❌ SELL        — Caution warranted. Consider reducing.",
            Signal.STRONG_SELL: "❌ STRONG SELL — Exit position. High downside risk.",
        }
        print(f"║  {action_map[sig]:<64}  ║")
        print(f"╚{'═'*66}╝{RS}\n")

        # disclaimer
        print(f"{Y}⚠️  DISCLAIMER: This is a simulation for educational purposes only.\n"
              f"   Not financial advice. Always consult a SEBI-registered advisor.{RS}\n")

    def _print_market_card(self, m: MarketSnapshot):
        chg     = m.price - m.prev_close
        chg_pct = chg / max(m.prev_close, 1) * 100
        chg_col = G if chg >= 0 else R
        rows = [
            ["Company",   m.company_name],
            ["Ticker",    m.ticker],
            ["Price",     f"₹{m.price:.2f}  {chg_col}({chg:+.2f} / {chg_pct:+.2f}%){RS}"],
            ["52W High",  f"₹{m.week52_high:.2f}"],
            ["52W Low",   f"₹{m.week52_low:.2f}"],
            ["P/E",       f"{m.pe_ratio:.2f}"],
            ["P/B",       f"{m.pb_ratio:.2f}"],
            ["EPS Growth",f"{m.eps_growth:+.2f}%"],
            ["RSI(14)",   f"{m.rsi:.2f}"],
            ["MACD",      f"{m.macd:+.4f}"],
            ["Beta",      f"{m.beta:.2f}"],
            ["Mkt Cap",   f"₹{m.market_cap:,.0f} Cr"],
            ["Sector",    m.sector],
            ["Div Yield", f"{m.dividend_yield:.2f}%"],
        ]
        print(f"\n{BR}{C}{'═'*50}")
        print(f"  📈  LIVE MARKET SNAPSHOT")
        print(f"{'═'*50}{RS}")
        print(tabulate(rows, tablefmt="simple"))
        print()


# ═══════════════════════════════════════════════════════════════════════════
# SECTION H : POPULAR INDIAN TICKERS (help text)
# ═══════════════════════════════════════════════════════════════════════════

POPULAR_NSE = {
    "RELIANCE.NS": "Reliance Industries",
    "TCS.NS":      "Tata Consultancy Services",
    "INFY.NS":     "Infosys",
    "HDFCBANK.NS": "HDFC Bank",
    "ICICIBANK.NS":"ICICI Bank",
    "SBIN.NS":     "State Bank of India",
    "WIPRO.NS":    "Wipro",
    "TATAMOTORS.NS":"Tata Motors",
    "ADANIENT.NS": "Adani Enterprises",
    "BAJFINANCE.NS":"Bajaj Finance",
    "NIFTY50":     "Nifty 50 Index (^NSEI)",
    "SENSEX":      "BSE Sensex (^BSESN)",
}


# ═══════════════════════════════════════════════════════════════════════════
# SECTION I : MAIN  —  interactive Colab entry point
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print(f"\n{BR}{M}{'╔'+'═'*62+'╗'}")
    print(f"║{'🇮🇳  SWARM INTELLIGENCE  —  INDIAN STOCK PREDICTOR':^62}║")
    print(f"║{'Powered by Multi-Agent Debate · Live NSE/BSE Data':^62}║")
    print(f"{'╚'+'═'*62+'╝'}{RS}\n")

    # ── popular tickers hint ──
    print(f"{C}Popular Indian tickers you can use:{RS}")
    for ticker, name in POPULAR_NSE.items():
        print(f"  {Y}{ticker:<20}{RS} {name}")

    # ── user inputs ──
    print()
    while True:
        try:
            n_agents = int(input(f"\n{BR}👥 How many agents do you want in the swarm? "
                                 f"(min 3, max 30): {RS}").strip())
            if 3 <= n_agents <= 30:
                break
            print(f"{R}Please enter a number between 3 and 30.{RS}")
        except ValueError:
            print(f"{R}Please enter a valid integer.{RS}")

    ticker = input(f"\n{BR}📈 Enter Indian stock ticker "
                   f"(e.g. RELIANCE.NS / TCS.NS / INFY.NS): {RS}").strip().upper()
    if not ticker:
        ticker = "TCS.NS"

    # handle common aliases
    if ticker == "NIFTY50":   ticker = "^NSEI"
    if ticker == "SENSEX":    ticker = "^BSESN"

    url = input(f"\n{BR}🌐 Paste a news URL for live sentiment "
                f"(or press Enter to skip): {RS}").strip()

    while True:
        try:
            rounds = int(input(f"\n{BR}🔁 How many debate rounds? (1–6, default 3): {RS}").strip() or "3")
            if 1 <= rounds <= 6:
                break
            print(f"{R}Enter between 1 and 6.{RS}")
        except ValueError:
            rounds = 3
            break

    # ── fetch data ──
    market = fetch_market_data(ticker)
    if market is None:
        print(f"{R}Exiting — could not fetch market data. "
              f"Check ticker symbol and internet connection.{RS}")
        return

    # ── scrape URL if provided ──
    if url:
        headlines, url_sent = scrape_url(url)
        market.url_headlines   = headlines
        # blend URL sentiment into news_sentiment
        if headlines:
            market.news_sentiment = (0.40 * market.news_sentiment + 0.60 * url_sent)

    # ── build swarm & run ──
    agents = create_agents(n_agents)
    print(f"\n{G}✅ Created {len(agents)} agents:{RS}")
    for a in agents:
        print(f"   {Y}• {a.name:<18}{RS} → {a.archetype}")

    swarm  = SwarmOrchestrator(agents, rounds=rounds)
    result = swarm.run(market)

    # ── post-simulation: round-by-round score evolution ──
    print(f"\n{C}{BR}📊 SCORE EVOLUTION ACROSS ROUNDS:{RS}")
    headers = ["Round"] + [f"R{i+1}" for i in range(rounds)]
    rows    = []
    for rnd_data in swarm.all_rounds_log:
        interim = swarm._aggregate(rnd_data["votes"])
        rows.append([f"Round {rnd_data['round']}",
                     f"{interim['signal'].emoji()} {interim['signal'].label()}",
                     f"{interim['weighted_score']:+.3f}",
                     f"{interim['confidence']*100:.0f}%"])
    print(tabulate(rows,
                   headers=["Round","Signal","Score","Conviction"],
                   tablefmt="rounded_outline"))

    print(f"\n{G}✅ Simulation complete!{RS}")


if __name__ == "__main__":
    main()
