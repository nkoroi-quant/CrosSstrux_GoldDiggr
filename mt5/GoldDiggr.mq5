//+------------------------------------------------------------------+
//|         GoldDiggr_MTF_Optimized_v11.3.mq5                       |
//|   H1 + M15 context | M5 + M1 entry precision                      |
//|   Cached server calls + richer management + pyramiding           |
//+------------------------------------------------------------------+
#property copyright "nkoroi-quant + integration"
#property link      "https://github.com/nkoroi-quant/CrosSstrux_GoldDiggr"
#property version   "12.0"
#property strict

#include <Trade/Trade.mqh>

CTrade trade;

// ========================= INPUTS =========================
input string API_URL                = "http://127.0.0.1:8000/analyze";
input string API_KEY                = "";
input string AssetName              = "XAUUSD";
input long   MagicNumber            = 260325;

input double BaseLot                = 0.01;
input double RiskPercentPerTrade    = 0.75;
input double MaxRiskPercentPerTrade = 2.00;
input double MaxLotCap              = 0.20;

input int    ATR_Period             = 14;
input double ATR_Multiplier         = 2.0;

input double MinProbability         = 58.0;
input double StrongProb             = 65.0;
input double MaxCDI                 = 0.60;

input bool   AllowOffHours          = false;
input int    SignalConfirmations    = 2;
input int    MaxPositions           = 3;
input int    MinBarsBetweenEntries  = 2;
input int    MaxSpreadPoints        = 350;
input int    HTTPTimeoutMs          = 10000;
input bool   EnableHttpDiagnostics  = true;
input int    ContextMaxAgeMins      = 5;
input int    HealthProbeSeconds     = 30;
input int    ContextRefreshCooldownSeconds = 10;
input double M1PrecisionThresholdBase     = 0.55;
input double M1PrecisionThresholdStrong    = 0.50;
input double M1PrecisionThresholdReversal   = 0.57;

input double TP1_RR                 = 1.0;
input double TrailStartRR           = 1.2;
input double BiasTrailATRWeak       = 0.70;
input double BiasTrailATRStrong     = 1.00;
input double TP2_ExtendATR          = 3.00;
input int    MinModifyPoints        = 30;

input bool   EnableScaleIn          = true;
input bool   EnablePyramiding       = true;
input double ScaleInLotFactor       = 0.50;
input double PyramidLotFactor       = 0.60;
input double ScaleInMinProb         = 72.0;
input double PyramidMinProb         = 74.0;
input int    PyramidCooldownBars    = 3;
input double PyramidMinBasketProfit = 0.0;

input double EquitySoftDDLimit      = 0.05;
input double EquityHardDDLimit      = 0.10;

input bool   UsePanel               = true;

// NEW efficiency inputs from v11.5
input bool   UseRichResponse        = false;      // default false for bandwidth
input int    RequestThrottleSeconds = 5;          // prevent request spam

// ========================= CONSTANTS =========================
#define CONTEXT_WINDOW 30
#define ENTRY_WINDOW    30
#define EQUITY_HISTORY_SIZE 32

// ========================= STRUCTS =========================
struct MarketFrame
{
   string bias;
   string sweep;
   double trend_strength;
   double momentum_points;
   double avg_range_points;
   double close_to_range;
};

struct SignalData
{
   bool   valid;
   string signal;               // BUY / SELL / NONE
   string regime;
   string market_state;
   string active_bias;
   string session;
   string h1_bias;
   string m15_bias;
   string m15_sweep;
   string m5_bias;
   string m5_sweep;
   double probability;
   double signal_confidence;
   double cdi;
   double risk_multiplier;
   double recommended_risk_pct;
   double dynamic_threshold;
   double cooldown_bars;
   double breakeven_at_rr;
   double trail_after_rr;
   double max_hold_bars;
   double max_spread_points;
   double m1_precision_score;
   double context_score;
   datetime context_bar_time;
};

// ========================= GLOBAL STATE =========================
MqlRates g_m1Rates[];
MqlRates g_m5Rates[];
MqlRates g_m15Rates[];
MqlRates g_h1Rates[];
int g_m1Count = 0;
int g_m5Count = 0;
int g_m15Count = 0;
int g_h1Count = 0;
datetime g_m1BarTime = 0;
datetime g_m5BarTime = 0;
datetime g_m15BarTime = 0;
datetime g_h1BarTime = 0;
datetime g_lastContextM15Bar = 0; // legacy alias retained for compatibility
datetime g_lastContextRefreshTime = 0;
datetime g_lastContextRefreshBar = 0;
datetime g_lastContextAttemptBar = 0;
datetime g_lastContextAttemptTime = 0;
string g_lastContextSource = "NONE";
string g_lastContextResponse = "";
datetime g_lastEntryM1Bar = 0;
datetime g_lastM1HeartbeatBar = 0;
datetime g_lastTradeM1Bar = 0;
datetime g_lastHealthProbe = 0;
bool g_lastHealthOk = false;
int g_consecutiveHttpFails = 0;
datetime g_lastDecisionTraceBar = 0;
datetime g_lastRequestTime = 0;

SignalData g_cachedSignal;
bool g_contextReady = false;

string g_confirmSignalName = "";
int    g_confirmSignalCount = 0;

string recentSignals[5];
int recentSignalCount = 0;

ulong partialClosedTickets[128];

int atrHandle = INVALID_HANDLE;
string PanelPrefix = "GoldDiggr_";
long   panelChartID = 0;
bool   panelCreated = false;

// Equity tracking
double equityHistory[EQUITY_HISTORY_SIZE];
int    equityHistoryCount = 0;
double equityPeak = 0.0;

// Performance tracking
int    totalClosedEvents = 0;
int    fullClosedTrades  = 0;
int    partialExitEvents = 0;
int    wins = 0, losses = 0;
double grossProfit = 0.0, grossLoss = 0.0;
double fullNetProfit = 0.0, netProfit = 0.0;

// Adaptive knobs
double dynamicMinProb         = 58.0;
double dynamicRiskFactor      = 1.0;
double dynamicOffHoursMinProb = 65.0;
double dynamicTrailATRWeak    = 0.70;
double dynamicTrailATRStrong  = 1.00;
double dynamicTP2ExtendATR    = 3.00;
double dynamicScaleInMinProb  = 72.0;
double dynamicPyramidMinProb  = 74.0;
double dynamicTP1_RR          = 1.0;
double dynamicTrailStartRR    = 1.2;

// ========================= LOG =========================
void Log(string msg) { Print("[GoldDiggr] ", msg); }

void LogM1Heartbeat()
{
   if(g_m1BarTime == 0) return;
   if(g_lastM1HeartbeatBar == g_m1BarTime) return;
   g_lastM1HeartbeatBar = g_m1BarTime;

   double pt = SymbolPoint();
   double spreadPts = (SymbolInfoDouble(_Symbol, SYMBOL_ASK) - SymbolInfoDouble(_Symbol, SYMBOL_BID)) / pt;
   MqlRates c = g_m1Rates[0];
   double rangePts = (c.high - c.low) / pt;
   double bodyPts  = MathAbs(c.close - c.open) / pt;
   double upperWickPts = (c.high - MathMax(c.open, c.close)) / pt;
   double lowerWickPts = (MathMin(c.open, c.close) - c.low) / pt;
   double buyPrec = GetM1PrecisionScore("BUY");
   double sellPrec = GetM1PrecisionScore("SELL");
   string cachedSignal = (g_contextReady ? g_cachedSignal.signal : "NONE");
   string cachedRegime = (g_contextReady ? g_cachedSignal.regime : "N/A");
   string cachedState  = (g_contextReady ? g_cachedSignal.market_state : "N/A");
   string cachedH1     = (g_contextReady ? g_cachedSignal.h1_bias : "N/A");
   string cachedM15    = (g_contextReady ? g_cachedSignal.m15_bias : "N/A");
   string cachedM5     = (g_contextReady ? g_cachedSignal.m5_bias : "N/A");
   double cachedConf   = (g_contextReady ? g_cachedSignal.signal_confidence : 0.0);
   double cachedProb   = (g_contextReady ? g_cachedSignal.probability : 0.0);
   int ctxAgeMins   = (g_contextReady && g_lastContextRefreshTime > 0) ? (int)((TimeCurrent() - g_lastContextRefreshTime) / 60) : -1;
   string ctxSource   = (g_contextReady ? g_lastContextSource : "NONE");

   Log(StringFormat(
      "M1 close @%s O=%.2f H=%.2f L=%.2f C=%.2f range=%.1f body=%.1f uw=%.1f lw=%.1f spread=%.1f pos=%d lastEntryBars=%d ctx=%s src=%s conf=%.2f prob=%.2f regime=%s state=%s h1=%s m15=%s m5=%s buyM1=%.2f sellM1=%.2f ctxAgeMins=%d",
      TimeToString(g_m1BarTime, TIME_DATE | TIME_MINUTES),
      c.open, c.high, c.low, c.close,
      rangePts, bodyPts, upperWickPts, lowerWickPts,
      spreadPts, CountOurPositions(), BarsSinceLastEntry(),
      cachedSignal, ctxSource, cachedConf, cachedProb, cachedRegime, cachedState, cachedH1, cachedM15, cachedM5,
      buyPrec, sellPrec, ctxAgeMins
   ));
}


// ========================= NEW: IsNewBar (v11.5 upgrade) =========================
bool IsNewBar()
{
   static datetime lastBar = 0;
   datetime currentBar = iTime(_Symbol, PERIOD_M1, 0);
   if(currentBar != lastBar)
   {
      lastBar = currentBar;
      return true;
   }
   return false;
}

string ResolveHealthUrl()
{
   string url = API_URL;
   int p = StringFind(url, "/analyze");
   if(p > 0)
      return StringSubstr(url, 0, p) + "/health";
   p = StringFind(url, "/predict");
   if(p > 0)
      return StringSubstr(url, 0, p) + "/health";
   return url + "/health";
}

string DescribeWebRequestError(int err)
{
   switch(err)
   {
      case 4006: return "transport blocked or no connection";
      case 4014: return "request timed out";
      case 4016: return "invalid request body";
      default:   return "web request failed";
   }
}

double GetAdaptiveM1PrecisionThreshold(string signal, string marketState, double signalConfidence)
{
   double threshold = M1PrecisionThresholdBase;

   if(marketState == "TRENDING" || marketState == "BREAKOUT")
      threshold = MathMin(threshold, M1PrecisionThresholdStrong);
   else if(marketState == "REVERSAL")
      threshold = MathMax(threshold, M1PrecisionThresholdReversal);
   else if(marketState == "TRANSITION")
      threshold = MathMax(threshold, M1PrecisionThresholdBase + 0.01);

   if(signalConfidence >= StrongProb)
      threshold -= 0.02;
   if(signalConfidence >= StrongProb + 5.0)
      threshold -= 0.02;

   return ClampDouble(threshold, 0.48, 0.62);
}

void LogDecisionTrace(string stage, string reason, string signal, double spreadPts, double precisionScore)
{
   if(!EnableHttpDiagnostics) return;
   Log(StringFormat("TRACE stage=%s signal=%s spread=%.1f precision=%.2f ctx=%s reason=%s",
                    stage, signal, spreadPts, precisionScore,
                    (g_contextReady ? g_cachedSignal.market_state : "N/A"), reason));
}

bool ProbeServerHealth(bool force = false)
{
   datetime now = TimeCurrent();
   if(!force && g_lastHealthProbe > 0 && (now - g_lastHealthProbe) < HealthProbeSeconds)
      return g_lastHealthOk;

   g_lastHealthProbe = now;
   string healthUrl = ResolveHealthUrl();
   char empty[];
   ArrayResize(empty, 0);
   char res[];
   string resp_headers = "";
   ResetLastError();
   int status = WebRequest("GET", healthUrl, "", HTTPTimeoutMs, empty, res, resp_headers);
   string response = CharArrayToString(res, 0, -1, CP_UTF8);
   g_lastHealthOk = (status == 200);

   if(EnableHttpDiagnostics)
      Log(StringFormat("HEALTH url=%s HTTP=%d err=%d response=%s", healthUrl, status, GetLastError(), response));

   return g_lastHealthOk;
}

// ========================= SMALL HELPERS =========================
double ClampDouble(double v, double lo, double hi)
{
   if(v < lo) return lo;
   if(v > hi) return hi;
   return v;
}

double SymbolPoint()
{
   double pt = SymbolInfoDouble(_Symbol, SYMBOL_POINT);
   return (pt > 0.0) ? pt : 0.01;
}

string ToUpper(string s)
{
   StringToUpper(s);
   return s;
}

bool IsNewerBar(datetime currentBar, datetime &lastSeen)
{
   if(currentBar <= 0) return false;
   if(currentBar == lastSeen) return false;
   lastSeen = currentBar;
   return true;
}

int CountOurPositions()
{
   int count = 0;
   for(int i = 0; i < PositionsTotal(); i++)
   {
      ulong ticket = PositionGetTicket(i);
      if(PositionSelectByTicket(ticket))
      {
         if(PositionGetString(POSITION_SYMBOL) == _Symbol &&
            PositionGetInteger(POSITION_MAGIC) == MagicNumber)
            count++;
      }
   }
   return count;
}

bool HasOpenPositionType(ENUM_POSITION_TYPE ptype)
{
   for(int i = 0; i < PositionsTotal(); i++)
   {
      ulong ticket = PositionGetTicket(i);
      if(PositionSelectByTicket(ticket))
      {
         if(PositionGetString(POSITION_SYMBOL) == _Symbol &&
            PositionGetInteger(POSITION_MAGIC) == MagicNumber &&
            (ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE) == ptype)
            return true;
      }
   }
   return false;
}

bool HasOppositePosition(ENUM_POSITION_TYPE ptype)
{
   for(int i = 0; i < PositionsTotal(); i++)
   {
      ulong ticket = PositionGetTicket(i);
      if(PositionSelectByTicket(ticket))
      {
         if(PositionGetString(POSITION_SYMBOL) == _Symbol &&
            PositionGetInteger(POSITION_MAGIC) == MagicNumber &&
            (ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE) != ptype)
            return true;
      }
   }
   return false;
}

bool IsPartialClosed(ulong ticket)
{
   for(int i = 0; i < ArraySize(partialClosedTickets); i++)
      if(partialClosedTickets[i] == ticket) return true;
   return false;
}

void MarkPartialClosed(ulong ticket)
{
   for(int i = 0; i < ArraySize(partialClosedTickets); i++)
   {
      if(partialClosedTickets[i] == 0)
      {
         partialClosedTickets[i] = ticket;
         return;
      }
   }
}

void PushRecentSignal(string s)
{
   int size = ArraySize(recentSignals);
   recentSignals[recentSignalCount % size] = s;
   recentSignalCount++;
}

string BuildSignalHistoryJson()
{
   int size = ArraySize(recentSignals);
   int count = MathMin(recentSignalCount, size);
   string json = "[";
   bool first = true;
   for(int i = 0; i < count; i++)
   {
      int idx = (recentSignalCount - count + i) % size;
      if(recentSignals[idx] == "") continue;
      if(!first) json += ",";
      json += "\"" + recentSignals[idx] + "\"";
      first = false;
   }
   json += "]";
   return json;
}

int BarsSinceLastEntry()
{
   if(g_lastTradeM1Bar == 0 || g_m1BarTime == 0) return 999;
   int secs = (int)(g_m1BarTime - g_lastTradeM1Bar);
   if(secs < 0) return 999;
   return secs / 60;
}

// ========================= MARKET DATA =========================
bool RefreshBuffers()
{
   ArraySetAsSeries(g_m1Rates, true);
   ArraySetAsSeries(g_m5Rates, true);
   ArraySetAsSeries(g_m15Rates, true);
   ArraySetAsSeries(g_h1Rates, true);

   g_m1Count = CopyRates(_Symbol, PERIOD_M1, 1, ENTRY_WINDOW, g_m1Rates);
   g_m5Count = CopyRates(_Symbol, PERIOD_M5, 1, CONTEXT_WINDOW, g_m5Rates);
   g_m15Count = CopyRates(_Symbol, PERIOD_M15, 1, CONTEXT_WINDOW, g_m15Rates);
   g_h1Count = CopyRates(_Symbol, PERIOD_H1, 1, CONTEXT_WINDOW, g_h1Rates);

   if(g_m1Count > 0) g_m1BarTime = g_m1Rates[0].time;
   if(g_m5Count > 0) g_m5BarTime = g_m5Rates[0].time;
   if(g_m15Count > 0) g_m15BarTime = g_m15Rates[0].time;
   if(g_h1Count > 0) g_h1BarTime = g_h1Rates[0].time;

   return (g_m1Count >= 10 && g_m5Count >= 10 && g_m15Count >= 10 && g_h1Count >= 10);
}

double GetATRValue()
{
   if(atrHandle == INVALID_HANDLE) return 0.0;
   double atrBuf[];
   ArraySetAsSeries(atrBuf, true);
   if(CopyBuffer(atrHandle, 0, 1, 1, atrBuf) < 1) return 0.0;
   return atrBuf[0];
}

void UpdateEquityHistory()
{
   double eq = AccountInfoDouble(ACCOUNT_EQUITY);
   if(eq <= 0.0) return;
   if(eq > equityPeak) equityPeak = eq;
   equityHistory[equityHistoryCount % EQUITY_HISTORY_SIZE] = eq;
   equityHistoryCount++;
}

double GetEquityCurveSlope()
{
   int n = MathMin(equityHistoryCount, EQUITY_HISTORY_SIZE);
   if(n < 6) return 0.0;
   int newest = (equityHistoryCount - 1) % EQUITY_HISTORY_SIZE;
   int older  = (equityHistoryCount - 6) % EQUITY_HISTORY_SIZE;
   double oldv = equityHistory[older];
   if(oldv <= 0.0) return 0.0;
   return (equityHistory[newest] - oldv) / oldv;
}

double GetEquityCurveFactor()
{
   double eq = AccountInfoDouble(ACCOUNT_EQUITY);
   if(eq <= 0.0) return 1.0;
   if(eq > equityPeak) equityPeak = eq;

   double dd = (equityPeak - eq) / equityPeak;
   double slope = GetEquityCurveSlope();
   double factor = 1.0;

   if(dd > EquityHardDDLimit || slope < -0.03) factor = 0.40;
   else if(dd > 0.08 || slope < -0.02) factor = 0.60;
   else if(dd > EquitySoftDDLimit || slope < -0.01) factor = 0.80;
   else if(dd < 0.01 && slope > 0.01) factor = 1.10;

   return ClampDouble(factor, 0.40, 1.10);
}

double GetVolatilityFactor(double atrPoints, double avgRangePoints)
{
   if(atrPoints <= 0.0) return 1.0;
   if(avgRangePoints <= 0.0) avgRangePoints = atrPoints;

   double ratio = atrPoints / avgRangePoints;
   double factor = 1.0;

   if(ratio < 0.70) factor = 0.90;
   else if(ratio < 0.95) factor = 1.05;
   else if(ratio < 1.25) factor = 1.15;
   else if(ratio < 1.60) factor = 0.95;
   else factor = 0.80;

   if(atrPoints < 60.0) factor *= 0.85;
   else if(atrPoints > 250.0) factor *= 0.80;

   return ClampDouble(factor, 0.60, 1.25);
}

// ========================= FRAME ANALYSIS =========================
void AnalyzeFrame(MqlRates &rates[], int count, MarketFrame &frame)
{
   frame.bias = "NEUTRAL";
   frame.sweep = "NONE";
   frame.trend_strength = 0.0;
   frame.momentum_points = 0.0;
   frame.avg_range_points = 0.0;
   frame.close_to_range = 0.5;

   if(count < 5) return;

   double pt = SymbolPoint();
   double firstClose = rates[count - 1].close;
   double lastClose = rates[0].close;
   double netMove = (lastClose - firstClose) / pt;
   double sumRange = 0.0;
   int up = 0, down = 0;

   for(int i = 0; i < count; i++)
   {
      sumRange += (rates[i].high - rates[i].low) / pt;
      if(i < count - 1)
      {
         if(rates[i].close > rates[i + 1].close) up++;
         else if(rates[i].close < rates[i + 1].close) down++;
      }
   }

   frame.avg_range_points = sumRange / count;
   frame.momentum_points = (rates[0].close - rates[MathMin(5, count - 1)].close) / pt;
   frame.trend_strength = (double)(up - down) / MathMax(count - 1, 1);

   double upperWick = rates[0].high - MathMax(rates[0].open, rates[0].close);
   double lowerWick = MathMin(rates[0].open, rates[0].close) - rates[0].low;
   double totalRange = MathMax(rates[0].high - rates[0].low, pt);
   frame.close_to_range = (rates[0].close - rates[0].low) / totalRange;

   if(rates[0].high > rates[1].high && rates[0].low > rates[1].low && netMove > 0)
      frame.bias = "UP";
   else if(rates[0].high < rates[1].high && rates[0].low < rates[1].low && netMove < 0)
      frame.bias = "DOWN";
   else if(MathAbs(netMove) > 15.0)
      frame.bias = (netMove > 0 ? "UP" : "DOWN");

   double prevHigh = rates[1].high;
   double prevLow = rates[1].low;
   for(int i = 2; i < MathMin(count, 6); i++)
   {
      if(rates[i].high > prevHigh) prevHigh = rates[i].high;
      if(rates[i].low < prevLow) prevLow = rates[i].low;
   }

   double sweepBuffer = 10.0 * pt;
   if(rates[0].low < (prevLow - sweepBuffer) && rates[0].close > prevLow)
      frame.sweep = "BULL";
   else if(rates[0].high > (prevHigh + sweepBuffer) && rates[0].close < prevHigh)
      frame.sweep = "BEAR";

   // Wick bias helps classify candle intent.
   if(frame.bias == "NEUTRAL")
   {
      if(rates[0].close > rates[0].open && lowerWick > upperWick)
         frame.bias = "UP";
      else if(rates[0].close < rates[0].open && upperWick > lowerWick)
         frame.bias = "DOWN";
   }

}

string GetCombinedBias(MarketFrame &h1, MarketFrame &m5)
{
   if(h1.bias == m5.bias && h1.bias != "NEUTRAL") return h1.bias;
   if(h1.bias != "NEUTRAL" && m5.bias == "NEUTRAL") return h1.bias;
   if(m5.bias != "NEUTRAL" && h1.bias == "NEUTRAL") return m5.bias;
   return "NEUTRAL";
}

string BuildMarketState(MarketFrame &h1, MarketFrame &m5, double cdi)
{
   if(m5.sweep != "NONE" && MathAbs(cdi) > 0.30) return "REVERSAL";
   if(h1.bias == m5.bias && h1.bias != "NEUTRAL")
   {
      if(MathAbs(m5.trend_strength) > 0.25 || MathAbs(m5.momentum_points) > 10.0)
         return "TRENDING";
      return "TRANSITION";
   }
   if(MathAbs(cdi) > 0.35) return "BREAKOUT";
   return "TRANSITION";
}

string BuildLiquidityState(MarketFrame &m5)
{
   if(m5.sweep != "NONE") return "SWEEPED";
   if(MathAbs(m5.trend_strength) > 0.30) return "CLEAN";
   return "UNCLEAR";
}

string BuildVolatilityState(double volExpansion)
{
   if(volExpansion >= 1.35) return "HIGH";
   if(volExpansion >= 1.10) return "EXPANDING";
   if(volExpansion <= 0.85) return "LOW";
   return "NORMAL";
}

// ========================= M1 ENTRY PRECISION =========================
double GetM1PrecisionScore(string signal)
{
   if(g_m1Count < 3 || g_m5Count < 5) return 0.0;
   double pt = SymbolPoint();
   MarketFrame m5;
   MarketFrame m1;
   AnalyzeFrame(g_m5Rates, g_m5Count, m5);
   AnalyzeFrame(g_m1Rates, g_m1Count, m1);
   double score = 0.0;

   MqlRates c0 = g_m1Rates[0];
   MqlRates c1 = g_m1Rates[1];
   MqlRates c2 = g_m1Rates[2];

   bool bullishEngulf = (c0.close > c0.open && c1.close < c1.open && c0.close > c1.open && c0.open <= c1.close);
   bool bearishEngulf = (c0.close < c0.open && c1.close > c1.open && c0.close < c1.open && c0.open >= c1.close);
   bool microBreakBuy = (c0.close > c1.high && c0.close > c0.open);
   bool microBreakSell = (c0.close < c1.low && c0.close < c0.open);

   double body = MathAbs(c0.close - c0.open) / MathMax(c0.high - c0.low, pt);
   double momentum = (c0.close - c2.close) / pt;

   if(signal == "BUY")
   {
      if(bullishEngulf) score += 0.35;
      if(microBreakBuy) score += 0.25;
      if(c0.close > c1.high) score += 0.10;
      if(m1.bias == "UP") score += 0.10;
      if(m5.bias == "UP") score += 0.15;
      if(m5.sweep == "BULL") score += 0.10;
      if(m5.trend_strength > 0.10) score += 0.10;
      if(momentum > 4.0) score += 0.10;
      score += MathMin(body, 0.20);
   }
   else if(signal == "SELL")
   {
      if(bearishEngulf) score += 0.35;
      if(microBreakSell) score += 0.25;
      if(c0.close < c1.low) score += 0.10;
      if(m1.bias == "DOWN") score += 0.10;
      if(m5.bias == "DOWN") score += 0.15;
      if(m5.sweep == "BEAR") score += 0.10;
      if(m5.trend_strength < -0.10) score += 0.10;
      if(momentum < -4.0) score += 0.10;
      score += MathMin(body, 0.20);
   }

   return ClampDouble(score, 0.0, 1.5);
}

bool M1EntryTrigger(string signal, double signalConfidence, string marketState, double &precisionScore)
{
   precisionScore = GetM1PrecisionScore(signal);
   double threshold = GetAdaptiveM1PrecisionThreshold(signal, marketState, signalConfidence);
   return (signal == "BUY" || signal == "SELL") && precisionScore >= threshold;
}

// ========================= SERVER REQUEST =========================
double GetM15Cdi()
{
   if(g_m15Count < 5) return 0.0;
   double mean = 0.0;
   for(int i = 0; i < MathMin(g_m15Count, 20); i++)
      mean += g_m15Rates[i].close;
   mean /= MathMin(g_m15Count, 20);
   double atr = GetATRValue();
   if(atr <= 0.0) atr = SymbolPoint() * 100.0;
   return (g_m15Rates[0].close - mean) / atr;
}

double GetM5Cdi()
{
   return GetM15Cdi();
}

string BuildRatesJson(MqlRates &rates[], int count)
{
   string json = "[";
   for(int i = 0; i < count; i++)
   {
      string candle = StringFormat(
         "{\"time\":\"%s\",\"open\":%.5f,\"high\":%.5f,\"low\":%.5f,\"close\":%.5f}",
         TimeToString(rates[i].time, TIME_DATE | TIME_MINUTES),
         rates[i].open, rates[i].high, rates[i].low, rates[i].close
      );
      json += candle;
      if(i < count - 1) json += ",";
   }
   json += "]";
   return json;
}

string BuildRequestJSON()
{
   MarketFrame h1;
   MarketFrame m15;
   MarketFrame m5Entry;
   AnalyzeFrame(g_h1Rates, g_h1Count, h1);
   AnalyzeFrame(g_m15Rates, g_m15Count, m15);
   AnalyzeFrame(g_m5Rates, g_m5Count, m5Entry);
   double pt = SymbolPoint();
   double atr = GetATRValue();
   double atrPoints = (pt > 0.0) ? atr / pt : 0.0;
   double avgRangePoints = m15.avg_range_points;
   double volFactor = GetVolatilityFactor(atrPoints, avgRangePoints);
   double curveFactor = GetEquityCurveFactor();
   double cdi = GetM15Cdi();
   double entryPrecisionBuy = GetM1PrecisionScore("BUY");
   double entryPrecisionSell = GetM1PrecisionScore("SELL");

   double spreadPts = (SymbolInfoDouble(_Symbol, SYMBOL_ASK) - SymbolInfoDouble(_Symbol, SYMBOL_BID)) / pt;
   double balance   = AccountInfoDouble(ACCOUNT_BALANCE);
   double equity    = AccountInfoDouble(ACCOUNT_EQUITY);
   double dd        = (balance > 0.0 && equity < balance) ? ((balance - equity) / balance) * 100.0 : 0.0;

   string json = "{";
   json += "\"asset\":\"" + AssetName + "\",";
   json += "\"analysis_mode\":\"H1_M15_CONTEXT__M5_M1_ENTRY\",";
   json += "\"client_time\":\"" + TimeToString(TimeCurrent(), TIME_DATE | TIME_MINUTES | TIME_SECONDS) + "\",";
   json += "\"m1_bar_time\":\"" + TimeToString(g_m1BarTime, TIME_DATE | TIME_MINUTES) + "\",";
   json += "\"context_timeframe_pair\":\"H1_M15\",";
   json += "\"entry_timeframe_pair\":\"M5_M1\",";
   json += "\"context_candles_h1\":" + BuildRatesJson(g_h1Rates, g_h1Count) + ",";
   json += "\"context_candles_m15\":" + BuildRatesJson(g_m15Rates, g_m15Count) + ",";
   json += "\"entry_candles_m5\":" + BuildRatesJson(g_m5Rates, g_m5Count) + ",";
   json += "\"entry_candles_m1\":" + BuildRatesJson(g_m1Rates, g_m1Count) + ",";
   json += "\"h1_bias\":\"" + h1.bias + "\",";
   json += "\"m15_bias\":\"" + m15.bias + "\",";
   json += "\"m5_bias\":\"" + m5Entry.bias + "\",";
   json += "\"m15_sweep\":\"" + m15.sweep + "\",";
   json += "\"m5_sweep\":\"" + m5Entry.sweep + "\",";
   json += "\"entry_precision_buy\":" + DoubleToString(entryPrecisionBuy, 4) + ",";
   json += "\"entry_precision_sell\":" + DoubleToString(entryPrecisionSell, 4) + ",";
   json += "\"m1_precision_score\":" + DoubleToString((entryPrecisionBuy + entryPrecisionSell) / 2.0, 4) + ",";
   json += "\"spread_points\":" + DoubleToString(spreadPts, 2) + ",";
   json += "\"confidence_threshold\":" + DoubleToString(MinProbability / 100.0, 4) + ",";
   json += "\"max_spread_points\":" + IntegerToString(MaxSpreadPoints) + ",";
   json += "\"drawdown_pct\":" + DoubleToString(dd, 2) + ",";
   json += "\"bars_since_last_trade\":" + IntegerToString(BarsSinceLastEntry()) + ",";
   json += "\"open_positions\":" + IntegerToString(CountOurPositions()) + ",";
   json += "\"max_positions\":" + IntegerToString(MaxPositions) + ",";
   json += "\"atr_points\":" + DoubleToString(atrPoints, 1) + ",";
   json += "\"volatility_factor\":" + DoubleToString(volFactor, 3) + ",";
   json += "\"equity_curve_factor\":" + DoubleToString(curveFactor, 3) + ",";
   json += "\"cdi\":" + DoubleToString(cdi, 3) + ",";
   json += "\"signal_history\":" + BuildSignalHistoryJson() + ",";
   json += "\"include_rich\":" + (UseRichResponse ? "true" : "false") + ",";
   json += "\"session_preference\":\"" + (AllowOffHours ? "Off-hours" : "Active") + "\",";
   json += "\"account_balance\":" + DoubleToString(balance, 2) + ",";
   json += "\"account_equity\":" + DoubleToString(equity, 2);
   json += "}";
   return json;
}

string ExtractStringField(string json, string key)
{
   string needle = "\"" + key + "\":";
   int start = StringFind(json, needle);
   if(start < 0) return "";
   start += StringLen(needle);
   while(start < StringLen(json) && StringGetCharacter(json, start) == ' ') start++;
   if(start >= StringLen(json)) return "";

   if(StringGetCharacter(json, start) == '"')
   {
      start++;
      int end = StringFind(json, "\"", start);
      if(end < 0) return "";
      return StringSubstr(json, start, end - start);
   }

   int end = start;
   while(end < StringLen(json))
   {
      ushort ch = (ushort)StringGetCharacter(json, end);
      if(ch == ',' || ch == '}' || ch == ']') break;
      end++;
   }
   string v = StringSubstr(json, start, end - start);
   StringTrimLeft(v);
   StringTrimRight(v);
   StringReplace(v, "\"", "");
   return v;
}

double ExtractDoubleField(string json, string key, double defaultValue = 0.0)
{
   string v = ExtractStringField(json, key);
   if(v == "" || v == "null") return defaultValue;
   double d = StringToDouble(v);
   if(d == 0.0 && v != "0" && v != "0.0") return defaultValue;
   return d;
}

bool SendRequest(string body, string &jsonOut, string urlOverride = "")
{
   char post[], res[];
   string resp_headers = "";
   string headers = "Content-Type: application/json\r\n";
   if(API_KEY != "")
      headers += "X-API-Key: " + API_KEY + "\r\n";

   int len = StringToCharArray(body, post, 0, WHOLE_ARRAY, CP_UTF8) - 1;
   if(len < 0)
   {
      jsonOut = "";
      return false;
   }
   ArrayResize(post, len);

   string targetUrl = (urlOverride == "" ? API_URL : urlOverride);

   if(EnableHttpDiagnostics)
      Log(StringFormat("POST url=%s bytes=%d", targetUrl, len));

   ResetLastError();
   int status = WebRequest("POST", targetUrl, headers, HTTPTimeoutMs, post, res, resp_headers);
   jsonOut = CharArrayToString(res, 0, -1, CP_UTF8);

   if(status != 200)
   {
      int err = GetLastError();
      g_lastHealthOk = ProbeServerHealth(true);
      g_consecutiveHttpFails++;
      Log(StringFormat("HTTP=%d err=%d response=%s", status, err, jsonOut));
      if(EnableHttpDiagnostics)
         Log(StringFormat("POST diagnostics: health=%s consecutiveFails=%d cause=%s", (g_lastHealthOk ? "OK" : "FAIL"), g_consecutiveHttpFails, DescribeWebRequestError(err)));
      return false;
   }

   g_consecutiveHttpFails = 0;
   return (StringLen(jsonOut) > 5);
}

bool WebRequestWithRetry(string method, string url, string body, string &result[])
{
   for(int attempt = 0; attempt < 3; attempt++)
   {
      char post[], res[];
      string resp_headers = "";
      string headers = "Content-Type: application/json
";
      if(API_KEY != "")
         headers += "X-API-Key: " + API_KEY + "
";

      int len = StringToCharArray(body, post, 0, WHOLE_ARRAY, CP_UTF8) - 1;
      if(len < 0)
         return -1;
      ArrayResize(post, len);

      ResetLastError();
      int status = WebRequest(method, url, headers, HTTPTimeoutMs, post, res, resp_headers);
      ArrayResize(result, 0);
      if(status == 200)
      {
         string full = CharArrayToString(res, 0, -1, CP_UTF8);
         int pieces = StringLen(full) > 0 ? 1 : 0;
         if(pieces > 0)
         {
            ArrayResize(result, 1);
            result[0] = full;
         }
         g_consecutiveHttpFails = 0;
         return 200;
      }

      Sleep(300 * (attempt + 1));
   }

   g_consecutiveHttpFails++;
   Log("WebRequestWithRetry failed after 3 attempts. Consecutive fails: " + IntegerToString(g_consecutiveHttpFails));
   return -1;
}

bool SendRequestWithRetry(string body, string &jsonOut, string urlOverride = "")
{
   string targetUrl = (urlOverride == "" ? API_URL : urlOverride);
   string result[];
   int status = WebRequestWithRetry("POST", targetUrl, body, result);
   if(status != 200 || ArraySize(result) <= 0)
      return false;

   jsonOut = result[0];
   return (StringLen(jsonOut) > 5);
}

// ========================= LOT SIZING =========================
double AdjustRiskByEquity(double lot)
{
   double balance = AccountInfoDouble(ACCOUNT_BALANCE);
   double equity  = AccountInfoDouble(ACCOUNT_EQUITY);
   if(balance <= 0.0) return lot;

   double dd = (balance - equity) / balance;
   if(dd > EquityHardDDLimit) lot *= 0.40;
   else if(dd > EquitySoftDDLimit) lot *= 0.75;
   else if(dd > 0.02) lot *= 0.90;
   else if(dd < 0.01) lot *= 1.05;

   lot *= GetEquityCurveFactor();
   return lot;
}

double CalculateDynamicLot(SignalData &d, double entryPrice, double slPrice)
{
   double minLot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
   double step   = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
   double maxLot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);

   if(MaxLotCap > 0.0)
      maxLot = MathMin(maxLot, MaxLotCap);

   if(step <= 0.0) step = 0.01;
   if(minLot <= 0.0) minLot = step;
   if(maxLot <= 0.0) maxLot = 100.0;

   double pt = SymbolPoint();
   double atr = GetATRValue();
   double atrPoints = (pt > 0.0) ? atr / pt : 0.0;
   MarketFrame m5;
   AnalyzeFrame(g_m5Rates, g_m5Count, m5);
   double avgRangePoints = m5.avg_range_points;
   double volFactor = GetVolatilityFactor(atrPoints, avgRangePoints);
   double curveFactor = GetEquityCurveFactor();

   double buffer = d.signal_confidence - d.dynamic_threshold;
   double confidenceFactor = 1.0;
   if(buffer >= 8.0) confidenceFactor = 1.45;
   else if(buffer >= 5.0) confidenceFactor = 1.25;
   else if(buffer >= 2.0) confidenceFactor = 1.10;
   else if(buffer >= 0.0) confidenceFactor = 1.00;
   else confidenceFactor = 0.80;

   double riskPct = RiskPercentPerTrade * dynamicRiskFactor * curveFactor * confidenceFactor;
   riskPct *= (volFactor >= 1.0 ? 1.0 : 0.90);
   if(d.recommended_risk_pct > 0.0)
      riskPct = MathMin(riskPct, d.recommended_risk_pct);
   riskPct = ClampDouble(riskPct, 0.20, MaxRiskPercentPerTrade);

   double equity = AccountInfoDouble(ACCOUNT_EQUITY);
   double riskMoney = equity * riskPct / 100.0;

   double tickValue = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);
   double tickSize  = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);
   double slDist    = MathAbs(entryPrice - slPrice);

   double lot = 0.0;
   if(tickValue > 0.0 && tickSize > 0.0 && slDist > 0.0)
      lot = riskMoney * tickSize / (slDist * tickValue);
   else
      lot = BaseLot;

   lot *= volFactor;
   lot *= d.risk_multiplier;
   lot = AdjustRiskByEquity(lot);
   lot = ClampDouble(lot, 0.0, maxLot);

   if(lot < minLot)
   {
      if(buffer >= 5.0)
         lot = MathMin(maxLot, minLot * 1.50);
      else if(buffer >= 2.0)
         lot = minLot;
      else
         return 0.0;
   }

   lot = MathFloor(lot / step) * step;
   lot = NormalizeDouble(lot, 2);
   if(lot < minLot)
   {
      if(buffer >= 5.0) lot = minLot;
      else return 0.0;
   }

   if(lot > maxLot) lot = maxLot;
   return lot;
}

// ========================= SIGNAL CACHE =========================
bool ContextAllowedByBias(string signal, string h1Bias, string m15Bias)
{
   if(h1Bias == "" || m15Bias == "" || h1Bias == "NULL" || m15Bias == "NULL")
      return false;

   if(signal == "BUY")
   {
      if(h1Bias == "DOWN" && m15Bias == "DOWN") return false;
      if(h1Bias == "DOWN" && m15Bias == "NEUTRAL") return false;
      return true;
   }
   if(signal == "SELL")
   {
      if(h1Bias == "UP" && m15Bias == "UP") return false;
      if(h1Bias == "UP" && m15Bias == "NEUTRAL") return false;
      return true;
   }
   return false;
}

bool BuildLocalContextSignal(SignalData &d)
{
   if(g_m15Count < 10 || g_h1Count < 10 || g_m5Count < 10 || g_m1Count < 10) return false;

   MarketFrame h1;
   MarketFrame m15;
   MarketFrame m5Exec;
   AnalyzeFrame(g_h1Rates, g_h1Count, h1);
   AnalyzeFrame(g_m15Rates, g_m15Count, m15);
   AnalyzeFrame(g_m5Rates, g_m5Count, m5Exec);
   double cdi = GetM15Cdi();
   string marketState = BuildMarketState(h1, m15, cdi);
   string bias = GetCombinedBias(h1, m15);
   string signal = "NONE";

   double buyEntry = GetM1PrecisionScore("BUY");
   double sellEntry = GetM1PrecisionScore("SELL");

   double localThreshold = GetAdaptiveM1PrecisionThreshold("BUY", marketState, 60.0);
   if(marketState == "REVERSAL")
      localThreshold = MathMax(localThreshold, M1PrecisionThresholdReversal);

   if(h1.bias == "UP" && m15.bias == "UP" && buyEntry >= localThreshold)
      signal = "BUY";
   else if(h1.bias == "DOWN" && m15.bias == "DOWN" && sellEntry >= localThreshold)
      signal = "SELL";
   else if(m15.sweep == "BULL" && h1.bias != "DOWN" && buyEntry >= localThreshold)
      signal = "BUY";
   else if(m15.sweep == "BEAR" && h1.bias != "UP" && sellEntry >= localThreshold)
      signal = "SELL";
   else if(m15.trend_strength > 0.10 && bias == "UP" && buyEntry >= localThreshold)
      signal = "BUY";
   else if(m15.trend_strength < -0.10 && bias == "DOWN" && sellEntry >= localThreshold)
      signal = "SELL";

   double precision = (signal == "BUY") ? buyEntry : ((signal == "SELL") ? sellEntry : 0.0);
   double prob = 52.0 + MathAbs(m15.trend_strength) * 12.0 + MathAbs(m5Exec.trend_strength) * 6.0 + precision * 18.0;
   if(marketState == "TRENDING") prob += 3.0;
   if(marketState == "BREAKOUT") prob += 4.5;
   if(m15.sweep != "NONE") prob += 2.0;
   prob = ClampDouble(prob, 0.0, 99.0);

   ZeroMemory(d);
   d.valid = true;
   d.context_bar_time = g_m15BarTime;
   d.signal = signal;
   d.regime = (cdi > 0.35 ? "high" : (cdi < -0.35 ? "low" : "mid"));
   d.market_state = marketState;
   d.active_bias = (signal == "BUY") ? "BULLISH" : ((signal == "SELL") ? "BEARISH" : (bias == "UP" ? "BULLISH" : (bias == "DOWN" ? "BEARISH" : "NEUTRAL")));
   d.session = "EAT";
   d.h1_bias = h1.bias;
   d.m15_bias = m15.bias;
   d.m15_sweep = m15.sweep;
   d.m5_bias = m5Exec.bias;
   d.m5_sweep = m5Exec.sweep;
   d.probability = prob;
   d.signal_confidence = prob;
   d.cdi = cdi;
   d.risk_multiplier = 1.0;
   d.recommended_risk_pct = RiskPercentPerTrade;
   d.dynamic_threshold = MinProbability;
   d.cooldown_bars = 2.0;
   d.breakeven_at_rr = TP1_RR;
   d.trail_after_rr = TrailStartRR;
   d.max_hold_bars = 24.0;
   d.max_spread_points = MaxSpreadPoints;
   d.m1_precision_score = precision;
   d.context_score = MathAbs(m15.trend_strength) + MathAbs(m5Exec.trend_strength) / 2.0 + precision;
   return true;
}

bool UpdateContextCache()
{
   if(g_m1Count < 10 || g_m5Count < 10 || g_m15Count < 10 || g_h1Count < 10) return false;
   if(g_m15BarTime == 0 || g_m1BarTime == 0) return false;

   // v11.5: refresh once per closed M1 bar, and throttle repeated server calls.
   if(g_lastContextAttemptBar == g_m1BarTime)
      return g_contextReady;

   if(g_lastRequestTime > 0 && (TimeCurrent() - g_lastRequestTime) < RequestThrottleSeconds)
      return g_contextReady;

   int ctxAgeMins = (g_lastContextRefreshTime > 0) ? (int)((TimeCurrent() - g_lastContextRefreshTime) / 60) : 9999;
   bool ctxStale = (!g_contextReady || ctxAgeMins > ContextMaxAgeMins);

   if(g_contextReady && !ctxStale)
      return true;

   if(g_contextReady && ctxStale)
      Log(StringFormat("Context stale (%d mins) -> refreshing on new M1 close", ctxAgeMins));
   else if(!g_contextReady)
      Log("No valid context yet -> requesting server context");

   g_lastContextAttemptBar = g_m1BarTime;
   g_lastContextAttemptTime = TimeCurrent();
   g_lastRequestTime = TimeCurrent();

   if(EnableHttpDiagnostics)
      ProbeServerHealth(false);

   string body = BuildRequestJSON();
   string json = "";
   string targetUrl = API_URL + (UseRichResponse ? "?include_rich=true" : "?include_rich=false");
   string result[];
   int requestStatus = WebRequestWithRetry("POST", targetUrl, body, result);
   bool requestOk = (requestStatus == 200 && ArraySize(result) > 0);
   if(requestOk)
      json = result[0];
   if(!requestOk)
   {
      Log("Context server call failed -> using local fallback context");
      if(g_lastHealthOk)
         Log("Health check is OK, so POST likely failed due to payload/endpoint mismatch or local request handling");

      SignalData local;
      if(!BuildLocalContextSignal(local))
      {
         g_contextReady = false;
         g_lastContextSource = "LOCAL";
         return false;
      }

      g_cachedSignal = local;
      g_contextReady = true;
      g_lastContextSource = "LOCAL";
      g_lastContextRefreshTime = TimeCurrent();
      g_lastContextRefreshBar = g_m1BarTime;
      g_lastContextM15Bar = g_lastContextRefreshBar;
      g_lastContextResponse = "";
      if(UsePanel) UpdatePanel(local.regime, local.probability, false);
      Log(StringFormat("CTX signal=%s conf=%.2f prob=%.2f regime=%s state=%s h1=%s m15=%s m5=%s m1=%.2f src=LOCAL",
                       local.signal, local.signal_confidence, local.probability, local.regime,
                       local.market_state, local.h1_bias, local.m15_bias, local.m5_bias, local.m1_precision_score));
      return true;
   }

   if(StringLen(g_lastContextResponse) > 0 && json == g_lastContextResponse)
      Log("Context response is identical to the previous refresh; backend may be stable or over-smoothed.");
   g_lastContextResponse = json;

   SignalData d;
   ZeroMemory(d);
   d.valid = true;
   d.context_bar_time = g_m15BarTime;
   d.signal = ToUpper(ExtractStringField(json, "last_signal"));
   if(d.signal == "") d.signal = "NONE";
   d.regime = ToUpper(ExtractStringField(json, "regime"));
   d.market_state = ToUpper(ExtractStringField(json, "market_state"));
   d.active_bias = ToUpper(ExtractStringField(json, "active_bias"));
   d.session = ExtractStringField(json, "session_label");
   if(d.session == "") d.session = ExtractStringField(json, "timezone");
   if(d.session == "") d.session = "EAT";
   d.probability = ExtractDoubleField(json, "probability", 0.0);
   if(d.probability <= 1.0) d.probability *= 100.0;
   d.signal_confidence = ExtractDoubleField(json, "signal_confidence", d.probability);
   if(d.signal_confidence <= 1.0) d.signal_confidence *= 100.0;
   d.cdi = ExtractDoubleField(json, "cdi", 0.0);
   d.risk_multiplier = ExtractDoubleField(json, "risk_multiplier", 1.0);
   d.recommended_risk_pct = ExtractDoubleField(json, "recommended_risk_pct", RiskPercentPerTrade);
   d.dynamic_threshold = ExtractDoubleField(json, "dynamic_threshold", MinProbability);
   if(d.dynamic_threshold <= 1.0) d.dynamic_threshold *= 100.0;
   d.cooldown_bars = ExtractDoubleField(json, "cooldown_bars", 2.0);
   d.breakeven_at_rr = ExtractDoubleField(json, "breakeven_at_rr", 1.0);
   d.trail_after_rr = ExtractDoubleField(json, "trail_after_rr", TrailStartRR);
   d.max_hold_bars = ExtractDoubleField(json, "max_hold_bars", 24.0);
   d.max_spread_points = ExtractDoubleField(json, "max_spread_points", MaxSpreadPoints);

   string h1Bias = ToUpper(ExtractStringField(json, "h1_bias"));
   string m15Bias = ToUpper(ExtractStringField(json, "m15_bias"));
   string m5BiasExec = ToUpper(ExtractStringField(json, "m5_bias"));
   string m15Sweep = ToUpper(ExtractStringField(json, "m15_sweep"));
   string m5SweepExec = ToUpper(ExtractStringField(json, "m5_sweep"));
   if(m15Bias == "") m15Bias = m5BiasExec;
   if(m15Sweep == "") m15Sweep = m5SweepExec;
   d.h1_bias = h1Bias;
   d.m15_bias = m15Bias;
   d.m15_sweep = m15Sweep;
   d.m5_bias = m5BiasExec;
   d.m5_sweep = m5SweepExec;

   double serverEntryPrecision = (d.signal == "BUY") ? ExtractDoubleField(json, "entry_precision_buy", 0.0)
                                                      : ExtractDoubleField(json, "entry_precision_sell", 0.0);
   d.m1_precision_score = (serverEntryPrecision > 0.0) ? serverEntryPrecision : GetM1PrecisionScore(d.signal);

   if((d.h1_bias == "" || d.h1_bias == "NULL") || (d.m15_bias == "" || d.m15_bias == "NULL"))
   {
      Log("Context fields incomplete from server -> using local fallback context");
      SignalData local;
      if(!BuildLocalContextSignal(local))
      {
         g_contextReady = false;
         g_lastContextSource = "LOCAL";
         return false;
      }

      g_cachedSignal = local;
      g_contextReady = true;
      g_lastContextSource = "LOCAL";
      g_lastContextRefreshTime = TimeCurrent();
      g_lastContextRefreshBar = g_m1BarTime;
      g_lastContextM15Bar = g_lastContextRefreshBar;
      g_lastContextResponse = json;
      Log(StringFormat("CTX signal=%s conf=%.2f prob=%.2f regime=%s state=%s h1=%s m15=%s m5=%s m1=%.2f src=LOCAL",
                       local.signal, local.signal_confidence, local.probability, local.regime,
                       local.market_state, local.h1_bias, local.m15_bias, local.m5_bias, local.m1_precision_score));
      return true;
   }

   g_cachedSignal = d;
   g_contextReady = true;
   g_lastContextSource = "SERVER";
   g_lastContextRefreshTime = TimeCurrent();
   g_lastContextRefreshBar = g_m1BarTime;
   g_lastContextM15Bar = g_lastContextRefreshBar;
   if(UsePanel) UpdatePanel(d.regime, d.probability, false);

   Log(StringFormat("CTX signal=%s conf=%.2f prob=%.2f regime=%s state=%s h1=%s m15=%s m5=%s m1=%.2f src=SERVER ageMins=0",
                    d.signal, d.signal_confidence, d.probability, d.regime,
                    d.market_state, d.h1_bias, d.m15_bias, d.m5_bias, d.m1_precision_score));
   return true;
}
// ========================= EXECUTION =========================
bool StopDistancesValid(ENUM_POSITION_TYPE type, double sl, double tp)
{
   double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   double pt  = SymbolInfoDouble(_Symbol, SYMBOL_POINT);
   int stopsLevel = (int)SymbolInfoInteger(_Symbol, SYMBOL_TRADE_STOPS_LEVEL);
   double minDist = (stopsLevel + 2) * pt;

   if(type == POSITION_TYPE_BUY)
   {
      if(sl > 0 && (bid - sl) < minDist) return false;
      if(tp > 0 && (tp - ask) < minDist) return false;
   }
   else
   {
      if(sl > 0 && (sl - ask) < minDist) return false;
      if(tp > 0 && (bid - tp) < minDist) return false;
   }
   return true;
}

bool ModifyPositionByTicket(ulong ticket, double sl, double tp, string why)
{
   if(!PositionSelectByTicket(ticket)) return false;
   string sym = PositionGetString(POSITION_SYMBOL);
   int digits = (int)SymbolInfoInteger(sym, SYMBOL_DIGITS);

   if(sl > 0.0) sl = NormalizeDouble(sl, digits);
   if(tp > 0.0) tp = NormalizeDouble(tp, digits);

   MqlTradeRequest req;
   MqlTradeResult  result;
   ZeroMemory(req);
   ZeroMemory(result);

   req.action  = TRADE_ACTION_SLTP;
   req.position = ticket;
   req.symbol   = sym;
   req.sl       = sl;
   req.tp       = tp;
   req.magic    = MagicNumber;

   bool sent = OrderSend(req, result);
   if(!sent || (result.retcode != TRADE_RETCODE_DONE && result.retcode != TRADE_RETCODE_DONE_PARTIAL))
   {
      Log(StringFormat("MODIFY failed ticket=%I64u ret=%d | %s", ticket, result.retcode, why));
      return false;
   }

   return true;
}

bool ClosePartialByTicket(ulong ticket, double volume)
{
   if(!PositionSelectByTicket(ticket)) return false;
   double vol = NormalizeDouble(volume, 2);
   double step = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
   double minVol = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
   if(step <= 0.0) step = 0.01;
   if(vol <= 0.0) return false;
   vol = MathFloor(vol / step) * step;
   if(vol < minVol) return false;

   bool ok = trade.PositionClosePartial(ticket, vol);
   if(ok)
      Log(StringFormat("PARTIAL CLOSE ticket=%I64u vol=%.2f", ticket, vol));
   return ok;
}

bool OpenPosition(SignalData &d)
{
   ENUM_POSITION_TYPE ptype = (d.signal == "BUY") ? POSITION_TYPE_BUY : POSITION_TYPE_SELL;

   if(HasOppositePosition(ptype))
   {
      Log("Opposite position open -> skip");
      return false;
   }

   bool sameDirection = HasOpenPositionType(ptype);
   int ourPosCount = CountOurPositions();

   if(ourPosCount >= MaxPositions)
   {
      Log("Max positions reached");
      return false;
   }

   if(BarsSinceLastEntry() < MinBarsBetweenEntries)
   {
      Log("Entry cooldown active");
      return false;
   }

   if(sameDirection && (!EnableScaleIn && !EnablePyramiding))
   {
      Log("Same-direction entries disabled");
      return false;
   }

   double atr = GetATRValue();
   if(atr <= 0.0)
   {
      Log("ATR not ready");
      return false;
   }

   double price = 0.0, sl = 0.0, tp = 0.0;
   if(d.signal == "BUY")
   {
      price = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
      sl = price - atr * ATR_Multiplier;
      tp = price + atr * ATR_Multiplier * 2.0;
   }
   else
   {
      price = SymbolInfoDouble(_Symbol, SYMBOL_BID);
      sl = price + atr * ATR_Multiplier;
      tp = price - atr * ATR_Multiplier * 2.0;
   }

   if(!StopDistancesValid(ptype, sl, tp))
   {
      Log("Entry rejected: invalid stops");
      return false;
   }

   double lot = CalculateDynamicLot(d, price, sl);
   if(lot <= 0.0)
   {
      Log("Lot size too small");
      return false;
   }

   if(sameDirection)
   {
      double requiredProb = MathMax(dynamicScaleInMinProb, dynamicPyramidMinProb);
      if(d.signal_confidence < requiredProb)
      {
         Log("Pyramid blocked: probability too low");
         return false;
      }

      if(BarsSinceLastEntry() < PyramidCooldownBars)
      {
         Log("Pyramid cooldown active");
         return false;
      }

      if(GetEquityCurveFactor() < 0.70)
      {
         Log("Pyramid blocked: equity curve weak");
         return false;
      }

      double basketProfit = 0.0;
      for(int i = 0; i < PositionsTotal(); i++)
      {
         ulong ticket = PositionGetTicket(i);
         if(PositionSelectByTicket(ticket) &&
            PositionGetString(POSITION_SYMBOL) == _Symbol &&
            PositionGetInteger(POSITION_MAGIC) == MagicNumber &&
            (ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE) == ptype)
         {
            basketProfit += PositionGetDouble(POSITION_PROFIT);
         }
      }

      bool basketHealthy = (basketProfit > PyramidMinBasketProfit) ||
                           (MathAbs(d.m1_precision_score) >= 0.55) ||
                           (d.signal_confidence >= requiredProb + 3.0);
      if(!basketHealthy)
      {
         Log("Pyramid blocked: basket not healthy");
         return false;
      }

      double factor = EnablePyramiding ? PyramidLotFactor : ScaleInLotFactor;
      if(GetEquityCurveFactor() > 1.05 && basketProfit > 0.0)
         factor = MathMin(1.0, factor + 0.10);
      lot *= factor;
   }

   double minLot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
   if(lot < minLot)
   {
      double buffer = d.signal_confidence - d.dynamic_threshold;
      if(buffer >= 5.0)
         lot = MathMin(SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX), minLot * 1.5);
      else if(buffer >= 2.0)
         lot = minLot;
      else
         return false;
   }

   double execPrice = (d.signal == "BUY") ? SymbolInfoDouble(_Symbol, SYMBOL_ASK) : SymbolInfoDouble(_Symbol, SYMBOL_BID);
   bool ok = false;
   if(d.signal == "BUY")
      ok = trade.Buy(lot, _Symbol, execPrice, sl, tp);
   else
      ok = trade.Sell(lot, _Symbol, execPrice, sl, tp);

   if(!ok)
   {
      Log(StringFormat("OrderSend failed ret=%d", trade.ResultRetcode()));
      return false;
   }

   g_lastTradeM1Bar = g_m1BarTime;
   Log(StringFormat("ENTRY %s lot=%.2f conf=%.2f thr=%.2f", d.signal, lot, d.signal_confidence, d.dynamic_threshold));
   return true;
}

// ========================= MANAGEMENT =========================
void ManagePositions()
{
   double atr = GetATRValue();
   if(atr <= 0.0) return;

   MarketFrame h1;
   MarketFrame m5;
   AnalyzeFrame(g_h1Rates, g_h1Count, h1);
   AnalyzeFrame(g_m5Rates, g_m5Count, m5);
   string bias = GetCombinedBias(h1, m5);
   string sweep = m5.sweep;
   double curveFactor = GetEquityCurveFactor();
   double trailCurveMult = (curveFactor < 0.70) ? 0.80 : (curveFactor > 1.05 ? 1.05 : 1.0);
   double tpCurveMult    = (curveFactor < 0.70) ? 0.85 : (curveFactor > 1.05 ? 1.15 : 1.0);

   double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   double pt  = SymbolPoint();

   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      ulong ticket = PositionGetTicket(i);
      if(!PositionSelectByTicket(ticket)) continue;
      if(PositionGetString(POSITION_SYMBOL) != _Symbol || PositionGetInteger(POSITION_MAGIC) != MagicNumber) continue;

      ENUM_POSITION_TYPE type = (ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE);
      double open = PositionGetDouble(POSITION_PRICE_OPEN);
      double sl   = PositionGetDouble(POSITION_SL);
      double tp   = PositionGetDouble(POSITION_TP);
      double vol  = PositionGetDouble(POSITION_VOLUME);
      double curPrice = (type == POSITION_TYPE_BUY) ? bid : ask;
      double risk = MathAbs(open - sl);
      if(risk <= 0.0) continue;
      double rr = MathAbs(curPrice - open) / risk;

      bool biasMatches  = (type == POSITION_TYPE_BUY && bias == "UP") || (type == POSITION_TYPE_SELL && bias == "DOWN");
      bool sweepAgainst = (type == POSITION_TYPE_BUY && sweep == "BEAR") || (type == POSITION_TYPE_SELL && sweep == "BULL");
      bool sweepWith    = (type == POSITION_TYPE_BUY && sweep == "BULL") || (type == POSITION_TYPE_SELL && sweep == "BEAR");

      if(rr >= TP1_RR && !IsPartialClosed(ticket))
      {
         double partialVol = MathFloor((vol / 2.0) / SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP)) * SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
         if(ClosePartialByTicket(ticket, partialVol))
            MarkPartialClosed(ticket);
      }

      double desiredSL = sl;
      double desiredTP = tp;
      bool wantModify = false;
      string why = "";

      if(rr >= 1.0)
      {
         if(type == POSITION_TYPE_BUY && (desiredSL == 0.0 || open > desiredSL))
         {
            desiredSL = open;
            wantModify = true;
            why += "BE ";
         }
         else if(type == POSITION_TYPE_SELL && (desiredSL == 0.0 || open < desiredSL))
         {
            desiredSL = open;
            wantModify = true;
            why += "BE ";
         }
      }

      if(rr >= dynamicTrailStartRR)
      {
         double trailATR = biasMatches ? dynamicTrailATRStrong : dynamicTrailATRWeak;
         trailATR *= trailCurveMult;

         if(type == POSITION_TYPE_BUY)
         {
            double trailSL = bid - atr * trailATR;
            if(trailSL > desiredSL)
            {
               desiredSL = trailSL;
               wantModify = true;
               why += (biasMatches ? "TRAIL_STRONG " : "TRAIL_WEAK ");
            }

            if(biasMatches || sweepWith)
            {
               double extTP = bid + atr * dynamicTP2ExtendATR * tpCurveMult;
               if(extTP > desiredTP)
               {
                  desiredTP = extTP;
                  wantModify = true;
                  why += "EXTEND_TP ";
               }
            }
            if(sweepAgainst)
            {
               double tighterSL = bid - atr * 0.55;
               if(tighterSL > desiredSL)
               {
                  desiredSL = tighterSL;
                  wantModify = true;
                  why += "ANTI_SWEEP ";
               }
            }
         }
         else
         {
            double trailSL = ask + atr * trailATR;
            if(desiredSL == 0.0 || trailSL < desiredSL)
            {
               desiredSL = trailSL;
               wantModify = true;
               why += (biasMatches ? "TRAIL_STRONG " : "TRAIL_WEAK ");
            }

            if(biasMatches || sweepWith)
            {
               double extTP = ask - atr * dynamicTP2ExtendATR * tpCurveMult;
               if(desiredTP == 0.0 || extTP < desiredTP)
               {
                  desiredTP = extTP;
                  wantModify = true;
                  why += "EXTEND_TP ";
               }
            }
            if(sweepAgainst)
            {
               double tighterSL = ask + atr * 0.55;
               if(desiredSL == 0.0 || tighterSL < desiredSL)
               {
                  desiredSL = tighterSL;
                  wantModify = true;
                  why += "ANTI_SWEEP ";
               }
            }
         }
      }

      if(wantModify)
      {
         double minMove = MinModifyPoints * pt;
         bool slChanged = (desiredSL > 0.0 && MathAbs(desiredSL - sl) >= minMove);
         bool tpChanged = (desiredTP > 0.0 && MathAbs(desiredTP - tp) >= minMove);
         if((slChanged || tpChanged) && StopDistancesValid(type, desiredSL, desiredTP))
            ModifyPositionByTicket(ticket, desiredSL, desiredTP, why);
      }
   }
}

// ========================= TRADE JOURNAL =========================
void OnTradeTransaction(const MqlTradeTransaction &trans, const MqlTradeRequest &request, const MqlTradeResult &result)
{
   if(trans.type != TRADE_TRANSACTION_DEAL_ADD) return;
   if(!HistoryDealSelect(trans.deal)) return;

   string sym = HistoryDealGetString(trans.deal, DEAL_SYMBOL);
   long magic = HistoryDealGetInteger(trans.deal, DEAL_MAGIC);
   if(sym != _Symbol || magic != MagicNumber) return;

   long entry = HistoryDealGetInteger(trans.deal, DEAL_ENTRY);
   double profit = HistoryDealGetDouble(trans.deal, DEAL_PROFIT);
   ulong positionId = (ulong)HistoryDealGetInteger(trans.deal, DEAL_POSITION_ID);
   bool positionStillOpen = PositionSelectByTicket(positionId);

   if(entry == DEAL_ENTRY_OUT || entry == DEAL_ENTRY_OUT_BY || entry == DEAL_ENTRY_INOUT)
   {
      totalClosedEvents++;
      netProfit += profit;

      if(positionStillOpen)
      {
         partialExitEvents++;
      }
      else
      {
         fullClosedTrades++;
         fullNetProfit += profit;

         if(profit >= 0.0)
         {
            wins++;
            grossProfit += profit;
         }
         else
         {
            losses++;
            grossLoss += MathAbs(profit);
         }

         UpdateEquityHistory();
      }
   }
}

// ========================= PANEL =========================
double GetEquityDrawdownPct()
{
   double eq = AccountInfoDouble(ACCOUNT_EQUITY);
   if(eq <= 0.0 || equityPeak <= 0.0) return 0.0;
   if(eq > equityPeak) equityPeak = eq;
   return ClampDouble(((equityPeak - eq) / equityPeak) * 100.0, 0.0, 100.0);
}

double CalculateWinRate()
{
   int closed = wins + losses;
   if(closed <= 0) return 0.0;
   return 100.0 * (double)wins / (double)closed;
}

void CreatePanel()
{
   if(!UsePanel || panelCreated) return;

   panelChartID = ChartID();

   ObjectCreate(panelChartID, PanelPrefix+"BG", OBJ_RECTANGLE_LABEL, 0, 0, 0);
   ObjectSetInteger(panelChartID, PanelPrefix+"BG", OBJPROP_XDISTANCE, 10);
   ObjectSetInteger(panelChartID, PanelPrefix+"BG", OBJPROP_YDISTANCE, 10);
   ObjectSetInteger(panelChartID, PanelPrefix+"BG", OBJPROP_XSIZE, 320);
   ObjectSetInteger(panelChartID, PanelPrefix+"BG", OBJPROP_YSIZE, 220);
   ObjectSetInteger(panelChartID, PanelPrefix+"BG", OBJPROP_BGCOLOR, clrBlack);
   ObjectSetInteger(panelChartID, PanelPrefix+"BG", OBJPROP_BORDER_TYPE, BORDER_FLAT);
   ObjectSetInteger(panelChartID, PanelPrefix+"BG", OBJPROP_COLOR, clrWhite);

   ObjectCreate(panelChartID, PanelPrefix+"Title", OBJ_LABEL, 0, 0, 0);
   ObjectSetInteger(panelChartID, PanelPrefix+"Title", OBJPROP_XDISTANCE, 20);
   ObjectSetInteger(panelChartID, PanelPrefix+"Title", OBJPROP_YDISTANCE, 20);
   ObjectSetString(panelChartID, PanelPrefix+"Title", OBJPROP_TEXT, "GoldDiggr v12.0");
   ObjectSetInteger(panelChartID, PanelPrefix+"Title", OBJPROP_COLOR, clrGold);
   ObjectSetInteger(panelChartID, PanelPrefix+"Title", OBJPROP_FONTSIZE, 12);

   ObjectCreate(panelChartID, PanelPrefix+"Conn", OBJ_LABEL, 0, 0, 0);
   ObjectSetInteger(panelChartID, PanelPrefix+"Conn", OBJPROP_XDISTANCE, 20);
   ObjectSetInteger(panelChartID, PanelPrefix+"Conn", OBJPROP_YDISTANCE, 45);

   ObjectCreate(panelChartID, PanelPrefix+"Regime", OBJ_LABEL, 0, 0, 0);
   ObjectSetInteger(panelChartID, PanelPrefix+"Regime", OBJPROP_XDISTANCE, 20);
   ObjectSetInteger(panelChartID, PanelPrefix+"Regime", OBJPROP_YDISTANCE, 70);

   ObjectCreate(panelChartID, PanelPrefix+"Prob", OBJ_LABEL, 0, 0, 0);
   ObjectSetInteger(panelChartID, PanelPrefix+"Prob", OBJPROP_XDISTANCE, 20);
   ObjectSetInteger(panelChartID, PanelPrefix+"Prob", OBJPROP_YDISTANCE, 95);

   ObjectCreate(panelChartID, PanelPrefix+"Signal", OBJ_LABEL, 0, 0, 0);
   ObjectSetInteger(panelChartID, PanelPrefix+"Signal", OBJPROP_XDISTANCE, 20);
   ObjectSetInteger(panelChartID, PanelPrefix+"Signal", OBJPROP_YDISTANCE, 120);

   ObjectCreate(panelChartID, PanelPrefix+"DD", OBJ_LABEL, 0, 0, 0);
   ObjectSetInteger(panelChartID, PanelPrefix+"DD", OBJPROP_XDISTANCE, 20);
   ObjectSetInteger(panelChartID, PanelPrefix+"DD", OBJPROP_YDISTANCE, 145);

   ObjectCreate(panelChartID, PanelPrefix+"Winrate", OBJ_LABEL, 0, 0, 0);
   ObjectSetInteger(panelChartID, PanelPrefix+"Winrate", OBJPROP_XDISTANCE, 20);
   ObjectSetInteger(panelChartID, PanelPrefix+"Winrate", OBJPROP_YDISTANCE, 170);

   panelCreated = true;
   Log("Panel created with live objects");
}

void UpdatePanel(string regime, double prob)
{
   UpdatePanel(regime, prob, false);
}

void UpdatePanel(string regime, double prob, bool elevated)
{
   if(!panelCreated) CreatePanel();
   if(!UsePanel) return;

   string connText = (g_contextReady && g_consecutiveHttpFails == 0) ? "CONNECTED ✓" : "OFFLINE (" + IntegerToString(g_consecutiveHttpFails) + ")";
   if(elevated) connText = connText + " / ALERT";
   color connColor = (g_contextReady && g_consecutiveHttpFails == 0) ? clrLime : clrRed;

   ObjectSetString(panelChartID, PanelPrefix+"Conn", OBJPROP_TEXT, "Status: " + connText);
   ObjectSetInteger(panelChartID, PanelPrefix+"Conn", OBJPROP_COLOR, connColor);

   if(g_contextReady)
   {
      ObjectSetString(panelChartID, PanelPrefix+"Regime", OBJPROP_TEXT, "Regime: " + regime);
      ObjectSetString(panelChartID, PanelPrefix+"Prob", OBJPROP_TEXT, "Probability: " + DoubleToString(prob, 1) + "%");
      ObjectSetString(panelChartID, PanelPrefix+"Signal", OBJPROP_TEXT, "Signal: " + g_cachedSignal.signal);
      ObjectSetString(panelChartID, PanelPrefix+"DD", OBJPROP_TEXT, "Equity DD: " + DoubleToString(GetEquityDrawdownPct(), 2) + "%");
      ObjectSetString(panelChartID, PanelPrefix+"Winrate", OBJPROP_TEXT, "Winrate: " + DoubleToString(CalculateWinRate(), 1) + "%");
   }
   ChartRedraw(panelChartID);
}

// ========================= PANEL / CLEANUP HELPERS =========================

void DeleteObjectsByPrefix(string prefix)
{
   long chart_id = 0;
   int total = ObjectsTotal(chart_id, 0, -1);
   for(int i = total - 1; i >= 0; i--)
   {
      string name = ObjectName(chart_id, i, 0, -1);
      if(StringFind(name, prefix) == 0)
         ObjectDelete(chart_id, name);
   }
}

// ========================= INIT / DEINIT =========================
int OnInit()
{
   trade.SetExpertMagicNumber((int)MagicNumber);
   trade.SetDeviationInPoints(20);
   trade.SetTypeFillingBySymbol(_Symbol);

   ArraySetAsSeries(g_m1Rates, true);
   ArraySetAsSeries(g_m5Rates, true);
   ArraySetAsSeries(g_h1Rates, true);

   dynamicMinProb         = MinProbability;
   dynamicRiskFactor      = 1.0;
   dynamicOffHoursMinProb = StrongProb;
   dynamicTrailATRWeak    = BiasTrailATRWeak;
   dynamicTrailATRStrong  = BiasTrailATRStrong;
   dynamicTP2ExtendATR    = TP2_ExtendATR;
   dynamicScaleInMinProb  = ScaleInMinProb;
   dynamicPyramidMinProb  = PyramidMinProb;
   dynamicTP1_RR          = TP1_RR;
   dynamicTrailStartRR    = TrailStartRR;

   atrHandle = iATR(_Symbol, PERIOD_M1, ATR_Period);
   if(atrHandle == INVALID_HANDLE)
   {
      Log("Failed to create ATR handle");
      return INIT_FAILED;
   }

   equityPeak = AccountInfoDouble(ACCOUNT_EQUITY);
   UpdateEquityHistory();
   RefreshBuffers();
   g_lastM1HeartbeatBar = g_m1BarTime;

   EventSetTimer(1);
   if(UsePanel) CreatePanel();

   Log("GoldDiggr MTF optimized v12.0 initialized successfully");
   return INIT_SUCCEEDED;
}

void OnDeinit(const int reason)
{
   EventKillTimer();
   if(atrHandle != INVALID_HANDLE) IndicatorRelease(atrHandle);
   if(UsePanel) DeleteObjectsByPrefix(PanelPrefix);
   panelCreated = false;
}

// ========================= TRADING CYCLE =========================
void ProcessTradingCycle()
{
   if(!RefreshBuffers()) return;

   LogM1Heartbeat();

   UpdateEquityHistory();
   ManagePositions();

   double spreadPts = (SymbolInfoDouble(_Symbol, SYMBOL_ASK) - SymbolInfoDouble(_Symbol, SYMBOL_BID)) / SymbolPoint();
   if(spreadPts > MaxSpreadPoints)
   {
      Log(StringFormat("Spread too high (%.1f pts)", spreadPts));
      return;
   }

   if(!UpdateContextCache())
   {
      if(UsePanel) UpdatePanel(g_contextReady ? g_cachedSignal.regime : "N/A", g_contextReady ? g_cachedSignal.probability : 0.0, false);
      return;
   }

   SignalData d = g_cachedSignal;
   int liveCtxAgeMins = (g_contextReady && g_lastContextRefreshTime > 0) ? (int)((TimeCurrent() - g_lastContextRefreshTime) / 60) : 9999;
   if(g_contextReady && liveCtxAgeMins > ContextMaxAgeMins)
      Log(StringFormat("Context age %d mins exceeds limit %d -> refreshing on next M1 close", liveCtxAgeMins, ContextMaxAgeMins));

   if(!d.valid || (d.signal != "BUY" && d.signal != "SELL"))
   {
      if(g_lastDecisionTraceBar != g_m1BarTime)
      {
         LogDecisionTrace("NO_SIGNAL", "context returned NONE", d.signal, spreadPts, d.m1_precision_score);
         g_lastDecisionTraceBar = g_m1BarTime;
      }
      PushRecentSignal("NONE");
      g_confirmSignalName = "";
      g_confirmSignalCount = 0;
      return;
   }

   PushRecentSignal(d.signal);

   if(d.probability < MinProbability || d.signal_confidence < d.dynamic_threshold)
   {
      if(g_lastDecisionTraceBar != g_m1BarTime)
      {
         LogDecisionTrace("BLOCKED", "low probability or confidence", d.signal, spreadPts, d.m1_precision_score);
         g_lastDecisionTraceBar = g_m1BarTime;
      }
      g_confirmSignalName = "";
      g_confirmSignalCount = 0;
      return;
   }

   if(d.cdi > MaxCDI || (d.max_spread_points > 0.0 && spreadPts > d.max_spread_points))
   {
      if(g_lastDecisionTraceBar != g_m1BarTime)
      {
         LogDecisionTrace("BLOCKED", "CDI or spread limit", d.signal, spreadPts, d.m1_precision_score);
         g_lastDecisionTraceBar = g_m1BarTime;
      }
      g_confirmSignalName = "";
      g_confirmSignalCount = 0;
      return;
   }

   if(!AllowOffHours && d.session == "Off-hours" && d.signal_confidence < dynamicOffHoursMinProb)
   {
      if(g_lastDecisionTraceBar != g_m1BarTime)
      {
         LogDecisionTrace("BLOCKED", "off-hours confidence filter", d.signal, spreadPts, d.m1_precision_score);
         g_lastDecisionTraceBar = g_m1BarTime;
      }
      g_confirmSignalName = "";
      g_confirmSignalCount = 0;
      return;
   }

   if(!ContextAllowedByBias(d.signal, d.h1_bias, d.m15_bias))
   {
      if(g_lastDecisionTraceBar != g_m1BarTime)
      {
         LogDecisionTrace("BLOCKED", "higher timeframe bias mismatch", d.signal, spreadPts, d.m1_precision_score);
         g_lastDecisionTraceBar = g_m1BarTime;
      }
      g_confirmSignalName = "";
      g_confirmSignalCount = 0;
      return;
   }

   double precisionScore = 0.0;
   bool m1Trigger = M1EntryTrigger(d.signal, d.signal_confidence, d.market_state, precisionScore);
   if(!m1Trigger)
   {
      if(g_lastDecisionTraceBar != g_m1BarTime)
      {
         LogDecisionTrace("BLOCKED", "M1 trigger not strong enough", d.signal, spreadPts, precisionScore);
         g_lastDecisionTraceBar = g_m1BarTime;
      }
      g_confirmSignalName = "";
      g_confirmSignalCount = 0;
      return;
   }

   if(d.m1_precision_score > 0.0 && precisionScore < d.m1_precision_score)
   {
      g_confirmSignalName = "";
      g_confirmSignalCount = 0;
      return;
   }

   if(d.h1_bias == "" || d.m15_bias == "" || d.h1_bias == "NULL" || d.m15_bias == "NULL")
   {
      g_confirmSignalName = "";
      g_confirmSignalCount = 0;
      return;
   }

   if(g_confirmSignalName != d.signal)
   {
      if(g_lastDecisionTraceBar != g_m1BarTime)
      {
         LogDecisionTrace("ARMING", "signal confirmation reset", d.signal, spreadPts, d.m1_precision_score);
         g_lastDecisionTraceBar = g_m1BarTime;
      }
      g_confirmSignalName = d.signal;
      g_confirmSignalCount = 1;
      return;
   }

   g_confirmSignalCount++;
   if(g_confirmSignalCount < SignalConfirmations)
      return;

   if(g_lastDecisionTraceBar != g_m1BarTime)
   {
      LogDecisionTrace("READY", "all gates passed, attempting entry", d.signal, spreadPts, d.m1_precision_score);
      g_lastDecisionTraceBar = g_m1BarTime;
   }

   if(OpenPosition(d))
   {
      g_confirmSignalName = "";
      g_confirmSignalCount = 0;
   }
}

void OnTick()
{
   if(!IsNewBar())
      return;

   ProcessTradingCycle();
}

// ========================= TIMER =========================
void OnTimer()
{
   if(!UsePanel) return;

   if(g_contextReady)
      UpdatePanel(g_cachedSignal.regime, g_cachedSignal.probability, false);
   else
      UpdatePanel("N/A", 0.0, false);
}
