//+------------------------------------------------------------------+
//|               GoldDiggr_v10_CONTEXT_ENGINE.mq5                   |
//|  Rolling 30-candle context, adaptive entries, safe management,   |
//|  trade journaling, quality tracking, pyramiding, and equity      |
//|  curve optimization                                              |
//+------------------------------------------------------------------+
#property strict

#include <Trade/Trade.mqh>

CTrade trade;

// ========================= INPUTS =========================
input string API_URL               = "http://127.0.0.1:8000/analyze";
input string AssetName             = "XAUUSD";
input long   MagicNumber           = 260325;

input double BaseLot               = 0.01;
input int    ATR_Period            = 14;
input double ATR_Multiplier        = 2.0;

input double MinProbability        = 58.0;
input double StrongProb            = 65.0;
input double MaxCDI                = 0.60;

input bool   AllowOffHours         = true;
input int    SignalConfirmations   = 2;
input int    MaxPositions          = 3;
input int    MinBarsBetweenEntries = 2;
input int    MaxSpreadPoints       = 350;
input int    HTTPTimeoutMs         = 10000;

input double TP1_RR                = 1.0;
input double TrailStartRR          = 1.2;
input double TrailATR              = 1.0;
input double BiasTrailATRWeak      = 0.70;
input double BiasTrailATRStrong    = 1.00;
input double TP2_ExtendATR         = 3.00;
input int    MinModifyPoints       = 30;

input bool   EnableScaleIn         = true;
input bool   EnablePyramiding      = true;
input double ScaleInLotFactor      = 0.50;
input double PyramidLotFactor      = 0.60;
input double ScaleInMinProb        = 72.0;
input double PyramidMinProb        = 74.0;
input int    PyramidCooldownBars   = 3;
input double PyramidMinBasketProfit= 0.0;

input double EquitySoftDDLimit     = 0.05;
input double EquityHardDDLimit     = 0.10;

// ========================= STATE =========================
#define MAX_CANDLES 30
#define HISTORY_DEPTH 5
#define EQUITY_HISTORY_SIZE 32

MqlRates g_candles[];
int      g_candleCount = 0;

datetime lastCandleTime      = 0;
datetime lastEntryCandleTime = 0;

string signalBuffer[5];
int    signalCount = 0;

string recentSignals[5];
int    recentSignalCount = 0;

ulong partialClosedTickets[128];

// Equity curve tracking
double equityHistory[EQUITY_HISTORY_SIZE];
int    equityHistoryCount = 0;
double equityPeak = 0.0;

// Trade quality tracking
int    totalClosedEvents  = 0;   // all exit deals, including partial exits
int    fullClosedTrades   = 0;   // only final fully closed trades
int    partialExitEvents   = 0;
int    wins                = 0;   // full closes only
int    losses              = 0;   // full closes only
double grossProfit         = 0.0; // full closes only
double grossLoss           = 0.0; // full closes only
double fullNetProfit       = 0.0; // full closes only
double netProfit           = 0.0; // all exit deals

// Phase 2 adaptive learning
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
int    consecutiveWins        = 0;
int    consecutiveLosses      = 0;

int atrHandle = INVALID_HANDLE;

// ========================= STRUCT =========================
struct SignalData
{
   string signal;
   double probability;
   double adjusted_probability;

   string regime;
   double cdi;
   double risk_mult;
   string session;

   string bias;
   string sweep;

   double trend_strength;
   double momentum_points;
   double avg_range_points;

   double context_score;
};

// ========================= LOG =========================
void Log(string msg)
{
   Print("[GoldDiggr] ", msg);
}

// ========================= INIT / DEINIT =========================
int OnInit()
{
   trade.SetExpertMagicNumber((int)MagicNumber);
   trade.SetDeviationInPoints(20);

   ArraySetAsSeries(g_candles, true);

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
   RefreshCandleBuffer();

   return INIT_SUCCEEDED;
}

void OnDeinit(const int reason)
{
   if(atrHandle != INVALID_HANDLE)
      IndicatorRelease(atrHandle);
}

// ========================= HELPERS =========================
double ClampDouble(double v, double lo, double hi)
{
   if(v < lo) return lo;
   if(v > hi) return hi;
   return v;
}

double SymbolPoint()
{
   double pt = SymbolInfoDouble(_Symbol, SYMBOL_POINT);
   if(pt <= 0.0)
      pt = 0.01;
   return pt;
}

bool IsNewCandle()
{
   datetime t = iTime(_Symbol, PERIOD_M1, 0);
   if(t != lastCandleTime)
   {
      lastCandleTime = t;
      return true;
   }
   return false;
}

bool RefreshCandleBuffer()
{
   ArraySetAsSeries(g_candles, true);
   g_candleCount = CopyRates(_Symbol, PERIOD_M1, 1, MAX_CANDLES, g_candles);

   if(g_candleCount < 20)
   {
      Log("Not enough candles yet for context");
      return false;
   }

   return true;
}

double GetATRValue()
{
   if(atrHandle == INVALID_HANDLE)
      return 0.0;

   double atrBuf[];
   ArraySetAsSeries(atrBuf, true);

   if(CopyBuffer(atrHandle, 0, 1, 1, atrBuf) < 1)
      return 0.0;

   return atrBuf[0];
}

void UpdateEquityHistory()
{
   double eq = AccountInfoDouble(ACCOUNT_EQUITY);
   if(eq <= 0.0)
      return;

   if(equityPeak <= 0.0 || eq > equityPeak)
      equityPeak = eq;

   equityHistory[equityHistoryCount % EQUITY_HISTORY_SIZE] = eq;
   equityHistoryCount++;
}

double GetEquityCurveSlope()
{
   int n = MathMin(equityHistoryCount, EQUITY_HISTORY_SIZE);
   if(n < 6)
      return 0.0;

   int newest = (equityHistoryCount - 1) % EQUITY_HISTORY_SIZE;
   int older  = (equityHistoryCount - 6) % EQUITY_HISTORY_SIZE;

   double oldv = equityHistory[older];
   if(oldv <= 0.0)
      return 0.0;

   return (equityHistory[newest] - oldv) / oldv;
}

double GetEquityCurveFactor()
{
   double eq = AccountInfoDouble(ACCOUNT_EQUITY);
   if(eq <= 0.0)
      return 1.0;

   if(equityPeak <= 0.0 || eq > equityPeak)
      equityPeak = eq;

   double dd = (equityPeak - eq) / equityPeak;
   double slope = GetEquityCurveSlope();

   double factor = 1.0;

   if(dd > EquityHardDDLimit || slope < -0.03)
      factor = 0.40;
   else if(dd > 0.08 || slope < -0.02)
      factor = 0.60;
   else if(dd > EquitySoftDDLimit || slope < -0.01)
      factor = 0.80;
   else if(dd < 0.01 && slope > 0.01)
      factor = 1.10;

   return ClampDouble(factor, 0.40, 1.10);
}

int CountOurPositions()
{
   int count = 0;
   for(int i = 0; i < PositionsTotal(); i++)
   {
      ulong ticket = PositionGetTicket(i);
      if(!PositionSelectByTicket(ticket))
         continue;

      string sym = PositionGetString(POSITION_SYMBOL);
      long magic = (long)PositionGetInteger(POSITION_MAGIC);

      if(sym == _Symbol && magic == MagicNumber)
         count++;
   }
   return count;
}

bool HasOpenPositionType(ENUM_POSITION_TYPE ptype)
{
   for(int i = 0; i < PositionsTotal(); i++)
   {
      ulong ticket = PositionGetTicket(i);
      if(!PositionSelectByTicket(ticket))
         continue;

      string sym = PositionGetString(POSITION_SYMBOL);
      long magic = (long)PositionGetInteger(POSITION_MAGIC);
      ENUM_POSITION_TYPE type = (ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE);

      if(sym == _Symbol && magic == MagicNumber && type == ptype)
         return true;
   }
   return false;
}

bool HasOppositePosition(ENUM_POSITION_TYPE ptype)
{
   for(int i = 0; i < PositionsTotal(); i++)
   {
      ulong ticket = PositionGetTicket(i);
      if(!PositionSelectByTicket(ticket))
         continue;

      string sym = PositionGetString(POSITION_SYMBOL);
      long magic = (long)PositionGetInteger(POSITION_MAGIC);
      ENUM_POSITION_TYPE type = (ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE);

      if(sym != _Symbol || magic != MagicNumber)
         continue;

      if(type != ptype)
         return true;
   }
   return false;
}

bool IsPartialClosed(ulong ticket)
{
   for(int i = 0; i < ArraySize(partialClosedTickets); i++)
      if(partialClosedTickets[i] == ticket)
         return true;
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

double BasketProfit(ENUM_POSITION_TYPE ptype)
{
   double sum = 0.0;

   for(int i = 0; i < PositionsTotal(); i++)
   {
      ulong ticket = PositionGetTicket(i);
      if(!PositionSelectByTicket(ticket))
         continue;

      string sym = PositionGetString(POSITION_SYMBOL);
      long magic = (long)PositionGetInteger(POSITION_MAGIC);
      ENUM_POSITION_TYPE type = (ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE);

      if(sym != _Symbol || magic != MagicNumber || type != ptype)
         continue;

      sum += PositionGetDouble(POSITION_PROFIT);
   }

   return sum;
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
      if(idx < 0)
         idx += size;

      if(recentSignals[idx] == "")
         continue;

      if(!first)
         json += ",";
      json += "\"" + recentSignals[idx] + "\"";
      first = false;
   }

   json += "]";
   return json;
}

void PushRecentSignal(string s)
{
   int size = ArraySize(recentSignals);
   recentSignals[recentSignalCount % size] = s;
   recentSignalCount++;
}

int BarsSinceLastEntry()
{
   if(lastEntryCandleTime == 0 || lastCandleTime == 0)
      return 999;

   int secs = (int)(lastCandleTime - lastEntryCandleTime);
   if(secs < 0)
      secs = 0;

   return secs / 60;
}

double NormalizeVolumeToStep(double volume)
{
   double minVol = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
   double step   = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
   double maxVol = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);

   if(step <= 0.0)
      step = 0.01;

   volume = MathFloor(volume / step) * step;
   volume = MathMax(volume, 0.0);
   volume = MathMin(volume, maxVol);

   if(volume < minVol)
      return 0.0;

   return NormalizeDouble(volume, 2);
}

double AdjustRiskByEquity(double lot)
{
   double balance = AccountInfoDouble(ACCOUNT_BALANCE);
   double equity  = AccountInfoDouble(ACCOUNT_EQUITY);

   if(balance <= 0.0)
      return lot;

   double dd = (balance - equity) / balance;

   if(dd > EquityHardDDLimit)
      lot *= 0.40;
   else if(dd > EquitySoftDDLimit)
      lot *= 0.75;
   else if(dd > 0.02)
      lot *= 0.90;
   else if(dd < 0.01)
      lot *= 1.05;

   lot *= GetEquityCurveFactor();

   return lot;
}

bool StopDistancesValid(ENUM_POSITION_TYPE type, double sl, double tp)
{
   double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   double pt  = SymbolPoint();

   int stopsLevel  = (int)SymbolInfoInteger(_Symbol, SYMBOL_TRADE_STOPS_LEVEL);
   int freezeLevel = (int)SymbolInfoInteger(_Symbol, SYMBOL_TRADE_FREEZE_LEVEL);
   int minLevelPts = MathMax(stopsLevel, freezeLevel) + 2;
   double minDist  = minLevelPts * pt;

   if(type == POSITION_TYPE_BUY)
   {
      if(sl > 0.0 && (bid - sl) < minDist)
         return false;
      if(tp > 0.0 && (tp - ask) < minDist)
         return false;
   }
   else
   {
      if(sl > 0.0 && (sl - ask) < minDist)
         return false;
      if(tp > 0.0 && (bid - tp) < minDist)
         return false;
   }

   return true;
}

// ========================= CONTEXT ENGINE =========================
double GetTrendStrength()
{
   if(g_candleCount < 3)
      return 0.0;

   int up = 0, down = 0;

   for(int i = g_candleCount - 1; i > 0; i--)
   {
      double newer = g_candles[i - 1].close;
      double older = g_candles[i].close;

      if(newer > older)
         up++;
      else if(newer < older)
         down++;
   }

   double denom = (double)MathMax(1, g_candleCount - 1);
   return (double)(up - down) / denom;
}

double GetMomentumPoints()
{
   if(g_candleCount < 6)
      return 0.0;

   int lookback = MathMin(5, g_candleCount - 1);
   double pt = SymbolPoint();

   return (g_candles[0].close - g_candles[lookback].close) / pt;
}

double GetAvgRangePoints()
{
   if(g_candleCount <= 0)
      return 0.0;

   double pt = SymbolPoint();
   double sum = 0.0;

   for(int i = 0; i < g_candleCount; i++)
      sum += (g_candles[i].high - g_candles[i].low) / pt;

   return sum / g_candleCount;
}

string GetBias()
{
   if(g_candleCount < 4)
      return "NEUTRAL";

   double h1 = g_candles[0].high;
   double h2 = g_candles[1].high;
   double h3 = g_candles[2].high;

   double l1 = g_candles[0].low;
   double l2 = g_candles[1].low;
   double l3 = g_candles[2].low;

   if(h1 > h2 && h2 > h3 && l1 > l2 && l2 > l3)
      return "UP";

   if(h1 < h2 && h2 < h3 && l1 < l2 && l2 < l3)
      return "DOWN";

   return "NEUTRAL";
}

string GetLiquiditySweepSignal()
{
   if(g_candleCount < 6)
      return "NONE";

   double pt = SymbolPoint();
   double sweepBuffer = 10.0 * pt;

   double prevHigh = g_candles[1].high;
   double prevLow  = g_candles[1].low;

   for(int i = 2; i <= 5; i++)
   {
      prevHigh = MathMax(prevHigh, g_candles[i].high);
      prevLow  = MathMin(prevLow,  g_candles[i].low);
   }

   double curHigh  = g_candles[0].high;
   double curLow   = g_candles[0].low;
   double curClose = g_candles[0].close;

   if(curLow < (prevLow - sweepBuffer) && curClose > prevLow)
      return "BULL";

   if(curHigh > (prevHigh + sweepBuffer) && curClose < prevHigh)
      return "BEAR";

   return "NONE";
}

double ComputeContextScore(string signal, string bias, string sweep, string session, double cdi, double trendStrength, double momentumPoints, double avgRangePoints)
{
   double score = 0.0;

   if(signal == "BUY")
   {
      if(bias == "UP") score += 25.0;
      if(bias == "DOWN") score -= 20.0;

      if(sweep == "BULL") score += 20.0;
      if(sweep == "BEAR") score -= 10.0;
   }
   else if(signal == "SELL")
   {
      if(bias == "DOWN") score += 25.0;
      if(bias == "UP") score -= 20.0;

      if(sweep == "BEAR") score += 20.0;
      if(sweep == "BULL") score -= 10.0;
   }

   if(MathAbs(trendStrength) > 0.15)
      score += 10.0;

   if(MathAbs(momentumPoints) > 8.0)
      score += 10.0;

   if(avgRangePoints > 80.0 && avgRangePoints < 250.0)
      score += 5.0;
   else if(avgRangePoints > 400.0)
      score -= 5.0;

   if(cdi < 0.35)
      score += 10.0;
   else if(cdi > MaxCDI)
      score -= 25.0;

   if(session == "Off-hours")
      score -= 10.0;

   return score;
}

bool IsSessionAllowed(string session, double adjustedProb, string signal, string bias, string sweep, double trendStrength)
{
   if(session != "Off-hours")
      return true;

   if(adjustedProb >= dynamicOffHoursMinProb)
      return true;

   if(signal == "BUY"  && (bias == "UP"   || sweep == "BULL" || trendStrength > 0.15))
      return true;

   if(signal == "SELL" && (bias == "DOWN" || sweep == "BEAR" || trendStrength < -0.15))
      return true;

   if(AllowOffHours && adjustedProb >= (dynamicMinProb + 5.0) && MathAbs(trendStrength) >= 0.15)
      return true;

   return false;
}

// ========================= PHASE 2 LEARNING =========================
void UpdateAdaptiveLearning()
{
   if(fullClosedTrades < 5)
      return;

   double winRate = (double)wins / (double)fullClosedTrades * 100.0;
   double profitFactor = (grossLoss > 0.0) ? (grossProfit / grossLoss) : 0.0;
   double curveFactor = GetEquityCurveFactor();

   bool weak  = (winRate < 45.0 || profitFactor < 1.0 || consecutiveLosses >= 3 || curveFactor < 0.70);
   bool strong = (winRate > 60.0 && profitFactor > 1.2 && consecutiveWins >= 3 && curveFactor > 1.02);

   if(weak)
   {
      dynamicMinProb         = MathMin(72.0, dynamicMinProb + 0.75);
      dynamicOffHoursMinProb = MathMin(80.0, dynamicOffHoursMinProb + 0.50);
      dynamicRiskFactor      = MathMax(0.50, dynamicRiskFactor - 0.05);
      dynamicScaleInMinProb   = MathMin(80.0, dynamicScaleInMinProb + 0.50);
      dynamicPyramidMinProb   = MathMin(80.0, dynamicPyramidMinProb + 0.50);
      dynamicTrailATRWeak     = MathMax(0.50, dynamicTrailATRWeak - 0.05);
      dynamicTrailATRStrong   = MathMax(0.70, dynamicTrailATRStrong - 0.03);
      dynamicTP2ExtendATR     = MathMax(2.00, dynamicTP2ExtendATR - 0.10);
      dynamicTP1_RR           = MathMax(0.80, dynamicTP1_RR - 0.05);
      dynamicTrailStartRR     = MathMax(1.00, dynamicTrailStartRR - 0.05);
   }
   else if(strong)
   {
      dynamicMinProb         = MathMax(MinProbability, dynamicMinProb - 0.25);
      dynamicOffHoursMinProb = MathMax(StrongProb, dynamicOffHoursMinProb - 0.25);
      dynamicRiskFactor      = MathMin(1.50, dynamicRiskFactor + 0.03);
      dynamicScaleInMinProb   = MathMax(ScaleInMinProb - 2.0, dynamicScaleInMinProb - 0.25);
      dynamicPyramidMinProb   = MathMax(PyramidMinProb, dynamicPyramidMinProb - 0.25);
      dynamicTrailATRWeak     = MathMin(BiasTrailATRWeak + 0.25, dynamicTrailATRWeak + 0.03);
      dynamicTrailATRStrong   = MathMin(BiasTrailATRStrong + 0.35, dynamicTrailATRStrong + 0.04);
      dynamicTP2ExtendATR     = MathMin(TP2_ExtendATR + 0.50, dynamicTP2ExtendATR + 0.10);
      dynamicTP1_RR           = MathMin(1.20, dynamicTP1_RR + 0.02);
      dynamicTrailStartRR     = MathMin(1.50, dynamicTrailStartRR + 0.02);
   }

   dynamicMinProb         = ClampDouble(dynamicMinProb, 50.0, 72.0);
   dynamicOffHoursMinProb = ClampDouble(dynamicOffHoursMinProb, 55.0, 80.0);
   dynamicRiskFactor      = ClampDouble(dynamicRiskFactor, 0.50, 1.50);
   dynamicScaleInMinProb   = ClampDouble(dynamicScaleInMinProb, 60.0, 80.0);
   dynamicPyramidMinProb   = ClampDouble(dynamicPyramidMinProb, 60.0, 80.0);
   dynamicTrailATRWeak     = ClampDouble(dynamicTrailATRWeak, 0.50, 1.20);
   dynamicTrailATRStrong   = ClampDouble(dynamicTrailATRStrong, 0.70, 1.50);
   dynamicTP2ExtendATR     = ClampDouble(dynamicTP2ExtendATR, 2.00, 4.00);
   dynamicTP1_RR           = ClampDouble(dynamicTP1_RR, 0.80, 1.20);
   dynamicTrailStartRR     = ClampDouble(dynamicTrailStartRR, 1.00, 1.50);

   Log(StringFormat("ADAPTIVE winRate=%.2f PF=%.2f curve=%.2f dynMin=%.2f dynRisk=%.2f dynOff=%.2f dynScaleIn=%.2f dynPyramid=%.2f TP1=%.2f TrailStart=%.2f trailW=%.2f trailS=%.2f tpExt=%.2f",
                    winRate, profitFactor, curveFactor, dynamicMinProb, dynamicRiskFactor, dynamicOffHoursMinProb,
                    dynamicScaleInMinProb, dynamicPyramidMinProb, dynamicTP1_RR, dynamicTrailStartRR,
                    dynamicTrailATRWeak, dynamicTrailATRStrong, dynamicTP2ExtendATR));
}

// ========================= JSON BUILD =========================
string BuildRequestJSON()
{
   double spreadPts = (SymbolInfoDouble(_Symbol, SYMBOL_ASK) - SymbolInfoDouble(_Symbol, SYMBOL_BID)) / SymbolPoint();
   double balance   = AccountInfoDouble(ACCOUNT_BALANCE);
   double equity    = AccountInfoDouble(ACCOUNT_EQUITY);
   double drawdown   = 0.0;

   if(balance > 0.0 && equity < balance)
      drawdown = ((balance - equity) / balance) * 100.0;

   string json = "{";
   json += "\"asset\":\"" + AssetName + "\",";
   json += "\"timeframe\":\"M1\",";
   json += "\"candles\":[";

   for(int i = g_candleCount - 1; i >= 0; i--)
   {
      json += "{";
      json += "\"time\":\"" + TimeToString(g_candles[i].time, TIME_DATE|TIME_MINUTES) + "\",";
      json += "\"open\":"  + DoubleToString(g_candles[i].open, 5) + ",";
      json += "\"high\":"  + DoubleToString(g_candles[i].high, 5) + ",";
      json += "\"low\":"   + DoubleToString(g_candles[i].low, 5) + ",";
      json += "\"close\":" + DoubleToString(g_candles[i].close, 5);
      json += "}";

      if(i > 0)
         json += ",";
   }

   json += "],";
   json += "\"spread_points\":" + DoubleToString(spreadPts, 2) + ",";
   json += "\"bid\":" + DoubleToString(SymbolInfoDouble(_Symbol, SYMBOL_BID), 5) + ",";
   json += "\"ask\":" + DoubleToString(SymbolInfoDouble(_Symbol, SYMBOL_ASK), 5) + ",";
   json += "\"confidence_threshold\":" + DoubleToString(MinProbability / 100.0, 4) + ",";
   json += "\"max_spread_points\":" + DoubleToString((double)MaxSpreadPoints, 2) + ",";
   json += "\"drawdown_pct\":" + DoubleToString(drawdown, 2) + ",";
   json += "\"daily_loss_pct\":0,";
   json += "\"winning_streak\":0,";
   json += "\"losing_streak\":0,";
   json += "\"bars_since_last_trade\":" + IntegerToString(BarsSinceLastEntry()) + ",";
   json += "\"open_positions\":" + IntegerToString(CountOurPositions()) + ",";
   json += "\"max_positions\":" + IntegerToString(MaxPositions) + ",";
   json += "\"min_persistence\":" + IntegerToString(SignalConfirmations) + ",";
   json += "\"cooldown_active\":" + string(BarsSinceLastEntry() < MinBarsBetweenEntries ? "true" : "false") + ",";
   json += "\"news_block\":false,";
   json += "\"signal_history\":" + BuildSignalHistoryJson() + ",";
   json += "\"session_preference\":\"" + string(AllowOffHours ? "Off-hours" : "Active") + "\",";
   json += "\"account_balance\":" + DoubleToString(balance, 2) + ",";
   json += "\"account_equity\":" + DoubleToString(equity, 2);
   json += "}";

   return json;
}

// ========================= JSON PARSE =========================
string Extract(string j, string key)
{
   string s = "\"" + key + "\":";
   int st = StringFind(j, s);
   if(st == -1)
      return "";

   st += StringLen(s);

   int en = StringFind(j, ",", st);
   if(en == -1)
      en = StringFind(j, "}", st);
   if(en == -1)
      return "";

   string v = StringSubstr(j, st, en - st);
   StringReplace(v, "\"", "");
   StringTrimLeft(v);
   StringTrimRight(v);
   return v;
}

bool SendRequest(string body, string &jsonOut, int &httpStatus, string &respHeaders)
{
   char post[];
   int len = StringToCharArray(body, post, 0, StringLen(body));

   uchar req[];
   ArrayResize(req, len);
   for(int i = 0; i < len; i++)
      req[i] = (uchar)post[i];

   for(int attempt = 1; attempt <= 2; attempt++)
   {
      uchar res[];
      string headers;
      ResetLastError();

      httpStatus = WebRequest(
         "POST",
         API_URL,
         "Content-Type: application/json\r\n",
         HTTPTimeoutMs,
         req,
         res,
         headers
      );

      respHeaders = headers;
      jsonOut = CharArrayToString(res);

      Log(StringFormat("HTTP=%d bytes=%d err=%d", httpStatus, ArraySize(res), GetLastError()));

      if(httpStatus != -1 && StringLen(jsonOut) > 5)
         return true;

      if(attempt == 1)
         Sleep(250);
   }

   return false;
}

// ========================= SIGNAL ENGINE =========================
bool GetSignal(SignalData &d)
{
   string body = BuildRequestJSON();

   int httpStatus = -1;
   string respHeaders = "";
   string json = "";

   if(!SendRequest(body, json, httpStatus, respHeaders))
   {
      Log("Empty server response");
      return false;
   }

   Log("RAW: " + json);

   string regime    = Extract(json, "regime");
   string elevatedS = Extract(json, "confirmed_elevated");
   bool elevated    = (elevatedS == "true");

   double prob = StringToDouble(Extract(json, "probability"));
   if(prob <= 1.0)
      prob *= 100.0;

   double cdi       = StringToDouble(Extract(json, "cdi"));
   double riskMult  = StringToDouble(Extract(json, "risk_multiplier"));
   if(riskMult <= 0.0)
      riskMult = 1.0;

   string session = "Active";
   if(StringFind(json, "\"session\":\"Off-hours\"") >= 0 || StringFind(json, "Off-hours") >= 0)
      session = "Off-hours";

   string bias  = GetBias();
   string sweep = GetLiquiditySweepSignal();
   double trendStrength  = GetTrendStrength();
   double momentumPoints = GetMomentumPoints();
   double avgRangePoints = GetAvgRangePoints();

   string sig = "NONE";
   bool strongContext = (prob >= StrongProb);
   bool mediumContext = (prob >= dynamicMinProb);

   // Tiered logic without requiring elevated as a gate
   if(regime == "low")
   {
      if(bias == "UP")
         sig = "BUY";
      else if(strongContext && sweep == "BULL")
         sig = "BUY";
      else if(mediumContext && trendStrength > -0.10)
         sig = "BUY";
      else if(mediumContext && momentumPoints > -150.0)
         sig = "BUY";
   }

   if(regime == "high")
   {
      if(bias == "DOWN")
         sig = "SELL";
      else if(strongContext && sweep == "BEAR")
         sig = "SELL";
      else if(mediumContext && trendStrength < 0.10)
         sig = "SELL";
      else if(mediumContext && momentumPoints < 150.0)
         sig = "SELL";
   }

   if(regime == "mid" && strongContext)
   {
      if(bias == "UP")
         sig = "BUY";
      else if(bias == "DOWN")
         sig = "SELL";
      else if(sweep == "BULL")
         sig = "BUY";
      else if(sweep == "BEAR")
         sig = "SELL";
      else if(trendStrength > 0.05)
         sig = "BUY";
      else if(trendStrength < -0.05)
         sig = "SELL";
   }

   // Momentum handling: only block runaway extremes
   double momentumAbs = MathAbs(momentumPoints);
   if(momentumAbs > 1200.0)
   {
      Log("Momentum EXTREME -> trend mode");

      if(momentumPoints < 0 && trendStrength < -0.10)
      {
         sig = "SELL";
         prob += 4.0;
      }
      else if(momentumPoints > 0 && trendStrength > 0.10)
      {
         sig = "BUY";
         prob += 4.0;
      }
      else
      {
         prob -= 4.0;
      }
   }
   else if(momentumAbs > 500.0)
   {
      Log("Momentum high -> reducing confidence");
      prob -= 3.0;
   }

   // Momentum alignment bonus
   if((momentumPoints > 0 && sig == "BUY") || (momentumPoints < 0 && sig == "SELL"))
      prob += 2.0;

   // Trend override when signal is still NONE
   if(sig == "NONE" && mediumContext)
   {
      if(trendStrength > 0.25)
      {
         sig = "BUY";
         Log("Trend override BUY");
      }
      else if(trendStrength < -0.25)
      {
         sig = "SELL";
         Log("Trend override SELL");
      }
   }

   double contextScore = ComputeContextScore(sig, bias, sweep, session, cdi, trendStrength, momentumPoints, avgRangePoints);
   double adjustedProb = ClampDouble(prob + (contextScore * 0.10), 0.0, 100.0);

   if(elevated)
   {
      adjustedProb += 2.5;
      Log("Elevated boost applied");
   }

   d.signal = sig;
   d.probability = prob;
   d.adjusted_probability = adjustedProb;
   d.regime = regime;
   d.cdi = cdi;
   d.risk_mult = riskMult;
   d.session = session;

   d.bias = bias;
   d.sweep = sweep;
   d.trend_strength = trendStrength;
   d.momentum_points = momentumPoints;
   d.avg_range_points = avgRangePoints;
   d.context_score = contextScore;

   Log(StringFormat("SIGNAL signal=%s prob=%.2f adj=%.2f regime=%s cdi=%.3f risk=%.3f session=%s bias=%s sweep=%s trend=%.3f mom=%.1f ctx=%.1f",
                    d.signal, d.probability, d.adjusted_probability, d.regime, d.cdi, d.risk_mult, d.session,
                    d.bias, d.sweep, d.trend_strength, d.momentum_points, d.context_score));

   return true;
}

// ========================= EXECUTION =========================
bool ModifyPositionByTicket(ulong ticket, double sl, double tp, string why)
{
   if(!PositionSelectByTicket(ticket))
      return false;

   string sym = PositionGetString(POSITION_SYMBOL);
   int digits = (int)SymbolInfoInteger(sym, SYMBOL_DIGITS);

   if(sl > 0.0) sl = NormalizeDouble(sl, digits);
   if(tp > 0.0) tp = NormalizeDouble(tp, digits);

   MqlTradeRequest req;
   MqlTradeResult  result;
   ZeroMemory(req);
   ZeroMemory(result);

   req.action   = TRADE_ACTION_SLTP;
   req.position = ticket;
   req.symbol   = sym;
   req.sl       = sl;
   req.tp       = tp;
   req.magic    = MagicNumber;

   bool sent = OrderSend(req, result);

   if(!sent || (result.retcode != TRADE_RETCODE_DONE && result.retcode != TRADE_RETCODE_DONE_PARTIAL))
   {
      Log(StringFormat("MODIFY failed ticket=%I64u ret=%d comment=%s reason=%s",
                       ticket, result.retcode, result.comment, why));
      return false;
   }

   Log(StringFormat("MODIFY ticket=%I64u SL=%.2f TP=%.2f | %s", ticket, sl, tp, why));
   return true;
}

bool ClosePartialByTicket(ulong ticket, double volume)
{
   if(!PositionSelectByTicket(ticket))
      return false;

   double vol = NormalizeVolumeToStep(volume);
   if(vol <= 0.0)
      return false;

   bool ok = trade.PositionClosePartial(ticket, vol);

   if(ok)
      Log(StringFormat("PARTIAL REQUEST ticket=%I64u vol=%.2f", ticket, vol));
   else
      Log(StringFormat("Partial close failed ticket=%I64u ret=%d desc=%s",
                       ticket, trade.ResultRetcode(), trade.ResultRetcodeDescription()));

   return ok;
}

bool OpenPosition(SignalData &d)
{
   ENUM_POSITION_TYPE ptype;
   if(d.signal == "BUY")
      ptype = POSITION_TYPE_BUY;
   else if(d.signal == "SELL")
      ptype = POSITION_TYPE_SELL;
   else
      return false;

   if(HasOppositePosition(ptype))
   {
      Log("Opposite position already open -> skip");
      return false;
   }

   bool sameDirection = HasOpenPositionType(ptype);
   int  ourPosCount   = CountOurPositions();

   if(ourPosCount >= MaxPositions)
   {
      Log("Max positions reached");
      return false;
   }

   if(sameDirection && (!EnableScaleIn || !EnablePyramiding))
   {
      Log("Pyramiding disabled -> skip same-direction add");
      return false;
   }

   if(lastEntryCandleTime > 0)
   {
      int secsSinceLastEntry = (int)(lastCandleTime - lastEntryCandleTime);
      if(secsSinceLastEntry < (MinBarsBetweenEntries * 60))
      {
         Log("Entry cooldown active");
         return false;
      }
   }

   double atr = GetATRValue();
   if(atr <= 0.0)
   {
      Log("ATR not ready");
      return false;
   }

   double lot = BaseLot * (d.adjusted_probability / 100.0) * d.risk_mult * dynamicRiskFactor;

   if(sameDirection)
   {
      double requiredProb = MathMax(dynamicScaleInMinProb, dynamicPyramidMinProb);

      if(d.adjusted_probability < requiredProb)
      {
         Log("Pyramid blocked: probability not strong enough");
         return false;
      }

      if(BarsSinceLastEntry() < PyramidCooldownBars)
      {
         Log("Pyramid cooldown active");
         return false;
      }

      double curveFactor = GetEquityCurveFactor();
      if(curveFactor < 0.70)
      {
         Log("Pyramid blocked: equity curve weak");
         return false;
      }

      double basketProfit = BasketProfit(ptype);
      bool basketHealthy = (basketProfit > PyramidMinBasketProfit) ||
                           (MathAbs(d.trend_strength) > 0.20) ||
                           (d.adjusted_probability >= (requiredProb + 3.0));

      if(!basketHealthy)
      {
         Log("Pyramid blocked: basket not healthy");
         return false;
      }

      double pyramidFactor = MathMax(ScaleInLotFactor, PyramidLotFactor);
      if(curveFactor > 1.05 && basketProfit > 0.0)
         pyramidFactor = MathMin(1.0, pyramidFactor + 0.10);

      lot *= pyramidFactor;
      Log("Pyramiding boost applied");
   }

   lot = AdjustRiskByEquity(lot);
   lot = NormalizeVolumeToStep(lot);

   if(lot <= 0.0)
   {
      Log("Lot size below minimum");
      return false;
   }

   double price = 0.0, sl = 0.0, tp = 0.0;

   if(d.signal == "BUY")
   {
      price = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
      sl    = price - (atr * ATR_Multiplier);
      tp    = price + (atr * ATR_Multiplier * 2.0);
   }
   else
   {
      price = SymbolInfoDouble(_Symbol, SYMBOL_BID);
      sl    = price + (atr * ATR_Multiplier);
      tp    = price - (atr * ATR_Multiplier * 2.0);
   }

   if(!StopDistancesValid(ptype, sl, tp))
   {
      Log("Entry rejected: invalid stop distances");
      return false;
   }

   bool ok = false;
   if(d.signal == "BUY")
      ok = trade.Buy(lot, _Symbol, price, sl, tp);
   else
      ok = trade.Sell(lot, _Symbol, price, sl, tp);

   if(!ok)
   {
      Log(StringFormat("OrderSend failed ret=%d desc=%s",
                       trade.ResultRetcode(), trade.ResultRetcodeDescription()));
      return false;
   }

   lastEntryCandleTime = lastCandleTime;

   Log(StringFormat("ENTRY REQUESTED signal=%s lot=%.2f adjProb=%.2f regime=%s cdi=%.3f session=%s bias=%s sweep=%s",
                    d.signal, lot, d.adjusted_probability, d.regime, d.cdi, d.session, d.bias, d.sweep));

   return true;
}

// ========================= MANAGEMENT =========================
void ManagePositions()
{
   double atr = GetATRValue();
   if(atr <= 0.0)
      return;

   string bias  = GetBias();
   string sweep = GetLiquiditySweepSignal();
   double momentumNow = GetMomentumPoints();
   double momentumAbs = MathAbs(momentumNow);
   double curveFactor = GetEquityCurveFactor();

   double trailCurveMult = 1.0;
   double tpCurveMult = 1.0;

   if(curveFactor < 0.70)
   {
      trailCurveMult = 0.80;
      tpCurveMult = 0.85;
   }
   else if(curveFactor > 1.05)
   {
      trailCurveMult = 1.05;
      tpCurveMult = 1.15;
   }

   double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   double pt  = SymbolPoint();

   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      ulong ticket = PositionGetTicket(i);
      if(!PositionSelectByTicket(ticket))
         continue;

      string sym = PositionGetString(POSITION_SYMBOL);
      long magic = (long)PositionGetInteger(POSITION_MAGIC);
      if(sym != _Symbol || magic != MagicNumber)
         continue;

      ENUM_POSITION_TYPE type = (ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE);
      double open = PositionGetDouble(POSITION_PRICE_OPEN);
      double sl   = PositionGetDouble(POSITION_SL);
      double tp   = PositionGetDouble(POSITION_TP);
      double vol  = PositionGetDouble(POSITION_VOLUME);

      double curPrice = (type == POSITION_TYPE_BUY) ? bid : ask;
      double risk = MathAbs(open - sl);
      if(risk <= 0.0)
         continue;

      double rr = MathAbs(curPrice - open) / risk;

      bool biasMatches  = (type == POSITION_TYPE_BUY  && bias == "UP")   || (type == POSITION_TYPE_SELL && bias == "DOWN");
      bool sweepAgainst = (type == POSITION_TYPE_BUY  && sweep == "BEAR") || (type == POSITION_TYPE_SELL && sweep == "BULL");
      bool sweepWith    = (type == POSITION_TYPE_BUY  && sweep == "BULL") || (type == POSITION_TYPE_SELL && sweep == "BEAR");

      // Multi-stage profit sharing: one partial exit only
      if(rr >= dynamicTP1_RR && !IsPartialClosed(ticket))
      {
         double partialVol = NormalizeVolumeToStep(vol / 2.0);
         if(partialVol > 0.0 && partialVol < vol)
         {
            if(ClosePartialByTicket(ticket, partialVol))
            {
               MarkPartialClosed(ticket);
               Log(StringFormat("PARTIAL TP ticket=%I64u rr=%.2f open=%.2f cur=%.2f vol=%.2f partial=%.2f",
                                ticket, rr, open, curPrice, vol, partialVol));
            }
         }
      }

      double desiredSL = sl;
      double desiredTP = tp;
      bool wantModify = false;
      string why = "";

      // Break-even protection once 1R is reached
      if(rr >= 1.0)
      {
         if(type == POSITION_TYPE_BUY)
         {
            if(open > desiredSL)
            {
               desiredSL = open;
               wantModify = true;
               why += "BE ";
            }
         }
         else
         {
            if(desiredSL == 0.0 || open < desiredSL)
            {
               desiredSL = open;
               wantModify = true;
               why += "BE ";
            }
         }
      }

      // Trailing and TP extension only when price has moved enough
      if(rr >= dynamicTrailStartRR)
      {
         if(type == POSITION_TYPE_BUY)
         {
            double trailATR = biasMatches ? dynamicTrailATRStrong : dynamicTrailATRWeak;
            trailATR *= trailCurveMult;

            double trailSL  = bid - (atr * trailATR);

            if(trailSL > desiredSL)
            {
               desiredSL = trailSL;
               wantModify = true;
               why += (biasMatches ? "TRAIL_STRONG " : "TRAIL_WEAK ");
            }

            if(biasMatches || sweepWith)
            {
               double extTP = bid + (atr * dynamicTP2ExtendATR * tpCurveMult);
               if(extTP > desiredTP)
               {
                  desiredTP = extTP;
                  wantModify = true;
                  why += "EXTEND_TP ";
               }
            }

            if((!biasMatches || sweepAgainst) && rr >= 1.0)
            {
               double tighter = bid - (atr * 0.50);
               if(tighter > desiredSL)
               {
                  desiredSL = tighter;
                  wantModify = true;
                  why += "WEAKEN_TIGHTEN ";
               }
            }
         }
         else // SELL
         {
            double trailATR = biasMatches ? dynamicTrailATRStrong : dynamicTrailATRWeak;
            trailATR *= trailCurveMult;

            double trailSL  = ask + (atr * trailATR);

            if(desiredSL == 0.0 || trailSL < desiredSL)
            {
               desiredSL = trailSL;
               wantModify = true;
               why += (biasMatches ? "TRAIL_STRONG " : "TRAIL_WEAK ");
            }

            if(biasMatches || sweepWith)
            {
               double extTP = ask - (atr * dynamicTP2ExtendATR * tpCurveMult);
               if(desiredTP == 0.0 || extTP < desiredTP)
               {
                  desiredTP = extTP;
                  wantModify = true;
                  why += "EXTEND_TP ";
               }
            }

            if((!biasMatches || sweepAgainst) && rr >= 1.0)
            {
               double tighter = ask + (atr * 0.50);
               if(desiredSL == 0.0 || tighter < desiredSL)
               {
                  desiredSL = tighter;
                  wantModify = true;
                  why += "WEAKEN_TIGHTEN ";
               }
            }
         }
      }

      // Extreme trend mode: let winners run
      if(rr >= 1.5 && momentumAbs > 1200.0)
      {
         if(type == POSITION_TYPE_BUY)
            desiredTP = bid + (atr * dynamicTP2ExtendATR * 1.50 * tpCurveMult);
         else
            desiredTP = ask - (atr * dynamicTP2ExtendATR * 1.50 * tpCurveMult);

         wantModify = true;
         why += "EXTREME_RUN ";
      }

      if(wantModify)
      {
         double minMove = MinModifyPoints * pt;

         bool slChanged = (desiredSL > 0.0 && MathAbs(desiredSL - sl) >= minMove);
         bool tpChanged = (desiredTP > 0.0 && MathAbs(desiredTP - tp) >= minMove);

         if(slChanged || tpChanged)
         {
            if(!StopDistancesValid(type, desiredSL, desiredTP))
            {
               Log(StringFormat("Modify skipped (invalid stops) ticket=%I64u rr=%.2f bias=%s sweep=%s", ticket, rr, bias, sweep));
               continue;
            }

            ModifyPositionByTicket(ticket, desiredSL, desiredTP, why);
         }
      }
   }
}

// ========================= TRADE JOURNAL =========================
string DealReasonToString(long reason)
{
   if(reason == DEAL_REASON_SL)      return "SL";
   if(reason == DEAL_REASON_TP)      return "TP";
   if(reason == DEAL_REASON_SO)      return "StopOut";
   if(reason == DEAL_REASON_EXPERT)  return "EA";
   if(reason == DEAL_REASON_CLIENT)  return "Manual";
   if(reason == DEAL_REASON_MOBILE)  return "Mobile";
   if(reason == DEAL_REASON_ROLLOVER) return "Rollover";
   return "Other";
}

string DealEntryToString(long entry)
{
   if(entry == DEAL_ENTRY_IN)     return "OPEN";
   if(entry == DEAL_ENTRY_OUT)    return "CLOSE";
   if(entry == DEAL_ENTRY_INOUT)  return "REVERSAL";
   if(entry == DEAL_ENTRY_OUT_BY) return "OUT_BY";
   return "UNKNOWN";
}

string DealTypeToString(long type)
{
   if(type == DEAL_TYPE_BUY)  return "BUY";
   if(type == DEAL_TYPE_SELL) return "SELL";
   return "OTHER";
}

void LogQualityStats()
{
   double winRate = (fullClosedTrades > 0) ? ((double)wins / (double)fullClosedTrades) * 100.0 : 0.0;
   double profitFactor = (grossLoss > 0.0) ? (grossProfit / grossLoss) : 0.0;

   Log(StringFormat("QUALITY exitDeals=%d fullTrades=%d partials=%d wins=%d losses=%d winRate=%.2f%% netAll=%.2f netFull=%.2f PF=%.2f",
                    totalClosedEvents, fullClosedTrades, partialExitEvents, wins, losses, winRate, netProfit, fullNetProfit, profitFactor));
}

void OnTradeTransaction(const MqlTradeTransaction &trans,
                        const MqlTradeRequest &request,
                        const MqlTradeResult &result)
{
   if(trans.type != TRADE_TRANSACTION_DEAL_ADD)
      return;

   if(!HistoryDealSelect(trans.deal))
      return;

   string sym = HistoryDealGetString(trans.deal, DEAL_SYMBOL);
   long magic = (long)HistoryDealGetInteger(trans.deal, DEAL_MAGIC);

   if(sym != _Symbol || magic != MagicNumber)
      return;

   long entry  = (long)HistoryDealGetInteger(trans.deal, DEAL_ENTRY);
   long reason = (long)HistoryDealGetInteger(trans.deal, DEAL_REASON);
   long dtype  = (long)HistoryDealGetInteger(trans.deal, DEAL_TYPE);

   ulong positionId = (ulong)HistoryDealGetInteger(trans.deal, DEAL_POSITION_ID);
   bool positionStillOpen = PositionSelectByTicket(positionId);

   double volume  = HistoryDealGetDouble(trans.deal, DEAL_VOLUME);
   double price   = HistoryDealGetDouble(trans.deal, DEAL_PRICE);
   double profit  = HistoryDealGetDouble(trans.deal, DEAL_PROFIT);
   double swap    = HistoryDealGetDouble(trans.deal, DEAL_SWAP);
   double comm    = HistoryDealGetDouble(trans.deal, DEAL_COMMISSION);
   datetime tm    = (datetime)HistoryDealGetInteger(trans.deal, DEAL_TIME);

   string entryTxt = DealEntryToString(entry);
   string reasonTxt = DealReasonToString(reason);
   string dealSide  = DealTypeToString(dtype);

   if(entry == DEAL_ENTRY_IN)
   {
      Log(StringFormat("OPEN  deal=%I64u side=%s vol=%.2f price=%.2f reason=%s time=%s",
                       trans.deal, dealSide, volume, price, reasonTxt, TimeToString(tm, TIME_DATE|TIME_SECONDS)));
   }
   else if(entry == DEAL_ENTRY_OUT || entry == DEAL_ENTRY_OUT_BY || entry == DEAL_ENTRY_INOUT)
   {
      totalClosedEvents++;
      netProfit += profit;

      if(positionStillOpen)
      {
         partialExitEvents++;
         Log(StringFormat("CLOSE deal=%I64u kind=PARTIAL_EXIT side=%s vol=%.2f price=%.2f P/L=%.2f swap=%.2f comm=%.2f reason=%s time=%s",
                          trans.deal, dealSide, volume, price, profit, swap, comm, reasonTxt, TimeToString(tm, TIME_DATE|TIME_SECONDS)));
      }
      else
      {
         fullClosedTrades++;
         fullNetProfit += profit;

         if(profit >= 0.0)
         {
            wins++;
            grossProfit += profit;
            consecutiveWins++;
            consecutiveLosses = 0;
         }
         else
         {
            losses++;
            grossLoss += MathAbs(profit);
            consecutiveLosses++;
            consecutiveWins = 0;
         }

         Log(StringFormat("CLOSE deal=%I64u kind=FULL_CLOSE side=%s vol=%.2f price=%.2f P/L=%.2f swap=%.2f comm=%.2f reason=%s time=%s",
                          trans.deal, dealSide, volume, price, profit, swap, comm, reasonTxt, TimeToString(tm, TIME_DATE|TIME_SECONDS)));

         UpdateEquityHistory();
         UpdateAdaptiveLearning();
      }

      LogQualityStats();
   }
   else
   {
      Log(StringFormat("DEAL deal=%I64u entry=%s side=%s vol=%.2f price=%.2f P/L=%.2f reason=%s time=%s",
                       trans.deal, entryTxt, dealSide, volume, price, profit, reasonTxt, TimeToString(tm, TIME_DATE|TIME_SECONDS)));
   }
}

// ========================= MAIN LOOP =========================
void OnTick()
{
   if(!IsNewCandle())
      return;

   RefreshCandleBuffer();
   UpdateEquityHistory();

   ManagePositions();

   double spreadPts = (SymbolInfoDouble(_Symbol, SYMBOL_ASK) - SymbolInfoDouble(_Symbol, SYMBOL_BID)) / SymbolPoint();
   if(spreadPts > MaxSpreadPoints)
   {
      Log(StringFormat("Blocked: spread too high (%.1f pts)", spreadPts));
      return;
   }

   SignalData d;
   if(!GetSignal(d))
   {
      PushRecentSignal("NONE");
      return;
   }

   PushRecentSignal(d.signal);

   if(d.signal == "NONE")
      return;

   if(d.adjusted_probability < dynamicMinProb)
   {
      Log(StringFormat("Rejected: adjusted probability too low (raw=%.2f adj=%.2f dynMin=%.2f)",
                       d.probability, d.adjusted_probability, dynamicMinProb));
      return;
   }

   if(d.cdi > MaxCDI)
   {
      Log(StringFormat("Rejected: CDI too high (%.3f)", d.cdi));
      return;
   }

   if(!IsSessionAllowed(d.session, d.adjusted_probability, d.signal, d.bias, d.sweep, d.trend_strength))
   {
      Log(StringFormat("Rejected: off-hours condition not strong enough (adj=%.2f dynOff=%.2f bias=%s trend=%.3f)",
                       d.adjusted_probability, dynamicOffHoursMinProb, d.bias, d.trend_strength));
      return;
   }

   if(signalCount < SignalConfirmations)
   {
      signalBuffer[signalCount % SignalConfirmations] = d.signal;
      signalCount++;
      Log("Waiting confirmation...");
      return;
   }

   signalBuffer[signalCount % SignalConfirmations] = d.signal;
   signalCount++;

   bool confirmed = true;
   for(int i = 1; i < SignalConfirmations; i++)
   {
      if(signalBuffer[i] != signalBuffer[0])
      {
         confirmed = false;
         break;
      }
   }

   if(!confirmed)
   {
      Log("Waiting confirmation...");
      return;
   }

   OpenPosition(d);
}