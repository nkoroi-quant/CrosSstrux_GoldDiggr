//+------------------------------------------------------------------+
//|               GoldDiggr_Integrated_v10.2.mq5                     |
//|  Full rich context engine with flattened stability improvements |
//+------------------------------------------------------------------+
#property copyright "nkoroi-quant + integration"
#property link      "https://github.com/nkoroi-quant/CrosSstrux_GoldDiggr"
#property version   "10.2"
#property strict

#include <Trade/Trade.mqh>

CTrade trade;

// ========================= INPUTS =========================
input string API_URL               = "http://127.0.0.1:8000/analyze";
input string API_KEY               = "";                    // Optional
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

input bool   UsePanel              = true;

// ========================= STATE =========================
#define MAX_CANDLES 30
#define EQUITY_HISTORY_SIZE 32

MqlRates g_candles[];
int      g_candleCount = 0;

datetime lastCandleTime      = 0;
datetime lastEntryCandleTime = 0;

string signalBuffer[5];
int    signalCount = 0;

string recentSignals[5];
int    recentSignalCount = 0;

ulong  partialClosedTickets[128];

// Equity tracking
double equityHistory[EQUITY_HISTORY_SIZE];
int    equityHistoryCount = 0;
double equityPeak = 0.0;

// Trade quality tracking
int    totalClosedEvents = 0;
int    fullClosedTrades  = 0;
int    partialExitEvents = 0;
int    wins = 0, losses = 0;
double grossProfit = 0.0, grossLoss = 0.0;
double fullNetProfit = 0.0, netProfit = 0.0;

// Adaptive learning
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

string PanelPrefix = "GoldDiggr_";

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
void Log(string msg) { Print("[GoldDiggr] ", msg); }

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

   EventSetTimer(1);
   if(UsePanel) CreatePanel();

   Log("GoldDiggr v10.2 Integrated initialized successfully");
   return INIT_SUCCEEDED;
}

void OnDeinit(const int reason)
{
   EventKillTimer();
   if(atrHandle != INVALID_HANDLE) IndicatorRelease(atrHandle);
   if(UsePanel) ObjectsDeleteAll(0, PanelPrefix);
}

// ========================= PANEL =========================
void CreatePanel() { Log("Panel enabled (basic)"); }
void UpdatePanel(string regime, double prob, bool elevated) { /* expand later if needed */ }

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
   return (pt > 0.0) ? pt : 0.01;
}

bool RefreshCandleBuffer()
{
   ArraySetAsSeries(g_candles, true);
   g_candleCount = CopyRates(_Symbol, PERIOD_M1, 1, MAX_CANDLES, g_candles);
   return (g_candleCount >= 20);
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

int CountOurPositions()
{
   int count = 0;
   for(int i = 0; i < PositionsTotal(); i++)
   {
      ulong ticket = PositionGetTicket(i);
      if(PositionSelectByTicket(ticket))
         if(PositionGetString(POSITION_SYMBOL) == _Symbol && PositionGetInteger(POSITION_MAGIC) == MagicNumber)
            count++;
   }
   return count;
}

bool HasOpenPositionType(ENUM_POSITION_TYPE ptype)
{
   for(int i = 0; i < PositionsTotal(); i++)
   {
      ulong ticket = PositionGetTicket(i);
      if(PositionSelectByTicket(ticket))
         if(PositionGetString(POSITION_SYMBOL) == _Symbol && 
            PositionGetInteger(POSITION_MAGIC) == MagicNumber && 
            (ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE) == ptype)
            return true;
   }
   return false;
}

bool HasOppositePosition(ENUM_POSITION_TYPE ptype)
{
   for(int i = 0; i < PositionsTotal(); i++)
   {
      ulong ticket = PositionGetTicket(i);
      if(PositionSelectByTicket(ticket))
         if(PositionGetString(POSITION_SYMBOL) == _Symbol && 
            PositionGetInteger(POSITION_MAGIC) == MagicNumber && 
            (ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE) != ptype)
            return true;
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
      if(partialClosedTickets[i] == 0)
      {
         partialClosedTickets[i] = ticket;
         return;
      }
}

double BasketProfit(ENUM_POSITION_TYPE ptype)
{
   double sum = 0.0;
   for(int i = 0; i < PositionsTotal(); i++)
   {
      ulong ticket = PositionGetTicket(i);
      if(PositionSelectByTicket(ticket))
         if(PositionGetString(POSITION_SYMBOL) == _Symbol && 
            PositionGetInteger(POSITION_MAGIC) == MagicNumber && 
            (ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE) == ptype)
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
      if(recentSignals[idx] != "")
      {
         if(!first) json += ",";
         json += "\"" + recentSignals[idx] + "\"";
         first = false;
      }
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
   if(lastEntryCandleTime == 0) return 999;
   int secs = (int)(lastCandleTime - lastEntryCandleTime);
   return (secs > 0) ? secs / 60 : 0;
}

double NormalizeVolumeToStep(double volume)
{
   double minVol = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
   double step   = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
   double maxVol = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);
   if(step <= 0.0) step = 0.01;
   volume = MathFloor(volume / step) * step;
   volume = MathMax(volume, 0.0);
   volume = MathMin(volume, maxVol);
   return (volume < minVol) ? 0.0 : NormalizeDouble(volume, 2);
}

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

bool StopDistancesValid(ENUM_POSITION_TYPE type, double sl, double tp)
{
   double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   double pt  = SymbolPoint();
   int stopsLevel = (int)SymbolInfoInteger(_Symbol, SYMBOL_TRADE_STOPS_LEVEL);
   int minLevelPts = stopsLevel + 2;
   double minDist = minLevelPts * pt;

   if(type == POSITION_TYPE_BUY)
   {
      if(sl > 0.0 && (bid - sl) < minDist) return false;
      if(tp > 0.0 && (tp - ask) < minDist) return false;
   }
   else
   {
      if(sl > 0.0 && (sl - ask) < minDist) return false;
      if(tp > 0.0 && (bid - tp) < minDist) return false;
   }
   return true;
}

// ========================= CONTEXT ENGINE =========================
double GetTrendStrength()
{
   if(g_candleCount < 3) return 0.0;
   int up = 0, down = 0;
   for(int i = g_candleCount - 1; i > 0; i--)
   {
      if(g_candles[i-1].close > g_candles[i].close) up++;
      else if(g_candles[i-1].close < g_candles[i].close) down++;
   }
   return (double)(up - down) / (g_candleCount - 1);
}

double GetMomentumPoints()
{
   if(g_candleCount < 6) return 0.0;
   double pt = SymbolPoint();
   return (g_candles[0].close - g_candles[5].close) / pt;
}

double GetAvgRangePoints()
{
   if(g_candleCount <= 0) return 0.0;
   double pt = SymbolPoint();
   double sum = 0.0;
   for(int i = 0; i < g_candleCount; i++)
      sum += (g_candles[i].high - g_candles[i].low) / pt;
   return sum / g_candleCount;
}

string GetBias()
{
   if(g_candleCount < 4) return "NEUTRAL";
   if(g_candles[0].high > g_candles[1].high && g_candles[1].high > g_candles[2].high &&
      g_candles[0].low  > g_candles[1].low  && g_candles[1].low  > g_candles[2].low)
      return "UP";
   if(g_candles[0].high < g_candles[1].high && g_candles[1].high < g_candles[2].high &&
      g_candles[0].low  < g_candles[1].low  && g_candles[1].low  < g_candles[2].low)
      return "DOWN";
   return "NEUTRAL";
}

string GetLiquiditySweepSignal()
{
   if(g_candleCount < 6) return "NONE";
   double pt = SymbolPoint();
   double sweepBuffer = 10.0 * pt;
   double prevHigh = g_candles[1].high;
   double prevLow  = g_candles[1].low;
   for(int i = 2; i <= 5; i++)
   {
      prevHigh = MathMax(prevHigh, g_candles[i].high);
      prevLow  = MathMin(prevLow, g_candles[i].low);
   }
   if(g_candles[0].low < (prevLow - sweepBuffer) && g_candles[0].close > prevLow) return "BULL";
   if(g_candles[0].high > (prevHigh + sweepBuffer) && g_candles[0].close < prevHigh) return "BEAR";
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
   if(MathAbs(trendStrength) > 0.15) score += 10.0;
   if(MathAbs(momentumPoints) > 8.0) score += 10.0;
   if(avgRangePoints > 80.0 && avgRangePoints < 250.0) score += 5.0;
   else if(avgRangePoints > 400.0) score -= 5.0;
   if(cdi < 0.35) score += 10.0;
   else if(cdi > MaxCDI) score -= 25.0;
   if(session == "Off-hours") score -= 10.0;
   return score;
}

bool IsSessionAllowed(string session, double adjustedProb, string signal, string bias, string sweep, double trendStrength)
{
   if(session != "Off-hours") return true;
   if(adjustedProb >= dynamicOffHoursMinProb) return true;
   if(signal == "BUY"  && (bias == "UP"   || sweep == "BULL" || trendStrength > 0.15)) return true;
   if(signal == "SELL" && (bias == "DOWN" || sweep == "BEAR" || trendStrength < -0.15)) return true;
   if(AllowOffHours && adjustedProb >= (dynamicMinProb + 5.0) && MathAbs(trendStrength) >= 0.15) return true;
   return false;
}

// ========================= ADAPTIVE LEARNING =========================
void UpdateAdaptiveLearning()
{
   if(fullClosedTrades < 5) return;
   double winRate = (double)wins / fullClosedTrades * 100.0;
   double profitFactor = (grossLoss > 0.0) ? grossProfit / grossLoss : 0.0;
   double curveFactor = GetEquityCurveFactor();

   bool weak  = (winRate < 45.0 || profitFactor < 1.0 || consecutiveLosses >= 3 || curveFactor < 0.70);
   bool strong = (winRate > 60.0 && profitFactor > 1.2 && consecutiveWins >= 3 && curveFactor > 1.02);

   if(weak)
   {
      dynamicMinProb         = MathMin(72.0, dynamicMinProb + 0.75);
      dynamicOffHoursMinProb = MathMin(80.0, dynamicOffHoursMinProb + 0.50);
      dynamicRiskFactor      = MathMax(0.50, dynamicRiskFactor - 0.05);
      dynamicScaleInMinProb  = MathMin(80.0, dynamicScaleInMinProb + 0.50);
      dynamicPyramidMinProb  = MathMin(80.0, dynamicPyramidMinProb + 0.50);
      dynamicTrailATRWeak    = MathMax(0.50, dynamicTrailATRWeak - 0.05);
      dynamicTrailATRStrong  = MathMax(0.70, dynamicTrailATRStrong - 0.03);
      dynamicTP2ExtendATR    = MathMax(2.00, dynamicTP2ExtendATR - 0.10);
      dynamicTP1_RR          = MathMax(0.80, dynamicTP1_RR - 0.05);
      dynamicTrailStartRR    = MathMax(1.00, dynamicTrailStartRR - 0.05);
   }
   else if(strong)
   {
      dynamicMinProb         = MathMax(MinProbability, dynamicMinProb - 0.25);
      dynamicOffHoursMinProb = MathMax(StrongProb, dynamicOffHoursMinProb - 0.25);
      dynamicRiskFactor      = MathMin(1.50, dynamicRiskFactor + 0.03);
      dynamicScaleInMinProb  = MathMax(ScaleInMinProb - 2.0, dynamicScaleInMinProb - 0.25);
      dynamicPyramidMinProb  = MathMax(PyramidMinProb, dynamicPyramidMinProb - 0.25);
      dynamicTrailATRWeak    = MathMin(BiasTrailATRWeak + 0.25, dynamicTrailATRWeak + 0.03);
      dynamicTrailATRStrong  = MathMin(BiasTrailATRStrong + 0.35, dynamicTrailATRStrong + 0.04);
      dynamicTP2ExtendATR    = MathMin(TP2_ExtendATR + 0.50, dynamicTP2ExtendATR + 0.10);
      dynamicTP1_RR          = MathMin(1.20, dynamicTP1_RR + 0.02);
      dynamicTrailStartRR    = MathMin(1.50, dynamicTrailStartRR + 0.02);
   }

   dynamicMinProb         = ClampDouble(dynamicMinProb, 50.0, 72.0);
   dynamicOffHoursMinProb = ClampDouble(dynamicOffHoursMinProb, 55.0, 80.0);
   dynamicRiskFactor      = ClampDouble(dynamicRiskFactor, 0.50, 1.50);
   dynamicScaleInMinProb  = ClampDouble(dynamicScaleInMinProb, 60.0, 80.0);
   dynamicPyramidMinProb  = ClampDouble(dynamicPyramidMinProb, 60.0, 80.0);
   dynamicTrailATRWeak    = ClampDouble(dynamicTrailATRWeak, 0.50, 1.20);
   dynamicTrailATRStrong  = ClampDouble(dynamicTrailATRStrong, 0.70, 1.50);
   dynamicTP2ExtendATR    = ClampDouble(dynamicTP2ExtendATR, 2.00, 4.00);
   dynamicTP1_RR          = ClampDouble(dynamicTP1_RR, 0.80, 1.20);
   dynamicTrailStartRR    = ClampDouble(dynamicTrailStartRR, 1.00, 1.50);

   Log(StringFormat("ADAPTIVE winRate=%.2f PF=%.2f curve=%.2f dynMin=%.2f", winRate, profitFactor, curveFactor, dynamicMinProb));
}

// ========================= JSON & REQUEST =========================
string BuildRequestJSON()
{
   MqlRates rates[];
   ArraySetAsSeries(rates, true);
   int copied = CopyRates(_Symbol, PERIOD_M1, 0, MAX_CANDLES, rates);
   if(copied < 20)
      copied = MathMax(20, g_candleCount);

   string json = "{\"asset\":\"" + AssetName + "\",\"timeframe\":\"M1\",\"candles\":[";

   for(int i = 0; i < copied; i++)
   {
      string candle = StringFormat("{\"time\":\"%s\",\"open\":%.5f,\"high\":%.5f,\"low\":%.5f,\"close\":%.5f}",
                                   TimeToString(rates[i].time, TIME_DATE|TIME_MINUTES),
                                   rates[i].open, rates[i].high, rates[i].low, rates[i].close);
      json += candle;
      if(i < copied - 1)
         json += ",";
   }

   json += "],";   // close candles array

   double spreadPts = (SymbolInfoDouble(_Symbol, SYMBOL_ASK) - SymbolInfoDouble(_Symbol, SYMBOL_BID)) / SymbolPoint();
   double balance   = AccountInfoDouble(ACCOUNT_BALANCE);
   double equity    = AccountInfoDouble(ACCOUNT_EQUITY);
   double dd        = (balance > 0.0 && equity < balance) ? ((balance - equity)/balance)*100.0 : 0.0;

   // Correct field list with proper commas
   json += "\"spread_points\":"      + DoubleToString(spreadPts, 2) + ",";
   json += "\"bid\":"                + DoubleToString(SymbolInfoDouble(_Symbol, SYMBOL_BID), 5) + ",";
   json += "\"ask\":"                + DoubleToString(SymbolInfoDouble(_Symbol, SYMBOL_ASK), 5) + ",";
   json += "\"confidence_threshold\":" + DoubleToString(MinProbability / 100.0, 4) + ",";
   json += "\"max_spread_points\":"   + IntegerToString(MaxSpreadPoints) + ",";
   json += "\"drawdown_pct\":"        + DoubleToString(dd, 2) + ",";
   json += "\"bars_since_last_trade\":" + IntegerToString(BarsSinceLastEntry()) + ",";
   json += "\"open_positions\":"      + IntegerToString(CountOurPositions()) + ",";
   json += "\"max_positions\":"       + IntegerToString(MaxPositions) + ",";
   
   // FIXED: Proper key for signal_history
   json += "\"signal_history\":"      + BuildSignalHistoryJson() + ",";
   
   json += "\"session_preference\":\"" + (AllowOffHours ? "Off-hours" : "Active") + "\",";
   json += "\"account_balance\":"     + DoubleToString(balance, 2) + ",";
   json += "\"account_equity\":"      + DoubleToString(equity, 2);

   json += "}";   // Final closing brace

   // Debug - keep for now
   Print("[DEBUG JSON] Length=", StringLen(json), " | Last 120 chars: ", StringSubstr(json, StringLen(json)-120));

   return json;
}

string Extract(string j, string key)
{
   string s = "\"" + key + "\":";
   int st = StringFind(j, s);
   if(st == -1) return "";
   st += StringLen(s);
   int en = StringFind(j, ",", st);
   if(en == -1) en = StringFind(j, "}", st);
   if(en == -1) en = StringLen(j);
   string v = StringSubstr(j, st, en - st);
   StringReplace(v, "\"", "");
   StringTrimLeft(v); 
   StringTrimRight(v);
   return v;
}

bool SendRequest(string body, string &jsonOut)
{
   char post[], res[];
   string resp_headers = "";

   string headers = "Content-Type: application/json\r\n";
   if(API_KEY != "") 
      headers += "X-API-Key: " + API_KEY + "\r\n";

   StringToCharArray(body, post);

   int status = WebRequest("POST", API_URL, headers, HTTPTimeoutMs, post, res, resp_headers);
   jsonOut = CharArrayToString(res);

   if(status != 200)
      Log(StringFormat("ERROR HTTP=%d | Response: %s", status, jsonOut));
   else
      Log(StringFormat("SUCCESS HTTP=200 | bytes=%d", ArraySize(res)));

   return (status == 200 && StringLen(jsonOut) > 5);
}

// ========================= SIGNAL ENGINE =========================
bool GetSignal(SignalData &d)
{
   string body = BuildRequestJSON();
   string json = "";
   if(!SendRequest(body, json))
   {
      Log("Empty server response");
      return false;
   }

   Log("RAW: " + json);

   string regime = Extract(json, "regime");
   double prob   = StringToDouble(Extract(json, "probability"));
   if(prob <= 1.0) prob *= 100.0;

   double cdi      = StringToDouble(Extract(json, "cdi"));
   double riskMult = StringToDouble(Extract(json, "risk_multiplier"));
   if(riskMult <= 0.0) riskMult = 1.0;

   string session = (StringFind(json, "Off-hours") >= 0) ? "Off-hours" : "Active";

   string bias  = GetBias();
   string sweep = GetLiquiditySweepSignal();
   double trendStrength  = GetTrendStrength();
   double momentumPoints = GetMomentumPoints();
   double avgRangePoints = GetAvgRangePoints();

   string sig = "NONE";
   bool strongContext = (prob >= StrongProb);
   bool mediumContext = (prob >= dynamicMinProb);

   if(regime == "low")
   {
      if(bias == "UP" || (strongContext && sweep == "BULL") || (mediumContext && trendStrength > -0.10))
         sig = "BUY";
   }
   if(regime == "high")
   {
      if(bias == "DOWN" || (strongContext && sweep == "BEAR") || (mediumContext && trendStrength < 0.10))
         sig = "SELL";
   }
   if(regime == "mid" && strongContext)
   {
      if(bias == "UP") sig = "BUY";
      else if(bias == "DOWN") sig = "SELL";
      else if(sweep == "BULL") sig = "BUY";
      else if(sweep == "BEAR") sig = "SELL";
      else if(trendStrength > 0.05) sig = "BUY";
      else if(trendStrength < -0.05) sig = "SELL";
   }

   double contextScore = ComputeContextScore(sig, bias, sweep, session, cdi, trendStrength, momentumPoints, avgRangePoints);
   double adjustedProb = ClampDouble(prob + contextScore * 0.10, 0.0, 100.0);

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

   Log(StringFormat("SIGNAL signal=%s prob=%.2f adj=%.2f regime=%s", sig, prob, adjustedProb, regime));
   return true;
}

// ========================= EXECUTION =========================
bool ModifyPositionByTicket(ulong ticket, double sl, double tp, string why)
{
   if(!PositionSelectByTicket(ticket)) return false;
   string sym = PositionGetString(POSITION_SYMBOL);
   int digits = (int)SymbolInfoInteger(sym, SYMBOL_DIGITS);
   if(sl > 0.0) sl = NormalizeDouble(sl, digits);
   if(tp > 0.0) tp = NormalizeDouble(tp, digits);

   MqlTradeRequest req; MqlTradeResult result;
   ZeroMemory(req); ZeroMemory(result);
   req.action   = TRADE_ACTION_SLTP;
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
   Log(StringFormat("MODIFY ticket=%I64u SL=%.2f TP=%.2f | %s", ticket, sl, tp, why));
   return true;
}

bool ClosePartialByTicket(ulong ticket, double volume)
{
   if(!PositionSelectByTicket(ticket)) return false;
   double vol = NormalizeVolumeToStep(volume);
   if(vol <= 0.0) return false;
   bool ok = trade.PositionClosePartial(ticket, vol);
   if(ok)
      Log(StringFormat("PARTIAL CLOSE ticket=%I64u vol=%.2f", ticket, vol));
   else
      Log(StringFormat("Partial close failed ticket=%I64u ret=%d", ticket, trade.ResultRetcode()));
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

   if(sameDirection && (!EnableScaleIn && !EnablePyramiding))
   {
      Log("Pyramiding/scale-in disabled");
      return false;
   }

   if(BarsSinceLastEntry() < MinBarsBetweenEntries)
   {
      Log("Entry cooldown active");
      return false;
   }

   double atr = GetATRValue();
   if(atr <= 0.0) { Log("ATR not ready"); return false; }

   double lot = BaseLot * (d.adjusted_probability / 100.0) * d.risk_mult * dynamicRiskFactor;

   if(sameDirection)
   {
      double requiredProb = MathMax(dynamicScaleInMinProb, dynamicPyramidMinProb);
      if(d.adjusted_probability < requiredProb)
      {
         Log("Pyramid blocked: probability too low");
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
      bool basketHealthy = (basketProfit > PyramidMinBasketProfit) || (MathAbs(d.trend_strength) > 0.20) || (d.adjusted_probability >= requiredProb + 3.0);
      if(!basketHealthy)
      {
         Log("Pyramid blocked: basket not healthy");
         return false;
      }
      double pyramidFactor = MathMax(ScaleInLotFactor, PyramidLotFactor);
      if(curveFactor > 1.05 && basketProfit > 0.0) pyramidFactor = MathMin(1.0, pyramidFactor + 0.10);
      lot *= pyramidFactor;
      Log("Pyramiding applied");
   }

   lot = AdjustRiskByEquity(lot);
   lot = NormalizeVolumeToStep(lot);
   if(lot <= 0.0) { Log("Lot size too small"); return false; }

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
      Log("Entry rejected: invalid stop distances");
      return false;
   }

   bool ok = (d.signal == "BUY") ? trade.Buy(lot, _Symbol, price, sl, tp) : trade.Sell(lot, _Symbol, price, sl, tp);
   if(!ok)
   {
      Log(StringFormat("OrderSend failed ret=%d", trade.ResultRetcode()));
      return false;
   }

   lastEntryCandleTime = lastCandleTime;
   Log(StringFormat("ENTRY %s lot=%.2f adjProb=%.2f", d.signal, lot, d.adjusted_probability));
   return true;
}

// ========================= MANAGEMENT =========================
void ManagePositions()
{
   double atr = GetATRValue();
   if(atr <= 0.0) return;

   string bias = GetBias();
   string sweep = GetLiquiditySweepSignal();
   double momentumAbs = MathAbs(GetMomentumPoints());
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

      // Partial TP1
      if(rr >= dynamicTP1_RR && !IsPartialClosed(ticket))
      {
         double partialVol = NormalizeVolumeToStep(vol / 2.0);
         if(partialVol > 0.0 && partialVol < vol)
            if(ClosePartialByTicket(ticket, partialVol))
               MarkPartialClosed(ticket);
      }

      double desiredSL = sl;
      double desiredTP = tp;
      bool wantModify = false;
      string why = "";

      // Break-even
      if(rr >= 1.0)
      {
         if(type == POSITION_TYPE_BUY && open > desiredSL)
         {
            desiredSL = open; wantModify = true; why += "BE ";
         }
         else if(type == POSITION_TYPE_SELL && (desiredSL == 0.0 || open < desiredSL))
         {
            desiredSL = open; wantModify = true; why += "BE ";
         }
      }

      // Trailing & TP extension
      if(rr >= dynamicTrailStartRR)
      {
         double trailATR = biasMatches ? dynamicTrailATRStrong : dynamicTrailATRWeak;
         trailATR *= trailCurveMult;

         if(type == POSITION_TYPE_BUY)
         {
            double trailSL = bid - atr * trailATR;
            if(trailSL > desiredSL)
            {
               desiredSL = trailSL; wantModify = true; why += (biasMatches ? "TRAIL_STRONG " : "TRAIL_WEAK ");
            }
            if(biasMatches || sweepWith)
            {
               double extTP = bid + atr * dynamicTP2ExtendATR * tpCurveMult;
               if(extTP > desiredTP)
               {
                  desiredTP = extTP; wantModify = true; why += "EXTEND_TP ";
               }
            }
         }
         else // SELL
         {
            double trailSL = ask + atr * trailATR;
            if(desiredSL == 0.0 || trailSL < desiredSL)
            {
               desiredSL = trailSL; wantModify = true; why += (biasMatches ? "TRAIL_STRONG " : "TRAIL_WEAK ");
            }
            if(biasMatches || sweepWith)
            {
               double extTP = ask - atr * dynamicTP2ExtendATR * tpCurveMult;
               if(desiredTP == 0.0 || extTP < desiredTP)
               {
                  desiredTP = extTP; wantModify = true; why += "EXTEND_TP ";
               }
            }
         }
      }

      if(wantModify)
      {
         double minMove = MinModifyPoints * pt;
         bool slChanged = (desiredSL > 0.0 && MathAbs(desiredSL - sl) >= minMove);
         bool tpChanged = (desiredTP > 0.0 && MathAbs(desiredTP - tp) >= minMove);
         if(slChanged || tpChanged)
            if(StopDistancesValid(type, desiredSL, desiredTP))
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
   long reason = HistoryDealGetInteger(trans.deal, DEAL_REASON);
   double profit = HistoryDealGetDouble(trans.deal, DEAL_PROFIT);
   ulong positionId = (ulong)HistoryDealGetInteger(trans.deal, DEAL_POSITION_ID);
   bool positionStillOpen = PositionSelectByTicket(positionId);

   if(entry == DEAL_ENTRY_IN)
   {
      // open log
   }
   else if(entry == DEAL_ENTRY_OUT || entry == DEAL_ENTRY_OUT_BY || entry == DEAL_ENTRY_INOUT)
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
         UpdateEquityHistory();
         UpdateAdaptiveLearning();
      }
   }
}

// ========================= MAIN TIMER =========================
void OnTimer()
{
   if(!RefreshCandleBuffer()) return;

   UpdateEquityHistory();
   ManagePositions();

   double spreadPts = (SymbolInfoDouble(_Symbol, SYMBOL_ASK) - SymbolInfoDouble(_Symbol, SYMBOL_BID)) / SymbolPoint();
   if(spreadPts > MaxSpreadPoints)
   {
      Log(StringFormat("Spread too high (%.1f pts)", spreadPts));
      return;
   }

   SignalData d;
   if(!GetSignal(d))
   {
      PushRecentSignal("NONE");
      return;
   }

   PushRecentSignal(d.signal);

   if(d.signal == "NONE") return;

   if(d.adjusted_probability < dynamicMinProb || d.cdi > MaxCDI ||
      !IsSessionAllowed(d.session, d.adjusted_probability, d.signal, d.bias, d.sweep, d.trend_strength))
      return;

   if(signalCount < SignalConfirmations)
   {
      signalBuffer[signalCount % SignalConfirmations] = d.signal;
      signalCount++;
      return;
   }

   signalBuffer[signalCount % SignalConfirmations] = d.signal;
   signalCount++;

   bool confirmed = true;
   for(int i = 1; i < SignalConfirmations; i++)
      if(signalBuffer[i] != signalBuffer[0]) { confirmed = false; break; }

   if(confirmed)
      OpenPosition(d);

   if(UsePanel) UpdatePanel(d.regime, d.adjusted_probability, false);
}