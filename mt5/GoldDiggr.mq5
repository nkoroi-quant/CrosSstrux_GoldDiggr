// GoldDiggr.mq5 - Updated with OnTimer(1s), visual panel for drift/fallback

#property strict
#property description "GoldDiggr v3.1 - CrossStrux EA"

#include <Trade\Trade.mqh>
CTrade trade;

input string API_URL = "http://localhost:8000/analyze";
input string API_KEY = "your-key"; // optional

// Panel objects
string PanelLabel = "GoldDiggrPanel";

void OnInit() {
   EventSetTimer(1); // 1 second updates
   CreatePanel();
}

void OnTimer() {
   // fetch latest candles from MT5, send to /analyze, parse response, execute
   // update panel with regime, drift status, fallback
   if (drift_psi > 0.15) ObjectSetString(0, PanelLabel+"Drift", OBJPROP_TEXT, "DRIFT HIGH - FALLBACK");
   // ... full trading logic with error handling + timeouts
}

void CreatePanel() {
   // visual objects for regime, probability, drift, etc.
}

void OnDeinit(const int reason) {
   EventKillTimer();
   ObjectsDeleteAll(0, "GoldDiggr");
}