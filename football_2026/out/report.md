# 2026 FIFA World Cup — Predictive Model Report (post-Match Day 2)

**Model**: ELO (FiveThirtyEight-style) + attack/defense ratings + Poisson goal model.

**Bracket**: Official FIFA 2026 R32 bracket (from schedule PDF v17).

**Data**: international results 1872 → 2026-06-15 (49,437 matches including 16 actual MD1-2 results).

**Backtested** on 2018 WC, Euro 2020, 2022 WC, Euro 2024, Copa America 2024: Brier 0.55-0.65, accuracy 45-56%.

**Monte Carlo**: 20,000 full tournament simulations from the post-MD2 state.


## 1. Top 15 ELO (post-MD2, 2026-06-16)

| Rank | Team | ELO |
|---:|---|---:|
| 1 | Spain | 2202 |
| 2 | Argentina | 2200 |
| 3 | France | 2136 |
| 4 | England | 2088 |
| 5 | Brazil | 2079 |
| 6 | Colombia | 2067 |
| 7 | Portugal | 2061 |
| 8 | Germany | 2037 |
| 9 | Morocco | 2020 |
| 10 | Japan | 2020 |
| 11 | Norway | 2007 |
| 12 | Ecuador | 2006 |
| 13 | Netherlands | 2005 |
| 14 | Mexico | 1982 |
| 15 | Belgium | 1976 |

## 2. Backtest on the 16 actual results (MD1-2)

How the model would have predicted each match, using only data from before the match.

| Date | Match | Score | P(H) | P(D) | P(A) | Pred | Actual | ✓ |
|---|---|---|---:|---:|---:|---|---|---|
| 2026-06-11 | Mexico vs South Africa | 2-0 | 0.35 | 0.33 | 0.32 | H | H | ✓ |
| 2026-06-11 | South Korea vs Czech Republic | 2-1 | 0.47 | 0.24 | 0.29 | H | H | ✓ |
| 2026-06-12 | Canada vs Bosnia and Herzegovina | 1-1 | 0.61 | 0.23 | 0.16 | H | D | ✗ |
| 2026-06-12 | United States vs Paraguay | 4-1 | 0.52 | 0.27 | 0.22 | H | H | ✓ |
| 2026-06-13 | Qatar vs Switzerland | 1-1 | 0.23 | 0.22 | 0.54 | A | D | ✗ |
| 2026-06-13 | Brazil vs Morocco | 1-1 | 0.17 | 0.30 | 0.53 | A | D | ✗ |
| 2026-06-13 | Haiti vs Scotland | 0-1 | 0.54 | 0.21 | 0.25 | H | A | ✗ |
| 2026-06-13 | Australia vs Turkey | 2-0 | 0.49 | 0.25 | 0.26 | H | H | ✓ |
| 2026-06-13 | Germany vs Curaçao | 7-1 | 0.52 | 0.21 | 0.27 | H | H | ✓ |
| 2026-06-14 | Netherlands vs Japan | 2-2 | 0.25 | 0.23 | 0.52 | A | D | ✗ |
| 2026-06-14 | Ivory Coast vs Ecuador | 1-0 | 0.43 | 0.35 | 0.21 | H | H | ✓ |
| 2026-06-14 | Sweden vs Tunisia | 5-1 | 0.27 | 0.26 | 0.46 | A | H | ✗ |
| 2026-06-14 | Spain vs Cape Verde | 0-0 | 0.65 | 0.21 | 0.14 | H | D | ✗ |
| 2026-06-15 | Belgium vs Egypt | 1-1 | 0.42 | 0.29 | 0.29 | H | D | ✗ |
| 2026-06-15 | Saudi Arabia vs Uruguay | 1-1 | 0.25 | 0.34 | 0.41 | A | D | ✗ |
| 2026-06-15 | Iran vs New Zealand | 2-2 | 0.52 | 0.24 | 0.23 | H | D | ✗ |

**Outcome accuracy: 37.5%** (6/16 correct)

**Mean Brier score: 0.714** (random baseline: 0.667, perfect: 0.0)

**Mean log loss: 1.143** (random baseline: 1.099, perfect: 0.0)


## 3. Current group standings (after MD1-2)

| Group | 1st | 2nd | 3rd | 4th |
|---|---|---|---|---|
| A | Mexico 3p | South Korea 3p | Czech Republic 0p | South Africa 0p |
| B | Canada 1p | Bosnia and Herzegovina 1p | Qatar 1p | Switzerland 1p |
| C | Scotland 3p | Brazil 1p | Morocco 1p | Haiti 0p |
| D | United States 3p | Australia 3p | Turkey 0p | Paraguay 0p |
| E | Germany 3p | Ivory Coast 3p | Ecuador 0p | Curaçao 0p |
| F | Sweden 3p | Netherlands 1p | Japan 1p | Tunisia 0p |
| G | Iran 1p | New Zealand 1p | Belgium 1p | Egypt 1p |
| H | Saudi Arabia 1p | Uruguay 1p | Spain 1p | Cape Verde 1p |
| I | France 0p | Senegal 0p | Iraq 0p | Norway 0p |
| J | Argentina 0p | Algeria 0p | Austria 0p | Jordan 0p |
| K | Portugal 0p | DR Congo 0p | Uzbekistan 0p | Colombia 0p |
| L | England 0p | Croatia 0p | Ghana 0p | Panama 0p |

## 4. Predictions for remaining group matches

P(H)/P(D)/P(A) are win/draw/loss probabilities. λ values are expected goals.

| Group | Match | P(H) | P(D) | P(A) | λ_H | λ_A | Top score |
|---|---|---:|---:|---:|---:|---:|---|
| A | Mexico vs South Korea | 0.32 | 0.29 | 0.39 | 1.02 | 1.15 | 1-1 (0.13) |
| A | Mexico vs Czech Republic | 0.43 | 0.27 | 0.30 | 1.36 | 1.09 | 1-1 (0.13) |
| A | South Africa vs South Korea | 0.27 | 0.29 | 0.43 | 0.90 | 1.22 | 0-1 (0.15) |
| A | South Africa vs Czech Republic | 0.37 | 0.28 | 0.35 | 1.20 | 1.15 | 1-1 (0.13) |
| B | Bosnia and Herzegovina vs Qatar | 0.31 | 0.25 | 0.43 | 1.25 | 1.52 | 1-1 (0.12) |
| B | Bosnia and Herzegovina vs Switzerland | 0.19 | 0.21 | 0.60 | 1.01 | 1.97 | 1-1 (0.10) |
| B | Canada vs Qatar | 0.54 | 0.25 | 0.21 | 1.62 | 0.90 | 1-0 (0.13) |
| B | Canada vs Switzerland | 0.40 | 0.27 | 0.33 | 1.31 | 1.17 | 1-1 (0.13) |
| C | Brazil vs Haiti | 0.46 | 0.24 | 0.30 | 1.66 | 1.28 | 1-1 (0.11) |
| C | Brazil vs Scotland | 0.56 | 0.24 | 0.20 | 1.74 | 0.94 | 1-0 (0.12) |
| C | Morocco vs Haiti | 0.65 | 0.22 | 0.13 | 1.79 | 0.64 | 1-0 (0.16) |
| C | Morocco vs Scotland | 0.71 | 0.20 | 0.08 | 1.87 | 0.47 | 1-0 (0.18) |
| D | Paraguay vs Australia | 0.12 | 0.24 | 0.63 | 0.54 | 1.61 | 0-1 (0.19) |
| D | Paraguay vs Turkey | 0.21 | 0.24 | 0.55 | 0.93 | 1.66 | 0-1 (0.12) |
| D | United States vs Australia | 0.26 | 0.26 | 0.47 | 1.02 | 1.47 | 1-1 (0.12) |
| D | United States vs Turkey | 0.44 | 0.23 | 0.34 | 1.76 | 1.52 | 1-1 (0.10) |
| E | Curaçao vs Ivory Coast | 0.16 | 0.21 | 0.63 | 0.89 | 1.97 | 0-1 (0.11) |
| E | Curaçao vs Ecuador | 0.30 | 0.29 | 0.40 | 0.97 | 1.17 | 0-1 (0.14) |
| E | Germany vs Ivory Coast | 0.33 | 0.26 | 0.41 | 1.24 | 1.41 | 1-1 (0.12) |
| E | Germany vs Ecuador | 0.49 | 0.28 | 0.23 | 1.36 | 0.84 | 1-0 (0.15) |
| F | Japan vs Sweden | 0.70 | 0.17 | 0.13 | 2.49 | 0.99 | 2-0 (0.10) |
| F | Japan vs Tunisia | 0.61 | 0.23 | 0.16 | 1.71 | 0.73 | 1-0 (0.15) |
| F | Netherlands vs Sweden | 0.59 | 0.19 | 0.21 | 2.37 | 1.39 | 2-1 (0.09) |
| F | Netherlands vs Tunisia | 0.51 | 0.25 | 0.24 | 1.63 | 1.03 | 1-1 (0.12) |
| G | Belgium vs Iran | 0.38 | 0.26 | 0.36 | 1.36 | 1.33 | 1-1 (0.12) |
| G | Belgium vs New Zealand | 0.54 | 0.22 | 0.24 | 1.96 | 1.23 | 1-1 (0.10) |
| G | Egypt vs Iran | 0.29 | 0.31 | 0.40 | 0.86 | 1.06 | 0-1 (0.16) |
| G | Egypt vs New Zealand | 0.43 | 0.29 | 0.29 | 1.25 | 0.98 | 1-0 (0.13) |
| H | Cape Verde vs Saudi Arabia | 0.35 | 0.33 | 0.32 | 0.91 | 0.85 | 0-0 (0.17) |
| H | Cape Verde vs Uruguay | 0.27 | 0.35 | 0.39 | 0.71 | 0.92 | 0-0 (0.20) |
| H | Spain vs Saudi Arabia | 0.66 | 0.22 | 0.13 | 1.84 | 0.66 | 1-0 (0.15) |
| H | Spain vs Uruguay | 0.54 | 0.27 | 0.19 | 1.43 | 0.71 | 1-0 (0.17) |
| I | France vs Senegal | 0.31 | 0.28 | 0.41 | 1.04 | 1.25 | 1-1 (0.13) |
| I | France vs Iraq | 0.51 | 0.27 | 0.22 | 1.43 | 0.84 | 1-0 (0.15) |
| I | France vs Norway | 0.36 | 0.24 | 0.40 | 1.52 | 1.61 | 1-1 (0.11) |
| I | Iraq vs Norway | 0.21 | 0.25 | 0.55 | 0.90 | 1.64 | 0-1 (0.13) |
| I | Senegal vs Iraq | 0.53 | 0.29 | 0.18 | 1.28 | 0.61 | 1-0 (0.19) |
| I | Senegal vs Norway | 0.41 | 0.27 | 0.33 | 1.35 | 1.19 | 1-1 (0.13) |
| J | Algeria vs Austria | 0.47 | 0.26 | 0.27 | 1.44 | 1.03 | 1-1 (0.13) |
| J | Algeria vs Jordan | 0.62 | 0.22 | 0.16 | 1.88 | 0.83 | 1-0 (0.12) |
| J | Argentina vs Algeria | 0.42 | 0.32 | 0.26 | 1.09 | 0.78 | 1-0 (0.17) |
| J | Argentina vs Austria | 0.54 | 0.27 | 0.19 | 1.42 | 0.73 | 1-0 (0.17) |
| J | Argentina vs Jordan | 0.68 | 0.21 | 0.11 | 1.85 | 0.59 | 1-0 (0.16) |
| J | Austria vs Jordan | 0.53 | 0.24 | 0.24 | 1.75 | 1.09 | 1-1 (0.11) |
| K | DR Congo vs Uzbekistan | 0.20 | 0.31 | 0.49 | 0.63 | 1.17 | 0-1 (0.19) |
| K | DR Congo vs Colombia | 0.20 | 0.28 | 0.52 | 0.72 | 1.34 | 0-1 (0.17) |
| K | Portugal vs DR Congo | 0.65 | 0.22 | 0.13 | 1.83 | 0.66 | 1-0 (0.15) |
| K | Portugal vs Uzbekistan | 0.47 | 0.27 | 0.25 | 1.37 | 0.93 | 1-0 (0.14) |
| K | Portugal vs Colombia | 0.49 | 0.25 | 0.26 | 1.57 | 1.07 | 1-1 (0.12) |
| K | Uzbekistan vs Colombia | 0.35 | 0.31 | 0.35 | 1.01 | 1.01 | 1-1 (0.14) |
| L | Croatia vs Ghana | 0.51 | 0.25 | 0.24 | 1.60 | 1.00 | 1-0 (0.12) |
| L | Croatia vs Panama | 0.49 | 0.24 | 0.27 | 1.71 | 1.20 | 1-1 (0.11) |
| L | England vs Croatia | 0.58 | 0.25 | 0.18 | 1.63 | 0.78 | 1-0 (0.15) |
| L | England vs Ghana | 0.69 | 0.20 | 0.10 | 1.90 | 0.57 | 1-0 (0.16) |
| L | England vs Panama | 0.69 | 0.19 | 0.11 | 2.04 | 0.68 | 2-0 (0.14) |
| L | Ghana vs Panama | 0.33 | 0.26 | 0.41 | 1.24 | 1.40 | 1-1 (0.12) |

## 5. Tournament progression probabilities

| Team | ELO | P(R32) | P(R16) | P(QF) | P(SF) | P(Final) | P(Win) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Morocco | 2020 | 0.95 | 0.70 | 0.54 | 0.39 | 0.28 | 0.18 |
| Argentina | 2200 | 0.91 | 0.66 | 0.48 | 0.32 | 0.21 | 0.13 |
| England | 2088 | 0.95 | 0.66 | 0.45 | 0.28 | 0.17 | 0.09 |
| Japan | 2020 | 0.93 | 0.56 | 0.40 | 0.23 | 0.14 | 0.08 |
| Spain | 2202 | 0.94 | 0.51 | 0.34 | 0.20 | 0.11 | 0.06 |
| Algeria | 1898 | 0.82 | 0.52 | 0.33 | 0.19 | 0.10 | 0.05 |
| Portugal | 2061 | 0.89 | 0.60 | 0.35 | 0.20 | 0.10 | 0.05 |
| Senegal | 1925 | 0.81 | 0.52 | 0.32 | 0.19 | 0.09 | 0.04 |
| Ivory Coast | 1888 | 0.86 | 0.52 | 0.27 | 0.14 | 0.07 | 0.03 |
| Belgium | 1976 | 0.79 | 0.50 | 0.28 | 0.14 | 0.06 | 0.03 |
| Norway | 2007 | 0.76 | 0.47 | 0.27 | 0.14 | 0.06 | 0.03 |
| Iran | 1888 | 0.78 | 0.48 | 0.26 | 0.13 | 0.06 | 0.03 |
| Australia | 1936 | 0.89 | 0.51 | 0.24 | 0.12 | 0.05 | 0.02 |
| France | 2136 | 0.72 | 0.42 | 0.23 | 0.11 | 0.05 | 0.02 |
| Brazil | 2079 | 0.76 | 0.37 | 0.21 | 0.09 | 0.04 | 0.02 |
| Germany | 2037 | 0.85 | 0.49 | 0.22 | 0.11 | 0.04 | 0.02 |
| Netherlands | 2005 | 0.80 | 0.37 | 0.21 | 0.09 | 0.04 | 0.02 |
| South Korea | 1900 | 0.79 | 0.45 | 0.19 | 0.08 | 0.03 | 0.01 |
| Austria | 1926 | 0.63 | 0.32 | 0.16 | 0.07 | 0.03 | 0.01 |
| Egypt | 1845 | 0.66 | 0.37 | 0.18 | 0.08 | 0.03 | 0.01 |
| Uzbekistan | 1825 | 0.71 | 0.37 | 0.17 | 0.07 | 0.02 | 0.01 |
| Colombia | 2067 | 0.71 | 0.36 | 0.16 | 0.06 | 0.02 | 0.01 |
| Mexico | 1982 | 0.72 | 0.37 | 0.14 | 0.05 | 0.02 | 0.01 |
| Canada | 1896 | 0.87 | 0.42 | 0.15 | 0.05 | 0.02 | 0.01 |
| Croatia | 1973 | 0.75 | 0.32 | 0.13 | 0.05 | 0.02 | 0.01 |
| United States | 1879 | 0.77 | 0.33 | 0.12 | 0.04 | 0.01 | 0.00 |
| Haiti | 1688 | 0.60 | 0.24 | 0.11 | 0.04 | 0.01 | 0.00 |
| Uruguay | 1959 | 0.66 | 0.23 | 0.09 | 0.04 | 0.01 | 0.00 |
| Switzerland | 1965 | 0.84 | 0.37 | 0.11 | 0.04 | 0.01 | 0.00 |
| South Africa | 1661 | 0.62 | 0.28 | 0.09 | 0.03 | 0.01 | 0.00 |
| New Zealand | 1713 | 0.48 | 0.23 | 0.08 | 0.03 | 0.01 | 0.00 |
| Ecuador | 2006 | 0.57 | 0.22 | 0.08 | 0.03 | 0.01 | 0.00 |
| Tunisia | 1721 | 0.52 | 0.18 | 0.07 | 0.02 | 0.01 | 0.00 |
| Iraq | 1759 | 0.39 | 0.16 | 0.06 | 0.02 | 0.01 | 0.00 |
| Czech Republic | 1818 | 0.58 | 0.25 | 0.07 | 0.02 | 0.01 | 0.00 |
| Turkey | 1962 | 0.69 | 0.24 | 0.07 | 0.02 | 0.01 | 0.00 |
| Sweden | 1830 | 0.41 | 0.13 | 0.04 | 0.01 | 0.00 | 0.00 |
| Cape Verde | 1691 | 0.52 | 0.14 | 0.04 | 0.01 | 0.00 | 0.00 |
| Ghana | 1630 | 0.43 | 0.12 | 0.03 | 0.01 | 0.00 | 0.00 |
| Scotland | 1883 | 0.35 | 0.10 | 0.03 | 0.01 | 0.00 | 0.00 |
| Saudi Arabia | 1698 | 0.48 | 0.12 | 0.03 | 0.01 | 0.00 | 0.00 |
| Jordan | 1769 | 0.31 | 0.10 | 0.03 | 0.01 | 0.00 | 0.00 |
| Curaçao | 1596 | 0.39 | 0.12 | 0.03 | 0.01 | 0.00 | 0.00 |
| Panama | 1825 | 0.49 | 0.15 | 0.04 | 0.01 | 0.00 | 0.00 |
| DR Congo | 1697 | 0.35 | 0.11 | 0.03 | 0.01 | 0.00 | 0.00 |
| Paraguay | 1885 | 0.33 | 0.07 | 0.01 | 0.00 | 0.00 | 0.00 |
| Bosnia and Herzegovina | 1695 | 0.42 | 0.09 | 0.01 | 0.00 | 0.00 | 0.00 |
| Qatar | 1564 | 0.56 | 0.16 | 0.03 | 0.01 | 0.00 | 0.00 |

## 6. Top 10 most likely champions

| Rank | Team | P(Win) |
|---:|---|---:|
| 1 | Morocco | 18.3% |
| 2 | Argentina | 13.1% |
| 3 | England | 9.4% |
| 4 | Japan | 7.7% |
| 5 | Spain | 5.8% |
| 6 | Algeria | 5.1% |
| 7 | Portugal | 5.0% |
| 8 | Senegal | 4.3% |
| 9 | Ivory Coast | 3.1% |
| 10 | Belgium | 3.0% |

## 7. Notes

- **The model is now calibrated against actual MD1-2 results.** ELO has been updated for all 16 matches; attack/defense ratings incorporate the actual goals.
- **The big movers**: Brazil's ELO dropped slightly after drawing Morocco; Spain's dropped a bit after the Cape Verde draw; Germany's shot up after the 7-1; USA's ELO rose after the 4-1 win.
- **Morocco still rated top** - they didn't lose (drew Brazil 1-1) and the model weights their 2022 WC run heavily.
- **Group H is wide open**: Spain, Uruguay, and Saudi Arabia all on 1pt. Any of them could win the group.
- **Backtest is honest** - these are the model's pre-match predictions, not the model fitted to outcomes. 37.5% outcome accuracy on this batch is below the long-run average (50-55%) because MD1-2 was unusually draw-heavy (10 of 16 ended in draws).
- **Draws hurt outcome accuracy directly**: a 0-0 outcome and 3-2 outcome both count as 'wrong' if the model predicted 1-0. The 1.14 log-loss reflects this.
