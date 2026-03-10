# Analyse Value Bets — XGBoost v4 vs Cotes Betclic
## 5 Ligues Majeures — Prochaine Journée (Mars 2026)

**Date de l'analyse** : 10 mars 2026

**Modèle** : XGBoost v4 (35 features, incluant statistiques avancées)

**Nombre de matchs analysés** : 47

---

## Value Bets Identifiés (Edge > 5%)

| # | Ligue | Match | Résultat | Prob. Modèle | Prob. Betclic | Cote | Edge | EV |
|:-:|:------|:------|:--------:|:---:|:---:|:---:|:---:|:---:|
| 1 | Bundesliga J26 | B. Leverkusen vs Bayern Munich | H | 36.9% | 19.9% | 4.70 | +17.0% | +73.4% |
| 2 | Premier League J30 | Nottingham Forest vs Fulham | A | 41.8% | 28.5% | 3.27 | +13.3% | +36.7% |
| 3 | Bundesliga J26 | Stuttgart vs RB Leipzig | H | 51.7% | 42.5% | 2.20 | +9.2% | +13.8% |
| 4 | Bundesliga J26 | Hoffenheim vs Wolfsburg | H | 73.4% | 64.5% | 1.45 | +9.0% | +6.5% |
| 5 | La Liga J28 | Atletico Madrid vs Getafe | H | 68.1% | 59.2% | 1.58 | +8.8% | +7.6% |
| 6 | Ligue 1 J26 | Rennes vs Lille | A | 40.6% | 31.9% | 2.93 | +8.7% | +19.0% |
| 7 | Premier League J30 | Chelsea vs Newcastle | H | 61.7% | 53.4% | 1.75 | +8.3% | +7.9% |
| 8 | Serie A J29 | Inter Milan vs Atalanta | H | 69.6% | 61.4% | 1.52 | +8.3% | +5.9% |
| 9 | Bundesliga J26 | Fribourg vs Union Berlin | D | 36.8% | 28.6% | 3.27 | +8.2% | +20.5% |
| 10 | Premier League J30 | Crystal Palace vs Leeds | H | 48.4% | 40.7% | 2.30 | +7.7% | +11.3% |
| 11 | Premier League J30 | Manchester United vs Aston Villa | D | 31.3% | 23.8% | 3.93 | +7.5% | +23.0% |
| 12 | Serie A J29 | Hellas Verone vs Genoa | A | 46.7% | 39.4% | 2.37 | +7.3% | +10.7% |
| 13 | Serie A J29 | Come vs Roma | H | 51.9% | 45.0% | 2.02 | +6.9% | +4.9% |
| 14 | Premier League J30 | Liverpool vs Tottenham | H | 77.6% | 70.9% | 1.32 | +6.7% | +2.5% |
| 15 | Bundesliga J26 | B. Leverkusen vs Bayern Munich | D | 27.4% | 20.8% | 4.50 | +6.5% | +23.1% |
| 16 | Premier League J30 | West Ham vs Manchester City | A | 63.4% | 57.1% | 1.64 | +6.3% | +4.0% |
| 17 | Ligue 1 J26 | Metz vs Toulouse | A | 53.5% | 47.4% | 1.97 | +6.1% | +5.4% |
| 18 | La Liga J28 | Girone vs Athletic Bilbao | H | 40.4% | 34.4% | 2.72 | +6.0% | +10.0% |
| 19 | Premier League J30 | Burnley vs Bournemouth | A | 56.9% | 51.3% | 1.82 | +5.6% | +3.5% |

---

## Premier League J30

| Match | H (Modèle) | H (Betclic) | D (Modèle) | D (Betclic) | A (Modèle) | A (Betclic) | Value Bet |
|:------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Burnley vs Bournemouth | 18.3% | 23.6% | 24.8% | 25.0% | 56.9% | 51.3% | A (+5.6%) |
| Sunderland vs Brighton | 31.7% | 28.5% | 28.2% | 28.4% | 40.1% | 43.1% | - |
| Chelsea vs Newcastle | 61.7% | 53.4% | 18.4% | 22.8% | 19.9% | 23.8% | H (+8.3%) |
| Arsenal vs Everton | 63.9% | 69.7% | 22.6% | 19.3% | 13.5% | 11.0% | - |
| West Ham vs Manchester City | 12.7% | 20.8% | 23.9% | 22.0% | 63.4% | 57.1% | A (+6.3%) |
| Manchester United vs Aston Villa | 43.9% | 54.0% | 31.3% | 23.8% | 24.8% | 22.2% | D (+7.5%) |
| Nottingham Forest vs Fulham | 31.1% | 44.0% | 27.1% | 27.4% | 41.8% | 28.5% | A (+13.3%) |
| Crystal Palace vs Leeds | 48.4% | 40.7% | 30.1% | 28.6% | 21.5% | 30.7% | H (+7.7%) |
| Liverpool vs Tottenham | 77.6% | 70.9% | 12.6% | 17.0% | 9.8% | 12.1% | H (+6.7%) |
| Brentford vs Wolverhampton | 60.9% | 60.6% | 23.8% | 22.5% | 15.3% | 17.0% | - |

## La Liga J28

| Match | H (Modèle) | H (Betclic) | D (Modèle) | D (Betclic) | A (Modèle) | A (Betclic) | Value Bet |
|:------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Alaves vs Villarreal | 27.9% | 28.5% | 28.5% | 27.5% | 43.5% | 44.1% | - |
| Girone vs Athletic Bilbao | 40.4% | 34.4% | 21.6% | 28.6% | 38.0% | 37.0% | H (+6.0%) |
| Atletico Madrid vs Getafe | 68.1% | 59.2% | 20.4% | 26.1% | 11.6% | 14.6% | H (+8.8%) |
| Real Oviedo vs Valencia | 36.2% | 32.8% | 30.1% | 30.1% | 33.7% | 37.1% | - |
| Real Madrid vs Elche | 77.1% | 76.7% | 14.8% | 14.4% | 8.1% | 8.9% | - |
| Majorque vs Espanyol | 43.0% | 40.3% | 29.2% | 30.3% | 27.8% | 29.4% | - |
| Barcelona vs Seville | 81.5% | 77.1% | 9.6% | 13.5% | 9.0% | 9.3% | - |
| Betis vs Celta Vigo | 42.9% | 43.1% | 29.3% | 29.0% | 27.7% | 27.9% | - |
| Real Sociedad vs Osasuna | 45.1% | 48.9% | 29.7% | 27.5% | 25.2% | 23.6% | - |
| Rayo Vallecano vs Levante | 54.7% | 55.3% | 26.0% | 25.0% | 19.2% | 19.7% | - |

## Serie A J29

| Match | H (Modèle) | H (Betclic) | D (Modèle) | D (Betclic) | A (Modèle) | A (Betclic) | Value Bet |
|:------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Torino vs Parme | 44.0% | 44.1% | 28.2% | 31.7% | 27.8% | 24.1% | - |
| Inter Milan vs Atalanta | 69.6% | 61.4% | 17.0% | 22.0% | 13.4% | 16.7% | H (+8.3%) |
| Naples vs Lecce | 67.4% | 68.7% | 19.9% | 21.2% | 12.7% | 10.1% | - |
| Udinese vs Juventus | 17.4% | 16.6% | 21.4% | 24.0% | 61.2% | 59.3% | - |
| Hellas Verone vs Genoa | 25.9% | 29.4% | 27.4% | 31.2% | 46.7% | 39.4% | A (+7.3%) |
| Pise vs Cagliari | 35.1% | 38.1% | 33.0% | 30.8% | 31.9% | 31.1% | - |
| Sassuolo vs Bologne | 30.8% | 34.9% | 28.5% | 28.6% | 40.7% | 36.4% | - |
| Come vs Roma | 51.9% | 45.0% | 20.9% | 27.6% | 27.2% | 27.4% | H (+6.9%) |
| Lazio vs Milan AC | 27.9% | 24.6% | 29.4% | 28.3% | 42.8% | 47.1% | - |
| Cremonese vs Fiorentina | 26.2% | 24.8% | 30.1% | 26.5% | 43.8% | 48.7% | - |

## Bundesliga J26

| Match | H (Modèle) | H (Betclic) | D (Modèle) | D (Betclic) | A (Modèle) | A (Betclic) | Value Bet |
|:------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| B. Monchengladbach vs St. Pauli | 42.7% | 48.0% | 30.2% | 27.9% | 27.2% | 24.1% | - |
| B. Leverkusen vs Bayern Munich | 36.9% | 19.9% | 27.4% | 20.8% | 35.7% | 59.3% | H (+17.0%), D (+6.5%) |
| Francfort vs Heidenheim | 65.9% | 62.4% | 20.4% | 20.6% | 13.6% | 17.0% | - |
| Dortmund vs Augsbourg | 70.0% | 67.4% | 18.0% | 18.7% | 12.0% | 13.9% | - |
| Hoffenheim vs Wolfsburg | 73.4% | 64.5% | 13.4% | 19.3% | 13.1% | 16.3% | H (+9.0%) |
| Hamburger SV vs Cologne | 45.6% | 45.6% | 26.5% | 26.5% | 27.9% | 27.9% | - |
| Werder Breme vs Mayence | 41.9% | 41.9% | 26.5% | 26.9% | 31.6% | 31.2% | - |
| Fribourg vs Union Berlin | 44.9% | 44.1% | 36.8% | 28.6% | 18.3% | 27.3% | D (+8.2%) |
| Stuttgart vs RB Leipzig | 51.7% | 42.5% | 21.3% | 23.8% | 27.0% | 33.7% | H (+9.2%) |

## Ligue 1 J26

| Match | H (Modèle) | H (Betclic) | D (Modèle) | D (Betclic) | A (Modèle) | A (Betclic) | Value Bet |
|:------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Marseille vs Auxerre | 66.9% | 64.9% | 20.0% | 20.5% | 13.0% | 14.6% | - |
| Lorient vs Lens | 24.8% | 22.5% | 29.5% | 25.8% | 45.7% | 51.7% | - |
| Angers vs Nice | 30.6% | 30.1% | 27.0% | 28.9% | 42.4% | 41.0% | - |
| Monaco vs Brest | 57.3% | 57.6% | 24.5% | 22.8% | 18.2% | 19.6% | - |
| Strasbourg vs Paris FC | 52.7% | 54.6% | 28.2% | 24.6% | 19.1% | 20.8% | - |
| Le Havre vs Lyon | 25.2% | 24.6% | 27.0% | 27.3% | 47.8% | 48.1% | - |
| Metz vs Toulouse | 21.8% | 24.3% | 24.7% | 28.3% | 53.5% | 47.4% | A (+6.1%) |
| Rennes vs Lille | 35.3% | 40.3% | 24.1% | 27.8% | 40.6% | 31.9% | A (+8.7%) |
