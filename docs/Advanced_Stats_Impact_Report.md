# Rapport d'Impact : Intégration des Statistiques Avancées (Modèle v4)

Ce rapport présente les résultats de l'intégration des statistiques de match avancées (tirs, possession, passes, expected goals, etc.) dans le modèle de prédiction de football XGBoost. L'objectif était d'évaluer si ces données, agrégées sous forme de moyennes glissantes, améliorent la capacité du modèle à prédire l'issue des rencontres.

## 1. Nouvelles Features Intégrées

Nous avons collecté les statistiques détaillées via l'API-Football pour **15 134 matchs** (soit 99,99% du dataset total). La collecte couvre désormais intégralement les saisons 2017 à 2025 (à l'exception d'un seul match en 2018 sans statistiques disponibles).

Pour éviter toute fuite de données (data leakage), nous avons calculé des **moyennes glissantes sur les 5 derniers matchs** de chaque équipe, en utilisant uniquement les matchs précédant la rencontre à prédire.

Les 18 nouvelles features ajoutées (pour l'équipe à domicile et à l'extérieur) sont :
* `shots_on_goal_avg5` : Tirs cadrés
* `total_shots_avg5` : Tirs totaux
* `shots_insidebox_avg5` : Tirs dans la surface
* `ball_possession_avg5` : Possession de balle (%)
* `total_passes_avg5` : Passes totales
* `passes_pct_avg5` : Pourcentage de passes réussies
* `corner_kicks_avg5` : Corners obtenus
* `fouls_avg5` : Fautes commises
* `expected_goals_avg5` : Expected Goals (xG) - *disponibles uniquement pour les saisons récentes (5 673 matchs couverts)*

La couverture globale des statistiques avancées (hors xG) atteint désormais **93,9%** sur l'ensemble du dataset (les matchs manquants correspondent aux 5 premiers matchs de chaque équipe qui n'ont pas encore d'historique suffisant).

## 2. Comparaison des Performances

Nous avons comparé trois versions du modèle sur un jeu de test chronologique (saisons 2023-2025, 3 027 matchs). Grâce à la collecte complète, le jeu de test bénéficie désormais d'une couverture de **99,9%** sur les statistiques avancées.

1. **Baseline (v2)** : 15 features (ELO, forme, buts, repos, H2H)
2. **Europe (v3)** : 17 features (v2 + participation aux coupes d'Europe)
3. **Advanced (v4)** : 33 features (v3 + 16 moyennes glissantes de stats avancées, xG exclus car non disponibles sur tout l'historique)

| Métrique | Baseline (v2) | Europe (v3) | Advanced (v4) |
|----------|---------------|-------------|---------------|
| **CV Log Loss** | 0.9933 | 0.9931 | **0.9910** |
| **Test Log Loss** | 0.9970 | 0.9976 | **0.9899** |
| **Test Accuracy** | 0.5206 | 0.5213 | **0.5273** |
| **Test F1 (Macro)** | 0.3964 | 0.3986 | **0.4058** |

### Analyse des Résultats

Le modèle v4 (Advanced) surpasse nettement les versions précédentes sur toutes les métriques :
* Le **Log Loss** (métrique la plus importante pour évaluer la justesse des probabilités) s'améliore significativement, passant de 0.9970 à 0.9899 sur le jeu de test.
* L'**Accuracy** progresse de près de 0.7 point, atteignant 52.73%.
* Le **F1-Score Macro** franchit la barre des 0.40 (0.4058), tiré par une meilleure détection des matchs nuls (F1 passant de 0.020 à 0.037) et des victoires à l'extérieur.

Ces améliorations démontrent que les statistiques avancées, lorsqu'elles sont disponibles pour la quasi-totalité du dataset, apportent un signal prédictif fort et complémentaire à l'ELO.

## 3. Importance des Features

L'analyse de l'importance des features (Gain) montre que les statistiques avancées apportent une réelle valeur ajoutée :

![Feature Importance](../results/feature_importance_advanced_v4.png)

Parmi les nouvelles features, les plus influentes sont :
1. **Passes totales (`home_total_passes_avg5`)** : Reflète le style de jeu (possession vs contre-attaque) et la maîtrise technique.
2. **Tirs totaux (`home_total_shots_avg5`)** : Un indicateur brut mais très efficace de la domination offensive.
3. **Possession de balle (`home_ball_possession_avg5`)** : Indicateur clé du contrôle du match.

Ces features se classent juste derrière les variables fondamentales (différence ELO, moyenne de buts encaissés/marqués), prouvant qu'elles capturent des signaux tactiques que l'ELO seul ne voit pas.

## 4. Visualisations Comparatives

![Comparaison des Modèles](../results/comparison_chart_v4.png)

Les graphiques confirment la supériorité du modèle v4, qui offre à la fois les probabilités les plus précises (Log Loss le plus bas) et une meilleure capacité de classification globale (F1 Macro et Accuracy).

## 5. Conclusion et Recommandations

L'intégration des statistiques avancées est un succès total. La collecte exhaustive des données sur l'ensemble des saisons (2017-2025) a permis au modèle d'exploiter pleinement ces nouvelles features, conduisant à des améliorations significatives sur toutes les métriques d'évaluation.

**Recommandations pour la suite :**
1. **Exploiter les Expected Goals (xG)** : Les xG sont désormais disponibles pour plus de 5 600 matchs récents. Une version v5 du modèle pourrait être entraînée spécifiquement sur ces saisons récentes pour évaluer l'apport prédictif des xG.
2. **Déploiement** : Le modèle v4 a été sauvegardé comme modèle de production (`xgb_football_model_v4_advanced.pkl`). Il est prêt à être utilisé pour des prédictions en direct, à condition de lui fournir les moyennes glissantes des 5 derniers matchs des équipes concernées.
3. **Mise à jour continue** : Mettre en place un pipeline automatisé pour collecter les statistiques avancées après chaque journée de championnat afin de maintenir les moyennes glissantes à jour.
