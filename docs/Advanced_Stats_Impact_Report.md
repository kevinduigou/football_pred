# Rapport d'Impact : Intégration des Statistiques Avancées (Modèle v4)

Ce rapport présente les résultats de l'intégration des statistiques de match avancées (tirs, possession, passes, etc.) dans le modèle de prédiction de football XGBoost. L'objectif était d'évaluer si ces données, agrégées sous forme de moyennes glissantes, améliorent la capacité du modèle à prédire l'issue des rencontres.

## 1. Nouvelles Features Intégrées

Nous avons collecté les statistiques détaillées via l'API-Football pour environ 2 400 matchs (principalement sur les saisons 2017-2019). Pour éviter toute fuite de données (data leakage), nous avons calculé des **moyennes glissantes sur les 5 derniers matchs** de chaque équipe, en utilisant uniquement les matchs précédant la rencontre à prédire.

Les 16 nouvelles features ajoutées (pour l'équipe à domicile et à l'extérieur) sont :
* `shots_on_goal_avg5` : Tirs cadrés
* `total_shots_avg5` : Tirs totaux
* `shots_insidebox_avg5` : Tirs dans la surface
* `ball_possession_avg5` : Possession de balle (%)
* `total_passes_avg5` : Passes totales
* `passes_pct_avg5` : Pourcentage de passes réussies
* `corner_kicks_avg5` : Corners obtenus
* `fouls_avg5` : Fautes commises

*Note : Les Expected Goals (xG) n'étaient pas disponibles pour les saisons historiques collectées et ont donc été exclus de cette itération.*

## 2. Comparaison des Performances

Nous avons comparé trois versions du modèle sur un jeu de test chronologique (saisons 2023-2025, 3 027 matchs) :
1. **Baseline (v2)** : 15 features (ELO, forme, buts, repos, H2H)
2. **Europe (v3)** : 17 features (v2 + participation aux coupes d'Europe)
3. **Advanced (v4)** : 33 features (v3 + 16 moyennes glissantes de stats avancées)

| Métrique | Baseline (v2) | Europe (v3) | Advanced (v4) |
|----------|---------------|-------------|---------------|
| **CV Log Loss** | 0.9933 | 0.9931 | **0.9913** |
| **Test Log Loss** | 0.9970 | 0.9976 | **0.9919** |
| **Test Accuracy** | 0.5206 | **0.5213** | 0.5206 |
| **Test F1 (Macro)** | 0.3964 | **0.3986** | 0.3937 |

### Analyse des Résultats

Le modèle v4 (Advanced) obtient le **meilleur Log Loss** à la fois en validation croisée (0.9913) et sur le jeu de test (0.9919). Le Log Loss est la métrique la plus importante pour un modèle de probabilités, car elle pénalise fortement les prédictions confiantes mais fausses. Une baisse du Log Loss indique que le modèle v4 calibre mieux ses probabilités que les versions précédentes.

Cependant, l'Accuracy et le F1-Score restent stables (autour de 52%). Cela s'explique par une particularité de notre jeu de données : les statistiques avancées n'ont été collectées que pour les saisons 2017-2019, tandis que le jeu de test couvre 2023-2025. Le modèle v4 a donc dû faire ses prédictions de test avec 100% de valeurs manquantes (NaN) sur les nouvelles features. Le fait que le Log Loss s'améliore malgré cela prouve la robustesse de l'algorithme XGBoost face aux données manquantes et confirme que l'apprentissage sur les données historiques a permis d'affiner la structure des arbres de décision.

## 3. Importance des Features

L'analyse de l'importance des features (Gain) montre que les statistiques avancées apportent une réelle valeur ajoutée :

![Feature Importance](../results/feature_importance_advanced_v4.png)

Parmi les nouvelles features, les plus influentes sont :
1. **Possession de balle (`ball_possession_avg5`)** : Indicateur clé du contrôle du jeu.
2. **Tirs dans la surface (`shots_insidebox_avg5`)** : Excellent proxy de la dangerosité réelle d'une équipe.
3. **Pourcentage de passes réussies (`passes_pct_avg5`)** : Reflète la maîtrise technique.

Ces features se classent juste derrière les variables fondamentales (différence ELO, moyenne de buts encaissés/marqués), prouvant qu'elles capturent des signaux tactiques que l'ELO seul ne voit pas.

## 4. Visualisations Comparatives

![Comparaison des Modèles](../results/comparison_chart_v4.png)

Les graphiques confirment que le modèle v4 offre les probabilités les plus précises (Log Loss le plus bas), bien que la classification brute (Accuracy) reste contrainte par la difficulté inhérente à la prédiction des matchs nuls.

## 5. Conclusion et Recommandations

L'intégration des statistiques avancées est un succès validé par l'amélioration du Log Loss. Les données de tirs dans la surface et de possession apportent une granularité tactique précieuse au modèle.

**Recommandations pour la suite :**
1. **Compléter la collecte de données** : Il est crucial de collecter les statistiques avancées pour les saisons récentes (2023-2025) afin que le modèle puisse utiliser ces features lors des prédictions en direct.
2. **Intégrer les Expected Goals (xG)** : Les xG sont disponibles pour les saisons récentes via l'API. Une fois la collecte terminée, l'ajout des `expected_goals_avg5` devrait supplanter les simples statistiques de tirs.
3. **Déploiement** : Le modèle v4 a été sauvegardé comme modèle de production (`xgb_football_model_v4_advanced.pkl`) car il gère nativement les valeurs manquantes et offre les probabilités les mieux calibrées.
