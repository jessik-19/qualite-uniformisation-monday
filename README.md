# 📊 Application Qualité & Uniformisation – Tableaux Monday  
**Analyse de cohérence, complétude et uniformisation des données issues de Monday.com**  
*Projet Data Quality / Python / Streamlit*

---

## 🎯 Objectif

Cette application vise à analyser la **qualité**, la **cohérence** et la **complétude** des données provenant de tableaux Monday.com utilisés par plusieurs pôles :

- Finance  
- Cash  
- Consolidation  

Elle permet en quelques secondes d'obtenir :

✔ un score global de complétude  
✔ un score par pôle  
✔ l'identification des incohérences  
✔ la détection des écritures différentes pour une même valeur  
✔ des tableaux de contrôle qualité prêts à corriger dans Monday  

---

## 🚀 Fonctionnalités principales

### 🔹 1. Import de fichiers Excel  
Compatible `.xlsx` et `.ods` (export brut Monday).  
Fusion automatique des pôles.

### 🔹 2. KPIs qualité générés automatiquement  
- Nombre total de tâches  
- Nombre de pôles distincts  
- Taux de complétude global  
- Taux de complétude par pôle  
- Nombre de valeurs distinctes par colonne  
- Pourcentage de cellules vides  

### 🔹 2. Heatmap de complétude (global & par pôle)
Visualisation claire de la qualité par colonne × pôle, permettant d’identifier :

- Colonnes à corriger en priorité  
- Colonnes trop hétérogènes  
- Pôles ayant un meilleur remplissage  

### 🔹 3. Détection d'incohérences d’écriture  
Repérage automatique :  
- des variantes orthographiques  
- majuscule/minuscule  
- valeurs proches mais différentes  
- valeurs propres vs valeurs brutes  

### 🔹 4. Analyse par volumétrie  
Graphiques générés automatiquement :

- Nombre de tâches par pôle  
- Colonnes les moins remplies  
- Colonnes les plus uniformes  
- Tableaux complets des incohérences  

---

## 🛠️ Stack Technique

- **Python**
  - Pandas
  - NumPy
  - Altair
- **Streamlit** pour l’interface interactive
- **Excel / Monday.com**
- **Data Cleaning / Data Quality**
- Gestion des variantes d’écriture  
- KPIs qualité automatiques



## 🚨 Confidentialité
Ce projet **ne contient aucune donnée sensible**.  
Seul **le code Python** est fourni.  
Les fichiers Excel d’origine ne sont pas inclus.

