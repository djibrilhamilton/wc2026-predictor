# ⚽ WC 2026 AI Predictor

**A full-stack machine learning project that predicts the 2026 FIFA World Cup: match scores, group standings, the official knockout bracket, and title probabilities, updated with real results as the tournament unfolds.**

🔗 **Live demo:** TODO-your-vercel-url
🇫🇷 / 🇺🇸 The interface is available in French and English.

---

## 🎯 Purpose

Football is a low-scoring, high-variance sport, so a single "predicted score" is misleading. This project treats prediction as a **probability problem**: instead of saying "France wins 2-1", it models the full distribution of scores and propagates the uncertainty through the entire tournament.

The goal was to build an end-to-end data product: statistical modeling, a live-updating data pipeline, and a polished, responsive interface that non-technical users can explore.

---

## ✨ Features

| Feature | Description |
|---|---|
| **Match probabilities** | Win / draw / loss odds plus the top most-likely scorelines for any of the 1,128 possible match-ups |
| **Group stage** | Predicted standings for all 12 groups, with expected points and the 8 best third-placed teams computed dynamically |
| **Knockout bracket** | Official FIFA 2026 structure (matches M73–M88) rendered as a connected SVG bracket from Round of 32 to the Final |
| **Match simulator** | Pick any two teams, toggle key-player injuries and see how the probabilities shift |
| **Radar comparison** | 5-dimension team profile (attack, defense, ELO, experience, trophies) |
| **Team analysis** | Stage-by-stage probabilities (R32 → Title) and head-to-head comparison |
| **Live updates** | Real results feed back into team ratings after every matchday |
| **Bilingual + responsive** | FR/EN switcher persisted in `localStorage`; layouts adapt to mobile |

---

## 🧠 Methodology

### 1. Dixon-Coles Poisson model
Goals for each team are modeled as Poisson-distributed:

```
λ_home = attack_home × defense_away × ELO_adjustment × scale
λ_away = attack_away × defense_home × ELO_adjustment × scale
P(i, j) = Poisson(i; λ_home) × Poisson(j; λ_away) × τ(i, j, ρ)
```

The **Dixon-Coles correction τ** (ρ = −0.12) adjusts the probabilities of low scores (0-0, 1-0, 0-1, 1-1), which plain independent Poisson models systematically get wrong. The 9×9 score matrix is normalized, and win / draw / loss probabilities are obtained by summing the relevant regions.

### 2. ELO ratings
Each team carries an ELO rating that adjusts expected goals through a soft logistic factor. After each real result, ratings are updated with a goal-difference multiplier:

```
expected = 1 / (1 + 10^((elo_away - elo_home) / 400))
Δelo     = K × goal_diff_multiplier × (result - expected)
```

### 3. Monte Carlo tournament simulation
The backend simulates the **entire tournament 50,000 times** (groups → best thirds → Round of 32 → Final) to estimate each team's probability of reaching every stage. This captures bracket effects that per-match predictions alone cannot.

### 4. Maximum likelihood fitting (optional)
`mle_fitter.py` fits attack/defense parameters on historical international results (Kaggle dataset + football-data.org), rather than relying on hand-set values.

### 5. Live Bayesian-style updating
After each matchday, `wc2026_live.py`:
- updates ELO ratings,
- shifts attack/defense parameters with a small learning rate (so one result never dominates),
- records recent form,
- deduplicates already-applied matches via a tracker file.

### 6. Injury impact model
Each key player has an estimated share of the team's attacking output. Marking a player as injured scales down the team's expected goals (floored to avoid unrealistic collapse), and the UI shows the before/after probability delta.

---

## 🏗️ Architecture

```
┌──────────────────────────┐        ┌───────────────────────────────┐
│  Data sources            │        │  Python backend               │
│  • football-data.org API │ ─────► │  • collectors/ (results, ELO) │
│  • ClubElo               │        │  • models/ (Dixon-Coles, MLE, │
│  • Kaggle historical CSV │        │    Monte Carlo, validator)    │
│  • EA FC squad ratings   │        │  • FastAPI endpoints          │
└──────────────────────────┘        └──────────────┬────────────────┘
                                                   │ exports predictions
                                                   ▼
                              ┌─────────────────────────────────────┐
                              │  React + Vite + TailwindCSS         │
                              │  • Client-side match engine         │
                              │  • Recharts, custom SVG bracket     │
                              │  • i18n context (FR/EN)             │
                              └──────────────┬──────────────────────┘
                                             │ git push → auto-deploy
                                             ▼
                                        Vercel (CDN)
```

The match engine is also implemented client-side in JavaScript, so the simulator and group predictions run instantly in the browser with no server round-trip.

---

## 🛠️ Tech stack

**Backend:** Python, FastAPI, NumPy, SciPy, pandas, scikit-learn
**Frontend:** React, Vite, TailwindCSS, Recharts, React Router
**Data:** football-data.org API, ClubElo, Kaggle international results, EA FC player ratings
**Deployment:** Vercel (continuous deployment from GitHub)

---

## 📁 Project structure

```
wc2026-predictor/
├── backend/
│   ├── collectors/        # match history, ELO, live WC results
│   ├── models/            # predictor, simulator, MLE fitter, validator
│   ├── api/main.py        # FastAPI endpoints
│   ├── data/              # teams, fitted params, cached results
│   └── run_predictions.py # CLI entry point
└── frontend/
    └── src/
        ├── pages/         # Dashboard, GroupStage, Knockout, Simulator, TeamAnalysis
        ├── components/    # MatchCard, GroupCard, RadarComparison, Flag...
        ├── context/       # LanguageContext (i18n)
        ├── data/          # tournament structure + prediction engine
        └── hooks/         # useIsMobile
```

---

## 🚀 Getting started

### Backend
```bash
cd backend
python -m venv venv
venv\Scripts\activate            # macOS/Linux: source venv/bin/activate
pip install numpy pandas scipy scikit-learn fastapi "uvicorn[standard]" requests python-dotenv

# Optional: add FOOTBALL_DATA_API_KEY to .env for live API results
python run_predictions.py --live          # fetch results + 50k simulations
python run_predictions.py --live --fast   # 10k simulations (quicker)
python run_predictions.py --fit           # refit parameters with MLE

uvicorn api.main:app --reload --port 8000 # optional API
```

### Frontend
```bash
cd frontend
npm install
npm run dev        # http://localhost:5173
npm run build
```

### Updating during the tournament
1. Add the new results to `KNOWN_RESULTS` in `backend/collectors/wc2026_live.py` (or let the API fetch them)
2. Run `python run_predictions.py --live`
3. `git push` and Vercel redeploys automatically

---

## 🔍 Engineering challenges & decisions

- **Realistic scorelines.** Early versions only produced 1-0 and 2-0. The cause was an under-scaled λ; recalibrating the scale factor and displaying the *full score distribution* (not just the mode) fixed it and made draws appear naturally.
- **Probabilities that sum to 100%.** Rounding each outcome independently broke totals; the draw probability is now derived as the remainder.
- **Consistent bracket.** Champion, finalists and bracket are all derived from a single computation so the UI can never contradict itself.
- **Official FIFA bracket.** Round of 32 pairings follow the real M73–M88 structure, including slots reserved for third-placed teams from specific groups.
- **Cross-platform flags.** Windows does not render flag emojis, so flags are served as images from flagcdn.com.
- **Mobile without touching desktop.** A `useIsMobile` hook adapts layouts below 768px while leaving the desktop UI unchanged.
- **Debugging in production.** Resolved module-scope vs. component-scope reference errors that only surfaced at runtime on Vercel.

---

## ⚠️ Limitations

Being upfront about what the model does *not* do:

- Team strength parameters are partly hand-calibrated rather than fully learned, unless `--fit` is run.
- Third-place slot assignment uses an eligibility-based heuristic, not FIFA's full table of 495 combinations (Annex C).
- Knockout winners in the bracket view are chosen by title probability (most-likely path), not by simulating each tie.
- No lineup, tactical, weather or fatigue data. Injuries are a simple attack-scaling approximation.
- Football remains highly random; the model gives probabilities, not certainties.

---

## 📊 Validation

`models/validator.py` backtests the approach on the 2022 World Cup.

| Metric | Result |
|---|---|
| Outcome accuracy (W/D/L) | TODO: fill in from your validator output |
| Brier score | TODO |
| Baseline (random / always home) | TODO |

---

## 🗺️ Roadmap

- [ ] Full FIFA Annex C third-place combination table
- [ ] Per-tie knockout simulation with extra time and penalties
- [ ] Time-decay weighting on historical matches
- [ ] Automated daily updates via GitHub Actions
- [ ] Calibration plots and model comparison dashboard

---

*Built for the love of football and statistics. ⚽*
