# Phased Delivery Milestones

The roadmap is structured into iterative phases so the team can deliver value while de-risking key assumptions.

## Phase 0 — Discovery & Alignment

- **Objectives**
  - Validate core quantitative research questions and target trading universes.
  - Audit existing data sources, infrastructure, and team capabilities.
- **Key Deliverables**
  - Documented problem statements and success metrics.
  - Current-state assessment captured in `docs/process/current_state.md`.
  - Prioritised backlog of foundational features and experiments.
- **Exit Criteria**
  - Stakeholders sign off on the quantified opportunity and constraints.
  - Feasibility of data acquisition and MT5 integration is confirmed.

## Phase 1 — Foundation Build-out

- **Objectives**
  - Stand up reproducible data ingestion, storage, and baseline modelling workflows.
  - Establish automated validation and monitoring for the ingestor and backtester.
- **Key Deliverables**
  - Production-ready data fetching scripts with scheduling guidance.
  - Baseline regression pipeline results published for key instrument pairs.
  - Documentation covering technical prerequisites and runbooks.
- **Exit Criteria**
  - Daily data refresh runs succeed for two consecutive weeks.
  - Baseline models achieve accuracy thresholds defined in the backlog.
  - Runbooks and onboarding guides are peer-reviewed.

## Phase 2 — Expansion & Optimisation

- **Objectives**
  - Layer advanced modelling techniques (tree ensembles, volatility models) on top of the baseline.
  - Integrate risk management, reporting, and alerting capabilities.
- **Key Deliverables**
  - Extended pipeline stages with comparative performance reporting.
  - Risk dashboards or notebooks that visualise exposure and PnL distributions.
  - Playbooks for scaling to additional broker accounts or asset classes.
- **Exit Criteria**
  - Advanced models demonstrate statistically significant uplift over baselines.
  - Operations team can respond to alerts using documented runbooks.
  - Governance review approves expansion to production trading.

> Continue refining phases beyond Phase 2 as new initiatives emerge (e.g., execution automation, portfolio rebalancing).
