---
title: 07 — Risks & Failure Modes
type: strategy
parent: "[[00-Index]]"
status: draft-v1
updated: 2026-05-04
tags: [knno, risk, premortem, failure-modes]
---

# 07 — Risks & Failure Modes

> [!warning]
> This document is a **pre-mortem**. Imagine it's December 2027 and KNNO has shut down. What killed it? The answers below are what we expect and what we'll fight to prevent.

---

## 1. Risk Severity Framework

| Severity | Likelihood | Definition                                                              |
| -------- | ---------- | ----------------------------------------------------------------------- |
| **Critical** | Any  | Single-point business killer. Forces shutdown or major pivot.           |
| **High** | M / H      | Materially threatens runway, brand, or core operations.                  |
| **Medium**| M         | Slows growth or hurts margins; recoverable with response.                |
| **Low**  | L          | Annoying; absorbable in normal operations.                               |

We track the top **8 critical/high risks** weekly in [[Technicals/Daily & Weekly Goals#Weekly Review Template]].

---

## 2. Critical Risks (the things that kill the business)

### 🔴 C-01 — Co-founder breakdown / split

| Field        | Detail                                                                                                |
| ------------ | ----------------------------------------------------------------------------------------------------- |
| Severity     | Critical                                                                                              |
| Likelihood   | **Higher than founders typically estimate** (industry data: ~45 % of co-founder businesses see significant founder conflict in Y1–2) |
| Trigger signs| Unresolved disagreements > 7 days; passive-aggressive comms; one founder doing > 65 % of the work; money fights |
| Impact       | Operational paralysis, brand inconsistency, eventual shutdown                                         |
| Mitigation   | (1) Operating agreement signed pre-launch, (2) named role ownership ([[03-Business Model & SWOT#Strategic Asks of the Co-Founders]]), (3) weekly check-in (15 min, just-the-two-of-you, not in the review meeting), (4) defined conflict resolution mechanism (e.g. one decision-maker per domain), (5) quarterly off-site (no laptops, just a long lunch), (6) buyout clause in operating agreement |
| Early warning| Burnout watch in [[Technicals/Daily & Weekly Goals#The Burnout Watch]] |

### 🔴 C-02 — Cash runway exhaustion

| Field        | Detail                                                                                                |
| ------------ | ----------------------------------------------------------------------------------------------------- |
| Severity     | Critical                                                                                              |
| Likelihood   | M-H (RM 15K is *tight*; the conservative scenario in [[09-Financial Projections (Conservative)]] is breakeven by month ~9) |
| Trigger      | 3 consecutive months of negative contribution; 1 large unexpected cost (equipment failure RM 4K+) without buffer |
| Impact       | Forced shutdown or panic-mode debt                                                                    |
| Mitigation   | (1) **3-month operating cash buffer** rule (never let working cash drop below RM 4.5K), (2) split founder funds: launch capital ≠ founder personal runway, (3) re-roast frequency tied to inventory not "schedule," (4) defer all non-essential spend until month 4 P&L is reviewed, (5) micro-loan / family bridge identified in advance (don't be searching for it the week you need it) |
| Early warning| Weekly cash position tracked in [[09-Financial Projections (Conservative)]]; trigger at RM 6K to start cost-cut conversation |

### 🔴 C-03 — Roasting consistency failure

| Field        | Detail                                                                                                |
| ------------ | ----------------------------------------------------------------------------------------------------- |
| Severity     | Critical (brand is built on roast quality)                                                            |
| Likelihood   | M (intermediate-skill founders × new machine × small batches)                                         |
| Trigger      | More than 2 batches in a row scoring < 3/4; customer complaints about flavour                        |
| Impact       | Brand erosion that's invisible until repeat-rate drops, then irreversible                             |
| Mitigation   | (1) Sample protocol (3 roasts before commit) in [[Technicals/Roasting Profiles#Sample Roasting]], (2) cup every batch within 72 h (no exceptions), (3) DTR tracked on every roast, (4) defect playbook taped to roaster, (5) **profile lock**: don't change a working profile mid-cycle, (6) mentor / advisor relationship with one experienced KL roaster (paid coffee + transparency = mentorship), (7) sample roaster purchase trigger in [[Technicals/Roasting Profiles#When to Buy Our Own Roaster]] |

### 🔴 C-04 — Tea-coffee crossover thesis fails

| Field        | Detail                                                                                                |
| ------------ | ----------------------------------------------------------------------------------------------------- |
| Severity     | Critical (positioning depends on it)                                                                  |
| Likelihood   | M (untested at scale in MY)                                                                           |
| Trigger      | Validation Experiment V1 fails (< 30 % of tea drinkers prefer co-ferment); cart conversion < 5 % at 2 venues |
| Impact       | Forced repositioning; loss of differentiation                                                         |
| Mitigation   | (1) Run V1 + V3 BEFORE capital committed, (2) defined go/no-go gates ([[02-Market Analysis & Validation#Stop / Go gates]]), (3) backup positioning prepared: "clean Indonesian specialty + light roast" without the tea-crossover frame, (4) menu design accommodates fallback (Gateway items work for both positions) |
| Early warning| Customer-question tracking: "I usually drink tea" frequency at the cart |

### 🔴 C-05 — Burnout (founder physical/mental)

| Field        | Detail                                                                                                |
| ------------ | ----------------------------------------------------------------------------------------------------- |
| Severity     | Critical                                                                                              |
| Likelihood   | H (small team + weekend-heavy schedule + multiple disciplines per founder)                            |
| Trigger      | Founder dreading next pop-up; chronic sleep < 6 hr; missing weekly review for "no time"               |
| Impact       | Quality drops invisibly first, then visibly. Recovery takes months.                                   |
| Mitigation   | (1) **Mandatory full off-day per week** (each founder), (2) max 5 pop-up days per founder per week, (3) weekly burnout check ([[Technicals/Daily & Weekly Goals#The Burnout Watch]]), (4) outsource what you can ($-cheap tasks: bookkeeping, basic logistics) by month 6, (5) ban "founder hustle" content — performing burnout normalises it |

---

## 3. High Risks (significant but not lethal)

### 🟠 H-01 — Reputation backlash from co-ferment "controversy"

| Detail | |
| ------ | --- |
| **What happens** | A vocal Malaysian / regional coffee critic (IG, podcast, Reddit) publicly attacks KNNO for "fake flavours" / "infusion-as-fermentation" / "not real coffee." |
| **Why it's plausible** | Co-ferments — especially fruit co-ferments — are the most controversial category in specialty right now. Some purists are aggressive online. |
| **Mitigation** | (1) **Transparency-first messaging**: process always disclosed, never sell infused as terroir, (2) document every lot's fermentation protocol publicly, (3) prepared response template (calm, factual, link to E-05 in [[04-Brand & Marketing Strategy#Education Library]]), (4) build credibility with cuppers / SCAM (Specialty Coffee Association Malaysia) early — they'll be the ones who defend you, (5) never punch down or argue publicly with critics |
| **Early warning** | Monitor IG mentions, hashtag #cofermentation weekly |

### 🟠 H-02 — Indonesian harvest failure / supply shock

| Detail | |
| ------ | --- |
| **What happens** | Climate shock (drought, flood, late rainfall) wipes out a key Indonesian harvest; Frinsa or Aceh lots become unavailable mid-year. |
| **Why it's plausible** | Climate volatility is the #1 risk to specialty coffee globally per ICO 2025 reports. |
| **Mitigation** | (1) 3-region portfolio (Aceh + Java + Bali / Sulawesi), (2) forward-buy 3-month inventory when prices favourable, (3) maintain 1 vetted non-Indonesian backup origin (Vietnam Da Lat / PNG / Philippines for proximity), (4) "guest series" model can sub-in (see [[03-Business Model & SWOT#Alternative Model — Importing Roasted Beans]]) |

### 🟠 H-03 — Forex (MYR/IDR/USD) shock

| Detail | |
| ------ | --- |
| **What happens** | Ringgit weakens 8–15 % against USD/IDR — green coffee landed cost rises sharply (RM 4–8/kg), eroding margin or forcing price hike. |
| **Mitigation** | (1) Forward-buy 3-month inventory when MYR strong, (2) pricing flexibility: signature drinks tested at RM 18 ceiling, (3) shift mix to lower-cost lots if needed (more washed, less premium co-ferment), (4) consider Indonesian rupiah purchases via fintech (Wise, Aspire) to avoid double-FX |
| **Early warning** | Track USD/MYR weekly; trigger conversation at any 5 % move |

### 🟠 H-04 — Equipment failure mid-pop-up

| Detail | |
| ------ | --- |
| **What happens** | Espresso machine, grinder, or cart electrical fails on a Saturday morning at a paid market. |
| **Mitigation** | (1) **Manual brewing fallback ready** (V60s + Comandante hand grinder + kettle on portable burner) — can serve filter for 2–3 hours without espresso, (2) backup espresso machine identified (loan or rental — friend roaster or Coffex 24-hr), (3) pre-event equipment check checklist ([[Technicals/Stock Management & Operations#Cart Load-Out Checklist]]), (4) service contract or "on-call" mechanic arranged with Coffex / Bean Brothers, (5) accept 1 day's loss as cost of doing business — communicate honestly with venue / customers |

### 🟠 H-05 — Pop-up venue cancellation / vendor fee inflation

| Detail | |
| ------ | --- |
| **What happens** | Riuh / Sunny Side Up either cancels event last-minute (weather, force majeure) or significantly raises vendor fees mid-year. |
| **Mitigation** | (1) Multi-venue portfolio (4+ confirmed venues by Q4), (2) "owned route" — recurring office-park or coffee-friend-cafe slot that doesn't depend on market organisers, (3) strong contracts that specify cancellation terms, (4) cash buffer to absorb 1 lost weekend |

### 🟠 H-06 — DBKL / KKM / regulatory action on mobile food vendors

| Detail | |
| ------ | --- |
| **What happens** | DBKL changes mobile hawker rules, raises licensing fees, or restricts certain venue types. KKM tightens "co-fermented" labelling rules. |
| **Mitigation** | (1) Legal counsel for licensing setup (one-time RM 1.5K), (2) maintain compliance margin (always 100 % licensed, never operate without papers), (3) join Malaysian F&B associations (early warning network), (4) packaging copy reviewed by food regulation specialist before any large print run |

### 🟠 H-07 — Competitor entrance into co-ferment cart space

| Detail | |
| ------ | --- |
| **What happens** | An established roaster (One Half, The Crackpots, Discover) launches a co-ferment-focused mobile arm. |
| **Mitigation** | (1) **Move fast on category ownership** — first 90 days post-launch is the brand-recall land grab, (2) education moat compounds; competitors must rebuild it, (3) producer relationships create exclusivity (we hold the lot, they don't), (4) framing: KNNO is *cart-native*, not a roaster's side hustle |
| **Early warning** | Monitor competitor IG quarterly; track "co-ferment" mentions from established brands |

### 🟠 H-08 — CoRoasting facility closure / terms change

| Detail | |
| ------ | --- |
| **What happens** | CoRoasting KL closes, raises rates 50 %+, or changes scheduling rules in ways that break our cadence. |
| **Mitigation** | (1) Backup co-roasting partners identified by month 3 ([[06-Suppliers & Co-Roasting#Backup co-roasting plan]]), (2) sample roaster purchased by month 9 if business is profitable (reduces dependency), (3) maintain warm relationship with at least one independent KL roaster (mentor + emergency capacity) |

---

## 4. Medium Risks (worth tracking, not panicking over)

| ID    | Risk                                                                          | Severity | Likelihood | Mitigation summary                                                                |
| ----- | ----------------------------------------------------------------------------- | -------- | ---------- | --------------------------------------------------------------------------------- |
| M-01  | Cart aesthetic looks DIY in early days, hurts brand                           | M        | M          | Invest in cart finish + cup design BEFORE first sale; defer flexible items        |
| M-02  | Online bag fulfilment delays / quality issues                                  | M        | M          | Test shipping flow with 5 friend orders before public launch; pack ≤ 24h post-roast |
| M-03  | One-product over-reliance (Strawberry Patch lot ends, no signature)            | M        | M          | Always have 2 signature-tier lots in rotation; menu transition planned 6+ wks ahead |
| M-04  | Customer education resistance ("just give me a normal coffee")                 | M        | M          | Gateway items on menu; never preach unprompted; education is *available* not *imposed* |
| M-05  | Workshop launch underperforms                                                  | M        | L-M        | Soft-launch with 8 known faces before public ticket sale; iterate 2× before scale |
| M-06  | IG / TikTok algorithm changes hurt organic reach                               | M        | H          | Diversify across IG + TikTok + RedNote + email (Y2); content quality > algorithm hacking |
| M-07  | Wholesale (Y2) pulls roasting capacity from cart                               | M        | M          | Cap B2B revenue at 15 % of total; always cart-first allocation                    |
| M-08  | Founders' day jobs interfere with operations                                   | M        | M          | Define ops boundaries with day jobs in writing; agree on "go full-time" trigger    |
| M-09  | Halal certification questions from customers                                   | M        | M          | Decide stance early (pursue or explain non-pursuit); consistent communication      |
| M-10  | Insurance gap → liability event                                                | M        | L          | Public liability insurance from day 1 (RM 600–1,200/yr)                            |
| M-11  | Cold brew goes off / makes a customer sick                                     | M (legal) | L         | Strict freshness windows ([[Technicals/Stock Management & Operations#Freshness windows]]); food handler cert |
| M-12  | Brand visual overlap with another KL brand                                     | M        | L          | Trademark search before signage print; quarterly KL brand scan                     |

---

## 5. Low Risks (note and forget)

- Loss of social media account (mitigation: backup founder credentials + email recovery + occasional CSV export)
- Domain expiration (mitigation: 5-year auto-renew; calendar alert)
- Single-day weather event (mitigation: indoor venue mix)
- Founder phone failure on cart day (mitigation: spare device with cart payment apps installed)
- Negative review (single one) (mitigation: respond once, calmly, then move on)

---

## 6. Risk Heatmap

```
                    LIKELIHOOD →
                    Low         Medium         High
                ┌────────────┬──────────────┬─────────────┐
       Critical │            │  C-04       │ C-01, C-02, │
                │            │              │ C-03, C-05  │
                ├────────────┼──────────────┼─────────────┤
S        High   │  H-04      │ H-01, H-02,  │ H-06        │
e               │            │ H-03, H-05,  │             │
v               │            │ H-07, H-08   │             │
e               ├────────────┼──────────────┼─────────────┤
r        Medium │ M-10, M-11 │ M-01, M-02,  │ M-06        │
i               │            │ M-03–M-09,   │             │
t               │            │ M-12         │             │
y               ├────────────┼──────────────┼─────────────┤
         Low    │ all others │              │             │
                └────────────┴──────────────┴─────────────┘
```

Top-right = top priority for active mitigation.

---

## 7. The Pre-Mortem Exercise (run quarterly)

> Adapted from Gary Klein's pre-mortem methodology.

### Process (45 min, both founders)

1. **(5 min) Set the scene:** "It's December 2027. KNNO has shut down. Why?"
2. **(15 min) Independent silent writing:** each founder writes 5 distinct reasons. No discussion.
3. **(10 min) Share + categorise:** map answers against the risk register.
4. **(10 min) Surface novel risks:** new ones from this round → add to register.
5. **(5 min) Pick one new mitigation per founder** for this quarter.

> The goal is not paranoia. It's **rehearsing the story before it has a chance to happen**.

---

## 8. Risk Mitigation Calendar

| When                          | What                                                                               |
| ----------------------------- | ---------------------------------------------------------------------------------- |
| Pre-launch (Jun 2026)         | Operating agreement signed (C-01); insurance live (M-10); validation V1+V3 (C-04)  |
| Launch month (Jul 2026)       | Backup espresso machine identified (H-04); cash buffer rule activated (C-02)       |
| Month 3 (Sep 2026)            | Backup co-roastery identified (H-08); first quarterly pre-mortem                   |
| Month 6 (Dec 2026)            | Forward-buy decision (H-02, H-03); founder check-in (C-01, C-05)                   |
| Month 9 (Mar 2027)            | Sample roaster purchase decision (C-03, H-08)                                      |
| Month 12 (Jun 2027)           | Year-1 risk register full review                                                   |

---

## 9. Crisis Playbooks (if X happens, do Y)

### Playbook: Equipment failure 1 hour before opening

1. Don't panic; assess what works
2. If espresso fails → switch to V60-only menu; shrink offering; communicate at queue
3. If grinder fails → use backup hand grinder; slow service to 1 cup / 90 sec; cap day at 25 cups
4. If electric fails → relocate within venue near another power source OR cancel + post a calm IG story
5. After event: full diagnostic; service appointment within 48 hr; don't roast/serve again on that machine until cleared

### Playbook: Bad cup served / customer complaint

1. Acknowledge immediately; offer a remake or refund (no negotiation)
2. Taste the dump cup yourself (root-cause data)
3. Apologise sincerely, briefly; do not over-explain or technicalise
4. If customer wants to leave, give them a 100g sampler with sincere card
5. Log the incident in the day log
6. Address root cause that night (grind, extraction, lot quality)
7. If pattern across 2+ cups same day: pull the lot from menu

### Playbook: Negative public review / IG callout

1. Read fully twice; don't reply for 4 hours
2. Distinguish: critique of *fact* (engage) vs critique of *taste* (acknowledge & move on)
3. If factual: respond once, calmly, with specifics ("Here's our process — link"). No defensiveness, no stacking responses
4. If taste-based: "Thanks for trying. Co-ferments aren't for everyone. Hope to see you next harvest." End.
5. Never reply more than once. Don't get baited.
6. Document the interaction in [[07-Risks & Failure Modes]] for pattern-tracking

### Playbook: Co-founder unable to operate (illness, family event, leave)

1. Other founder takes the cart day(s) solo; cap menu to 4 items
2. Roast schedule extended by partner-week duration
3. Communicate honestly with customers: "Smaller crew this week."
4. After 2 weeks: assess timeline; bring in trusted helper at fair pay if needed
5. Quarterly review revisits cover-arrangements

### Playbook: Cash runway crisis (working cash < RM 6K)

1. Emergency 24-hr meeting; honest review
2. Cut all discretionary spend (content, marketing experiments, future stock orders)
3. Push price ceiling to RM 18 on all signature drinks
4. Reduce roast frequency to bare minimum; prioritise cart over online fulfilment
5. Reach out to identified bridge sources (family / micro-loan / friend with deferred terms)
6. Re-do financials with current run-rate; pre-decide RM-floor for "shut down vs. continue" decision
7. Communicate to customers and supporters honestly *only if* shutdown is < 30 days away

---

## My Notes & Thoughts

- _Co-founder breakdown is the highest-likelihood killer. The operating agreement isn't optional. Get it signed before spending a ringgit on equipment._
- _The cash buffer rule (RM 4.5K floor) feels paranoid until the month it saves you._
- _Run the pre-mortem at your first quarterly even if everything is going well — especially then. The quarter when things look fine is the quarter blind spots grow._
- _Read the playbooks once now, calmly. Don't wait until you need them._
