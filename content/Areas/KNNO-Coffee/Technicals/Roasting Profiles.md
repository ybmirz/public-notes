---
title: Roasting Profiles — Light Roasts for Co-Ferments
type: technical
parent: "[[00-Index]]"
status: draft-v1
updated: 2026-05-04
tags: [knno, roasting, technical, profiles, co-ferment]
---

# Technicals — Roasting Profiles

> [!warning]
> This document is a **working philosophy + starting profile library**. Every profile must be re-validated when the green changes — even from the same producer. The numbers below are *anchors*, not gospel.

---

## 1. Roasting Philosophy (the immovable principles)

1. **Light roast, always.** Drop temp ≤ 198 °C bean / equivalent Agtron 70+. We do **not** chase second-crack character on co-ferments. Ever.
2. **Development time matters more than total time.** Target dev-time ratio (DTR) of **15–22 %**.
3. **Preserve fermentation character — don't bake it out.** Long Maillard, short development.
4. **No burnt character. No baked character.** Either is a disqualifying defect.
5. **Reproduce, then refine.** Three roasts of a new lot before a final profile is locked.
6. **Document every roast.** Even a "bad" one is data — see [[Technicals/Stock Management & Operations#Roast Log]].

---

## 2. Why Light Roast for Co-Ferments

Co-fermented coffees are **chemically delicate**. The fermentation produces volatile aromatics (esters, alcohols, organic acids) that:

- Are **destroyed by darker roasts** — caramelisation and Maillard reactions overpower fermentation flavours.
- Need **less heat to develop** because fermentation has already done part of the "flavour work."
- Need **clear extraction** — under-roast = grassy, over-roast = burnt-fruit.

The light roast preserves what the producer paid for. Anything else is paying for a Ferrari and driving it 30 km/h.

---

## 3. Roast Targets (the numbers)

| Variable                        | Target window           | Notes                                                                  |
| ------------------------------- | ----------------------- | ---------------------------------------------------------------------- |
| **Charge temp**                 | 175–185 °C              | Lower for smaller batch / lighter green                                |
| **Turnaround time**             | 1:30–2:00               | Indicator of momentum control                                          |
| **First Crack (FC) start**      | 8:30–10:30              | Earlier crack = use lower charge / softer ramp                         |
| **First Crack temp**            | 192–198 °C              | Bean probe; calibrate per machine                                      |
| **Drop time (after FC)**        | 1:30–2:30               | Development time                                                       |
| **Drop temp**                   | 200–205 °C              | Light end                                                              |
| **Total roast time**            | 10:00–13:00             | Longer = baked risk; shorter = under-developed                         |
| **DTR (development ratio)**     | **15–22 %**             | Sweet spot for co-ferments specifically                                |
| **Weight loss**                 | 11.5–13.5 %             | Lower for naturals/anaerobic; higher for washed                        |
| **Agtron (whole bean / ground)**| 75–85 / 80–90           | Use a colourimeter every batch                                         |

> [!info]
> Calibrate to your **specific co-roasting machine** (likely a Probat L12, Diedrich IR-12, or Giesen W6 at CoRoasting KL). All numbers above shift slightly per machine. **Lock the calibration in your first 5 roasts.**

---

## 4. Profile Library (starting recipes)

### 🌸 Profile A — Fruit Co-Ferment (Anaerobic Natural)

**Use for:** Frinsa fruit-co-ferment lots, Aceh anaerobic strawberry/jasmine lots, similar.

| Phase            | Time          | Bean temp     | Heat / air                                               |
| ---------------- | ------------- | ------------- | -------------------------------------------------------- |
| Charge           | 0:00          | 178 °C        | Heat 80 %, air 30 %                                      |
| Turnaround       | 1:45          | ~96 °C (low)  | Hold heat                                                |
| Yellowing        | 4:30          | 145 °C        | Heat 70 %, air 40 %                                      |
| Maillard begins  | 6:30          | 165 °C        | Heat 60 %, air 50 %                                      |
| First Crack      | 9:30 ± 30 s   | 195 °C        | Heat 35 %, air 70 % (don't stall the momentum)           |
| Development      | 9:30 → 11:30  | 195 → 202 °C  | Heat 30 %, air 75 %                                      |
| Drop             | 11:30         | 202 °C        | Cool fast — full air                                     |

- **Total:** 11:30
- **DTR:** 17.4 %
- **Target cup:** Strawberry, jasmine, candied citrus, light cream-soda body
- **Agtron target:** ~80 / 85

### 🍇 Profile B — Carbonic Maceration (CM)

**Use for:** Bali Kintamani CM, Java Halu CM, similar.

| Phase            | Time          | Bean temp     | Heat / air                                               |
| ---------------- | ------------- | ------------- | -------------------------------------------------------- |
| Charge           | 0:00          | 180 °C        | Heat 85 %, air 30 %                                      |
| Turnaround       | 1:40          | ~98 °C        | Hold                                                     |
| Yellowing        | 4:30          | 148 °C        | Heat 70 %, air 40 %                                      |
| Maillard         | 6:45          | 168 °C        | Heat 60 %, air 50 %                                      |
| First Crack      | 9:45 ± 30 s   | 196 °C        | Heat 35 %, air 70 %                                      |
| Development      | 9:45 → 12:00  | 196 → 203 °C  | Heat 30 %, air 75 %                                      |
| Drop             | 12:00         | 203 °C        | Cool fast                                                |

- **Total:** 12:00
- **DTR:** 18.7 %
- **Target cup:** Lychee, white grape, white wine acidity, light oolong finish
- **Agtron target:** ~78 / 83

### 🌾 Profile C — Yeast Inoculated

**Use for:** Lots labelled with specific yeast strain (e.g. *Saccharomyces cerevisiae*, *Lalvin EC1118*).

| Phase            | Time          | Bean temp     | Heat / air                                               |
| ---------------- | ------------- | ------------- | -------------------------------------------------------- |
| Charge           | 0:00          | 178 °C        | Heat 80 %, air 30 %                                      |
| Turnaround       | 1:45          | ~96 °C        | Hold                                                     |
| Yellowing        | 4:45          | 145 °C        | Heat 70 %, air 40 %                                      |
| Maillard         | 6:45          | 165 °C        | Heat 60 %, air 50 %                                      |
| First Crack      | 9:30 ± 30 s   | 194 °C        | Heat 35 %, air 70 %                                      |
| Development      | 9:30 → 11:15  | 194 → 201 °C  | Heat 30 %, air 75 %                                      |
| Drop             | 11:15         | 201 °C        | Cool fast                                                |

- **Total:** 11:15
- **DTR:** 15.6 %
- **Target cup:** Clean stone fruit, beer-like (subtle), bright clean acidity
- **Agtron target:** ~82 / 87

### 🫧 Profile D — Washed (Gateway / Espresso Anchor)

**Use for:** Aceh Gayo washed, Toraja washed — the daily-driver lot.

| Phase            | Time          | Bean temp     | Heat / air                                               |
| ---------------- | ------------- | ------------- | -------------------------------------------------------- |
| Charge           | 0:00          | 185 °C        | Heat 90 %, air 30 %                                      |
| Turnaround       | 1:30          | ~99 °C        | Hold                                                     |
| Yellowing        | 4:30          | 148 °C        | Heat 75 %, air 40 %                                      |
| Maillard         | 6:30          | 168 °C        | Heat 65 %, air 50 %                                      |
| First Crack      | 9:00 ± 30 s   | 196 °C        | Heat 40 %, air 65 %                                      |
| Development      | 9:00 → 11:00  | 196 → 204 °C  | Heat 35 %, air 70 %                                      |
| Drop             | 11:00         | 204 °C        | Cool fast                                                |

- **Total:** 11:00
- **DTR:** 18.2 %
- **Target cup:** Cocoa, brown sugar, almond, citrus finish, smooth body
- **Agtron target:** ~75 / 80
- **Espresso variant:** drop +30 s later (11:30) for a slightly fuller body — separate batch.

### 🌰 Profile E — Espresso (House Washed, slightly developed)

**Use for:** Pulling clean espresso from washed Indonesian. Same green as Profile D but ~30 s longer dev.

| Phase            | Time          | Bean temp     | Heat / air                                               |
| ---------------- | ------------- | ------------- | -------------------------------------------------------- |
| (Identical to D until FC) |       |               |                                                          |
| Development      | 9:00 → 11:30  | 196 → 205 °C  | Heat 35 %, air 70 %                                      |
| Drop             | 11:30         | 205 °C        | Cool fast                                                |

- **Total:** 11:30
- **DTR:** 21.7 %
- **Target cup (espresso):** Bittersweet chocolate, almond, brown sugar, smooth crema
- **Agtron target:** ~72 / 78

---

## 5. Defect Playbook (what to taste for, what caused it)

| Defect                      | Tastes like                                | Most likely cause                                          | Fix                                           |
| --------------------------- | ------------------------------------------ | ---------------------------------------------------------- | --------------------------------------------- |
| **Baked**                   | Flat, papery, cardboard finish             | DTR too high; long flat development; low energy through FC | Higher heat through Maillard; shorter dev     |
| **Under-developed**         | Grassy, hay, sour green-apple sharpness    | DTR < 14 %; FC too late; insufficient heat through dev     | Push heat through FC; extend dev by 30–45 s   |
| **Burnt / roasty**          | Ash, smoke, bitter after-taste             | Drop temp too high; over-roast                             | Drop earlier; lower roast end-point           |
| **Tipping** (charred ends)  | Acrid, ash on chocolate-bar finish         | Charge too high; aggressive early heat                     | Lower charge by 5–10 °C; slower early ramp    |
| **Scorching** (face burn)   | Sharp, smoky, uneven                       | Drum over-loaded; insufficient airflow at charge           | Reduce batch size; open air earlier           |
| **Stalled roast**           | Dull, lifeless, low acidity                | Heat dropped too low post-FC; stalled momentum             | Maintain heat through dev; don't over-modulate |
| **Quaker (white beans)**    | Peanut, dry, low sweetness                 | Under-ripe cherries in lot; not always your fault          | Hand-sort post-roast; report back to importer |
| **Lost ferment character**  | "Just a coffee" — no fruit, no ferment     | Roast too dark; or green was over-fermented + you over-corrected | Lighter drop; check green for over-ferment  |

---

## 6. Cupping Protocol (post-roast QC)

> Every roast batch gets cupped within 24–72 h of roast (rest period). No exceptions.

### Setup
- 8.25 g coffee, 150 ml water, 93 °C
- Grind: standard cupping (slightly coarser than V60)
- Steep: 4 min, then break crust
- Score: SCA 100-point sheet, but our 4-point internal:

| Internal score | Acceptance                                                                  |
| -------------- | --------------------------------------------------------------------------- |
| 4              | Excellent — meets target; can sell as signature                              |
| 3              | Good — meets target; can sell as house drip                                  |
| 2              | Off-target but clean — repurpose to milk-drink only                          |
| 1              | Defective — do NOT serve; root-cause + log; possible brewing-class material  |

### What to taste for (in order)
1. **Cleanliness** — any dirty, papery, dusty, or chemical notes? If yes → defect investigation.
2. **Sweetness** — present? Caramel / fruit / honey?
3. **Acidity** — bright, structured? Or sharp / sour / under-developed?
4. **Body** — appropriate to the lot (light for co-ferment, fuller for espresso)?
5. **Fermentation character** — present and pleasant? Or muted / overwhelming?
6. **Finish** — clean, lingering, balanced? Or short / harsh / lingering bitter?

---

## 7. Sample Roasting (the new-lot ritual)

> When evaluating a *new* green lot from a producer / importer, roast 3 samples before committing to a profile.

### Sample roast 1 — "Find first crack"

- Charge 200 g (or whatever your sample roaster takes)
- Use a generic light-roast curve (Profile A or D as starting point)
- Note: First crack timing, momentum, weight loss
- Cup → score on the 4-point internal scale

### Sample roast 2 — "Refine to target"

- Adjust based on roast 1 (e.g. shorter dev, slightly higher drop)
- Cup again — does it hit target profile?

### Sample roast 3 — "Confirm reproducibility"

- Repeat roast 2 exactly
- Cup blind against roast 2 — should score within 0.2 points
- If yes: lock the profile, scale to production batch. If no: investigate variable (green moisture, ambient temp, machine drift).

---

## 8. Co-Roasting Workflow at CoRoasting KL

> See also [[06-Suppliers & Co-Roasting#CoRoasting KL — Operating Notes]].

### Pre-session checklist
- [ ] Green weighed and pre-loaded into batch bins (labelled with lot ID)
- [ ] Roast log sheet ready (printed or tablet)
- [ ] Cupping samples planned (which roasts will be cupped Day +2)
- [ ] Calibration roast first (5–10 min "warm the machine" with a low-stakes batch)
- [ ] Probe + colourimeter checked
- [ ] Ventilation + RH noted (affects roast)

### During session
- Roast in order: lightest profile → darkest (no "dark to light" — leftover heat affects light profiles)
- Log every roast with: lot ID, batch weight, charge temp, FC time, drop time, drop temp, weight after, % loss, ambient T/RH, your subjective notes
- Resist the urge to "fix" mid-roast on a familiar profile — log the result and adjust the *next* batch

### Post-session
- Cool batches to room temp (15–20 min in cooling tray) before bagging
- Single-batch bag in degassing valve bags; label with lot + roast date
- Update [[Technicals/Stock Management & Operations#Roasted Inventory Log]]
- Schedule cupping for 24–72 h later (block calendar slot)

---

## 9. Roast Log Template (use this every roast)

```
ROAST LOG — KNNO COFFEE
Date: ____________  Time: __________  Roaster: ______________
Machine: __________________________
Ambient T/RH: ______ °C / ______ %

Lot ID: _____________________
Producer: ___________________
Process: ____________________
Profile target: _____________  (A / B / C / D / E)

Batch in: _______ g     Batch out: _______ g    Loss: ______ %
Charge T: _______ °C
Turnaround: _____ s     Yellowing: _____ s
First Crack: _____ s @ _____ °C
Drop: _____ s @ _____ °C
DTR: _____ %

Visual: Agtron WB ____ / GR ____

Subjective during roast:
________________________________________________________

Cupping (Day +2): ____________
Score (1–4): _____   Notes: __________________________________
________________________________________________________

Action / next-batch adjustment:
________________________________________________________
```

---

## 10. Roast Volume Planning

| Period    | Roast frequency | Weekly green volume (kg) | Batches/week |
| --------- | --------------- | ------------------------ | ------------ |
| Launch (Jul–Aug 2026) | 1× / week | 6–8 kg                  | 1–2 batches |
| Q4 2026   | 2× / week       | 12–18 kg                 | 2–4 batches  |
| Q1 2027   | 2–3× / week     | 18–25 kg                 | 4–6 batches  |
| Q2 2027   | 3× / week       | 25–35 kg                 | 6–8 batches  |

Cross-ref: [[Technicals/Stock Management & Operations#Roast Schedule]] and [[09-Financial Projections (Conservative)]].

---

## 11. When to Buy Our Own Roaster (the trigger)

**Move from co-roasting to own roaster when ALL three are true:**

1. We're consistently roasting > 30 kg/week
2. CoRoasting slot wait-time exceeds 2 weeks for 2 consecutive months
3. We have RM 30K+ in retained earnings *or* a confirmed financing path

**Likely first machine:** 1 kg sample roaster (RM 8–15K — Aillio Bullet, Sandbox, or used Probat sample) OR a 5 kg production roaster (RM 35–55K — used Diedrich, refurbished Probat).

> [!warning]
> Don't buy a roaster on revenue alone. Buy it when **you can already roast better than the co-roastery facility limits allow**. If consistency hasn't been proven on rented gear, owning gear won't fix it.

---

## My Notes & Thoughts

- _DTR is the single most important variable for our brand. If you can only track one number per roast, track this._
- _The defect playbook is your friend. Print it and tape it to the roaster bench. The first time you serve a "baked" cup, you'll know exactly what to look for next session._
- _Profile A (Strawberry Patch) is the brand. Master it before everything else. Roast it 5+ times before launch._
- _Consider keeping a private "rejected lots" cupping diary — coffees you tasted and decided not to buy. The pattern of what you reject is as informative as what you accept._
