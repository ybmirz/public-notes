---
title: Stock Management & Operations
type: technical
parent: "[[00-Index]]"
status: draft-v1
updated: 2026-05-04
tags: [knno, stock, inventory, operations, cart, sop]
---

# Technicals — Stock Management & Operations

> [!info]
> "Operations" at KNNO is small enough to fit on three sheets of paper. The point of this doc is to **make the small things automatic** so brain-bandwidth stays free for craft + customer.

---

## 1. Inventory Math

### Green coffee — the core formula

```
Monthly green needed (kg) = (Cups per month × 0.015 kg/cup) ÷ 0.87 (post-roast loss factor)
                          + (Bags per month × 0.20 kg/bag)   ÷ 0.87
                          + (Workshop pax × 0.05 kg/pax)
                          + 15 % buffer for QC / waste / sampling
```

### Worked example (Base scenario, month 4 — see [[09-Financial Projections (Conservative)]])

- 12 pop-up days × 45 cups = 540 cups
- 540 cups × 15 g = 8.1 kg (roasted)
- ÷ 0.87 = **9.3 kg green** for cup
- 15 bags × 200 g = 3.0 kg (roasted) ÷ 0.87 = **3.4 kg green** for bags
- Buffer: **+1.9 kg**
- **Total: ~14.6 kg green per month** at base scenario

### Roasted coffee — par levels

| Stock state                        | Roasted on hand (kg) | When to roast next     |
| ---------------------------------- | -------------------- | ---------------------- |
| **Healthy**                        | 6–10                 | Schedule next slot     |
| **Re-order trigger**               | 3–5                  | Book co-roastery slot in 3 days |
| **Critical (RED)**                 | < 2                  | Immediate re-roast within 48 hr |
| **Stockout — DO NOT REACH**        | < 0.5 of any signature SKU | Eat the cost; pull SKU from menu rather than serve stale |

### Freshness windows (the hard rules)

| Coffee state                | Best window         | Hard cut-off           | Action at cut-off        |
| --------------------------- | ------------------- | ---------------------- | ------------------------ |
| Roasted whole bean (sealed bag) | Days 7–28 from roast | Day 35              | Move to staff drinks; do not sell |
| Roasted whole bean (opened bag at cart) | Days 7–14    | Day 21              | Move to staff / blend out |
| Roasted ground (espresso hopper) | < 10 min after grind | 60 min            | Dump and re-grind        |
| Roasted ground (filter dose)| < 5 min before brew | 15 min                 | Dump and re-grind        |
| Cold brew (refrigerated, sealed) | 5 days from steep | 7 days              | Dump                     |
| Cold brew (on tap at cart)  | 1 day open / 8 hr at room temp | 24 hr open  | Dump                     |
| Steamed milk leftover       | 0 (never re-use)    | 0                      | Dump immediately         |

> [!warning]
> **The freshness rules are non-negotiable.** This is the brand. A "saved" RM 4 of stale coffee costs us a customer for life.

---

## 2. Lot Master File (the canonical lot record)

> Maintained as a spreadsheet OR as a simple `.md` file per lot in `Technicals/Lots/`. One row per lot.

| Field                  | Example                              |
| ---------------------- | ------------------------------------ |
| Lot ID                 | KNNO-2026-001                        |
| Producer               | Frinsa Estate (Wildan Mustofa)       |
| Region                 | Pangalengan, West Java               |
| Variety                | Lini S 795                           |
| Altitude               | 1,400–1,600 masl                     |
| Process                | Anaerobic Natural · 96 h · fruit co-ferment (strawberry) |
| Harvest                | Jul 2026                             |
| Sourcing channel       | Direct from Frinsa                   |
| FOB price              | USD 22 / kg                          |
| Landed cost (KL)       | RM 142 / kg green                    |
| Quantity bought        | 15 kg                                |
| Date received          | 2026-06-12                           |
| Profile assigned       | Profile A (Roasting Profiles)        |
| First roast date       | 2026-06-15                           |
| Cup score (sample R3)  | 3.5 / 4                              |
| Status                 | Active — Signature                   |
| Last roast date        | 2026-06-28                           |
| Estimated end-of-stock | 2026-08-10                           |

---

## 3. Roast Schedule (weekly cadence)

### Year-1 default cadence

| Day            | Activity                                                          |
| -------------- | ----------------------------------------------------------------- |
| **Monday**     | Inventory count + plan week's roast                                |
| **Tuesday**    | CoRoasting session (3–4 hr)                                        |
| **Wednesday**  | Cupping + QC of Tuesday's batches                                  |
| **Thursday**   | Bag + label for online orders + cart re-stock                     |
| **Friday**     | Online drop ships out + content production                         |
| **Sat–Sun**    | Pop-up days (the brand engine)                                     |

### Why Tuesday roast?

- Roast Tuesday → cupped Wednesday → degassed by Sunday cart day (4–5 days post-roast = peak window).
- If you only get 1 cart day per week, push roast to Wednesday.

---

## 4. Cart Load-Out Checklist (the pre-pop-up SOP)

> Print this. Tape it inside the cart kit box. Tick every line every time.

### Coffee
- [ ] Filter lot — 600 g whole bean, sealed (~30 cups @ 18 g)
- [ ] Filter lot — backup 300 g
- [ ] Espresso lot — 500 g whole bean, sealed
- [ ] Cold brew — 4 L pre-batched, in keg + 1 L in spare bottle

### Equipment (cart kit)
- [ ] Espresso machine (1 group, 240V) — packed, gasket checked
- [ ] Espresso grinder — burrs checked, calibrated
- [ ] Filter grinder — burrs checked, calibrated
- [ ] V60 drippers ×2 + filters (50 ct)
- [ ] Cupping spoons ×4 (for staff side use)
- [ ] Scales ×2 (one per brew station)
- [ ] Thermometer + 2 spare batteries
- [ ] Kettle (gooseneck, 1 L) + spare 600 ml
- [ ] Knockbox + group brush + cleaning brushes
- [ ] Tamper, distribution tool, dosing funnel
- [ ] Milk pitchers ×3 (small, medium, training)

### Service ware
- [ ] Cups (12 oz) ×60
- [ ] Cup lids ×60
- [ ] Cup sleeves ×60 (pre-printed today's lot story)
- [ ] Water cups (small) ×80
- [ ] Wood serving trays ×4 (for flights)
- [ ] Stirrers / paddles ×100

### Consumables
- [ ] Whole milk ×4 L (cold pack)
- [ ] Oat milk ×1 L (cold pack)
- [ ] Mineral water ×6 L (jug + small bottles)
- [ ] Sugar packets ×30 (only at customer request)
- [ ] Napkins ×100
- [ ] Wet wipes / sanitiser
- [ ] Bin liners ×3
- [ ] Spare gas cylinder (if portable burner) / extension cable + adapter

### Brand assets
- [ ] Cart sign (today's location-specific or default)
- [ ] Today's menu card (printed)
- [ ] Lot story card (printed for today's signature lot)
- [ ] QR code holder (acrylic block)
- [ ] Stamp + ink (for cup date)
- [ ] Apron ×2 (clean)
- [ ] Business cards / postcards ×30
- [ ] Bag samples for sale ×6 (200g) + ×6 (100g)

### Admin
- [ ] Card reader (charged) + backup QR FPX
- [ ] Cash float — RM 200 in small notes
- [ ] Receipt book / order pad
- [ ] Day log sheet (cups served / observations)
- [ ] First aid kit
- [ ] Vendor permit / SSM cert copy (laminated)

### Pre-departure check
- [ ] Weather forecast checked (if outdoor)
- [ ] Vendor contact / load-in time confirmed
- [ ] Parking / load-in route mapped
- [ ] Backup partner contact in case of equipment failure
- [ ] IG story scheduled / drafted for "live now"

---

## 5. Cart Service SOP

### Standard service sequence (single cup)

```
1. Greet (eye contact). "What brings you in today?" — open question, 3–5 s.
2. Confirm order. Repeat lot name back: "Strawberry Patch — got it."
3. Brew. (Standard times → §1 Freshness windows + Profile docs.)
4. Pour into cup. Stamp date. Apply sleeve.
5. Place water cup IN FRONT of customer first, then coffee.
6. Hand off line: "Frinsa, West Java — see if you taste strawberry first."
7. Receipt / QR code: "Scan if you want this lot at home."
8. Goodbye line: "Come back if you don't taste strawberry. We'll talk."
```

### Standard service sequence (flight)

```
1. Greet, take order.
2. Pull espresso first (28 s) → set on tray (slot 1).
3. Pour V60 (3:00) → set on tray (slot 2).
4. Pour cold brew over ice → set on tray (slot 3).
5. Place printed flight card explaining order (espresso → V60 → cold brew).
6. Walk customer through: "Same coffee. Three temperatures. Notice how the flavour changes."
7. Stamp + sleeve nothing. The tray IS the brand statement.
8. Goodbye line: same as above.
```

### Service speed targets

| Drink    | Target prep time | Cap (re-think the bar if exceeded) |
| -------- | ---------------- | ---------------------------------- |
| Espresso | 90 s             | 2 min                              |
| V60      | 4 min            | 5 min                              |
| Latte    | 2 min            | 3 min                              |
| Flight   | 6 min            | 8 min                              |
| Cold brew (pre-batched) | 30 s | 60 s                       |

> [!tip]
> **One barista cap: 50 cups in a 6-hour day.** Beyond that, queue management collapses and the brand calm breaks. Add a 2nd barista or close orders.

---

## 6. Equipment List & Maintenance Schedule

### The launch kit (RM 8–12K total, see [[09-Financial Projections (Conservative)#Launch Capex Breakdown]])

| Item                                     | Spec / model suggestion              | Cost (RM)    | Maintenance              |
| ---------------------------------------- | ------------------------------------ | ------------ | ------------------------ |
| Cart frame (custom or modular)           | Steel + timber, 1.2m × 0.6m          | 1,800–3,000  | Tighten bolts monthly    |
| 1-group espresso machine (commercial-grade portable) | Sage Dual Boiler (gateway) → ECM Synchronika or used La Marzocco Linea Mini | 3,000–6,500  | Backflush daily; descale monthly; gasket every 6 mo |
| Espresso grinder                         | Eureka Mignon Specialita / Niche Zero | 1,800–2,800  | Burr clean weekly; replace burrs annually |
| Filter grinder                           | Comandante (manual) initially, then Mahlkönig EK43 (year 2) | 800 (manual) → 8,000 (EK43) | Burr clean weekly |
| Kettle (electric gooseneck)              | Brewista or Hario EVKG                | 350–500      | Descale monthly          |
| V60 drippers                             | Hario plastic ×2 + glass ×1           | 120          | Hand-wash daily          |
| Scale                                    | Acaia Pearl S ×2                      | 600 (used)   | Calibrate monthly        |
| Knock box, brushes, tamper, distribution | Generic                               | 250          | Clean daily              |
| Cold brew kegs (5 L)                     | Mini-keg ×2                           | 350          | Sanitise after each use  |
| Pitchers (milk)                          | 350ml, 600ml ×2 each                  | 180          | Hand-wash daily          |
| Cooler box (cart-day milk)               | 30L                                   | 220          | Wipe weekly              |
| **Total**                                |                                       | **~8,800–13,500** |                       |

### Daily maintenance (end of pop-up day)
- [ ] Backflush espresso machine (2 cycles plain water)
- [ ] Wipe down all surfaces (food-safe sanitiser)
- [ ] Empty knock box, rinse
- [ ] Empty drip tray
- [ ] Detach milk steam wand, soak overnight
- [ ] Wipe grinder hopper exterior; if changed beans, brush burrs
- [ ] Sanitise milk pitchers, drying rack
- [ ] Bin all used filters, paper

### Weekly maintenance
- [ ] Backflush espresso machine with cleaner (Cafiza tablet)
- [ ] Strip + soak espresso group (gaskets, screen)
- [ ] Brush + vacuum grinder burr chambers
- [ ] Descale kettle
- [ ] Sanitise cold brew kegs
- [ ] Inspect cart frame for wear

### Monthly maintenance
- [ ] Descale espresso machine
- [ ] Replace shower screen (if needed)
- [ ] Burr inspection (espresso grinder)
- [ ] Cup count audit + restock all consumables to par
- [ ] Equipment service log update

### Quarterly
- [ ] Replace espresso machine gasket
- [ ] Replace burrs if >100 kg processed
- [ ] Deep-clean cart frame (dismantle + sand if needed)
- [ ] Review tool SOPs with co-founder

---

## 7. Hygiene & Compliance

### Required licensing (KL / DBKL — confirm with consultant)

- [ ] SSM business registration (Sole Prop or Sdn Bhd)
- [ ] DBKL Mobile Hawker / Roving Hawker licence (or equivalent)
- [ ] MOH food handler certificate (typhoid jab + course) — both founders
- [ ] Halal status decision (if pursued, JAKIM application)
- [ ] Liability insurance (~RM 600–1,200/year)
- [ ] Vendor agreement signed per market (separate per venue)

### Daily hygiene SOP
- Wash hands before service start, after every break, between tasks
- Apron clean every shift; hair tied back
- No eating / drinking from customer-facing cups
- Sanitiser at-arm's-reach throughout service
- Open wounds → cover and glove
- Sick day → don't show up. Period.

### Allergen disclosure
- Milk used (cow + oat); always state if asked
- No tree nuts, no soy, no gluten on cart by default
- Cross-contact possible at pop-up venues — disclose

---

## 8. Cash & Payments

### Float setup (per pop-up day)
- RM 200 in small notes (RM 1, 5, 10)
- Coin tray (~RM 30 in 50sen + RM 1)
- Card reader (TouchNGo / Boost / GrabPay enabled QR + FPX)
- Backup: founder phone with PayNet QR

### End-of-day reconciliation
- [ ] Card terminal reads day total
- [ ] QR/FPX day total via app
- [ ] Cash counted; reconcile against day log sheet
- [ ] Variance > RM 10 → investigate before tomorrow
- [ ] Day total + breakdown logged in [[Technicals/Daily & Weekly Goals#Daily Log Template]]

### Banking cadence
- Cash deposited weekly (or bi-weekly if low volume)
- Card / QR auto-settled to business account next-day
- Monthly: reconcile all rails against invoices

---

## 9. Roasted Inventory Log (template)

```
ROASTED INVENTORY — KNNO COFFEE
Updated: ____________

Lot ID | Roast date | Original (kg) | On hand (kg) | Days since roast | Status
-------|-----------|---------------|--------------|------------------|--------
KNNO-2026-001 (Frinsa Strawberry) | 2026-06-15 | 12.5 | 4.2 | 13 | Active
KNNO-2026-002 (Aceh Washed)       | 2026-06-15 | 5.0  | 1.8 | 13 | Active
KNNO-2026-003 (Bali CM)           | 2026-06-22 | 4.0  | 3.6 | 6  | Active

Trigger states:
- Frinsa: BELOW PAR — schedule re-roast within 5 days
- Aceh: HEALTHY
- Bali: HEALTHY

Total active stock: 9.6 kg
Estimated days of supply at current draw: 14 days
```

---

## 10. Green Inventory Log (template)

```
GREEN INVENTORY — KNNO COFFEE
Updated: ____________

Lot ID | Producer       | Process          | Bought | Used | On hand | %ile age (vs. harvest)
-------|----------------|------------------|--------|------|---------|----------------------
G-001  | Frinsa Estate  | Anaerobic strawberry | 15 kg | 9 kg | 6 kg  | 6 mo from harvest
G-002  | Aceh Permata   | Washed Grade 1   | 10 kg | 6 kg | 4 kg    | 4 mo from harvest
G-003  | Bali Kintamani | CM 72h           | 8 kg  | 2 kg | 6 kg    | 3 mo from harvest

Total green on hand: 16 kg
Months of supply at current draw: ~1.6 mo
Reorder triggers: NONE this week (target ≥ 1.5 mo on hand)
```

---

## 11. Storage Setup

### Green storage (room temp, dry, dark)
- GrainPro bags inside Pacojet plastic bins or food-safe drums
- Storage temp: 18–25 °C, RH < 60 %
- Off the floor (pallet or shelf, ≥ 15 cm clearance)
- Labelled: lot ID, producer, weight, date received

### Roasted storage (degassing valve bags)
- 1 kg or 2 kg degassing-valve bags (foil-lined)
- Cool, dry, ambient (20–25 °C)
- Labelled: lot ID, roast date, profile used, batch #

### At-cart storage
- Sealed cans / mason jars (UV-resistant)
- Open quantities only (≤ 1 kg per SKU per cart day)

> [!warning]
> Do not freeze roasted coffee in standard freezers (humidity damages aroma). Vacuum-freeze for long-term storage only.

---

## 12. Wastage & Loss Tracking

| Source of waste                | Acceptable threshold (per pop-up day) | What to do if exceeded            |
| ------------------------------ | ------------------------------------- | --------------------------------- |
| Stale roasted coffee dumped    | < 200 g                               | Re-evaluate roast schedule        |
| Brewed-and-not-served coffee   | < 100 ml                              | Improve brew-on-order discipline  |
| Cold brew dumped               | < 200 ml                              | Reduce batch size                 |
| Milk dumped                    | < 100 ml                              | Better order forecasting          |
| Customer-rejected drinks       | 0–1                                   | Investigate (taste it first)      |
| Equipment downtime             | 0 minutes                             | Pre-event checklist failed; fix process |

Cumulative waste log lives in [[Technicals/Daily & Weekly Goals#Weekly Review Template]].

---

## 13. Pop-Up Day Schedule (Sample 8-Hour Day)

| Time   | Activity                                                                 |
| ------ | ------------------------------------------------------------------------ |
| 08:30  | Load van / car at storage                                                 |
| 09:30  | Arrive venue, load-in begins                                              |
| 10:00  | Cart setup (electric, water, signage, equipment power-on)                |
| 10:30  | Calibrate espresso (4–6 test pulls, taste each)                           |
| 10:45  | Calibrate filter (1 brew, taste, adjust grind)                            |
| 11:00  | **Open** — first cup served                                               |
| 14:00  | Mid-day check: stock, equipment, water level, milk replenish              |
| 17:00  | Final check: bag re-stock, last cold-brew refill                          |
| 18:30  | **Last call** announced                                                   |
| 19:00  | **Close**                                                                  |
| 19:00–19:45 | Daily SOP cleanup + reconciliation + day log               |
| 19:45  | Load-out                                                                  |
| 20:30  | Back at storage; unload; prep tomorrow if multi-day event                 |

---

## My Notes & Thoughts

- _The cart load-out checklist will save you the first time you forget the *thermometer* or *cup sleeves*. Print it. Use it. Don't skip it because "you remember."_
- _The freshness rules will be tested first by you. The first time you face dumping RM 30 of coffee at end-of-day, the rule is the brand. Hold it._
- _Equipment maintenance is boring until something breaks at 11:00 on a Saturday. Build the habit early._
- _Track waste honestly. The number is a leading indicator of process quality._
