---
created: <% tp.file.creation_date() %>
tags:
---

# <% moment(tp.file.title,'YYYY-MM-DD').format("dddd, MMMM DD, YYYY") %>

<< [[Personal/Journals/Dailies/<% moment(tp.date.now("YYYY-MM-DD", -1)).format("YYYY") %>/<% tp.date.now("YYYY-MM-DD", -1) %>|Yesterday]] | [[Personal/Journals/Dailies/<% moment(tp.date.now("YYYY-MM-DD", 1)).format("YYYY") %>/<% tp.date.now("YYYY-MM-DD", 1) %>|Tomorrow]] >>

---

## 🌅 Start of day

### Timebox / today's calendar
- **Quick link:** [Open today in Google Calendar](https://calendar.google.com/calendar/u/0/r/customday/<% tp.date.now("YYYY") %>/<% tp.date.now("M") %>/<% tp.date.now("D") %>)
- **In Obsidian:** If you use the [Obsidian Google Calendar](https://github.com/yukigasai/obsidian-google-calendar) plugin, run **Insert gCal Events** (or **Open Google Calendar**) from the command palette to inject today's events here or open the calendar pane.

*(Paste or type your timebox below after planning in Google Calendar.)*
| Time       | Block / focus |
| ---------- | -------------- |
|            |                |
|            |                |

### Intention
- **One sentence for today:** 
- **Top 1–3 priorities:** 1)  2)  3)

---

## 📅 Mid-day check-in (optional)
### 🌜 How ya feeling?
- 
### 📖 A question or idea to ponder
- 
### 🔧 Technical challenge or puzzle
- 

---

## 🚀 Tasks to accomplish today
#### General work
- [ ] 
#### IoTWatt
- [ ] 
#### Kota App
- [ ] 
#### AMOU
- [ ] 
#### Coffee
- [ ] 
#### Personal / Knowledge
- [ ]  

---

## ✅ Habit Tracker
| Habit                                              | Status                |
| -------------------------------------------------- | --------------------- |
| Code for 2-3 hour on a project                     | [ ] #habit/coding     |
| Make Coffee before 9am                             | [ ] #habit/philosophy |
| Exercise - usual daily breakdown                   | [ ] #habit/exercise   |
| Write or summarize notes (Books/Research/Yappings) | [ ] #habit/notes      |
| Reflect or journal (5 min)                         | [ ] #habit/reflection |
| Pick up a book for atleast 1 hour                  | [ ] #habit/reflection |



---

## 📝 Notes

### Work notes by focus
*(Jot as you switch context—links to project/area notes live in [[Work Focuses|Work Focuses]].)*
- **General:** 
- **IoTWatt:** 
- **Kota App:** 
- **AMOU:** 
- **Coffee:** 

### Knowledge Management

#### Notes created today
```dataview
List FROM "" WHERE file.cday = date("<% tp.date.now('YYYY-MM-DD') %>") SORT file.ctime asc
```

#### Notes last touched today
```dataview
List FROM "" WHERE file.mday = date("<% tp.date.now('YYYY-MM-DD') %>") SORT file.mtime asc
```
---

## 📊 Task Overview
```tasks
not done
due today
group by filename
```

---

## 🌙 End of day reflection

*(Fill this before you close the day—keeps Obsidian as your accountability layer.)*

- **Shipped / done:** 
- **Deferred / didn’t get to:** 
- **One win:** 
- **One thing to improve tomorrow:** 
- **Tomorrow’s focus (1–2 items):** 