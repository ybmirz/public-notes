---
created: <% tp.file.creation_date() %>
tags:
---

# <% moment(tp.file.title,'YYYY-MM-DD').format("dddd, MMMM DD, YYYY") %>

<< [[Content/Resources/Dailies/<% tp.date.now("YYYY-MM-DD", -1) %>|Yesterday]] | [[Content/Resources/Dailies/<% tp.date.now("YYYY-MM-DD", 1) %>|Tomorrow]] >>

---

## 📅 Daily Reflection
### 🌜 How ya feeling today?
- 
### 📖 A question or idea to ponder (e.g., ethics, systems, meaning)
- 

##### A Technical Challenge or puzzle I'm currently facing
- 

### 🚀 Tasks to accomplish today
#### Project X
- [x]  ✅ 2025-11-16
#### Project Y
- [x]  ✅ 2025-11-16
#### Project Z
- [x]  ✅ 2025-11-16
#### Personal/Knowledge
- [ ] 
- [ ]  

### 👎 One challenge in my projects or workflow today is...
- 

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

### Project Notes

### Knowledge Management

#### Notes created today
```dataview
List FROM "" WHERE file.cday = date("<%tp.date.now('YYYY-MM-DD')%>") SORT file.ctime asc
```

#### Notes last touched today
```dataview
List FROM "" WHERE file.mday = date(" <%tp.date.now('YYYY-MM-DD')%> ") SORT file.mtime asc
```
---

## 📊 Task Overview
```tasks
not done
due today
group by filename
````