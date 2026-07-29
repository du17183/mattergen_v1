# 第6章训练—测试泄漏证据

| Cohort | Seeds | n | Harm | Harm rate | 数据资格 |
|---|---:|---:|---:|---:|---|
| Training overlap | 20000–20063 | 64 | 0 | 0% | diagnostic only |
| Held-out | 20064–20255 | 192 | 31 | 16.146% | supplementary only |
| Mixed | 20000–20255 | 256 | 31 | 12.109% | invalid for independent claims |

单侧Fisher exact p=6.8659e-5。overlap与held-out平均相对降力分别约−28.15%与−27.32%，但安全率明显不同。准确结论是：重叠没有明显夸大平均改善，却显著高估安全性。
