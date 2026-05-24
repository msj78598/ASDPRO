# ASDPRO Loss Detection App

واجهة Streamlit لتحليل قراءات العدادات باستخدام النموذج المدرب، ثم مراجعة النتائج عبر لجنة فنية وقواعد V/I لاستخراج حالات فاقد عالية الثقة.

## التشغيل

```powershell
pip install -r requirements.txt
.\run_app.ps1
```

ثم افتح:

```text
http://localhost:8501
```

## الملفات المطلوبة

- `app.py`: واجهة المستخدم.
- `predict_loss.py`: منطق التنبؤ والمراجعة الفنية.
- `models/random_forest_best.joblib`: النموذج المدرب المطلوب للتشغيل.
- `requirements.txt`: مكتبات التشغيل.
- `run_app.ps1`: تشغيل الواجهة محليًا.

لا يحتوي هذا المستودع على بيانات تدريب أو ملفات نتائج.
