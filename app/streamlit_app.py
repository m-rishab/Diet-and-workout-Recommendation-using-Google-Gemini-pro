"""Diet & Workout Recommendation — Streamlit frontend.

Run:  streamlit run app/streamlit_app.py
"""

import json
import os
import sys
import time

import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import backend as B  # noqa: E402

# ---------------------------------------------------------------------------
# Page config & session state
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Diet & Workout Recommendation",
    page_icon="\U0001F957",
    layout="wide",
    initial_sidebar_state="expanded",
)

ACCENT = "#C6FF3E"

_FORM_DEFAULTS = {
    "f_name": "",
    "f_age": 25,
    "f_gender": "Male",
    "f_units": "kg_cm",
    "f_w_kg": 72.0,
    "f_h_cm": 175.0,
    "f_w_lb": 155.0,
    "f_h_ft": 5.7,
    "f_diet": "Veg",
    "f_disease": "None",
    "f_disease_other": "",
    "f_region": "North",
    "f_state": "Delhi",
    "f_allergics": "None",
    "f_foodtype": "Home-cooked",
}


def _init_session() -> None:
    for k, v in _FORM_DEFAULTS.items():
        if k not in st.session_state:
            st.session_state[k] = v
    if "plan" not in st.session_state:
        st.session_state.plan = dict(B.EMPTY_PLAN)
    if "metrics" not in st.session_state:
        st.session_state.metrics = None
    if "inputs" not in st.session_state:
        st.session_state.inputs = None
    if "generated_name" not in st.session_state:
        st.session_state.generated_name = ""
    if "coach_hist" not in st.session_state:
        st.session_state.coach_hist = []
    if "from_cache" not in st.session_state:
        st.session_state.from_cache = False
    if "last_error" not in st.session_state:
        st.session_state.last_error = ""


_init_session()


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------
def _inputs() -> dict:
    disease_val = (
        f"{st.session_state.f_disease} - {st.session_state.f_disease_other}"
        if st.session_state.f_disease == "Other"
        else st.session_state.f_disease
    )
    w_kg = st.session_state.f_w_kg
    h_cm = st.session_state.f_h_cm
    if st.session_state.f_units == "Imperial":
        w_kg = st.session_state.f_w_lb * 0.453592
        h_cm = st.session_state.f_h_ft * 30.48
    return {
        "name": st.session_state.f_name,
        "age": str(st.session_state.f_age),
        "gender": st.session_state.f_gender,
        "weight": f"{w_kg:.1f}",
        "height": f"{h_cm:.1f}",
        "veg_or_nonveg": st.session_state.f_diet,
        "disease": disease_val,
        "region": st.session_state.f_region,
        "state": st.session_state.f_state,
        "allergics": st.session_state.f_allergics,
        "foodtype": st.session_state.f_foodtype,
    }


def _get_weight_height() -> tuple[float, float]:
    """Return (weight_kg, height_cm) from current form values."""
    if st.session_state.f_units == "Imperial":
        return st.session_state.f_w_lb * 0.453592, st.session_state.f_h_ft * 30.48
    return st.session_state.f_w_kg, st.session_state.f_h_cm


def _constraints() -> str:
    lines = []
    allergics = st.session_state.f_allergics
    disease = st.session_state.f_disease
    if allergics and allergics.lower() not in ("none", "no", "nil", ""):
        lines.append(f"NEVER suggest foods containing: {allergics}.")
    if disease and disease.lower() not in ("none", "no", "nil", ""):
        lines.append(f"Avoid foods that may trigger or worsen: {disease}.")
    lines.append("Add a disclaimer: This plan is not a substitute for professional medical advice.")
    return "\n".join(lines)


def _call_llm(prompt: str, stream: bool = True) -> str:
    last_err = None
    for _attempt in range(3):
        try:
            if stream:
                return "".join(
                    c.content if hasattr(c, "content") else str(c)
                    for c in B._get_model().stream(prompt)
                )
            resp = B._get_model().invoke(prompt)
            return resp.content if hasattr(resp, "content") else str(resp)
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            msg = str(exc)
            if any(s in msg for s in ("503", "504", "overloaded", "over capacity", "timeout")):
                if _attempt < 2:
                    time.sleep(4)
                    continue
            break
    raise last_err


def _generate_plan() -> None:
    inputs = _inputs()
    ckey = B.cache_key(inputs)
    st.session_state.last_error = ""

    cached = B.get_cached_plan(ckey)
    if cached:
        st.session_state.plan = B._enrich_plan(
            cached["plan"], state=inputs["state"], region=inputs["region"]
        )
        st.session_state.metrics = cached["metrics"]
        st.session_state.generated_name = inputs["name"]
        st.session_state.from_cache = True
        _cache_plan_photos(st.session_state.plan)
        return
    st.session_state.from_cache = False

    weight_kg, height_cm = _get_weight_height()
    metrics = B._compute_metrics(height_cm, weight_kg, int(inputs["age"]), inputs["gender"])
    prompt = B._build_prompt(inputs, _constraints())

    raw = _call_llm(prompt, stream=True)
    plan = None
    for attempt in range(2):
        try:
            plan = B._normalize_plan(B._extract_json(raw))
            break
        except Exception:
            if attempt == 0:
                retry = (
                    raw + "\n\n[SYSTEM] Your reply was not valid JSON. "
                    "Return ONLY a single JSON object matching the required schema. "
                    "No markdown fences, no prose."
                )
                raw = _call_llm(retry, stream=False)
            else:
                st.session_state.last_error = "The model did not return valid JSON."

    if plan is None:
        return

    plan = B._enrich_plan(plan, state=inputs["state"], region=inputs["region"])

    st.session_state.plan = plan
    st.session_state.metrics = metrics
    st.session_state.generated_name = inputs["name"]
    st.session_state.inputs = inputs
    B.save_plan(inputs, ckey, plan, metrics)
    _cache_plan_photos(plan)


def _fill_example():
    for k, v in {
        "f_name": "Rahul Sharma",
        "f_age": 28,
        "f_gender": "Male",
        "f_units": "kg_cm",
        "f_w_kg": 72.0,
        "f_h_cm": 175.0,
        "f_w_lb": 155.0,
        "f_h_ft": 5.7,
        "f_diet": "Veg",
        "f_disease": "None",
        "f_disease_other": "",
        "f_region": "North",
        "f_state": "Delhi",
        "f_allergics": "None",
        "f_foodtype": "Home-cooked",
    }.items():
        st.session_state[k] = v


def _load_saved(plan_id: int) -> None:
    row = B.get_saved_plan(plan_id)
    if row:
        inputs = row.get("inputs", {}) or {}
        st.session_state.plan = B._enrich_plan(
            row["plan"], state=inputs.get("state", ""), region=inputs.get("region", "")
        )
        st.session_state.metrics = row["metrics"]
        st.session_state.generated_name = row["user_name"]
        st.session_state.from_cache = False
        st.rerun()
    else:
        st.warning("Could not load that plan.")


def _cache_plan_photos(plan: dict) -> None:
    meal_names = [
        m.get("name", "")
        for sec in ("breakfasts", "lunches", "dinners")
        for m in plan.get(sec, [])
        if m.get("name")
    ]
    workout_names = [
        w.get("name", "")
        for w in plan.get("workouts", [])
        if w.get("name")
    ]
    try:
        B.cache_meal_images(meal_names, max_workers=3)
    except Exception:  # noqa: BLE001
        pass
    try:
        B.cache_workout_images(workout_names, max_workers=3)
    except Exception:  # noqa: BLE001
        pass


# ---------------------------------------------------------------------------
# Sidebar — user profile form
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("Diet & Workout")
    st.caption("Powered by NVIDIA NIM (Nemotron-3-Super-120B)")

    with st.form("profile_form"):
        st.text_input("Name", key="f_name")
        col_a, col_b = st.columns(2)
        col_a.number_input("Age", min_value=5, max_value=120, step=1, key="f_age")
        col_b.selectbox("Gender", ["Male", "Female"], key="f_gender")

        st.radio("Units", ["kg_cm", "Imperial"], horizontal=True, key="f_units")
        if st.session_state.f_units == "Imperial":
            st.number_input("Weight (lb)", min_value=20.0, max_value=700.0, key="f_w_lb")
            st.number_input("Height (ft)", min_value=3.0, max_value=8.0, step=0.1, key="f_h_ft")
        else:
            st.number_input("Weight (kg)", min_value=10.0, max_value=400.0, key="f_w_kg")
            st.number_input("Height (cm)", min_value=60.0, max_value=250.0, key="f_h_cm")

        st.radio("Diet", B.DIET_OPTIONS, horizontal=True, key="f_diet")
        st.selectbox("Medical condition", B.DISEASE_CHIPS, key="f_disease")
        if st.session_state.f_disease == "Other":
            st.text_input("Specify condition", key="f_disease_other")

        st.selectbox("Region", list(B.REGION_STATES.keys()), key="f_region")
        _state_opts = B.REGION_STATES.get(st.session_state.f_region, [])
        if st.session_state.f_state not in _state_opts:
            st.session_state.f_state = _state_opts[0] if _state_opts else ""
        st.selectbox("State", _state_opts, key="f_state")
        st.text_input("Allergies (or None)", key="f_allergics")
        st.selectbox("Food preference", B.FOOD_TYPES, key="f_foodtype")

        submitted = st.form_submit_button("Build My Plan", type="primary")

    st.button("Fill Example", on_click=_fill_example, width='stretch')

    history = B.all_saved_plans()
    if history:
        with st.expander(f"Saved plans ({len(history)})"):
            options = {f"{h['user_name']} — {h['created_at'][:10]} (#{h['id']})": h["id"] for h in history}
            choice = st.selectbox("Load a past plan", [""] + list(options.keys()), key="saved_pick")
            if choice and st.button("Load plan", width='stretch'):
                _load_saved(options[choice])

    if st.button("Clear all history", width='stretch'):
        n = B.clear_history()
        st.success(f"Deleted {n} saved plan(s).")
        st.rerun()


# ---------------------------------------------------------------------------
# Generation trigger
# ---------------------------------------------------------------------------
if submitted:
    weight_kg, height_cm = _get_weight_height()
    inputs = _inputs()
    if not all([inputs["name"].strip(), inputs["region"], inputs["state"],
                inputs["foodtype"], weight_kg > 0, height_cm > 0]):
        st.sidebar.error("Please fill all required fields.")
    else:
        with st.spinner("Building your plan — this can take up to a minute..."):
            try:
                _generate_plan()
            except Exception as exc:  # noqa: BLE001
                st.session_state.last_error = str(exc)
        if st.session_state.last_error:
            st.error(f"That did not work: {st.session_state.last_error}")
        else:
            tag = " (from cache)" if st.session_state.from_cache else ""
            st.success(f"Plan ready for {inputs['name'] or 'you'}!{tag}")

# ---------------------------------------------------------------------------
# Main area — render the plan
# ---------------------------------------------------------------------------
plan = st.session_state.plan
metrics = st.session_state.metrics
has_plan = bool(plan.get("breakfasts") or plan.get("restaurants") or plan.get("notes"))

st.title("Diet & Workout Recommendation")
st.caption(
    "Personalised meal, restaurant and workout plan by NVIDIA NIM. "
    "Not a substitute for professional medical advice."
)

if not has_plan:
    hero = B.image_path("hero")
    if hero:
        st.image(hero, width='stretch')
    st.info("Fill in your details in the sidebar and press **Build My Plan** to get a personalised diet + workout recommendation.")
    st.stop()

# ── Metrics banner ──────────────────────────────────────────────────────────
if metrics:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("BMI", f"{metrics['bmi']:.2f}", metrics["bmi_category"])
    c2.metric("BMR", f"{metrics['bmr']:.0f} kcal/day")
    c3.metric("TDEE", f"{metrics['tdee']:.0f} kcal/day")
    c4.metric("Daily target", f"{metrics['daily_target']:.0f} kcal")
    if st.session_state.generated_name:
        st.markdown(f"#### {st.session_state.generated_name}'s personalised plan")

# ── Tabs ────────────────────────────────────────────────────────────────────
_tabs = st.tabs(["Meals", "Restaurants", "Workouts", "Weekly Split", "Insights", "Ask the Coach"])

# Meals
with _tabs[0]:
    for section, title, cat_key in (
        ("breakfasts", "Breakfast", "breakfast"),
        ("lunches", "Lunch", "lunch"),
        ("dinners", "Dinner", "dinner"),
    ):
        items = plan.get(section, [])
        if not items:
            continue
        st.subheader(title)
        cols = st.columns(min(len(items), 5))
        for idx, item in enumerate(items):
            with cols[idx % 5]:
                img = B.meal_image_path(item.get("name", ""))
                if img:
                    st.image(img, width=200)
                st.markdown(f"**{item.get('name', '')}**")
                if "kcal" in item:
                    st.caption(f"{item.get('kcal', 0)} kcal · {item.get('protein_g', 0)}g protein")
                if item.get("why"):
                    st.caption(item["why"])
        st.divider()

# Restaurants
with _tabs[1]:
    rs = plan.get("restaurants", [])
    if rs:
        cols = st.columns(min(len(rs), 3))
        for idx, r in enumerate(rs):
            with cols[idx % 3]:
                img = B.image_path("restaurant", idx % 3)
                if img:
                    st.image(img, width=250)
                st.markdown(f"**{r.get('name', '')}**")
                tags = " · ".join(x for x in (r.get("cuisine", ""), r.get("price", "")) if x)
                st.caption(tags)
                if r.get("why"):
                    st.caption(r["why"])
        st.divider()
    else:
        st.info("No restaurants found.")

# Workouts
with _tabs[2]:
    ws = plan.get("workouts", [])
    if ws:
        for i in range(0, len(ws), 3):
            cols = st.columns(3)
            for col, w in zip(cols, ws[i : i + 3]):
                with col:
                    img = B.workout_image_path(w.get("name", ""))
                    if img:
                        st.image(img, width=200)
                    st.markdown(f"**{w.get('name', '')}** — *{w.get('target', '')}*")
                    st.caption(f"{w.get('sets', '')} sets × {w.get('reps', '')}  |  {w.get('tips', '')}")
    st.divider()
    img = B.image_path("yoga", 0)
    if img:
        st.image(img, width=250)

# Weekly split
with _tabs[3]:
    split = plan.get("weekly_split", [])
    if split:
        day_map = {d["day"]: d["focus"] for d in split}
        table = [
            [f"Day {i + 1}: {d}", day_map.get(d, "—")]
            for i, d in enumerate(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
        ]
        st.table(table)
        st.caption("Adjust rest days to your recovery needs.")

# Insights
with _tabs[4]:
    left, right = st.columns(2, gap="large")
    with left:
        if metrics:
            st.subheader("Health snapshot")
            st.markdown(
                f"**BMI {metrics['bmi']:.1f}** — *{metrics['bmi_category']}* (healthy range 18.5–24.9)"
            )
            gauge = max(0, min(100, int((metrics["bmi"] - 14) * 4)))
            bar_color = "#DC2626" if metrics["bmi_category"] == "Obesity" else ACCENT
            st.markdown(
                f"<div style='height:10px;border-radius:5px;background:#333;border:1px solid #555'>"
                f"<div style='height:10px;width:{gauge}%;background:{bar_color};border-radius:5px'></div></div>",
                unsafe_allow_html=True,
            )
            if plan.get("bmi_summary"):
                st.info(plan["bmi_summary"])
            elif metrics["bmi_category"] == "Normal weight":
                st.info("Maintain calories and focus on protein + strength training.")
            else:
                advice = (
                    "small calorie surplus (~+300 kcal)"
                    if "Underweight" in metrics["bmi_category"]
                    else "calorie deficit (~-500 kcal/day)"
                )
                st.info(f"Aim for a {advice} for gradual, healthy progress.")

        if metrics:
            st.subheader("Daily calorie targets")
            m1, m2, m3 = st.columns(3)
            m1.metric("BMR", f"{metrics['bmr']:.0f}")
            m2.metric("TDEE", f"{metrics['tdee']:.0f}")
            m3.metric("Target", f"{metrics['daily_target']:.0f}")

            st.subheader("Macro targets")
            m1, m2, m3 = st.columns(3)
            m1.metric("Protein", f"{metrics['daily_target'] * 0.30 / 4:.0f} g")
            m2.metric("Carbs", f"{metrics['daily_target'] * 0.45 / 4:.0f} g")
            m3.metric("Fats", f"{metrics['daily_target'] * 0.25 / 9:.0f} g")

    with right:
        st.subheader("Coach notes")
        for n in plan.get("notes", []):
            st.markdown(f"- {n}")

        bmis = B.all_bmis()
        if len(bmis) > 1:
            st.subheader("BMI trend")
            fig = go.Figure(
                go.Scatter(
                    x=[b["created_at"][:10] for b in bmis],
                    y=[b["bmi"] for b in bmis],
                    mode="lines+markers",
                    line=dict(color=ACCENT, width=2),
                    marker=dict(size=6),
                )
            )
            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="#0A0A0A",
                margin=dict(l=20, r=20, t=30, b=30),
                height=280,
            )
            st.plotly_chart(fig, width='stretch')

# Ask the Coach
with _tabs[5]:
    st.caption("Ask questions about your plan — the AI answers using your plan data.")
    for msg in st.session_state.coach_hist:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
    q = st.chat_input("Ask the coach...")
    if q:
        st.session_state.coach_hist.append({"role": "user", "content": q})
        context = f"Plan data:\n{json.dumps(plan)}\n\nUser question: {q}"
        with st.chat_message("user"):
            st.write(q)
        with st.spinner("Thinking..."):
            try:
                reply = _call_llm(context, stream=False)
                st.session_state.coach_hist.append({"role": "assistant", "content": reply})
                with st.chat_message("assistant"):
                    st.write(reply)
            except Exception as exc:  # noqa: BLE001
                err = f"Coach error: {exc}"
                st.session_state.coach_hist.append({"role": "assistant", "content": err})
                with st.chat_message("assistant"):
                    st.write(err)

# ── Export ──────────────────────────────────────────────────────────────────
with st.expander("Export / share this plan"):
    export_cols = st.columns(3)
    with export_cols[0]:
        st.download_button(
            "Download JSON",
            data=json.dumps(plan, indent=2, ensure_ascii=False),
            file_name="nutrifit_plan.json",
            mime="application/json",
            width='stretch',
        )
    with export_cols[1]:
        st.download_button(
            "Download PDF",
            data=B.pdf_bytes(plan, metrics),
            file_name="nutrifit_plan.pdf",
            mime="application/pdf",
            width='stretch',
        )
    with export_cols[2]:
        summary = (
            f"{st.session_state.generated_name}'s Diet & Workout plan: "
            f"BMI {metrics['bmi']:.1f} ({metrics['bmi_category']})"
            if metrics
            else "Diet & Workout plan"
        )
        st.text_input("Share summary", value=summary)
