"""Diet and Workout Recommendation App - built with Reflex + NVIDIA NIM."""

import asyncio
import os
from datetime import date

import pandas as pd
import plotly.graph_objects as go
import reflex as rx
from langchain_core.prompts import PromptTemplate
from langchain_nvidia_ai_endpoints import ChatNVIDIA

# --------------------------------------------------------------------------
# Model setup (NVIDIA NIM via langchain-nvidia-ai-endpoints)
# --------------------------------------------------------------------------
NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY", "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")

model = ChatNVIDIA(
    model="nvidia/nemotron-3-super-120b-a12b",
    nvidia_api_key=NVIDIA_API_KEY,
    temperature=0.6,
    top_p=1,
    max_completion_tokens=2048,
)

prompt_template_resto = PromptTemplate(
    input_variables=['name', 'age', 'gender', 'weight', 'height', 'veg_or_nonveg', 'disease', 'region', 'state', 'allergics', 'foodtype'],
    template="Diet Recommendation System:\n"
             "I want you to recommend 6 restaurant names, 6 breakfast names, 5 dinner names, and 6 workout names, "
             "based on the following criteria:\n"
             "Person name: {name}\n"
             "Person age: {age}\n"
             "Person gender: {gender}\n"
             "Person weight: {weight}\n"
             "Person height: {height}\n"
             "Person veg_or_nonveg: {veg_or_nonveg}\n"
             "Person generic disease: {disease}\n"
             "Person region: {region}\n"
             "Person state: {state}\n"
             "Person allergics: {allergics}\n"
             "Person foodtype: {foodtype}.",
)

USAGE_COUNTS = [10, 20, 15, 25, 30, 5, 10, 8, 15, 18, 8, 15, 10, 20, 22]


def _usage_weeks() -> list[str]:
    """List of ISO week labels (Mondays) from Jan 2024 until today."""
    return [d.strftime("%Y-%m-%d") for d in pd.date_range(date(2024, 1, 1), date.today(), freq="W-MON")]


class State(rx.State):
    """App state."""

    # Form fields
    name: str = ""
    age: str = ""
    gender: str = "Male"
    weight: str = ""
    height: str = ""
    veg_or_nonveg: str = "Veg"
    disease: str = ""
    region: str = ""
    person_state: str = ""
    allergics: str = ""
    foodtype: str = ""

    # Results
    recommendations: str = ""
    error: str = ""
    bmi_value: float = 0.0
    bmi_category: str = ""
    bmi_fig: go.Figure = go.Figure()
    usage_counts: list[int] = USAGE_COUNTS

    show_results: bool = False
    processing: bool = False

    @rx.var
    def usage_fig(self) -> go.Figure:
        """Weekly usage tracking chart."""
        weeks = _usage_weeks()
        counts = self.usage_counts[: len(weeks)]
        fig = go.Figure(
            data=[go.Bar(x=weeks, y=counts, text=counts, textposition="auto")]
        )
        fig.update_layout(
            title="Weekly Usage Tracking",
            xaxis_title="Week",
            yaxis_title="Usage Count",
        )
        return fig

    @rx.var
    def bmi_display(self) -> str:
        return f"Your BMI is {self.bmi_value:.2f}, which falls under the category: {self.bmi_category}"

    def _all_filled(self) -> bool:
        return all(
            [
                self.name, self.age, self.gender, self.weight, self.height,
                self.veg_or_nonveg, self.disease, self.region, self.person_state,
                self.allergics, self.foodtype,
            ]
        )

    @rx.event
    def set_name(self, value: str):
        self.name = value

    @rx.event
    def set_age(self, value: str):
        self.age = value

    @rx.event
    def set_gender(self, value: str):
        self.gender = value

    @rx.event
    def set_weight(self, value: str):
        self.weight = value

    @rx.event
    def set_height(self, value: str):
        self.height = value

    @rx.event
    def set_veg_or_nonveg(self, value: str):
        self.veg_or_nonveg = value

    @rx.event
    def set_disease(self, value: str):
        self.disease = value

    @rx.event
    def set_region(self, value: str):
        self.region = value

    @rx.event
    def set_person_state(self, value: str):
        self.person_state = value

    @rx.event
    def set_allergics(self, value: str):
        self.allergics = value

    @rx.event
    def set_foodtype(self, value: str):
        self.foodtype = value

    @rx.event
    def fill_example(self):
        """Auto-fill the form with demo data."""
        self.name = "Rahul Sharma"
        self.age = "28"
        self.gender = "Male"
        self.weight = "72"
        self.height = "175"
        self.veg_or_nonveg = "Veg"
        self.disease = "None"
        self.region = "North"
        self.person_state = "Delhi"
        self.allergics = "None"
        self.foodtype = "Home-cooked"

    @rx.event
    async def get_recommendations(self):
        """Validate the form, call the NVIDIA model and compute BMI."""
        self.error = ""
        if not self._all_filled():
            self.error = "Please fill in all the form fields."
            self.show_results = False
            yield
            return

        self.processing = True
        self.show_results = False
        yield

        input_data = {
            'name': self.name,
            'age': self.age,
            'gender': self.gender,
            'weight': self.weight,
            'height': self.height,
            'veg_or_nonveg': self.veg_or_nonveg,
            'disease': self.disease,
            'region': self.region,
            'state': self.person_state,
            'allergics': self.allergics,
            'foodtype': self.foodtype,
        }
        prompt = prompt_template_resto.format(**input_data)

        try:
            loop = asyncio.get_running_loop()
            resp = await loop.run_in_executor(None, model.invoke, prompt)
            self.recommendations = resp.content
        except Exception as exc:
            self.error = f"Something went wrong while generating recommendations: {exc}"
            self.processing = False
            yield
            return

        # BMI calculation and 3D visualization
        try:
            height_m = float(self.height) / 100.0
            bmi = float(self.weight) / (height_m ** 2)
            self.bmi_value = bmi

            if bmi < 18.5:
                category, color = "Underweight", "blue"
            elif bmi < 25:
                category, color = "Normal weight", "green"
            elif bmi < 30:
                category, color = "Overweight", "yellow"
            else:
                category, color = "Obesity", "red"
            self.bmi_category = category

            fig = go.Figure(
                data=[
                    go.Scatter3d(
                        x=[int(self.age)],
                        y=[float(self.weight)],
                        z=[bmi],
                        mode='markers',
                        marker=dict(size=12, color=color, opacity=0.8),
                        text=[f"Age: {self.age}<br>Weight: {self.weight} kg<br>BMI: {bmi:.2f}<br>Category: {category}"],
                        hoverinfo='text',
                    )
                ]
            )
            fig.update_layout(
                title='3D BMI Visualization',
                scene=dict(
                    xaxis_title='Age',
                    yaxis_title='Weight (kg)',
                    zaxis_title='BMI',
                    xaxis=dict(backgroundcolor="rgb(200, 200, 230)", gridcolor="white", showbackground=True, zerolinecolor="white"),
                    yaxis=dict(backgroundcolor="rgb(230, 200, 230)", gridcolor="white", showbackground=True, zerolinecolor="white"),
                    zaxis=dict(backgroundcolor="rgb(230, 230, 200)", gridcolor="white", showbackground=True, zerolinecolor="white"),
                ),
                margin=dict(r=10, l=10, b=10, t=30),
            )
            self.bmi_fig = fig
        except (ValueError, TypeError):
            pass

        # Slide the usage counter forward
        self.usage_counts = self.usage_counts[1:] + [self.usage_counts[-1] + 1]

        self.processing = False
        self.show_results = True
        yield


# --------------------------------------------------------------------------
# UI helpers
# --------------------------------------------------------------------------
def _field(label: str, component: rx.Component) -> rx.Component:
    return rx.vstack(
        rx.text(label, font_weight="600"),
        component,
        align="start",
        width="100%",
        spacing="1",
    )


def _form_fields() -> list[rx.Component]:
    return [
        _field("Name:", rx.input(placeholder="Enter your name", value=State.name, on_change=State.set_name, width="100%")),
        _field("Age:", rx.input(placeholder="Enter your age", value=State.age, on_change=State.set_age, width="100%")),
        _field("Gender:", rx.select(items=["Male", "Female"], value=State.gender, on_change=State.set_gender, width="100%")),
        _field("Weight (kg):", rx.input(placeholder="Enter your weight in kg", value=State.weight, on_change=State.set_weight, width="100%")),
        _field("Height (cm):", rx.input(placeholder="Enter your height in cm", value=State.height, on_change=State.set_height, width="100%")),
        _field("Veg or Non-Veg:", rx.select(items=["Veg", "Non-Veg"], value=State.veg_or_nonveg, on_change=State.set_veg_or_nonveg, width="100%")),
        _field("Disease:", rx.input(placeholder="Enter any generic disease", value=State.disease, on_change=State.set_disease, width="100%")),
        _field("Region:", rx.input(placeholder="Enter your region", value=State.region, on_change=State.set_region, width="100%")),
        _field("State:", rx.input(placeholder="Enter your state", value=State.person_state, on_change=State.set_person_state, width="100%")),
        _field("Allergics:", rx.input(placeholder="Enter any allergies", value=State.allergics, on_change=State.set_allergics, width="100%")),
        _field("Food Type:", rx.input(placeholder="Enter your preferred food type", value=State.foodtype, on_change=State.set_foodtype, width="100%")),
    ]


def index() -> rx.Component:
    """Main page."""
    return rx.container(
        rx.vstack(
            rx.heading("Diet and Workout Recommendation", size="8", text_align="center"),
            rx.text("Personalized recommendations powered by NVIDIA NIM (Nemotron-3-Super-120B)", color="gray.11", text_align="center"),

            rx.card(
                rx.vstack(
                    rx.button(
                        "Fill Example Data",
                        on_click=State.fill_example,
                        variant="soft",
                        width="100%",
                    ),
                    rx.grid(
                        *_form_fields(),
                        grid_template_columns="repeat(2, 1fr)",
                        gap="4",
                        width="100%",
                    ),
                    rx.button(
                        "Get Recommendations",
                        on_click=State.get_recommendations,
                        loading=State.processing,
                        disabled=State.processing,
                        width="100%",
                        size="3",
                    ),
                    rx.cond(
                        State.error != "",
                        rx.text(State.error, color="red", weight="medium"),
                    ),
                    spacing="4",
                ),
                width="100%",
            ),

            rx.cond(
                State.show_results,
                rx.vstack(
                    rx.heading("Recommendations", size="5"),
                    rx.markdown(State.recommendations, width="100%"),
                    rx.text(State.bmi_display, font_weight="600"),
                    rx.plotly.gl3d(data=State.bmi_fig, width="100%", height="400px"),
                    spacing="4",
                    width="100%",
                ),
            ),

            rx.vstack(
                rx.heading("Weekly Usage Tracking", size="5"),
                rx.plotly.basic(data=State.usage_fig, width="100%", height="300px"),
                spacing="4",
                align="start",
                width="100%",
            ),

            rx.text("Built with Reflex and NVIDIA NIM", color="gray.9", size="1"),
            spacing="6",
            width="100%",
        ),
        max_width="56em",
        padding="1.5em",
    )


app = rx.App()
app.add_page(index, route="/", title="Diet and Workout Recommendation")