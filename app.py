import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Нечеткий прогноз погоды",
    layout="wide"
)

st.title("Прогнозирование вероятности осадков")
st.write("Нечеткая система на основе алгоритма Мамдани")

variables = {
    "Температура": {
        "Низкая": ([-20, 0, 15], [1, 1, 0]),
        "Средняя": ([0, 18, 30], [0, 1, 0]),
        "Высокая": ([20, 30, 40], [0, 1, 1])
    },
    "Влажность": {
        "Низкая": ([0, 20, 50], [1, 1, 0]),
        "Средняя": ([25, 50, 75], [0, 1, 0]),
        "Высокая": ([55, 80, 100], [0, 1, 1])
    },
    "Давление": {
        "Низкое": ([950, 980, 1010], [1, 1, 0]),
        "Среднее": ([990, 1013, 1030], [0, 1, 0]),
        "Высокое": ([1015, 1040, 1050], [0, 1, 1])
    },
    "Осадки": {
        "Низкая": ([0, 25, 50], [1, 1, 0]),
        "Средняя": ([25, 50, 75], [0, 1, 0]),
        "Высокая": ([50, 75, 100], [0, 1, 1])
    }
}

ranges = {
    "Температура": (-20, 40),
    "Влажность": (0, 100),
    "Давление": (950, 1050),
    "Осадки": (0, 100)
}


def membership(value, points):
    xs, ys = points
    return np.interp(value, xs, ys)


rules = [
    ([("Температура", "Низкая"),
      ("Влажность", "Высокая")], "AND", "Высокая"),

    ([("Температура", "Средняя"),
      ("Влажность", "Высокая")], "AND", "Высокая"),

    ([("Температура", "Высокая"),
      ("Давление", "Высокое")], "AND", "Низкая"),

    ([("Температура", "Высокая"),
      ("Влажность", "Низкая")], "AND", "Низкая"),

    ([("Давление", "Низкое"),
      ("Влажность", "Высокая")], "AND", "Высокая"),

    ([("Давление", "Среднее"),
      ("Влажность", "Средняя")], "AND", "Средняя"),

    ([("Температура", "Низкая"),
      ("Давление", "Низкое")], "AND", "Высокая"),

    ([("Температура", "Средняя"),
      ("Давление", "Высокое")], "AND", "Низкая"),

    ([("Температура", "Высокая"),
      ("Влажность", "Средняя")], "AND", "Средняя"),

    ([("Давление", "Низкое"),
      ("Влажность", "Высокая")], "OR", "Высокая"),

    ([("Влажность", "Низкая")], "AND", "Низкая"),

    ([("Влажность", "Средняя")], "AND", "Средняя"),

    ([("Влажность", "Высокая")], "AND", "Высокая")
]



def mamdani(values):
    fuzzy = {}

    for variable in values:
        fuzzy[variable] = {}
        for term, points in variables[variable].items():
            fuzzy[variable][term] = float(
                membership(values[variable], points)
            )

    x = np.linspace(0, 100, 1001)
    aggregated = np.zeros_like(x)

    active_rules = []

    for number, (conditions, operator, result) in enumerate(
        rules, start=1
    ):
        degrees = [
            fuzzy[var][term]
            for var, term in conditions
        ]

        if operator == "AND":
            strength = min(degrees)
        else:
            strength = max(degrees)

        output_membership = membership(
            x, variables["Осадки"][result]
        )

        clipped = np.minimum(strength, output_membership)

        aggregated = np.maximum(aggregated, clipped)

        if strength > 0:
            active_rules.append((number, strength, result))

    area = np.trapezoid(aggregated, x)

    if area <= 0:
        return fuzzy, None, aggregated, active_rules

    centroid = (
        np.trapezoid(x * aggregated, x) / area
    )

    return fuzzy, centroid, aggregated, active_rules


st.header("Входные данные")

with st.form("weather_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        temp_text = st.text_input(
            "Температура (°C)",
            value="20"
        )

    with col2:
        humidity_text = st.text_input(
            "Влажность (%)",
            value="75"
        )

    with col3:
        pressure_text = st.text_input(
            "Давление (гПа)",
            value="1005"
        )

    calculate = st.form_submit_button(
        "Рассчитать прогноз"
    )


def show_membership_graph(variable, current_value=None):
    minimum, maximum = ranges[variable]
    x = np.linspace(minimum, maximum, 500)

    fig, ax = plt.subplots(figsize=(7, 3))

    for term, points in variables[variable].items():
        ax.plot(
            x,
            membership(x, points),
            label=term,
            linewidth=2
        )

    if current_value is not None:
        ax.axvline(
            current_value,
            color="black",
            linestyle="--",
            label="Ввод"
        )

    ax.set_title(variable)
    ax.set_xlabel("Значение")
    ax.set_ylabel("Степень принадлежности")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    ax.legend()

    return fig


st.header("Функции принадлежности")

for name in variables:
    fig = show_membership_graph(name)
    st.pyplot(fig)
    plt.close(fig)



if calculate:
    try:
        temp = float(temp_text.replace(",", "."))
        humidity = float(humidity_text.replace(",", "."))
        pressure = float(pressure_text.replace(",", "."))

        values = {
            "Температура": temp,
            "Влажность": humidity,
            "Давление": pressure
        }

        for variable, value in values.items():
            minimum, maximum = ranges[variable]

            if not np.isfinite(value):
                raise ValueError(
                    f"{variable}: требуется конечное число"
                )

            if not minimum <= value <= maximum:
                raise ValueError(
                    f"{variable}: допустимо от "
                    f"{minimum} до {maximum}"
                )

        fuzzy, result, aggregated, active = mamdani(values)

        if result is None:
            st.warning("Не активировано ни одно правило.")
        else:
            st.header("Результаты расчета")

            if result < 35:
                category = "Низкая"
                message = "Осадки маловероятны."
            elif result < 65:
                category = "Средняя"
                message = "Возможны осадки."
            else:
                category = "Высокая"
                message = "Повышенная вероятность осадков."

            st.metric(
                "Расчетный индекс осадков",
                f"{result:.2f} / 100"
            )

            st.success(
                f"Категория: {category}. {message}"
            )

            st.caption(
                "Индекс модели не является "
                "калиброванной метеорологической вероятностью."
            )

            st.subheader("Степени принадлежности")

            for variable, terms in fuzzy.items():
                st.write(f"**{variable}**")
                for term, degree in terms.items():
                    st.write(f"{term}: {degree:.3f}")

            st.subheader("Активированные правила")

            for number, strength, output in active:
                st.write(
                    f"Правило {number}: "
                    f"сила {strength:.3f}, "
                    f"результат — {output}"
                )

            st.subheader("Агрегация и центроид")

            x = np.linspace(0, 100, 1001)
            fig, ax = plt.subplots(figsize=(9, 4))

            ax.fill_between(
                x, aggregated, alpha=0.4,
                label="Агрегированное множество"
            )

            ax.plot(
                x, aggregated,
                linewidth=2
            )

            ax.axvline(
                result,
                color="red",
                linestyle="--",
                label=f"Центроид: {result:.2f}"
            )

            ax.set_xlim(0, 100)
            ax.set_ylim(0, 1.05)
            ax.set_xlabel("Индекс осадков")
            ax.set_ylabel("Принадлежность")
            ax.grid(alpha=0.3)
            ax.legend()

            st.pyplot(fig)
            plt.close(fig)

    except ValueError as error:
        st.error(
            "Ошибка ввода. Проверьте данные: "
            + str(error)
        )
