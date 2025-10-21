# anomaly_iqr.py
# С помощью GenAI_1_28.py находит аномально короткие/длинные предложения через IQR
# и выводит их с кратким анализом.

import re
import sys
import numpy as np
import matplotlib.pyplot as plt
import os


try:
    import GenAI_1_28 as base   # должен лежать рядом: GenAI_1_28.py
except ModuleNotFoundError:
    print("[error] Не найден файл GenAI_1_28.py рядом со скриптом. Переименуй исходник и положи рядом.")
    sys.exit(1)

def compute_iqr_bounds(lengths: list[int]) -> tuple[float, float, float, float]:
    """Q1, Q3, IQR, (lower, upper) по классическому правилу 1.5*IQR.
       Если IQR==0 (все длины одинаковые), слегка расширяем границы, чтобы не получить пустой результат."""
    q1 = float(np.percentile(lengths, 25))
    q3 = float(np.percentile(lengths, 75))
    iqr = q3 - q1
    if iqr == 0.0:
        # все предложения одной длины — расширим границы на 1, чтобы хоть что-то отфильтровать при желании
        lower = q1 - 1.0
        upper = q3 + 1.0
    else:
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
    return q1, q3, iqr, lower, upper

def sentence_features(s: str) -> dict:
    """Грубые признаки для пояснений."""
    return {
        "chars": len(s),
        "commas": s.count(","),
        "semicolons": s.count(";"),
        "dashes": s.count("—") + s.count("-"),
        "parens": s.count("(") + s.count(")"),
        "quotes": s.count('"') + s.count("«") + s.count("»") + s.count("'"),
        "digits": sum(c.isdigit() for c in s),
        "abbr_like": len(re.findall(r"\b\w+\.", s)),  # аббревиатуры вида "т.д."
    }

def explain_anomaly(length: int, avg_len: float, feats: dict) -> str:
    msgs = []
    # общая причина по отношению к среднему
    if length <= max(1, int(avg_len * 0.5)):
        msgs.append("Очень короткое относительно среднего — возможно, обрыв фразы или заголовок.")
    elif length >= int(avg_len * 1.5) + 1:
        msgs.append("Очень длинное относительно среднего — похоже на слитие нескольких мыслей.")

    # структурные причины
    if feats["commas"] + feats["semicolons"] >= 2:
        msgs.append("Много перечислений/вставных конструкций (много запятых/точек с запятой).")
    if feats["dashes"] >= 1:
        msgs.append("Есть тире — часто удлиняет предложение пояснениями.")
    if feats["parens"] >= 2:
        msgs.append("Есть скобки — вставные конструкции увеличивают длину.")
    if feats["abbr_like"] >= 1:
        msgs.append("Есть аббревиатуры вида 'слово.' — токенизация могла укорачивать счёт слов.")
    if feats["digits"] >= 1:
        msgs.append("Есть цифры — в базовом коде они отфильтровываются (t.isalpha()), поэтому длина могла занижаться.")

    if not msgs:  # если ничего не зацепили — дайте нейтральное
        msgs.append("На границе типичных длин — редкий, но возможный случай.")
    return " ".join(msgs)

def main():
    # 0) берём вход/язык из чужого модуля
    text = base.read_text(base.INPUT_FILE)
    sentences, lengths = base.tokenize_and_lengths(text, base.LANG)

    if not sentences:
        print("[info] Нет данных для анализа (пустой или отсутствующий input.txt).")
        return

    avg_len = float(np.mean(lengths))
    q1, q3, iqr, lower, upper = compute_iqr_bounds(lengths)

    print("\n=== Статистика по длинам (в словах) ===")
    print(f"Всего предложений: {len(sentences)}")
    print(f"Средняя длина: {avg_len:.2f}")
    print(f"Q1={q1:.2f}, Q3={q3:.2f}, IQR={iqr:.2f}")
    print(f"Границы выбросов: < {lower:.2f} или > {upper:.2f}\n")

    # находим выбросы (короткие/длинные)
    outliers = [(i, s, l) for i, (s, l) in enumerate(zip(sentences, lengths), start=1)
                if (l < lower or l > upper)]

    if not outliers:
        print("Выбросов по правилу IQR не найдено.")
        return
    
    base_dir = os.path.dirname(os.path.abspath(base.__file__))
    
    all_hist_path = os.path.join(base_dir, "hist_all_iqr.png")
    base.histogram(lengths, all_hist_path)  # отрисовка столбиков их функцией

    ax = plt.gca()
    ymin, ymax = ax.get_ylim()

    # вертикальные линии: нижняя/верхняя граница и квартиль Q1/Q3
    for x, style, label in [(lower, "--", "lower"), (q1, ":", "Q1"),
                            (q3, ":", "Q3"), (upper, "--", "upper")]:
        ax.axvline(x=x, linestyle=style, linewidth=1)
        ax.text(x, ymax*0.98, label, rotation=90, va="top", ha="right", fontsize=8)

    # "rug"-риски по оси X для каждого выброса (короткая вертикальная черта у низа столбца)
    for _, _, l in outliers:
        ax.plot([l, l], [0, ymax*0.05], linewidth=1)
    
    scatter_path = os.path.join(base_dir, "sent_len_scatter.png")
    x = np.arange(1, len(lengths) + 1)
    y = np.array(lengths)

    plt.figure()
    plt.scatter(x, y, s=25, label="sentences")
    mask = (y < lower) | (y > upper)
    plt.scatter(x[mask], y[mask], s=45, marker="D", label="outliers")

    # горизонтальные линии для IQR-границ и квартилей
    for yline, style, label in [(lower, "--", "lower"), (q1, ":", "Q1"),
                                (q3, ":", "Q3"), (upper, "--", "upper")]:
        plt.axhline(yline, linestyle=style, linewidth=1)

        plt.text(x.max()*1.01, yline, label, va="center", fontsize=8)

    idx_out = np.where(mask)[0]
    for i in idx_out[:10]:
        plt.annotate(f"#{i+1}", (x[i], y[i]), textcoords="offset points",
                     xytext=(5, 5), fontsize=8)

    plt.xlabel("Sentence index")
    plt.ylabel("Length (words)")
    plt.title("Sentence lengths with IQR outliers")
    plt.legend()
    plt.tight_layout()
    plt.savefig(scatter_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[info] Диаграмма индексов/длин сохранена в {scatter_path}")

    # Выводим с анализом причин
    print(f"Найдено выбросов: {len(outliers)}\n")
    for idx, s, l in outliers:
        feats = sentence_features(s)
        why = explain_anomaly(l, avg_len, feats)
        print(f"[{idx}] {l} слов | {feats['chars']} символов | причины: {why}")
        print(f"     «{s}»\n")

    # сохраняем их в файл
    try:
        with open("outliers.txt", "w", encoding="utf-8") as f:
            f.write(f"Средняя длина: {avg_len:.2f}\n")
            f.write(f"Q1={q1:.2f}, Q3={q3:.2f}, IQR={iqr:.2f}\n")
            f.write(f"Границы: < {lower:.2f} или > {upper:.2f}\n\n")
            for idx, s, l in outliers:
                feats = sentence_features(s)
                why = explain_anomaly(l, avg_len, feats)
                f.write(f"[{idx}] {l} слов | {feats['chars']} симв. | {why}\n")
                f.write(s.replace("\n", " ") + "\n\n")
        print("[info] Выбросы сохранены в outliers.txt")
    except Exception as e:
        print(f"[warn] Не удалось сохранить outliers.txt: {e}")

if __name__ == "__main__":
    main()
